import logging
import json
import math
import inspect
import re
import sys
from collections.abc import Mapping, Sequence
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import comfy.lora
import comfy.lora_convert
import comfy.model_management
import comfy.utils
import folder_paths
import torch

_LOG = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# DoRA decompose debug config (set per-node run via kwargs)
# --------------------------------------------------------------------------------------

_DORA_DECOMP_CFG: Dict[str, Any] = {
    "dbg": False,
    "dbg_n": 30,
    "dbg_stack": 10,
    "slice_fix": True,
    "adaln_swap_fix": True,
    "call_i": 0,
}


def _set_dora_decomp_cfg(*, dbg: bool, dbg_n: int, dbg_stack: int, slice_fix: bool, adaln_swap_fix: bool) -> None:
    _DORA_DECOMP_CFG["dbg"] = bool(dbg)
    try:
        _DORA_DECOMP_CFG["dbg_n"] = max(0, int(dbg_n))
    except Exception:
        _DORA_DECOMP_CFG["dbg_n"] = 30
    try:
        _DORA_DECOMP_CFG["dbg_stack"] = max(2, min(64, int(dbg_stack)))
    except Exception:
        _DORA_DECOMP_CFG["dbg_stack"] = 10
    _DORA_DECOMP_CFG["slice_fix"] = bool(slice_fix)
    _DORA_DECOMP_CFG["adaln_swap_fix"] = bool(adaln_swap_fix)
    # reset counter each time node runs (so logs are deterministic)
    _DORA_DECOMP_CFG["call_i"] = 0


def _patch_comfy_weight_decompose() -> None:
    """
    Patch ComfyUI DoRA normalization to:
      - normalize using norm(V) where V = W0 + alpha*delta (DoRA definition)
      - reshape dora_scale onto the active norm axis before division so non-square
        targets do not broadcast into an unintended outer product
      - slice dora_scale for sliced qkv offsets (common in Flux2) to prevent axis mismatch blow-ups
      - emit debug logs controlled from node settings (no env vars)
    """
    try:
        import comfy.weight_adapter.base as wa_base  # lazy import (avoid load-order issues)
    except Exception:
        return

    if getattr(wa_base, "_dora_weight_decompose_patched_by_dora_loader", False):
        return

    orig = getattr(wa_base, "weight_decompose", None)
    if orig is None:
        return

    if not hasattr(wa_base, "_dora_weight_decompose_orig_by_dora_loader"):
        wa_base._dora_weight_decompose_orig_by_dora_loader = orig

    def _find_ctx(max_depth: int):
        """
        Try to recover (key, offset) from adapter stack frames without inspect.stack().
        LoRAAdapter.calculate_weight commonly has locals: key, offset.
        """
        key = None
        offset = None
        caller = None
        try:
            f = sys._getframe(2)
        except Exception:
            f = None
        depth = 0
        while f is not None and depth < max_depth:
            loc = getattr(f, "f_locals", {}) or {}
            if caller is None:
                caller = f"{getattr(f, 'f_code', None).co_filename if getattr(f, 'f_code', None) else '?'}:{getattr(f, 'f_lineno', -1)}:{getattr(getattr(f, 'f_code', None), 'co_name', '?')}"
            if key is None and "key" in loc:
                key = loc.get("key")
            if offset is None and "offset" in loc:
                offset = loc.get("offset")
            if key is not None and offset is not None:
                break
            f = getattr(f, "f_back", None)
            depth += 1
        return key, offset, caller

    def weight_decompose_fixed(*args, **kwargs):
        if not getattr(wa_base, "_dora_weight_decompose_first_call_logged", False):
            wa_base._dora_weight_decompose_first_call_logged = True
            _LOG.warning("[DoRA Power LoRA Loader] weight_decompose_fixed invoked (DoRA normalization patch active).")

        if len(args) >= 4:
            dora_scale, weight, lora_diff, alpha = args[:4]
        else:
            dora_scale = kwargs.get("dora_scale")
            weight = kwargs.get("weight")
            lora_diff = kwargs.get("lora_diff")
            alpha = kwargs.get("alpha")
        if dora_scale is None or weight is None or lora_diff is None or alpha is None:
            raise TypeError("weight_decompose_fixed missing required arguments (dora_scale, weight, lora_diff, alpha)")

        strength = args[4] if len(args) >= 5 else kwargs.get("strength", 1.0)
        intermediate_dtype = args[5] if len(args) >= 6 else kwargs.get("intermediate_dtype", getattr(weight, "dtype", torch.float32))
        function = args[6] if len(args) >= 7 else kwargs.get("function")
        if function is None:
            raise TypeError("weight_decompose_fixed missing required argument 'function'")

        cfg = _DORA_DECOMP_CFG
        call_i = int(cfg.get("call_i", 0))
        cfg["call_i"] = call_i + 1
        do_dbg = bool(cfg.get("dbg", False)) and call_i < int(cfg.get("dbg_n", 30))

        # IMPORTANT: do DoRA math in fp32 so tiny LoRA deltas don't underflow to 0 in fp16.
        math_dtype = torch.float32
        dora_scale_local = comfy.model_management.cast_to_device(dora_scale, weight.device, math_dtype)

        try:
            a = float(alpha) if not isinstance(alpha, torch.Tensor) else float(alpha.item())
        except Exception:
            a = 1.0

        # lora_diff_scaled in fp32
        lora_diff_scaled = lora_diff.to(device=weight.device, dtype=math_dtype) * a

        # delta in fp32 (do NOT cast to fp16 here)
        try:
            delta32 = function(lora_diff_scaled)
        except Exception:
            # fallback: keep behavior if some backend insists on fp16, but still measure in fp32
            delta32 = function(lora_diff_scaled.to(dtype=intermediate_dtype)).to(dtype=math_dtype)

        if not isinstance(delta32, torch.Tensor):
            delta32 = torch.as_tensor(delta32, device=weight.device, dtype=math_dtype)
        else:
            delta32 = delta32.to(device=weight.device, dtype=math_dtype)

        weight32 = weight.to(dtype=math_dtype)
        weight_calc32 = weight32 + delta32

        # swap_scale_shift is applied to delta for adaLN_modulation weights.
        # If dora_scale is in unswapped ordering, apply the same swap so magnitude aligns.
        if bool(cfg.get("adaln_swap_fix", True)) and dora_scale_local is not None:
            try:
                fn_name = getattr(function, "__name__", "") or ""
                if "swap_scale_shift" in fn_name:
                    # Apply the exact same transform Comfy uses for this weight.
                    ds = dora_scale_local
                    if ds.ndim == 1:
                        ds2 = ds[:, None]
                        ds2 = function(ds2)
                        ds = ds2[:, 0]
                    else:
                        ds = function(ds)
                    dora_scale_local = ds
                    if do_dbg:
                        _LOG.warning("[DoRA Power LoRA Loader] DoRA dbg[%d] applied adaLN swap_scale_shift fix (fn=%s).", call_i, fn_name)
                else:
                    # Fallback heuristic for builds where function name isn't preserved.
                    ktmp, _, _ = _find_ctx(6)
                    key_hint = ktmp if isinstance(ktmp, str) else ""
                    if ("adaLN_modulation" in key_hint) or ("adaln_modulation" in key_hint.lower()):
                        n0 = int(dora_scale_local.shape[0])
                        if n0 == int(weight_calc32.shape[0]) and (n0 % 2) == 0:
                            h = n0 // 2
                            if dora_scale_local.ndim == 1:
                                dora_scale_local = torch.cat([dora_scale_local[h:], dora_scale_local[:h]], dim=0)
                            else:
                                dora_scale_local = torch.cat([dora_scale_local[h:, ...], dora_scale_local[:h, ...]], dim=0)
                            if do_dbg:
                                _LOG.warning("[DoRA Power LoRA Loader] DoRA dbg[%d] applied adaLN half-swap fallback (N=%d).", call_i, n0)
            except Exception:
                pass

        key_ctx = None
        off_ctx = None
        caller_ctx = None
        if bool(cfg.get("slice_fix", True)):
            try:
                if hasattr(dora_scale_local, "ndim") and int(dora_scale_local.ndim) != 1:
                    need = False
                else:
                    ds0 = int(dora_scale_local.shape[0]) if hasattr(dora_scale_local, "shape") else -1
                    need = (ds0 > 0) and (ds0 not in (int(weight_calc32.shape[0]), int(weight_calc32.shape[1])))
            except Exception:
                need = False
            if need or do_dbg:
                key_ctx, off_ctx, caller_ctx = _find_ctx(int(cfg.get("dbg_stack", 10)))
                if do_dbg and need and (off_ctx is None):
                    _LOG.warning("[DoRA Power LoRA Loader] DoRA dbg[%d] slice-fix needed but offset not found (key=%r).", call_i, key_ctx)
                if isinstance(off_ctx, tuple) and len(off_ctx) >= 2 and dora_scale_local is not None and hasattr(dora_scale_local, "shape"):
                    a, b = off_ctx[0], off_ctx[1]
                    try:
                        a = int(a)
                        b = int(b)
                        ds_len = int(dora_scale_local.shape[0])
                        if 0 <= a < b <= ds_len:
                            if (b - a) == int(weight_calc32.shape[0]):
                                dora_scale_local = dora_scale_local[a:b]
                            elif (b - a) == int(weight_calc32.shape[1]):
                                dora_scale_local = dora_scale_local[a:b]
                    except Exception:
                        pass

        wd_on_output_axis = int(dora_scale_local.shape[0]) == int(weight_calc32.shape[0])

        wc32 = weight_calc32
        if wd_on_output_axis:
            weight_norm = (
                wc32.reshape(wc32.shape[0], -1)
                .norm(dim=1, keepdim=True)
                .reshape(wc32.shape[0], *[1] * (wc32.dim() - 1))
            )
        else:
            weight_norm = (
                wc32.transpose(0, 1)
                .reshape(wc32.shape[1], -1)
                .norm(dim=1, keepdim=True)
                .reshape(wc32.shape[1], *[1] * (wc32.dim() - 1))
                .transpose(0, 1)
            )

        weight_norm = weight_norm + torch.finfo(torch.float32).eps
        if wd_on_output_axis:
            dora_scale_local = dora_scale_local.reshape(wc32.shape[0], *[1] * (wc32.dim() - 1))
        else:
            dora_scale_local = dora_scale_local.reshape(1, wc32.shape[1], *[1] * (wc32.dim() - 2))
        scale32 = dora_scale_local / weight_norm

        weight_dora32 = weight_calc32 * scale32

        if do_dbg:
            try:
                ld_max = float(lora_diff_scaled.abs().max().item())
                ld_rms = float((lora_diff_scaled.pow(2).mean().sqrt()).item())
                d_max = float(delta32.abs().max().item())
                d_rms = float((delta32.pow(2).mean().sqrt()).item())
                w_max = float(weight32.abs().max().item())
                upd_max = float((weight_dora32 - weight32).abs().max().item())
                _LOG.warning(
                    "[DoRA Power LoRA Loader] DoRA dbg[%d] key=%r off=%r axis=%s w=%s ds=%s alpha=%g strength=%g "
                    "lora_diff(max/rms)=%g/%g delta(max/rms)=%g/%g update(max)=%g w(max)=%g delta/w=%g "
                    "norm(min/max)=%g/%g scale(max)=%g wc(max)=%g caller=%s",
                    call_i,
                    key_ctx,
                    off_ctx,
                    ("out" if wd_on_output_axis else "in"),
                    tuple(weight_calc32.shape),
                    tuple(dora_scale_local.shape),
                    float(a),
                    float(strength) if not isinstance(strength, torch.Tensor) else float(strength.item()),
                    ld_max,
                    ld_rms,
                    d_max,
                    d_rms,
                    upd_max,
                    w_max,
                    (d_max / max(w_max, 1e-12)),
                    float(weight_norm.min().item()),
                    float(weight_norm.max().item()),
                    float(scale32.abs().max().item()),
                    float(wc32.abs().max().item()),
                    caller_ctx,
                )
            except Exception:
                pass

        try:
            s = float(strength) if not isinstance(strength, torch.Tensor) else float(strength.item())
        except Exception:
            s = 1.0
        if s != 1.0:
            out32 = weight32 + s * (weight_dora32 - weight32)
        else:
            out32 = weight_dora32
        weight[:] = out32.to(dtype=weight.dtype)
        return weight

    wa_base.weight_decompose = weight_decompose_fixed
    wa_base._dora_weight_decompose_patched_by_dora_loader = True
    _LOG.warning("[DoRA Power LoRA Loader] patched ComfyUI weight_decompose for correct DoRA normalization (norm(V) + broadcast-shape + slice fix).")

    patched_refs = 0
    for m in list(sys.modules.values()):
        if m is None:
            continue
        try:
            if getattr(m, "weight_decompose", None) is orig:
                setattr(m, "weight_decompose", weight_decompose_fixed)
                patched_refs += 1
        except Exception:
            pass
    if patched_refs:
        _LOG.warning("[DoRA Power LoRA Loader] patched %d cached weight_decompose references across sys.modules.", patched_refs)


def _patch_comfy_lora_calculate_weight_fp32() -> None:
    """
    Force fp32 intermediate matmul path inside weight_adapter.lora calculate_weight().

    Some mixed-precision/quantized stacks can flush tiny LoRA products to zero while building
    lora_diff. Force intermediate_dtype=torch.float32 before lora_diff is computed.

    This patch tries to stay robust across Comfy variants:
      - if signature has intermediate_dtype, set by position OR kwarg (without double-pass)
      - otherwise replace any positional torch.dtype argument with torch.float32
      - patch all adapter classes in comfy.weight_adapter.lora that expose calculate_weight
    """
    try:
        import comfy.weight_adapter.lora as wa_lora
    except Exception:
        return

    if getattr(wa_lora, "_dora_loader_patched_calc_weight_fp32", False):
        return

    patched = 0

    def _wrap_calc(orig):
        try:
            sig = inspect.signature(orig)
            param_names = list(sig.parameters.keys())
            has_intermediate = "intermediate_dtype" in sig.parameters
            idx_intermediate = param_names.index("intermediate_dtype") if has_intermediate else -1

            # calculate_weight is an instance method; callers pass args *without* the leading
            # `self`. When patching by positional index, compensate for the missing self slot so
            # we overwrite the actual intermediate_dtype argument instead of original_weight.
            idx_intermediate_call = idx_intermediate
            if has_intermediate and param_names and param_names[0] == "self":
                idx_intermediate_call = idx_intermediate - 1
        except Exception:
            sig = None
            has_intermediate = False
            idx_intermediate = -1
            idx_intermediate_call = -1

        def calculate_weight_fixed(self, *args, **kwargs):
            a = list(args)

            if has_intermediate and idx_intermediate >= 0:
                # If intermediate_dtype was provided positionally, overwrite that slot and
                # DO NOT also pass it as a kwarg (would be "multiple values").
                if idx_intermediate_call >= 0 and len(a) > idx_intermediate_call:
                    a[idx_intermediate_call] = torch.float32
                    # If upstream also set it as a kwarg, drop it to avoid duplicates.
                    if "intermediate_dtype" in kwargs:
                        kwargs.pop("intermediate_dtype", None)
                else:
                    # Otherwise, force via kwarg.
                    kwargs["intermediate_dtype"] = torch.float32
            else:
                for i, v in enumerate(a):
                    if isinstance(v, torch.dtype):
                        a[i] = torch.float32
                if sig is not None and "intermediate_dtype" in getattr(sig, "parameters", {}):
                    kwargs["intermediate_dtype"] = torch.float32

            return orig(self, *a, **kwargs)

        return calculate_weight_fixed

    for _, obj in list(wa_lora.__dict__.items()):
        try:
            if not isinstance(obj, type):
                continue
            orig = getattr(obj, "calculate_weight", None)
            if orig is None or not callable(orig):
                continue
            setattr(obj, "calculate_weight", _wrap_calc(orig))
            patched += 1
        except Exception:
            continue

    if patched:
        wa_lora._dora_loader_patched_calc_weight_fp32 = True
        _LOG.warning(
            "[DoRA Power LoRA Loader] patched %d weight_adapter.lora calculate_weight() methods: forcing fp32 intermediate_dtype "
            "(fixes lora_diff flush-to-zero on mixed-precision/quantized models).",
            patched,
        )


_patch_comfy_weight_decompose()
_patch_comfy_lora_calculate_weight_fp32()

# --------------------------------------------------------------------------------------
# Flexible optional inputs (Power LoRA Loader-style)
# --------------------------------------------------------------------------------------


class AnyType(str):
    # ComfyUI type checks typically use != comparisons; returning False makes this "match anything".
    def __ne__(self, __value: object) -> bool:  # noqa: D401
        return False


any_type = (AnyType("*"),)


class FlexibleOptionalInputType(dict):
    """
    Dict-like object that claims it contains any key, and returns a fallback type spec for unknown keys.
    This enables dynamic widgets / arbitrary optional inputs (rgthree Power LoRA Loader pattern).
    """

    def __init__(self, fallback_type, data: Union[dict, None] = None):
        super().__init__(data or {})
        self._fallback_type = fallback_type

    def __contains__(self, key: object) -> bool:
        return True

    def __getitem__(self, key: str):
        if dict.__contains__(self, key):
            return dict.__getitem__(self, key)
        return self._fallback_type


_LORA_MAGNITUDE_VECTOR_RE = re.compile(
    r"^(?P<base>.+?)\.lora_magnitude_vector(?:\.(?P<adapter>[A-Za-z0-9_-]+))?(?:\.weight)?$"
)


def _normalize_diffusers_dora_magnitude_keys(lora_sd: Dict[str, Any], verbose: bool = False) -> int:
    """
    Normalize PEFT/Diffusers DoRA magnitude keys into ComfyUI-style `.dora_scale` keys.

    Examples that get rewritten:
      - `... .lora_magnitude_vector`
      - `... .lora_magnitude_vector.weight`
      - `... .lora_magnitude_vector.default`
      - `... .lora_magnitude_vector.default.weight`
      - `... .lora_magnitude_vector.default_0`
      - `... .lora_magnitude_vector.default_0.weight`

    Diffusers/PEFT commonly stores DoRA magnitude tensors under `lora_magnitude_vector`, while
    ComfyUI's LoRA loader expects `dora_scale`. If these keys are not normalized, the direction
    matrices may load but the DoRA magnitude vectors are left behind as "unloaded keys".
    """
    renamed = 0
    collisions = 0
    examples: List[str] = []
    for key in list(lora_sd.keys()):
        m = _LORA_MAGNITUDE_VECTOR_RE.match(str(key))
        if not m:
            continue
        new_key = m.group("base") + ".dora_scale"
        value = lora_sd[key]
        if new_key in lora_sd:
            if new_key != key:
                collisions += 1
                if len(examples) < 10:
                    examples.append(f"collision {key} -> {new_key}")
            lora_sd.pop(key, None)
            continue
        lora_sd[new_key] = value
        if new_key != key:
            lora_sd.pop(key, None)
            renamed += 1
            if len(examples) < 10:
                examples.append(f"{key} -> {new_key}")
    if verbose and (renamed or collisions):
        _LOG.info(
            "[DoRA Power LoRA Loader] normalized diffusers/PEFT DoRA magnitude keys: renamed=%d collisions=%d",
            renamed,
            collisions,
        )
        for ex in examples:
            _LOG.info("[DoRA Power LoRA Loader] magnitude-key normalize example: %s", ex)
    return renamed


# Key suffixes we treat as "belongs to base module X"
_BASE_SUFFIXES = [
    ".lora_up.weight",
    ".lora_down.weight",
    ".lora_A.weight",
    ".lora_B.weight",
    ".lora_A.default.weight",
    ".lora_B.default.weight",
    "_lora.up.weight",
    "_lora.down.weight",
    ".lora.up.weight",
    ".lora.down.weight",
    ".lora_linear_layer.up.weight",
    ".lora_linear_layer.down.weight",
    ".lora_B",  # mochi style
    ".lora_A",  # mochi style
    ".alpha",
    ".dora_scale",
    ".w_norm",
    ".b_norm",
    ".diff",
    ".diff_b",
    ".set_weight",
    ".reshape_weight",
]

_SCALEABLE_SUFFIXES = (
    ".alpha",
    ".diff",
    ".diff_b",
    ".set_weight",
    ".reshape_weight",
    # mochi-style (no .weight suffix)
    ".lora_A",
    ".lora_B",
)

# When we need to linearly scale a LoRA's *effect* (delta), scaling BOTH matrices is quadratic.
# For broadcasts we scale only the "up" side (or alpha when present) to keep scaling linear.
_UP_ONLY_SCALE_SUFFIXES = (
    ".lora_up.weight",
    ".lora_A.weight",
    ".lora_A.default.weight",
    "_lora.up.weight",
    ".lora.up.weight",
    ".lora_linear_layer.up.weight",
    # mochi-style (no .weight suffix)
    ".lora_A",
)

_LORA_DIRECTION_SUFFIX_PAIRS = (
    (".lora_up.weight", ".lora_down.weight"),
    (".lora_B.weight", ".lora_A.weight"),
    (".lora_B.default.weight", ".lora_A.default.weight"),
    ("_lora.up.weight", "_lora.down.weight"),
    (".lora.up.weight", ".lora.down.weight"),
    (".lora_linear_layer.up.weight", ".lora_linear_layer.down.weight"),
    # mochi-style (no .weight suffix)
    (".lora_B", ".lora_A"),
)

# Only broadcast true LoRA "delta" parameters.
# IMPORTANT: do NOT broadcast DoRA-only params like dora_scale / w_norm / b_norm by default.
_BROADCAST_DELTA_SUFFIXES = (
    ".lora_up.weight",
    ".lora_down.weight",
    ".lora_A.weight",
    ".lora_B.weight",
    ".lora_A.default.weight",
    ".lora_B.default.weight",
    "_lora.up.weight",
    "_lora.down.weight",
    ".lora.up.weight",
    ".lora.down.weight",
    ".lora_linear_layer.up.weight",
    ".lora_linear_layer.down.weight",
    # mochi style
    ".lora_A",
    ".lora_B",
    # scalar strength
    ".alpha",
    # some exporters use diff-style deltas
    ".diff",
    ".diff_b",
    ".set_weight",
    ".reshape_weight",
)

# For OneTrainer DoRA exports, modulation modules often include DoRA params.
# If we broadcast only A/B/alpha and then delete the source prefix, we effectively strip DoRA
# for these modules -> can destabilize / cause pink outputs.
_BROADCAST_DORA_SUFFIXES = _BROADCAST_DELTA_SUFFIXES + (
    ".dora_scale",
    ".w_norm",
    ".b_norm",
)

_AUTO_STRENGTH_RATIO_FLOOR = 0.30
_AUTO_STRENGTH_RATIO_CEILING = 1.50
_AUTO_STRENGTH_DISPLAY_RATIO_EPS = 1e-3
_AUTO_STRENGTH_EPS = 1e-8
_AUTO_STRENGTH_ANALYSIS_MIN_NUMEL = 65536


class _AutoStrengthAnalysisDeviceError(RuntimeError):
    pass


def _normalize_auto_strength_device(value: Any) -> str:
    try:
        mode = str(value).strip().lower()
    except Exception:
        return "auto"
    return mode if mode in ("auto", "cpu", "gpu") else "auto"


def _torch_device_or_none(value: Any) -> Optional[torch.device]:
    try:
        device = torch.device(value)
    except Exception:
        return None
    if device.type == "meta":
        return None
    return device


def _torch_device_available(device: Optional[torch.device]) -> bool:
    if device is None:
        return False
    try:
        resolved = torch.device(device)
    except Exception:
        return False
    if resolved.type == "cpu":
        return False
    if resolved.type == "cuda":
        return bool(torch.cuda.is_available())
    if resolved.type == "xpu":
        xpu = getattr(torch, "xpu", None)
        return bool(getattr(xpu, "is_available", lambda: False)())
    if resolved.type == "mps":
        backends = getattr(torch, "backends", None)
        mps = getattr(backends, "mps", None)
        return bool(getattr(mps, "is_available", lambda: False)())
    return True


def _auto_strength_cast_float32(tensor: torch.Tensor, analysis_device: Optional[torch.device] = None) -> Optional[torch.Tensor]:
    if not isinstance(tensor, torch.Tensor):
        return None
    cpu = torch.device("cpu")
    target = analysis_device if analysis_device is not None else cpu
    try:
        if target.type == "cpu":
            # Materialize owned CPU float32 storage before any analysis work.
            return tensor.to(device=cpu, dtype=torch.float32, copy=True)

        if tensor.device.type == "cpu":
            # File-backed safetensor storage must be owned on CPU before H2D copies.
            cpu_owned = tensor.to(device=cpu, dtype=torch.float32, copy=True)
            return cpu_owned.to(device=target, dtype=torch.float32, non_blocking=False, copy=True)

        if tensor.dtype == torch.float32:
            try:
                if tensor.device == target:
                    return tensor
            except Exception:
                pass

        return tensor.to(device=target, dtype=torch.float32, non_blocking=False, copy=False)
    except _AutoStrengthAnalysisDeviceError:
        raise
    except Exception as exc1:
        try:
            if target.type == "cpu":
                return tensor.to(device=cpu, dtype=torch.float32, copy=True)
            cpu_owned = tensor.to(device=cpu, dtype=torch.float32, copy=True)
            return cpu_owned.to(device=target, dtype=torch.float32, non_blocking=False, copy=True)
        except Exception as exc2:
            if _auto_strength_is_device_failure(exc1, target) or _auto_strength_is_device_failure(exc2, target):
                raise _AutoStrengthAnalysisDeviceError from None
            return None


def _auto_strength_is_device_failure(exc: BaseException, analysis_device: Optional[torch.device]) -> bool:
    if analysis_device is None or getattr(analysis_device, "type", "cpu") == "cpu":
        return False
    msg = str(exc).lower()
    return any(
        token in msg
        for token in (
            "out of memory",
            "cuda",
            "xpu",
            "mps",
            "hip",
            "device-side assert",
            "cublas",
            "cudnn",
        )
    )


def _auto_strength_resolve_analysis_device(
    analysis_device_mode: str,
    load_device: Any,
    weight: Optional[torch.Tensor] = None,
) -> torch.device:
    mode = _normalize_auto_strength_device(analysis_device_mode)
    cpu = torch.device("cpu")
    model_device = _torch_device_or_none(load_device)
    if mode in ("auto", "cpu"):
        return cpu
    if mode == "gpu":
        if not _torch_device_available(model_device):
            return cpu
        return model_device if model_device is not None else cpu
    return cpu


def _auto_strength_get_analysis_load_device(model: Any, clip: Any = None) -> Any:
    cpu_fallback = None
    for root in (model, clip):
        for candidate in (
            root,
            getattr(root, "model", None),
            getattr(root, "diffusion_model", None),
            getattr(root, "cond_stage_model", None),
        ):
            if candidate is None:
                continue
            load_device = getattr(candidate, "load_device", None)
            if load_device is not None:
                device = _torch_device_or_none(load_device)
                if device is None:
                    continue
                if device.type != "cpu" and _torch_device_available(device):
                    return device
                if cpu_fallback is None:
                    cpu_fallback = device
    return cpu_fallback


def _auto_strength_safe_number(value: Any) -> Optional[float]:
    try:
        v = float(value)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def _auto_strength_describe_device(value: Any) -> str:
    device = _torch_device_or_none(value)
    if device is not None:
        return str(device)
    if value is None:
        return "cpu"
    try:
        return str(value)
    except Exception:
        return "unknown"


def _auto_strength_json_dumps(value: Any, *, pretty: bool = False) -> str:
    kwargs = {"ensure_ascii": False, "sort_keys": False}
    if pretty:
        kwargs["indent"] = 2
    else:
        kwargs["separators"] = (",", ":")
    return json.dumps(value, **kwargs)


def _src_has_dora_params(lora_sd: Dict[str, Any], base: str) -> bool:
    p = base + "."
    for k in lora_sd.keys():
        if k.startswith(p) and (k.endswith(".dora_scale") or k.endswith(".w_norm") or k.endswith(".b_norm")):
            return True
    return False


def _keymap_dest_key(v: Any) -> str:
    """
    Normalize a key_map value into a comparable "destination" key so we can dedupe alias bases.
    key_map values can be:
      - "some.weight"
      - ("some.weight", slice_tuple)
      - ("some.weight", None, patch_fn)
    """
    try:
        if v is None:
            return "__NONE__"
        if isinstance(v, tuple) and len(v) > 0:
            # include slice info when present to avoid merging genuinely distinct sliced mappings
            dest0 = str(v[0])
            if dest0.endswith(".weight"):
                dest0 = dest0[:-7]
            sl = None
            if len(v) > 1 and isinstance(v[1], tuple):
                sl = v[1]
            return f"{dest0}|{sl}" if sl is not None else str(dest0)
        dest0 = str(v)
        if dest0.endswith(".weight"):
            dest0 = dest0[:-7]
        return dest0
    except Exception:
        return repr(v)


def _target_preference(base: str) -> int:
    """
    Lower is better. Prefer canonical lora_unet_* bases over diffusion_model.* aliases.
    """
    if base.startswith("lora_unet_"):
        return 0
    if base.startswith("lora_"):
        return 1
    if base.startswith("diffusion_model."):
        return 2
    return 3


def _dedupe_targets_by_dest(key_map: Dict[str, Any], targets: List[str]) -> List[str]:
    """
    Dedupe target bases that map to the same destination in key_map (alias bases).
    Keep the most preferred base among aliases.
    Preserve original order as much as possible.
    """
    best_for_dest: Dict[str, str] = {}
    for b in targets:
        dest = _keymap_dest_key(key_map.get(b))
        cur = best_for_dest.get(dest)
        if cur is None or _target_preference(b) < _target_preference(cur):
            best_for_dest[dest] = b

    chosen = set(best_for_dest.values())
    out: List[str] = []
    for b in targets:
        if b in chosen and b not in out:
            out.append(b)
    # ensure preference within the preserved order if multiple kept
    out.sort(key=lambda x: (_target_preference(x), targets.index(x)))
    return out


def _delete_prefix_keys(lora_sd: Dict[str, Any], prefix: str) -> int:
    """
    Deletes all keys that start with prefix.
    Returns number of keys deleted.
    """
    to_del = [k for k in lora_sd.keys() if k.startswith(prefix)]
    for k in to_del:
        lora_sd.pop(k, None)
    return len(to_del)


def _rename_prefix_keys(lora_sd: Dict[str, Any], from_prefix: str, to_prefix: str, delete_from: bool = False) -> int:
    """
    Rename all keys that start with from_prefix to to_prefix + rest.
    Returns number of keys created.
    """
    created = 0
    keys = list(lora_sd.keys())
    for k in keys:
        if not k.startswith(from_prefix):
            continue
        nk = to_prefix + k[len(from_prefix) :]
        if nk not in lora_sd:
            lora_sd[nk] = lora_sd[k]
            created += 1
        if delete_from:
            lora_sd.pop(k, None)
    return created


def _clone_base_block(
    lora_sd: Dict[str, Any],
    from_base: str,
    to_base: str,
    scale: float = 1.0,
    allow_suffixes: Optional[Tuple[str, ...]] = None,
) -> int:
    """
    Clone all entries under from_base.* to to_base.* (by prefix), preserving suffixes.
    Returns number of keys created.
    """
    created = 0
    prefix = from_base + "."
    keys = list(lora_sd.keys())
    # If alpha exists, scale ONLY alpha (linear). Otherwise scale only "up" side.
    has_alpha = any(k.startswith(prefix) and k.endswith(".alpha") for k in keys)
    for k in keys:
        if not k.startswith(prefix):
            continue
        if allow_suffixes is not None and not k.endswith(allow_suffixes):
            continue
        nk = to_base + k[len(from_base) :]
        if nk in lora_sd:
            continue
        v = lora_sd[k]
        if isinstance(v, torch.Tensor):
            # IMPORTANT:
            # - Do NOT scale DoRA magnitude vectors (dora_scale). Those are not a delta.
            # - Do NOT scale BOTH LoRA matrices; that changes strength quadratically.
            # - Prefer scaling alpha, otherwise scale only the "up" side.
            vv = v
            if scale != 1.0:
                if k.endswith(".alpha"):
                    vv = v * scale
                elif (not has_alpha) and k.endswith(_UP_ONLY_SCALE_SUFFIXES):
                    vv = v * scale
            # Always clone broadcasted tensors to avoid any in-place casts/mutations downstream
            lora_sd[nk] = vv.clone()
        else:
            lora_sd[nk] = v
        created += 1
    return created


# --------------------------------------------------------------------------------------
# Z-Image Turbo / Lumina2 compatibility
# --------------------------------------------------------------------------------------

_ZIMAGE_QKV_COMPONENTS = ("to_q", "to_k", "to_v")

_ZIMAGE_QKV_MATRIX_FAMILIES = (
    {
        "name": "diffusers2",
        "up_suffix": ".lora_B.weight",
        "down_suffix": ".lora_A.weight",
    },
    {
        "name": "diffusers2_default",
        "up_suffix": ".lora_B.default.weight",
        "down_suffix": ".lora_A.default.weight",
    },
    {
        "name": "regular",
        "up_suffix": ".lora_up.weight",
        "down_suffix": ".lora_down.weight",
    },
    {
        "name": "diffusers1",
        "up_suffix": "_lora.up.weight",
        "down_suffix": "_lora.down.weight",
    },
    {
        "name": "diffusers3",
        "up_suffix": ".lora.up.weight",
        "down_suffix": ".lora.down.weight",
    },
    {
        "name": "transformers",
        "up_suffix": ".lora_linear_layer.up.weight",
        "down_suffix": ".lora_linear_layer.down.weight",
    },
)

_ZIMAGE_QKV_CAT_SUFFIXES = (
    ".dora_scale",
    ".w_norm",
    ".b_norm",
    ".diff",
    ".diff_b",
    ".set_weight",
    ".reshape_weight",
)

_ZIMAGE_ATTN_ALIAS_REWRITES = (
    (".attention.to.q.", ".attention.to_q."),
    (".attention.to.k.", ".attention.to_k."),
    (".attention.to.v.", ".attention.to_v."),
    (".attention.to.out.0.", ".attention.to_out.0."),
    (".attention.to.out.", ".attention.to_out."),
)

_ZIMAGE_UNDERSCORE_PREFIX_REWRITES = (
    ("lora_unet_", ""),
    ("lycoris_", ""),
    ("diffusion_model_", "diffusion_model."),
    ("base_model_model_", "base_model.model."),
    ("base_model_", "base_model."),
    ("transformer_", "transformer."),
    ("model_", "model."),
    ("unet_", "unet."),
)

_ZIMAGE_UNDERSCORE_ATTN_REWRITES = (
    (".attention_to_q", ".attention.to_q"),
    ("_attention_to_q", ".attention.to_q"),
    (".attention_to_k", ".attention.to_k"),
    ("_attention_to_k", ".attention.to_k"),
    (".attention_to_v", ".attention.to_v"),
    ("_attention_to_v", ".attention.to_v"),
    (".attention_to_out_0", ".attention.to_out.0"),
    ("_attention_to_out_0", ".attention.to_out.0"),
    (".attention_out", ".attention.out"),
    ("_attention_out", ".attention.out"),
)


def _looks_like_zimage_lumina2_model(model, model_sd_keys: Optional[Set[str]] = None) -> bool:
    """
    Best-effort detection for Z-Image Turbo / Lumina2 architectures.
    Avoid relying on a specific ComfyUI class existing across builds.
    """
    try:
        model_core = getattr(model, "model", model)
        cls_name = type(model_core).__name__.lower()
        if ("lumina2" in cls_name) or ("zimage" in cls_name) or ("z_image" in cls_name):
            return True
    except Exception:
        pass

    keys = model_sd_keys
    if keys is None:
        try:
            sd = getattr(getattr(model, "model", model), "state_dict", None)
            if callable(sd):
                keys = set(sd().keys())
        except Exception:
            keys = None

    if not keys:
        return False

    has_qkv = any(k.startswith("diffusion_model.layers.") and ".attention.qkv.weight" in k for k in keys)
    has_out = any(k.startswith("diffusion_model.layers.") and ".attention.out.weight" in k for k in keys)
    has_ff = any(k.startswith("diffusion_model.layers.") and ".feed_forward.w1.weight" in k for k in keys)
    has_adaln = any(k.startswith("diffusion_model.layers.") and ".adaLN_modulation." in k for k in keys)
    return has_qkv and has_out and (has_ff or has_adaln)


def _looks_like_zimage_attention_lora(lora_sd: Dict[str, Any]) -> bool:
    for k in lora_sd.keys():
        ks = _normalize_zimage_attention_key_string(str(k))
        has_layers = ks.startswith("layers.") or ".layers." in ks
        if not has_layers or ".attention." not in ks:
            continue
        if (
            ".attention.qkv." in ks
            or ".attention.out." in ks
            or ".attention.to_q." in ks
            or ".attention.to_k." in ks
            or ".attention.to_v." in ks
            or ".attention.to.q." in ks
            or ".attention.to.k." in ks
            or ".attention.to.v." in ks
            or ".attention.to_out.0." in ks
            or ".attention.to.out.0." in ks
        ):
            return True
    return False


def _normalize_zimage_attention_key_string(key: str) -> str:
    nk = key
    for old, new in _ZIMAGE_ATTN_ALIAS_REWRITES:
        if old in nk:
            nk = nk.replace(old, new)

    while True:
        changed = False
        for old, new in _ZIMAGE_UNDERSCORE_PREFIX_REWRITES:
            if nk.startswith(old):
                nk = new + nk[len(old) :]
                changed = True
                break
        if not changed:
            break

    nk = re.sub(r"(^|[._])layers_(\d+)_", lambda m: f"{m.group(1)}layers.{m.group(2)}.", nk)

    for old, new in _ZIMAGE_UNDERSCORE_ATTN_REWRITES:
        if old in nk:
            nk = nk.replace(old, new)

    return nk


def _zimage_add_key_aliases(key_map: Dict[str, Any], base: str, target: Any) -> int:
    aliases: List[str] = []

    def _add_alias(alias: Optional[str]) -> None:
        if alias and alias not in aliases:
            aliases.append(alias)

    _add_alias(base)
    if base.endswith(".weight"):
        _add_alias(base[: -len(".weight")])

    base_no_weight = base[: -len(".weight")] if base.endswith(".weight") else base
    if base_no_weight.startswith("diffusion_model."):
        stem = base_no_weight[len("diffusion_model.") :]
        stem_u = stem.replace(".", "_")
        for alias in (
            stem,
            f"diffusion_model.{stem}",
            f"transformer.{stem}",
            f"base_model.model.{stem}",
            f"model.{stem}",
            f"unet.{stem}",
            f"lora_unet_{stem_u}",
            f"lycoris_{stem_u}",
        ):
            _add_alias(alias)

    created = 0
    for alias in aliases:
        if alias not in key_map:
            key_map[alias] = target
            created += 1
    return created


def _augment_key_map_with_zimage_lumina2_aliases(
    key_map: Dict[str, Any],
    model,
    model_sd_keys: Optional[Set[str]],
    verbose: bool = False,
) -> int:
    """
    Add exact ZiT/Lumina2 aliases into key_map.

    This complements ComfyUI's generic map with aliases commonly found in trainer exports:
      - transformer.*
      - base_model.model.*
      - bare bases
      - lora_unet_* / lycoris_*
    """
    added = 0

    # First try ComfyUI's dedicated mapper if present in this build.
    z_to_diffusers = getattr(comfy.utils, "z_image_to_diffusers", None)
    if callable(z_to_diffusers):
        try:
            model_core = getattr(model, "model", model)
            model_cfg = getattr(model_core, "model_config", None)
            unet_cfg = getattr(model_cfg, "unet_config", None)
            if unet_cfg is not None:
                diffusers_keys = z_to_diffusers(unet_cfg, output_prefix="diffusion_model.")
                for diff_key, target in diffusers_keys.items():
                    if not str(diff_key).endswith(".weight"):
                        continue
                    added += _zimage_add_key_aliases(key_map, str(diff_key), target)
                if verbose and diffusers_keys:
                    _LOG.info(
                        "[DoRA Power LoRA Loader] ZiT/Lumina2 compat: added %s key aliases via z_image_to_diffusers().",
                        added,
                    )
        except Exception as e:
            if verbose:
                _LOG.warning(
                    "[DoRA Power LoRA Loader] ZiT/Lumina2 compat: z_image_to_diffusers() failed (%r); using state_dict fallback.",
                    e,
                )

    # Fallback / supplement from the actual live state_dict keys.
    if model_sd_keys:
        for k in model_sd_keys:
            ks = str(k)
            if not ks.startswith("diffusion_model.layers.") or not ks.endswith(".weight"):
                continue
            added += _zimage_add_key_aliases(key_map, ks, ks)

    if verbose and added:
        _LOG.info("[DoRA Power LoRA Loader] ZiT/Lumina2 compat: total key aliases added=%s", added)

    return added


def _make_scalar_tensor_like(value: float, ref: Optional[torch.Tensor]) -> torch.Tensor:
    if isinstance(ref, torch.Tensor):
        return torch.tensor(float(value), dtype=ref.dtype, device=ref.device)
    return torch.tensor(float(value), dtype=torch.float32)


def _tensor_scalar_to_float(v: Any, default: float = 1.0) -> float:
    try:
        if isinstance(v, torch.Tensor):
            return float(v.item())
        return float(v)
    except Exception:
        return float(default)


def _normalize_zimage_attention_component_aliases(lora_sd: Dict[str, Any], verbose: bool = False) -> int:
    created = 0
    keys = list(lora_sd.keys())
    for k in keys:
        ks = str(k)
        nk = _normalize_zimage_attention_key_string(ks)
        if nk != ks and nk not in lora_sd:
            v = lora_sd[k]
            lora_sd[nk] = v.clone() if isinstance(v, torch.Tensor) else v
            created += 1
    if verbose and created:
        _LOG.info("[DoRA Power LoRA Loader] ZiT/Lumina2 compat: normalized %s attention-key aliases.", created)
    return created


def _cat_dim0_if_compatible(tensors: Sequence[torch.Tensor]) -> Optional[torch.Tensor]:
    if not tensors:
        return None
    first = tensors[0]
    if not isinstance(first, torch.Tensor):
        return None

    prepared: List[torch.Tensor] = []
    tail = tuple(first.shape[1:])
    for t in tensors:
        if not isinstance(t, torch.Tensor):
            return None
        if t.ndim != first.ndim:
            return None
        if tuple(t.shape[1:]) != tail:
            return None
        if t.device != first.device or t.dtype != first.dtype:
            t = t.to(device=first.device, dtype=first.dtype)
        prepared.append(t)

    try:
        return torch.cat(prepared, dim=0)
    except Exception:
        return None


def _collect_zimage_attention_bases(lora_sd: Dict[str, Any]) -> Set[str]:
    bases: Set[str] = set()
    pat = re.compile(r"^(?P<base>.+\.attention)\.(?:to_q|to_k|to_v|qkv|to_out(?:\.0)?|out)\.")
    for k in lora_sd.keys():
        m = pat.match(str(k))
        if m:
            bases.add(m.group("base"))
    return bases


def _fuse_zimage_attention_qkv_for_family(
    lora_sd: Dict[str, Any],
    attention_base: str,
    up_suffix: str,
    down_suffix: str,
    family_name: str,
    verbose: bool = False,
) -> Tuple[int, int]:
    """
    Represent split Q/K/V LoRAs against a fused QKV weight as a single larger-rank LoRA.

    For each component i ∈ {Q,K,V} with delta_i = alpha_i * up_i @ down_i,
    the fused target is represented exactly as:

        fused_up   = block_diag(alpha_q*up_q, alpha_k*up_k, alpha_v*up_v)
        fused_down = cat([down_q, down_k, down_v], dim=0)

    yielding:
        fused_up @ fused_down = cat([delta_q, delta_k, delta_v], dim=0)

    This preserves each component independently and avoids the incorrect “just concatenate both
    matrices” shortcut, which is not mathematically equivalent for standard LoRA factorization.
    """
    fused_base = f"{attention_base}.qkv"
    fused_up_key = fused_base + up_suffix
    fused_down_key = fused_base + down_suffix

    if fused_up_key in lora_sd and fused_down_key in lora_sd:
        return (0, 0)

    comp_rows = []
    any_present = False
    processed: List[str] = []

    for comp in _ZIMAGE_QKV_COMPONENTS:
        comp_base = f"{attention_base}.{comp}"
        up_key = comp_base + up_suffix
        down_key = comp_base + down_suffix
        if up_key in lora_sd or down_key in lora_sd:
            any_present = True
        if up_key not in lora_sd or down_key not in lora_sd:
            if any_present and verbose:
                _LOG.warning(
                    "[DoRA Power LoRA Loader] ZiT/Lumina2 compat: incomplete QKV family %s under %s (missing %s or %s); leaving split keys untouched.",
                    family_name,
                    attention_base,
                    up_key,
                    down_key,
                )
            return (0, 0)

        up = lora_sd[up_key]
        down = lora_sd[down_key]
        if not isinstance(up, torch.Tensor) or not isinstance(down, torch.Tensor) or up.ndim != 2 or down.ndim != 2:
            if verbose:
                _LOG.warning(
                    "[DoRA Power LoRA Loader] ZiT/Lumina2 compat: cannot fuse %s %s because tensors are not 2D LoRA matrices.",
                    attention_base,
                    family_name,
                )
            return (0, 0)

        if int(up.shape[1]) != int(down.shape[0]):
            if verbose:
                _LOG.warning(
                    "[DoRA Power LoRA Loader] ZiT/Lumina2 compat: rank mismatch for %s %s (%s vs %s); leaving split keys untouched.",
                    attention_base,
                    family_name,
                    tuple(up.shape),
                    tuple(down.shape),
                )
            return (0, 0)

        alpha_key = comp_base + ".alpha"
        alpha = _tensor_scalar_to_float(lora_sd.get(alpha_key, 1.0), default=1.0)

        up_scaled = up if alpha == 1.0 else (up * alpha)
        comp_rows.append((comp, up_scaled, down, alpha_key))
        processed.extend([up_key, down_key])
        if alpha_key in lora_sd:
            processed.append(alpha_key)

    if not any_present:
        return (0, 0)

    in_dim = int(comp_rows[0][2].shape[1])
    up_ref = comp_rows[0][1]
    down_ref = comp_rows[0][2]
    prepared_ups: List[torch.Tensor] = []
    prepared_downs: List[torch.Tensor] = []
    for _, up, down, _ in comp_rows:
        if int(down.shape[1]) != in_dim:
            if verbose:
                _LOG.warning(
                    "[DoRA Power LoRA Loader] ZiT/Lumina2 compat: input-dim mismatch while fusing %s %s; leaving split keys untouched.",
                    attention_base,
                    family_name,
                )
            return (0, 0)
        if up.device != up_ref.device or up.dtype != up_ref.dtype:
            up = up.to(device=up_ref.device, dtype=up_ref.dtype)
        if down.device != down_ref.device or down.dtype != down_ref.dtype:
            down = down.to(device=down_ref.device, dtype=down_ref.dtype)
        prepared_ups.append(up)
        prepared_downs.append(down)

    try:
        fused_up = torch.block_diag(*prepared_ups)
        fused_down = torch.cat(prepared_downs, dim=0)
    except Exception as e:
        if verbose:
            _LOG.warning(
                "[DoRA Power LoRA Loader] ZiT/Lumina2 compat: block-diag fusion failed for %s %s (%r); leaving split keys untouched.",
                attention_base,
                family_name,
                e,
            )
        return (0, 0)

    lora_sd[fused_up_key] = fused_up
    lora_sd[fused_down_key] = fused_down
    lora_sd[fused_base + ".alpha"] = _make_scalar_tensor_like(1.0, fused_down)

    created = 3

    # Fuse any per-output first-dimension-attached auxiliary tensors when present.
    for suffix in _ZIMAGE_QKV_CAT_SUFFIXES:
        fused_aux_key = fused_base + suffix
        if fused_aux_key in lora_sd:
            continue

        comp_keys = [f"{attention_base}.{comp}{suffix}" for comp in _ZIMAGE_QKV_COMPONENTS]
        present = [k for k in comp_keys if k in lora_sd]
        if not present:
            continue
        if len(present) != 3:
            if verbose:
                _LOG.warning(
                    "[DoRA Power LoRA Loader] ZiT/Lumina2 compat: partial auxiliary QKV tensors for %s%s; leaving originals untouched.",
                    attention_base,
                    suffix,
                )
            continue

        fused_aux = _cat_dim0_if_compatible([lora_sd[k] for k in comp_keys])
        if fused_aux is None:
            if verbose:
                _LOG.warning(
                    "[DoRA Power LoRA Loader] ZiT/Lumina2 compat: incompatible auxiliary tensor shapes while fusing %s%s; leaving originals untouched.",
                    attention_base,
                    suffix,
                )
            continue

        lora_sd[fused_aux_key] = fused_aux
        processed.extend(comp_keys)
        created += 1

    for k in processed:
        lora_sd.pop(k, None)

    if verbose:
        total_rank = int(fused_down.shape[0])
        _LOG.info(
            "[DoRA Power LoRA Loader] ZiT/Lumina2 compat: fused %s split Q/K/V -> %s (family=%s, rank=%s, created=%s).",
            attention_base,
            fused_base,
            family_name,
            total_rank,
            created,
        )

    return (1, created)


def _remap_zimage_attention_out_prefixes(lora_sd: Dict[str, Any], verbose: bool = False) -> Tuple[int, int]:
    remapped_groups = 0
    created = 0
    keys = list(lora_sd.keys())
    pat = re.compile(r"^(?P<base>.+\.attention)\.to_out\.0\.")
    done: Set[str] = set()

    for k in keys:
        m = pat.match(str(k))
        if not m:
            continue
        src_base = m.group("base") + ".to_out.0"
        if src_base in done:
            continue
        done.add(src_base)

        dst_base = m.group("base") + ".out"
        src_prefix = src_base + "."
        dst_prefix = dst_base + "."

        if any(str(x).startswith(dst_prefix) for x in lora_sd.keys()):
            continue

        n = _rename_prefix_keys(lora_sd, src_prefix, dst_prefix, delete_from=True)
        if n:
            remapped_groups += 1
            created += n
            if verbose:
                _LOG.info("[DoRA Power LoRA Loader] ZiT/Lumina2 compat: remapped %s -> %s (%s keys).", src_base, dst_base, n)

    return (remapped_groups, created)


def _apply_zimage_lumina2_compat(
    lora_sd: Dict[str, Any],
    model,
    model_sd_keys: Optional[Set[str]],
    key_map: Optional[Dict[str, Any]],
    verbose: bool = False,
) -> None:
    """
    Normalize ZiT/Lumina2 LoRA exports into the native fused-attention form expected by the model.

    This is intentionally conservative: it only activates when the live model strongly looks like
    Lumina2/Z-Image Turbo, or when both the model and the LoRA show strong ZiT-style attention cues.
    """
    model_is_zimage = _looks_like_zimage_lumina2_model(model, model_sd_keys)
    lora_is_zimage = _looks_like_zimage_attention_lora(lora_sd)

    if not model_is_zimage:
        return
    if not lora_is_zimage:
        # Still add aliases for native ZiT keys if this is a Lumina2 model; it is cheap and safe.
        if key_map is not None:
            _augment_key_map_with_zimage_lumina2_aliases(key_map, model, model_sd_keys, verbose=verbose)
        return

    if key_map is not None:
        _augment_key_map_with_zimage_lumina2_aliases(key_map, model, model_sd_keys, verbose=verbose)

    alias_created = _normalize_zimage_attention_component_aliases(lora_sd, verbose=verbose)

    fused_groups = 0
    fused_keys = 0
    for attention_base in sorted(_collect_zimage_attention_bases(lora_sd)):
        for family in _ZIMAGE_QKV_MATRIX_FAMILIES:
            g, n = _fuse_zimage_attention_qkv_for_family(
                lora_sd,
                attention_base,
                up_suffix=family["up_suffix"],
                down_suffix=family["down_suffix"],
                family_name=family["name"],
                verbose=verbose,
            )
            fused_groups += g
            fused_keys += n

    out_groups, out_keys = _remap_zimage_attention_out_prefixes(lora_sd, verbose=verbose)

    if verbose and (alias_created or fused_groups or out_groups):
        _LOG.info(
            "[DoRA Power LoRA Loader] ZiT/Lumina2 compat summary: alias_keys=%s fused_qkv_groups=%s fused_keys=%s remapped_out_groups=%s remapped_out_keys=%s.",
            alias_created,
            fused_groups,
            fused_keys,
            out_groups,
            out_keys,
        )


def _pick_flux2_broadcast_targets(key_map: Dict[str, str]) -> Tuple[List[str], List[str], List[str]]:
    """Derive broadcast destinations from the current model's key_map (Flux2 varies across builds)."""
    bases = list(key_map.keys())

    def _is_modulation_base(b: str) -> bool:
        bl = b.lower()
        return "modulation" in bl or "stream_modulation" in bl

    mods = [b for b in bases if _is_modulation_base(b)]
    if mods:
        img = [b for b in mods if "img" in b.lower() or "image" in b.lower()]
        txt = [b for b in mods if "txt" in b.lower() or "text" in b.lower() or "context" in b.lower()]
        single = [b for b in mods if "single" in b.lower()]
        return (sorted(img), sorted(txt), sorted(single))

    # Fallback for Flux2 builds where the modulation layers are exposed under norm/adaln modules.
    re_img = re.compile(r"^transformer\.transformer_blocks\.\d+\.norm1\.linear$")
    re_txt = re.compile(r"^transformer\.transformer_blocks\.\d+\.norm1_context\.linear$")
    re_single = re.compile(r"^transformer\.single_transformer_blocks\.\d+\.norm\.linear$")
    img = [b for b in bases if re_img.match(b)]
    txt = [b for b in bases if re_txt.match(b)]
    single = [b for b in bases if re_single.match(b)]
    return (sorted(img), sorted(txt), sorted(single))


def _infer_flux_block_counts(model_sd_keys: Optional[Set[str]]) -> Tuple[int, int]:
    """
    Infer (n_double, n_single) from diffusion_model.* block keys.
    Fallback if model_config isn't accessible.
    """
    if not model_sd_keys:
        return (0, 0)

    re_double = re.compile(r"^diffusion_model\.double_blocks\.(\d+)\.")
    re_single = re.compile(r"^diffusion_model\.single_blocks\.(\d+)\.")
    max_d = -1
    max_s = -1
    for k in model_sd_keys:
        m = re_double.match(k)
        if m:
            i = int(m.group(1))
            if i > max_d:
                max_d = i
            continue
        m = re_single.match(k)
        if m:
            i = int(m.group(1))
            if i > max_s:
                max_s = i
    return (max_d + 1 if max_d >= 0 else 0, max_s + 1 if max_s >= 0 else 0)


def _iter_tensors(obj: Any, path: str = ""):
    """Yield (path, tensor) pairs from arbitrary nested objects."""
    if isinstance(obj, torch.Tensor):
        yield path, obj
        return
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else str(k)
            yield from _iter_tensors(v, p)
        return
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        for i, v in enumerate(obj):
            p = f"{path}[{i}]"
            yield from _iter_tensors(v, p)
        return


def _tensor_health_report(tensors: List[Tuple[str, torch.Tensor]], topn: int = 20):
    nan = []
    inf = []
    mags = []
    for k, t in tensors:
        try:
            if torch.isnan(t).any():
                nan.append(k)
            if torch.isinf(t).any():
                inf.append(k)
            mags.append((k, float(t.detach().abs().max().item())))
        except Exception:
            continue
    mags.sort(key=lambda x: x[1], reverse=True)
    return nan, inf, mags[:topn]


def _log_lora_tensor_health(tag: str, lora_sd: Dict[str, Any], verbose: bool):
    tensors = [(k, v) for k, v in lora_sd.items() if isinstance(v, torch.Tensor)]
    nan, inf, top = _tensor_health_report(tensors)
    if nan or inf:
        _LOG.warning(
            "[DoRA Power LoRA Loader] %s: LoRA file contains NaN/Inf (nan=%d inf=%d). This will produce pink.",
            tag,
            len(nan),
            len(inf),
        )
        if verbose:
            for k in nan[:50]:
                _LOG.warning("[DoRA Power LoRA Loader] %s: NaN key: %s", tag, k)
            for k in inf[:50]:
                _LOG.warning("[DoRA Power LoRA Loader] %s: Inf key: %s", tag, k)
    if verbose and top:
        _LOG.info("[DoRA Power LoRA Loader] %s: top max|x| in lora_sd:", tag)
        for k, m in top:
            _LOG.info("  %12.4g  %s", m, k)


def _suffix_tensor_stats(sd: Dict[str, Any], suffix: str) -> Tuple[int, int, float, List[str]]:
    """
    Returns (count, zero_count, max_abs, dtypes) for tensors whose key endswith(suffix).
    zero_count counts tensors whose max_abs == 0.
    """
    n = 0
    z = 0
    mx = 0.0
    dtypes: Set[str] = set()
    for k, v in sd.items():
        if not isinstance(v, torch.Tensor):
            continue
        if not str(k).endswith(suffix):
            continue
        n += 1
        dtypes.add(str(v.dtype))
        try:
            m = float(v.detach().abs().max().item())
            if m == 0.0:
                z += 1
            if m > mx:
                mx = m
        except Exception:
            continue
    return (n, z, mx, sorted(dtypes))


def _log_lora_direction_stats(tag: str, lora_sd: Dict[str, Any], verbose: bool) -> None:
    """Targeted stats for direction matrices (up/down). Helps distinguish 'missing/ignored' vs 'all zeros'."""
    if not verbose:
        return
    suffix_groups = _LORA_DIRECTION_SUFFIX_PAIRS
    for up_s, down_s in suffix_groups:
        ups = [v for k, v in lora_sd.items() if str(k).endswith(up_s) and isinstance(v, torch.Tensor)]
        downs = [v for k, v in lora_sd.items() if str(k).endswith(down_s) and isinstance(v, torch.Tensor)]
        if not ups and not downs:
            continue

        def _summ(ts):
            if not ts:
                return (0, 0, 0.0)
            zero = 0
            mx = 0.0
            for t in ts:
                try:
                    m = float(t.detach().abs().max().item())
                    if m == 0.0:
                        zero += 1
                    if m > mx:
                        mx = m
                except Exception:
                    continue
            return (len(ts), zero, mx)

        nu, zu, mu = _summ(ups)
        nd, zd, md = _summ(downs)
        _LOG.info(
            "[DoRA Power LoRA Loader] %s: dir-mats %s/%s: up n=%d zero=%d max|x|=%g | down n=%d zero=%d max|x|=%g",
            tag,
            up_s,
            down_s,
            nu,
            zu,
            mu,
            nd,
            zd,
            md,
        )


def _log_loaded_tensor_health(tag: str, loaded: Any, verbose: bool):
    tensors = list(_iter_tensors(loaded, path="loaded"))
    nan, inf, top = _tensor_health_report(tensors)
    if nan or inf:
        _LOG.warning(
            "[DoRA Power LoRA Loader] %s: loaded patches contain NaN/Inf (nan=%d inf=%d). Pink is expected.",
            tag,
            len(nan),
            len(inf),
        )
        if verbose:
            for k in nan[:50]:
                _LOG.warning("[DoRA Power LoRA Loader] %s: NaN patch path: %s", tag, k)
            for k in inf[:50]:
                _LOG.warning("[DoRA Power LoRA Loader] %s: Inf patch path: %s", tag, k)
    if verbose and top:
        _LOG.info("[DoRA Power LoRA Loader] %s: top max|x| in loaded patches:", tag)
        for k, m in top:
            _LOG.info("  %12.4g  %s", m, k)


def _unwrap_key_map_target(v: Any) -> Tuple[Optional[str], Optional[Tuple[int, int, int]]]:
    """Return (dest_key, slice_tuple) from a key_map value."""
    try:
        if v is None:
            return (None, None)
        if isinstance(v, str):
            return (v, None)
        if isinstance(v, tuple) and len(v) >= 1 and isinstance(v[0], str):
            # Common Comfy pattern: (dest_key, (dim, start, length))
            sl = None
            if len(v) >= 2 and isinstance(v[1], tuple) and len(v[1]) == 3:
                try:
                    sl = (int(v[1][0]), int(v[1][1]), int(v[1][2]))
                except Exception:
                    sl = None
            return (v[0], sl)
    except Exception:
        return (None, None)
    return (None, None)


def _auto_strength_destination_group(
    base: str,
    key_map: Dict[str, Any],
    model_state_dict: Optional[Dict[str, Any]],
    clip_state_dict: Optional[Dict[str, Any]],
) -> Optional[str]:
    dest, _ = _unwrap_key_map_target(key_map.get(base))
    if not dest:
        return None
    if model_state_dict is not None and dest in model_state_dict:
        return "model"
    if clip_state_dict is not None and dest in clip_state_dict:
        return "clip"
    return None


def _auto_strength_tensor_rms(tensor: torch.Tensor, analysis_device: Optional[torch.device] = None) -> Optional[float]:
    try:
        t = _auto_strength_cast_float32(tensor, analysis_device)
        if t is None:
            return None
        n = int(t.numel())
        if n <= 0:
            return None
        return float(t.norm().item()) / (float(n) ** 0.5)
    except _AutoStrengthAnalysisDeviceError:
        raise
    except Exception as exc:
        if _auto_strength_is_device_failure(exc, analysis_device):
            raise _AutoStrengthAnalysisDeviceError from None
        return None


def _auto_strength_get_destination_weight(
    base: str,
    key_map: Dict[str, Any],
    model_state_dict: Optional[Dict[str, Any]],
    clip_state_dict: Optional[Dict[str, Any]],
) -> Optional[torch.Tensor]:
    dest, sl = _unwrap_key_map_target(key_map.get(base))
    if not dest:
        return None

    weight = None
    if model_state_dict is not None:
        weight = model_state_dict.get(dest)
    if weight is None and clip_state_dict is not None:
        weight = clip_state_dict.get(dest)
    if not isinstance(weight, torch.Tensor):
        return None

    if sl is None:
        return weight

    try:
        dim, start, length = int(sl[0]), int(sl[1]), int(sl[2])
        if length <= 0:
            return None
        if dim < 0 or dim >= weight.ndim:
            return None
        if start < 0 or (start + length) > int(weight.shape[dim]):
            return None
        return weight.narrow(dim, start, length)
    except Exception:
        return None


def _auto_strength_destination_family(weight: Optional[torch.Tensor]) -> str:
    """
    Return a coarse destination family for auto-strength cohorting.

    Invariant: auto-strength ratios must compare like with like. Pooling SDXL spatial
    conv kernels and large projection matrices into one mean biases the ratio toward
    whichever family dominates parameter count, even if their update magnitudes are
    not semantically comparable.
    """
    if not isinstance(weight, torch.Tensor):
        return "unknown"
    try:
        ndim = int(weight.ndim)
    except Exception:
        return "unknown"

    if ndim <= 0:
        return "unknown"
    if ndim == 1:
        return "vector"
    if ndim == 2:
        return "linear"
    try:
        spatial = tuple(int(x) for x in weight.shape[2:])
    except Exception:
        spatial = ()
    if spatial:
        return "conv:" + "x".join(str(x) for x in spatial)
    return f"tensor:{ndim}d"


def _auto_strength_measure_dora_effect(
    weight: torch.Tensor,
    delta: torch.Tensor,
    dora_scale: torch.Tensor,
    analysis_device: Optional[torch.device] = None,
) -> Optional[float]:
    """
    Return RMS(update) for the actual DoRA weight path applied by Comfy.

    Invariant: for DoRA, layer scores must reflect the post-normalization weight update
    against the destination base weight. Ranking by the raw low-rank delta alone is not
    comparable across bases because DoRA normalizes V = W0 + Δ against W0 and rescales
    by dora_scale, so equal ||Δ|| can produce radically different final updates.
    """
    try:
        weight32 = _auto_strength_cast_float32(weight, analysis_device)
        delta32 = _auto_strength_cast_float32(delta, analysis_device)
        dora_scale32 = _auto_strength_cast_float32(dora_scale, analysis_device)
        if weight32 is None or delta32 is None or dora_scale32 is None:
            return None
        delta32 = delta32.reshape(weight32.shape)
    except _AutoStrengthAnalysisDeviceError:
        raise
    except Exception as exc:
        if _auto_strength_is_device_failure(exc, analysis_device):
            raise _AutoStrengthAnalysisDeviceError from None
        return None

    if dora_scale32.ndim != 1 or weight32.ndim < 2:
        return None

    try:
        if int(dora_scale32.shape[0]) == int(weight32.shape[0]):
            weight_calc32 = weight32 + delta32
            weight_norm = (
                weight_calc32.reshape(weight_calc32.shape[0], -1)
                .norm(dim=1, keepdim=True)
                .reshape(weight_calc32.shape[0], *[1] * (weight_calc32.dim() - 1))
            )
            dora_scale32 = dora_scale32.reshape(weight_calc32.shape[0], *[1] * (weight_calc32.dim() - 1))
        elif int(dora_scale32.shape[0]) == int(weight32.shape[1]):
            weight_calc32 = weight32 + delta32
            weight_norm = (
                weight_calc32.transpose(0, 1)
                .reshape(weight_calc32.shape[1], -1)
                .norm(dim=1, keepdim=True)
                .reshape(weight_calc32.shape[1], *[1] * (weight_calc32.dim() - 1))
                .transpose(0, 1)
            )
            dora_scale32 = dora_scale32.reshape(1, weight_calc32.shape[1], *[1] * (weight_calc32.dim() - 2))
        else:
            return None

        weight_norm = weight_norm + torch.finfo(torch.float32).eps
        weight_dora32 = weight_calc32 * (dora_scale32 / weight_norm)
        return _auto_strength_tensor_rms(weight_dora32 - weight32, analysis_device=analysis_device)
    except _AutoStrengthAnalysisDeviceError:
        raise
    except Exception as exc:
        if _auto_strength_is_device_failure(exc, analysis_device):
            raise _AutoStrengthAnalysisDeviceError from None
        return None


def _auto_strength_measure_base_delta_on_device(
    lora_sd: Dict[str, Any],
    base: str,
    key_map: Dict[str, Any],
    model_state_dict: Optional[Dict[str, Any]],
    clip_state_dict: Optional[Dict[str, Any]],
    analysis_device: Optional[torch.device],
) -> Optional[float]:
    """
    Return a comparable magnitude score for a single base's update.

    Supported cases:
      - standard LoRA / DoRA low-rank pairs / direct deltas: RMS(delta)
      - DoRA low-rank pairs: RMS(actual post-normalization DoRA update)
      - direct delta tensors (.diff / .diff_b / .set_weight / .reshape_weight)

    Invariant: scores must be comparable across destination tensor sizes. Using raw
    Frobenius norms violates that invariant because ||ΔW||_F scales with sqrt(numel),
    which over-boosts smaller spatial layers in mixed architectures like SDXL. We
    therefore compare RMS update magnitude (scaled Frobenius / sqrt(numel)).

    Returns None when the base has no measurable linear delta representation.
    """
    prefix = base + "."
    direct_norms: List[float] = []
    dest_weight = _auto_strength_get_destination_weight(base, key_map, model_state_dict, clip_state_dict)
    dora_scale = lora_sd.get(base + ".dora_scale")
    has_dora = isinstance(dora_scale, torch.Tensor)

    for suffix in (".diff", ".diff_b", ".set_weight", ".reshape_weight"):
        key = base + suffix
        tensor = lora_sd.get(key)
        if not isinstance(tensor, torch.Tensor):
            continue
        if has_dora and isinstance(dest_weight, torch.Tensor):
            dora_rms = _auto_strength_measure_dora_effect(dest_weight, tensor, dora_scale, analysis_device=analysis_device)
            if dora_rms is not None:
                direct_norms.append(dora_rms)
                continue
        rms = _auto_strength_tensor_rms(tensor, analysis_device=analysis_device)
        if rms is not None:
            direct_norms.append(rms)

    for up_suffix, down_suffix in _LORA_DIRECTION_SUFFIX_PAIRS:
        up = lora_sd.get(base + up_suffix)
        down = lora_sd.get(base + down_suffix)
        if not isinstance(up, torch.Tensor) or not isinstance(down, torch.Tensor):
            continue
        try:
            up_cast = _auto_strength_cast_float32(up, analysis_device)
            down_cast = _auto_strength_cast_float32(down, analysis_device)
            if up_cast is None or down_cast is None:
                continue
            up_mat = up_cast.reshape(int(up.shape[0]), -1)
            down_mat = down_cast.reshape(int(down.shape[0]), -1)
        except _AutoStrengthAnalysisDeviceError:
            raise
        except Exception as exc:
            if _auto_strength_is_device_failure(exc, analysis_device):
                raise _AutoStrengthAnalysisDeviceError from None
            continue
        if up_mat.ndim != 2 or down_mat.ndim != 2:
            continue
        if int(up_mat.shape[1]) != int(down_mat.shape[0]):
            continue

        try:
            delta = up_mat @ down_mat
        except _AutoStrengthAnalysisDeviceError:
            raise
        except Exception as exc:
            if _auto_strength_is_device_failure(exc, analysis_device):
                raise _AutoStrengthAnalysisDeviceError from None
            continue

        alpha = _tensor_scalar_to_float(lora_sd.get(base + ".alpha"), default=1.0)
        rank = max(1, int(down_mat.shape[0]))
        scale = (alpha / float(rank)) if (base + ".alpha") in lora_sd else 1.0
        try:
            delta = delta * float(scale)
        except _AutoStrengthAnalysisDeviceError:
            raise
        except Exception as exc:
            if _auto_strength_is_device_failure(exc, analysis_device):
                raise _AutoStrengthAnalysisDeviceError from None
            pass

        if has_dora and isinstance(dest_weight, torch.Tensor):
            dora_rms = _auto_strength_measure_dora_effect(dest_weight, delta, dora_scale, analysis_device=analysis_device)
            if dora_rms is not None:
                direct_norms.append(dora_rms)
                continue

        delta_rms = _auto_strength_tensor_rms(delta, analysis_device=analysis_device)
        if delta_rms is not None:
            direct_norms.append(delta_rms)

    if not direct_norms:
        # Best-effort fallback for exotic exports that still expose one-side linear tensors.
        vals: List[float] = []
        for key, tensor in lora_sd.items():
            if not str(key).startswith(prefix) or not isinstance(tensor, torch.Tensor):
                continue
            if str(key).endswith(_UP_ONLY_SCALE_SUFFIXES):
                rms = _auto_strength_tensor_rms(tensor, analysis_device=analysis_device)
                if rms is not None:
                    vals.append(rms)
        if vals:
            return float(sum(vals) / len(vals))
        return None

    return float(sum(direct_norms) / len(direct_norms))


def _auto_strength_measure_base_delta(
    lora_sd: Dict[str, Any],
    base: str,
    key_map: Dict[str, Any],
    model_state_dict: Optional[Dict[str, Any]],
    clip_state_dict: Optional[Dict[str, Any]],
    analysis_device_mode: str = "auto",
    analysis_load_device: Any = None,
    verbose: bool = False,
) -> Optional[float]:
    dest_weight = _auto_strength_get_destination_weight(base, key_map, model_state_dict, clip_state_dict)
    analysis_device = _auto_strength_resolve_analysis_device(analysis_device_mode, analysis_load_device, dest_weight)
    try:
        return _auto_strength_measure_base_delta_on_device(
            lora_sd=lora_sd,
            base=base,
            key_map=key_map,
            model_state_dict=model_state_dict,
            clip_state_dict=clip_state_dict,
            analysis_device=analysis_device,
        )
    except _AutoStrengthAnalysisDeviceError:
        if analysis_device is None or analysis_device.type == "cpu":
            return None
        if verbose:
            _LOG.warning(
                "[DoRA Power LoRA Loader] auto-strength: base=%s analysis device %s failed; retrying on CPU",
                base,
                analysis_device,
            )
        return _auto_strength_measure_base_delta_on_device(
            lora_sd=lora_sd,
            base=base,
            key_map=key_map,
            model_state_dict=model_state_dict,
            clip_state_dict=clip_state_dict,
            analysis_device=torch.device("cpu"),
        )


def _auto_strength_analyze_base_targets(
    lora_sd: Dict[str, Any],
    lora_bases: Iterable[str],
    key_map: Dict[str, Any],
    model_state_dict: Optional[Dict[str, Any]],
    clip_state_dict: Optional[Dict[str, Any]],
    analysis_device_mode: str,
    analysis_load_device: Any,
    strength_model: float,
    strength_clip: float,
    ratio_floor: float,
    ratio_ceiling: float,
    logical_groups: Optional[Dict[str, Tuple[str, float]]] = None,
    verbose: bool = False,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Compute per-base target strengths and preserve a structured report that matches
    the loader's real logical-group-aware measurement model.

    Invariants:
      - auto-strength must modulate the same *linear delta* that standard LoRA and
        DoRA feed into Comfy's patch loader.
      - auto-strength must be invariant to synthetic compat expansion. Broadcasting one
        logical source into N per-block bases must not change its measured target just
        because the loader expanded the keys before comfy.lora.load_lora(...).

    We therefore compute a base-local score from the linear update representation, fold
    compat-broadcast clones back into their logical source groups for measurement, and
    then convert those absolute targets into per-base redistribution ratios before
    comfy.lora.load_lora(...).
    """
    ratio_floor = max(0.0, float(ratio_floor))
    ratio_ceiling = max(ratio_floor, float(ratio_ceiling))
    logical_groups = logical_groups or {}

    grouped_norms: Dict[Tuple[str, str], Dict[str, List[float]]] = {}
    base_groups: Dict[str, str] = {}
    base_families: Dict[str, str] = {}
    base_cohorts: Dict[str, Tuple[str, str]] = {}
    base_norms: Dict[str, float] = {}
    base_logical_ids: Dict[str, str] = {}
    base_logical_scales: Dict[str, float] = {}
    logical_norms: Dict[Tuple[str, str, str], float] = {}
    logical_members: Dict[Tuple[str, str, str], List[str]] = {}
    skipped_zero_strength_members: Dict[Tuple[str, str, str], List[str]] = {}
    skipped_zero_strength_bases: List[str] = []
    targets: Dict[str, float] = {}

    for base in lora_bases:
        group = _auto_strength_destination_group(base, key_map, model_state_dict, clip_state_dict)
        if group is None:
            continue
        base_groups[base] = group
        dest_weight = _auto_strength_get_destination_weight(base, key_map, model_state_dict, clip_state_dict)
        family = _auto_strength_destination_family(dest_weight)
        base_families[base] = family
        cohort = (group, family)
        base_cohorts[base] = cohort
        global_strength = float(strength_model if group == "model" else strength_clip)
        targets[base] = global_strength

        logical_id = base
        logical_scale = 1.0
        lg = logical_groups.get(base)
        if isinstance(lg, tuple) and len(lg) >= 2:
            try:
                logical_id = str(lg[0])
                logical_scale = abs(float(lg[1]))
            except Exception:
                logical_id = base
                logical_scale = 1.0
        if not (logical_scale > _AUTO_STRENGTH_EPS):
            logical_scale = 1.0
        base_logical_ids[base] = logical_id
        base_logical_scales[base] = logical_scale
        logical_key = (cohort[0], cohort[1], logical_id)

        if abs(global_strength) < _AUTO_STRENGTH_EPS:
            skipped_zero_strength_members.setdefault(logical_key, []).append(base)
            skipped_zero_strength_bases.append(base)
            continue

        logical_members.setdefault(logical_key, []).append(base)

        norm = _auto_strength_measure_base_delta(
            lora_sd=lora_sd,
            base=base,
            key_map=key_map,
            model_state_dict=model_state_dict,
            clip_state_dict=clip_state_dict,
            analysis_device_mode=analysis_device_mode,
            analysis_load_device=analysis_load_device,
            verbose=verbose,
        )
        if norm is None or not (norm > _AUTO_STRENGTH_EPS):
            continue
        base_norms[base] = norm
        logical_norm = float(norm / logical_scale)
        logical_norms[logical_key] = logical_norm
        grouped_norms.setdefault(cohort, {}).setdefault(logical_id, []).append(logical_norm)

    group_means: Dict[Tuple[str, str], Optional[float]] = {}
    for cohort, vals_by_logical in grouped_norms.items():
        logical_vals = [float(sum(vals) / len(vals)) for vals in vals_by_logical.values() if vals]
        group_means[cohort] = float(sum(logical_vals) / len(logical_vals)) if logical_vals else None
        for logical_id, vals in vals_by_logical.items():
            if vals:
                logical_norms[(cohort[0], cohort[1], logical_id)] = float(sum(vals) / len(vals))

    for base, group in base_groups.items():
        global_strength = float(strength_model if group == "model" else strength_clip)
        if abs(global_strength) < _AUTO_STRENGTH_EPS:
            targets[base] = 0.0
            continue

        family = base_families.get(base, "unknown")
        cohort = base_cohorts.get(base, (group, family))
        logical_id = base_logical_ids.get(base, base)
        norm = logical_norms.get((cohort[0], cohort[1], logical_id))
        mean_norm = group_means.get(cohort)
        if norm is None or mean_norm is None or not (norm > _AUTO_STRENGTH_EPS):
            targets[base] = global_strength
            continue

        ratio = mean_norm / norm
        ratio = max(ratio_floor, min(ratio_ceiling, ratio))
        targets[base] = float(global_strength * ratio)

    measured = len(base_norms)
    total = len(base_groups)
    analyzable = sum(len(v) for v in logical_members.values())
    measured_logical = sum(1 for logical_key in logical_members.keys() if logical_norms.get(logical_key, 0.0) > _AUTO_STRENGTH_EPS)

    cohorts_report: List[Dict[str, Any]] = []
    for cohort in sorted(set(base_cohorts.values())):
        group, family = cohort
        cohort_members = [k for k in logical_members.keys() if k[0] == group and k[1] == family]
        measured_members = [k for k in cohort_members if logical_norms.get(k, 0.0) > _AUTO_STRENGTH_EPS]
        total_bases = sum(len(logical_members.get(k, [])) for k in cohort_members)
        measured_bases = sum(sum(1 for base in logical_members.get(k, []) if base in base_norms) for k in cohort_members)
        skipped_bases = sum(
            len(skipped_zero_strength_members.get(k, []))
            for k in skipped_zero_strength_members.keys()
            if k[0] == group and k[1] == family
        )
        cohorts_report.append(
            {
                "group": group,
                "family": family,
                "mean_norm": _auto_strength_safe_number(group_means.get(cohort)),
                "logical_count": len(cohort_members),
                "measured_logical_count": len(measured_members),
                "base_count": total_bases,
                "skipped_zero_strength_base_count": skipped_bases,
                "measured_base_count": measured_bases,
            }
        )

    logical_reports: List[Dict[str, Any]] = []
    for logical_key, members in logical_members.items():
        group, family, logical_id = logical_key
        global_strength = float(strength_model if group == "model" else strength_clip)
        logical_norm = logical_norms.get(logical_key)
        cohort_mean = group_means.get((group, family))
        ratio_raw = None
        ratio_applied = None
        if logical_norm is not None and cohort_mean is not None and logical_norm > _AUTO_STRENGTH_EPS:
            ratio_raw = float(cohort_mean / logical_norm)
            ratio_applied = float(max(ratio_floor, min(ratio_ceiling, ratio_raw)))
        fallback_to_global = ratio_applied is None
        target_strength = float(global_strength if fallback_to_global else global_strength * ratio_applied)
        bases_report = []
        measured_base_count = 0
        for base in sorted(members):
            base_target = float(targets.get(base, global_strength))
            base_ratio = None
            if abs(global_strength) > _AUTO_STRENGTH_EPS:
                base_ratio = float(base_target / global_strength)
            if base in base_norms:
                measured_base_count += 1
            bases_report.append(
                {
                    "base": base,
                    "norm": _auto_strength_safe_number(base_norms.get(base)),
                    "logical_scale": _auto_strength_safe_number(base_logical_scales.get(base, 1.0)),
                    "measured": base in base_norms,
                    "ratio_applied": _auto_strength_safe_number(base_ratio),
                    "target_strength": base_target,
                }
            )

        logical_reports.append(
            {
                "group": group,
                "family": family,
                "logical_id": logical_id,
                "fanout": len(members),
                "measured_base_count": measured_base_count,
                "mean_norm": _auto_strength_safe_number(logical_norm),
                "cohort_mean_norm": _auto_strength_safe_number(cohort_mean),
                "ratio_raw": _auto_strength_safe_number(ratio_raw),
                "ratio_applied": _auto_strength_safe_number(ratio_applied),
                "global_strength": global_strength,
                "target_strength": target_strength,
                "fallback_to_global": fallback_to_global,
                "bases": bases_report,
            }
        )

    logical_reports.sort(
        key=lambda item: (
            -abs((_auto_strength_safe_number(item.get("ratio_applied")) or 1.0) - 1.0),
            str(item.get("group") or ""),
            str(item.get("family") or ""),
            str(item.get("logical_id") or ""),
        )
    )

    report = {
        "schema": 1,
        "kind": "dora_auto_strength_report",
        "analysis_device_mode": str(analysis_device_mode),
        "analysis_load_device": _auto_strength_describe_device(analysis_load_device),
        "strength_model": float(strength_model),
        "strength_clip": float(strength_clip),
        "ratio_floor": ratio_floor,
        "ratio_ceiling": ratio_ceiling,
        "mapped_bases": total,
        "analyzable_bases": analyzable,
        "measured_bases": measured,
        "logical_groups_total": len(logical_members),
        "logical_groups_measured": measured_logical,
        "logical_groups_skipped_zero_strength": len(skipped_zero_strength_members),
        "cohorts": cohorts_report,
        "logical_groups": logical_reports,
        "skipped_zero_strength_bases": sorted(skipped_zero_strength_bases),
        "unmeasured_bases": sorted(
            base for base in base_groups.keys()
            if base not in base_norms and base not in skipped_zero_strength_bases
        ),
    }

    if verbose:
        cohort_summary = {
            f"{group}/{family}": mean
            for (group, family), mean in sorted(group_means.items())
        }
        _LOG.info(
            "[DoRA Power LoRA Loader] auto-strength: measured %s/%s mapped bases (%s/%s logical groups) (cohort_means=%s ratio_floor=%s ratio_ceiling=%s)",
            measured,
            total,
            measured_logical,
            len(logical_members),
            cohort_summary,
            ratio_floor,
            ratio_ceiling,
        )
        sample = logical_reports[:20]
        for item in sample:
            _LOG.info(
                "[DoRA Power LoRA Loader] auto-strength: logical=%s group=%s family=%s fanout=%s mean_norm=%s cohort_mean=%s ratio=%s target=%s",
                item.get("logical_id"),
                item.get("group"),
                item.get("family"),
                item.get("fanout"),
                item.get("mean_norm"),
                item.get("cohort_mean_norm"),
                item.get("ratio_applied"),
                item.get("target_strength"),
            )

    return targets, report


def _auto_strength_compute_base_targets(
    lora_sd: Dict[str, Any],
    lora_bases: Iterable[str],
    key_map: Dict[str, Any],
    model_state_dict: Optional[Dict[str, Any]],
    clip_state_dict: Optional[Dict[str, Any]],
    analysis_device_mode: str,
    analysis_load_device: Any,
    strength_model: float,
    strength_clip: float,
    ratio_floor: float,
    ratio_ceiling: float,
    logical_groups: Optional[Dict[str, Tuple[str, float]]] = None,
    verbose: bool = False,
) -> Dict[str, float]:
    targets, _ = _auto_strength_analyze_base_targets(
        lora_sd=lora_sd,
        lora_bases=lora_bases,
        key_map=key_map,
        model_state_dict=model_state_dict,
        clip_state_dict=clip_state_dict,
        analysis_device_mode=analysis_device_mode,
        analysis_load_device=analysis_load_device,
        strength_model=strength_model,
        strength_clip=strength_clip,
        ratio_floor=ratio_floor,
        ratio_ceiling=ratio_ceiling,
        logical_groups=logical_groups,
        verbose=verbose,
    )
    return targets
def _auto_strength_targets_to_ratios(
    base_strengths: Dict[str, float],
    key_map: Dict[str, Any],
    model_state_dict: Optional[Dict[str, Any]],
    clip_state_dict: Optional[Dict[str, Any]],
    strength_model: float,
    strength_clip: float,
) -> Dict[str, float]:
    """
    Convert absolute per-base targets into redistribution ratios relative to the
    caller's global model/clip strengths.

    Invariant: enabling auto-strength must be a no-op when every base's computed
    ratio is 1.0. That requires preserving Comfy's normal outer patch strength path
    instead of baking the caller's global strength into the tensors themselves.
    """
    ratios: Dict[str, float] = {}
    for base, target in base_strengths.items():
        group = _auto_strength_destination_group(base, key_map, model_state_dict, clip_state_dict)
        if group is None:
            continue
        global_strength = float(strength_model if group == "model" else strength_clip)
        if abs(global_strength) <= _AUTO_STRENGTH_EPS:
            ratios[base] = 1.0
            continue
        try:
            ratios[base] = float(target) / global_strength
        except Exception:
            ratios[base] = 1.0
    return ratios


def _apply_base_strength_ratios(
    lora_sd: Dict[str, Any],
    base_ratios: Dict[str, float],
) -> Tuple[Dict[str, Any], bool]:
    """
    Bake per-base redistribution ratios into the LoRA tensors themselves.

    Rules:
      - if .alpha exists, scale ONLY .alpha
      - otherwise scale only one linear factor (.lora_up / .lora_A / equivalent)
      - direct delta tensors (.diff / .set_weight / ...) are scaled directly
      - DoRA magnitude tensors (.dora_scale / .w_norm / .b_norm) are never scaled

    We intentionally scale only the relative auto-strength ratio, not the caller's
    global model/clip strength. The outer patch strength must still be applied by
    model.add_patches()/clip.add_patches() so DoRA keeps Comfy's normal post-
    normalization strength mixing semantics.
    """
    if not base_ratios:
        return (lora_sd, False)

    scaled = dict(lora_sd)
    changed = False
    all_keys = list(lora_sd.keys())
    keys_by_base: Dict[str, List[Any]] = {}
    has_alpha_by_base: Dict[str, bool] = {}
    for base in base_ratios.keys():
        prefix = base + "."
        keys = [k for k in all_keys if str(k).startswith(prefix)]
        keys_by_base[base] = keys
        has_alpha_by_base[base] = any(str(k) == (base + ".alpha") for k in keys)

    for base, ratio in base_ratios.items():
        try:
            ratio_f = float(ratio)
        except Exception:
            continue
        if abs(ratio_f - 1.0) <= _AUTO_STRENGTH_EPS:
            continue

        keys = keys_by_base.get(base) or []
        if not keys:
            continue

        has_alpha = has_alpha_by_base.get(base, False)
        for k in keys:
            v = lora_sd.get(k)
            if not isinstance(v, torch.Tensor):
                continue
            ks = str(k)

            if ks.endswith((".dora_scale", ".w_norm", ".b_norm")):
                continue

            should_scale = False
            if ks.endswith(".alpha"):
                should_scale = True
            elif ks.endswith((".diff", ".diff_b", ".set_weight", ".reshape_weight")):
                should_scale = True
            elif (not has_alpha) and ks.endswith(_UP_ONLY_SCALE_SUFFIXES):
                should_scale = True

            if should_scale:
                scaled[k] = v * ratio_f
                changed = True

    return (scaled, changed)


def _fix_onetrainer_output_axis_dora_mats(
    lora_sd: Dict[str, Any],
    key_map: Dict[str, Any],
    model_state_dict: Optional[Dict[str, Any]],
    clip_state_dict: Optional[Dict[str, Any]],
    verbose: bool,
) -> None:
    """
    OneTrainer 'Apply on output axis (DoRA only)' can store direction mats transposed.
    If they don't match the destination weight layout, Comfy effectively applies only dora_scale
    -> lora_diff becomes identically zero (your exact symptom).
    """
    sd_model = model_state_dict or {}
    sd_clip = clip_state_dict or {}
    pair_suffixes = _LORA_DIRECTION_SUFFIX_PAIRS
    fixed = 0
    checked = 0
    examples: List[str] = []
    dora_bases = [k[: -len(".dora_scale")] for k in lora_sd.keys() if str(k).endswith(".dora_scale")]
    for base in dora_bases:
        up_key = down_key = None
        for us, ds in pair_suffixes:
            uk = base + us
            dk = base + ds
            if uk in lora_sd and dk in lora_sd:
                up_key, down_key = uk, dk
                break
        if up_key is None:
            continue

        up = lora_sd.get(up_key)
        down = lora_sd.get(down_key)
        if not isinstance(up, torch.Tensor) or not isinstance(down, torch.Tensor) or up.ndim != 2 or down.ndim != 2:
            continue

        dest, sl = _unwrap_key_map_target(key_map.get(base))
        if not dest or not dest.endswith(".weight"):
            continue

        w = sd_model.get(dest)
        if w is None:
            w = sd_clip.get(dest)
        if not isinstance(w, torch.Tensor) or w.ndim < 2:
            continue

        out_dim = int(w.shape[0])
        in_dim = int(w.shape[1])
        # If mapping is a slice (e.g. qkv), compare against the effective slice shape.
        if sl is not None:
            dim, _start, length = sl
            if dim == 0:
                out_dim = int(length)
            elif dim == 1:
                in_dim = int(length)

        # Robust rank inference: pick the shared "rank-like" dimension.
        u0, u1 = int(up.shape[0]), int(up.shape[1])
        d0, d1 = int(down.shape[0]), int(down.shape[1])
        dims = {u0, u1, d0, d1}
        cand = [x for x in dims if x not in (out_dim, in_dim)]
        if cand:
            r = min(cand)
        else:
            r = min(u0, u1, d0, d1)

        checked += 1

        # Standard: up=(out,r), down=(r,in)
        if u0 == out_dim and u1 == r and d0 == r and d1 == in_dim:
            continue

        # Swapped: up=(r,in), down=(out,r) -> swap
        if u0 == r and u1 == in_dim and d0 == out_dim and d1 == r:
            lora_sd[up_key], lora_sd[down_key] = down, up
            fixed += 1
            if len(examples) < 10:
                examples.append(f"swap  base={base} W=({out_dim},{in_dim}) up={tuple(up.shape)} down={tuple(down.shape)}")
            continue

        # Transposed: up=(in,r), down=(r,out) -> up=(out,r), down=(r,in)
        if u0 == in_dim and u1 == r and d0 == r and d1 == out_dim:
            lora_sd[up_key] = down.transpose(0, 1).contiguous()  # (out,r)
            lora_sd[down_key] = up.transpose(0, 1).contiguous()  # (r,in)
            fixed += 1
            if len(examples) < 10:
                examples.append(f"xpose base={base} W=({out_dim},{in_dim}) up={tuple(up.shape)} down={tuple(down.shape)}")
            continue

        # Transposed+swapped: up=(r,out), down=(in,r) -> transpose each
        if u0 == r and u1 == out_dim and d0 == in_dim and d1 == r:
            lora_sd[up_key] = up.transpose(0, 1).contiguous()  # (out,r)
            lora_sd[down_key] = down.transpose(0, 1).contiguous()  # (r,in)
            fixed += 1
            if len(examples) < 10:
                examples.append(f"xpose2 base={base} W=({out_dim},{in_dim}) up={tuple(up.shape)} down={tuple(down.shape)}")
            continue

    if verbose:
        _LOG.info("[DoRA Power LoRA Loader] OneTrainer output-axis DoRA mat-fix: checked=%d fixed=%d", checked, fixed)
        for ex in examples:
            _LOG.info("[DoRA Power LoRA Loader] OneTrainer mat-fix example: %s", ex)


def _get_unet_config_counts(model) -> Tuple[int, int]:
    """
    Best-effort read of Flux unet_config depth from the live model instance.
    Returns (depth, depth_single_blocks) or (0,0) if unavailable.
    """
    try:
        core = getattr(model, "model", None)
        cfg = getattr(core, "model_config", None)
        unet_cfg = getattr(cfg, "unet_config", None)
        if isinstance(unet_cfg, dict):
            d = int(unet_cfg.get("depth", 0) or 0)
            s = int(unet_cfg.get("depth_single_blocks", 0) or 0)
            return (d, s)
        # dict-like / attr-like fallback
        d = int(getattr(unet_cfg, "depth", 0) or 0)
        s = int(getattr(unet_cfg, "depth_single_blocks", 0) or 0)
        return (d, s)
    except Exception:
        return (0, 0)


def _apply_flux2_onetrainer_dora_compat(
    lora_sd: Dict[str, Any],
    model,
    model_sd_keys: Optional[Set[str]],
    key_map: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
    broadcast_auto_scale: bool = True,
    broadcast_scale: float = 1.0,
    broadcast_modulations: bool = True,
    broadcast_include_dora_scale: bool = False,
    auto_strength_logical_groups: Optional[Dict[str, Tuple[str, float]]] = None,
) -> None:
    """
    Flux2 / OneTrainer DoRA compat:
      1) rename time_guidance_embed -> time_text_embed (ComfyUI/diffusers mapping expects time_text_embed.*)
      2) broadcast global modulation LoRAs (double_stream_modulation_*/single_stream_modulation)
         onto per-block diffusers keys that ComfyUI actually maps:
           - transformer_blocks.{i}.norm1.linear
           - transformer_blocks.{i}.norm1_context.linear
           - single_transformer_blocks.{i}.norm.linear
    """
    # 1) time_guidance_embed -> time_text_embed
    # Only do it if the source prefix exists and the target prefix doesn't already exist (avoid double-mapping).
    if any(k.startswith("transformer.time_guidance_embed.") for k in lora_sd.keys()) and not any(
        k.startswith("transformer.time_text_embed.") for k in lora_sd.keys()
    ):
        n = _rename_prefix_keys(
            lora_sd,
            "transformer.time_guidance_embed.",
            "transformer.time_text_embed.",
            delete_from=True,
        )
        if verbose:
            _LOG.info("[DoRA Power LoRA Loader] flux2 compat: renamed %s keys time_guidance_embed -> time_text_embed", n)

    # 2) Broadcast global modulations if present (OneTrainer exports globals; ComfyUI expects per-block keys).
    # Choose destinations from the *current* model's key_map rather than hardcoding.
    if not key_map:
        return
    if not broadcast_modulations:
        return

    src_img = "transformer.double_stream_modulation_img.linear"
    src_txt = "transformer.double_stream_modulation_txt.linear"
    src_single = "transformer.single_stream_modulation.linear"

    # If key_map directly supports the source base, keep source keys untouched.
    src_img_is_mappable = src_img in key_map
    src_txt_is_mappable = src_txt in key_map
    src_single_is_mappable = src_single in key_map

    have_img = any(k.startswith(src_img + ".") for k in lora_sd.keys())
    have_txt = any(k.startswith(src_txt + ".") for k in lora_sd.keys())
    have_single = any(k.startswith(src_single + ".") for k in lora_sd.keys())

    img_targets, txt_targets, single_targets = _pick_flux2_broadcast_targets(key_map)

    # Dedupe alias bases (e.g. diffusion_model.* and lora_unet_* pointing to the same dest weight)
    img_targets = _dedupe_targets_by_dest(key_map, img_targets)
    txt_targets = _dedupe_targets_by_dest(key_map, txt_targets)
    single_targets = _dedupe_targets_by_dest(key_map, single_targets)

    if verbose:
        def _dump_targets(name: str, tgs: List[str]):
            for t in tgs:
                _LOG.info("[DoRA Power LoRA Loader] flux2 compat: %s target %s -> %r", name, t, key_map.get(t))

        _dump_targets("img", img_targets)
        _dump_targets("txt", txt_targets)
        _dump_targets("single", single_targets)

    # Helpful debug: print actual targets (not only counts).
    if verbose:
        _LOG.info("[DoRA Power LoRA Loader] flux2 compat: img_targets=%s", img_targets)
        _LOG.info("[DoRA Power LoRA Loader] flux2 compat: txt_targets=%s", txt_targets)
        _LOG.info("[DoRA Power LoRA Loader] flux2 compat: single_targets=%s", single_targets)

    # Choose suffix set: include DoRA magnitude params only when explicitly requested.
    def _pick_suffixes(src_base: str) -> Tuple[str, ...]:
        if broadcast_include_dora_scale and _src_has_dora_params(lora_sd, src_base):
            return _BROADCAST_DORA_SUFFIXES
        return _BROADCAST_DELTA_SUFFIXES

    suf_img = _pick_suffixes(src_img)
    suf_txt = _pick_suffixes(src_txt)
    suf_single = _pick_suffixes(src_single)

    def _scale_for(n: int) -> float:
        if n <= 0:
            return 1.0
        return (float(broadcast_scale) / float(n)) if broadcast_auto_scale else float(broadcast_scale)

    def _register_auto_strength_group(source_base: str, targets: List[str], scale: float) -> None:
        if auto_strength_logical_groups is None or not targets:
            return
        try:
            scale_f = float(scale)
        except Exception:
            scale_f = 1.0
        if not (abs(scale_f) > _AUTO_STRENGTH_EPS):
            scale_f = 1.0
        logical_id = f"broadcast:{source_base}"
        for dst in targets:
            auto_strength_logical_groups[dst] = (logical_id, scale_f)

    scale_img = _scale_for(len(img_targets))
    scale_txt = _scale_for(len(txt_targets))
    scale_single = _scale_for(len(single_targets))

    if verbose:
        _LOG.info(
            "[DoRA Power LoRA Loader] flux2 compat: broadcast targets: img=%s txt=%s single=%s",
            len(img_targets),
            len(txt_targets),
            len(single_targets),
        )
        _LOG.info(
            "[DoRA Power LoRA Loader] flux2 compat: broadcast scales: img=%s txt=%s single=%s (auto=%s base=%s)",
            scale_img,
            scale_txt,
            scale_single,
            broadcast_auto_scale,
            broadcast_scale,
        )

    # Only broadcast into targets that the file doesn't already define.
    def _has_any_base(prefix_base: str) -> bool:
        p = prefix_base + "."
        return any(k.startswith(p) for k in lora_sd.keys())

    # For Flux2: these are typically GLOBAL modulation modules, not per-block.
    # If we have exactly one unique destination, rename into that canonical base instead of cloning/broadcasting.
    if have_img and (not src_img_is_mappable) and img_targets:
        if any(_has_any_base(t) for t in img_targets):
            if verbose:
                _LOG.info("[DoRA Power LoRA Loader] flux2 compat: img modulation targets already present; dropping source %s", src_img)
            _delete_prefix_keys(lora_sd, src_img + ".")
        elif len(img_targets) == 1:
            dst = img_targets[0]
            n = _rename_prefix_keys(lora_sd, src_img + ".", dst + ".", delete_from=True)
            if verbose:
                _LOG.info("[DoRA Power LoRA Loader] flux2 compat: renamed %s keys %s -> %s", n, src_img, dst)
        else:
            created = 0
            for dst in img_targets:
                created += _clone_base_block(lora_sd, src_img, dst, scale=scale_img, allow_suffixes=suf_img)
            _register_auto_strength_group(src_img, img_targets, scale_img)
            if verbose:
                _LOG.info("[DoRA Power LoRA Loader] flux2 compat: broadcast %s -> %s targets (keys=%s)", src_img, len(img_targets), created)
            _delete_prefix_keys(lora_sd, src_img + ".")

    if have_txt and (not src_txt_is_mappable) and txt_targets:
        if any(_has_any_base(t) for t in txt_targets):
            if verbose:
                _LOG.info("[DoRA Power LoRA Loader] flux2 compat: txt modulation targets already present; dropping source %s", src_txt)
            _delete_prefix_keys(lora_sd, src_txt + ".")
        elif len(txt_targets) == 1:
            dst = txt_targets[0]
            n = _rename_prefix_keys(lora_sd, src_txt + ".", dst + ".", delete_from=True)
            if verbose:
                _LOG.info("[DoRA Power LoRA Loader] flux2 compat: renamed %s keys %s -> %s", n, src_txt, dst)
        else:
            created = 0
            for dst in txt_targets:
                created += _clone_base_block(lora_sd, src_txt, dst, scale=scale_txt, allow_suffixes=suf_txt)
            _register_auto_strength_group(src_txt, txt_targets, scale_txt)
            if verbose:
                _LOG.info("[DoRA Power LoRA Loader] flux2 compat: broadcast %s -> %s targets (keys=%s)", src_txt, len(txt_targets), created)
            _delete_prefix_keys(lora_sd, src_txt + ".")

    if have_single and (not src_single_is_mappable) and single_targets:
        if any(_has_any_base(t) for t in single_targets):
            if verbose:
                _LOG.info("[DoRA Power LoRA Loader] flux2 compat: single modulation targets already present; dropping source %s", src_single)
            _delete_prefix_keys(lora_sd, src_single + ".")
        elif len(single_targets) == 1:
            dst = single_targets[0]
            n = _rename_prefix_keys(lora_sd, src_single + ".", dst + ".", delete_from=True)
            if verbose:
                _LOG.info("[DoRA Power LoRA Loader] flux2 compat: renamed %s keys %s -> %s", n, src_single, dst)
        else:
            created = 0
            for dst in single_targets:
                created += _clone_base_block(lora_sd, src_single, dst, scale=scale_single, allow_suffixes=suf_single)
            _register_auto_strength_group(src_single, single_targets, scale_single)
            if verbose:
                _LOG.info(
                    "[DoRA Power LoRA Loader] flux2 compat: broadcast %s -> %s targets (keys=%s)",
                    src_single,
                    len(single_targets),
                    created,
                )
            _delete_prefix_keys(lora_sd, src_single + ".")


def _extract_lora_bases(keys: Iterable[str]) -> Set[str]:
    bases: Set[str] = set()
    for key in keys:
        for suffix in _BASE_SUFFIXES:
            if key.endswith(suffix):
                bases.add(key[: -len(suffix)])
                break
    return bases


def _candidate_base_variants(base: str) -> List[str]:
    """
    Generate plausible variants to improve matching across naming conventions.
    """
    variants = [base]

    # Drop common leading prefixes if present.
    drop_prefixes = (
        "diffusion_model.",
        "model.",
        "base_model.model.",
        "base_model.",
        "unet.",
        "transformer.",
        "text_encoder.",
        "text_encoder_2.",
        "clip_l.",
        "clip_g.",
        "clip_h.",
        "t5xxl.transformer.",
        "hydit_clip.transformer.bert.",
        "text_encoders.",
    )

    for prefix in drop_prefixes:
        if base.startswith(prefix):
            variants.append(base[len(prefix) :])

    # Flux/Flux2 trainer vs ComfyUI naming rewrite.
    # ComfyUI's Modulation uses `lin`, while some trainers export `linear`.
    rewrite_variants: List[str] = []
    for v in variants:
        # .linear -> .lin (end or segment)
        if v.endswith(".linear"):
            rewrite_variants.append(v[: -len(".linear")] + ".lin")
        if ".linear." in v:
            rewrite_variants.append(v.replace(".linear.", ".lin."))
        # ZiT/Lumina2 export variants.
        if ".attention.to.q" in v:
            rewrite_variants.append(v.replace(".attention.to.q", ".attention.to_q"))
        if ".attention.to.k" in v:
            rewrite_variants.append(v.replace(".attention.to.k", ".attention.to_k"))
        if ".attention.to.v" in v:
            rewrite_variants.append(v.replace(".attention.to.v", ".attention.to_v"))
        if v.endswith(".to_out.0"):
            rewrite_variants.append(v[: -len(".to_out.0")] + ".out")
        if ".to_out.0." in v:
            rewrite_variants.append(v.replace(".to_out.0.", ".out."))

    variants.extend(rewrite_variants)

    # De-dup while preserving order.
    out: List[str] = []
    seen: Set[str] = set()
    for variant in variants:
        if variant and variant not in seen:
            seen.add(variant)
            out.append(variant)
    return out


def _pick_best_match(candidates: List[str], prefer_contains: Optional[str] = None) -> Optional[str]:
    if not candidates:
        return None
    if prefer_contains:
        preferred = [candidate for candidate in candidates if prefer_contains in candidate]
        if preferred:
            candidates = preferred
    # Shortest key tends to be the most direct (least extra prefixing).
    return sorted(candidates, key=len)[0]


def _find_weight_key_for_base(sd_keys: Set[str], sd_key_list: List[str], base: str) -> Optional[str]:
    """
    Map a LoRA base name (e.g. transformer.foo.bar) to an actual state_dict weight key.
    Returns the weight key ending in '.weight'.
    """
    variants = _candidate_base_variants(base)

    # Fast exact tries first.
    exact_try = []
    for variant in variants:
        exact_try.extend(
            [
                f"{variant}.weight",
                f"{variant}.lin.weight",  # extra safety if variant already ends with ".linear" and rewrite didn't trigger
                f"diffusion_model.{variant}.weight",
                f"diffusion_model.transformer.{variant}.weight",
                f"transformer.{variant}.weight",
                f"base_model.model.{variant}.weight",
                f"base_model.{variant}.weight",
                f"model.{variant}.weight",
            ]
        )

    for key in exact_try:
        if key in sd_keys:
            return key

    # Suffix scan fallback for unresolved bases.
    suffix_candidates = []
    for variant in variants:
        suffix = f"{variant}.weight"
        for key in sd_key_list:
            if key.endswith(suffix):
                suffix_candidates.append(key)

    return _pick_best_match(suffix_candidates, prefer_contains="diffusion_model.")


def _extend_key_map_with_dynamic_matches(
    key_map: Dict[str, Any],
    lora_bases: Set[str],
    model_sd_keys: Optional[Set[str]],
    model_sd_list: Optional[List[str]],
    clip_sd_keys: Optional[Set[str]],
    clip_sd_list: Optional[List[str]],
    verbose: bool = False,
) -> Tuple[int, List[str]]:
    """
    Add base->weight mappings into key_map for any lora_bases not already present.
    Returns: (num_added, unresolved_bases)
    """
    added = 0
    unresolved = []

    for base in sorted(lora_bases):
        if base in key_map:
            continue

        found = None

        # Prefer model match first.
        if model_sd_keys is not None and model_sd_list is not None:
            found = _find_weight_key_for_base(model_sd_keys, model_sd_list, base)

        # Else try clip.
        if found is None and clip_sd_keys is not None and clip_sd_list is not None:
            found = _find_weight_key_for_base(clip_sd_keys, clip_sd_list, base)

        if found is not None:
            key_map[base] = found
            added += 1
            if verbose:
                _LOG.info("[DoRA Power LoRA Loader] map: %s -> %s", base, found)
        else:
            unresolved.append(base)
            if verbose:
                _LOG.warning("[DoRA Power LoRA Loader] unresolved LoRA base: %s", base)

    return added, unresolved


def _parse_lora_stack_kwargs(kwargs: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Supports two input conventions:

    1) rgthree-style:
       kwargs contains keys like LORA_1, LORA_2... whose values are dicts:
         { on: bool, lora: str, strength: float, strengthTwo?: float }

    2) our simple-per-field style (used by this node's JS UI):
       lora_{i}_enabled, lora_{i}_name, lora_{i}_strength_model, lora_{i}_strength_clip

    Returns a normalized list of entries:
      { on, lora, strength_model, strength_clip }
    """
    entries: List[Dict[str, Any]] = []

    # 1) rgthree-style dict widgets
    for k, v in kwargs.items():
        ku = str(k).upper()
        if not ku.startswith("LORA_"):
            continue
        if not isinstance(v, dict):
            continue
        if "lora" not in v:
            continue
        strength_model = v.get("strength", 0.0)
        strength_clip = v.get("strengthTwo", None)
        if strength_clip is None:
            strength_clip = strength_model
        entries.append(
            {
                "on": bool(v.get("on", True)),
                "lora": v.get("lora"),
                "strength_model": float(strength_model),
                "strength_clip": float(strength_clip),
            }
        )

    # 2) our per-field convention
    idx_re = re.compile(r"^lora_(\d+)_name$", re.IGNORECASE)
    indices: Set[int] = set()
    for k in kwargs.keys():
        m = idx_re.match(str(k))
        if m:
            indices.add(int(m.group(1)))

    for i in sorted(indices):
        name = kwargs.get(f"lora_{i}_name")
        if name is None or name in ("", "None", "NONE"):
            continue
        enabled = kwargs.get(f"lora_{i}_enabled", True)
        sm = kwargs.get(f"lora_{i}_strength_model", kwargs.get(f"lora_{i}_strength", 0.0))
        sc = kwargs.get(f"lora_{i}_strength_clip", kwargs.get(f"lora_{i}_strength_two", None))
        if sc is None:
            sc = sm
        entries.append(
            {
                "on": bool(enabled),
                "lora": name,
                "strength_model": float(sm),
                "strength_clip": float(sc),
            }
        )

    return entries



def _auto_strength_report_line_for_group(item: Dict[str, Any]) -> str:
    logical_id = str(item.get("logical_id") or "?")
    fanout = int(item.get("fanout") or 0)
    ratio = _auto_strength_safe_number(item.get("ratio_applied"))
    target = _auto_strength_safe_number(item.get("target_strength"))
    mean_norm = _auto_strength_safe_number(item.get("mean_norm"))
    cohort_mean = _auto_strength_safe_number(item.get("cohort_mean_norm"))
    if ratio is None:
        ratio_text = "global"
    else:
        ratio_text = f"{ratio:.3f}x"
    target_text = "?" if target is None else f"{target:.4f}"
    norm_text = "?" if mean_norm is None else f"{mean_norm:.6g}"
    cohort_text = "?" if cohort_mean is None else f"{cohort_mean:.6g}"
    return (
        f"    - {item.get('group')}/{item.get('family')} :: {logical_id} "
        f"(fanout={fanout}, ratio={ratio_text}, target={target_text}, norm={norm_text}, cohort={cohort_text})"
    )


def _auto_strength_report_split_groups(logical_groups: Iterable[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    boosts: List[Dict[str, Any]] = []
    pullbacks: List[Dict[str, Any]] = []
    neutral: List[Dict[str, Any]] = []

    for item in logical_groups:
        ratio = _auto_strength_safe_number(item.get("ratio_applied"))
        if ratio is None or abs(ratio - 1.0) <= _AUTO_STRENGTH_DISPLAY_RATIO_EPS:
            neutral.append(item)
        elif ratio > 1.0:
            boosts.append(item)
        else:
            pullbacks.append(item)

    boosts.sort(
        key=lambda item: (
            -(_auto_strength_safe_number(item.get("ratio_applied")) or 1.0),
            str(item.get("group") or ""),
            str(item.get("family") or ""),
            str(item.get("logical_id") or ""),
        )
    )
    pullbacks.sort(
        key=lambda item: (
            (_auto_strength_safe_number(item.get("ratio_applied")) or 1.0),
            str(item.get("group") or ""),
            str(item.get("family") or ""),
            str(item.get("logical_id") or ""),
        )
    )
    neutral.sort(
        key=lambda item: (
            str(item.get("group") or ""),
            str(item.get("family") or ""),
            str(item.get("logical_id") or ""),
        )
    )

    return boosts, pullbacks, neutral


def _build_auto_strength_row_text_report(row: Dict[str, Any]) -> str:
    idx = int(row.get("row_index", 0)) + 1
    lora_name = str(row.get("lora_name") or "None")
    status = str(row.get("status") or "unknown")
    lines = [f"[{idx}] {lora_name}", f"  Status           : {status}"]
    if row.get("status_detail"):
        lines.append(f"  Detail           : {row.get('status_detail')}")
    lines.append(f"  Strength model   : {float(row.get('strength_model', 0.0)):.4f}")
    lines.append(f"  Strength clip    : {float(row.get('strength_clip', 0.0)):.4f}")

    report = row.get("report") if isinstance(row.get("report"), dict) else None
    if report is None:
        return "\n".join(lines)

    lines.extend(
        [
            f"  Analysis device  : {report.get('analysis_device_mode')} (load_device={report.get('analysis_load_device')})",
            f"  Ratio window     : {float(report.get('ratio_floor', 0.0)):.4f} .. {float(report.get('ratio_ceiling', 0.0)):.4f}",
            f"  Bases            : mapped={int(report.get('mapped_bases', 0))} analyzable={int(report.get('analyzable_bases', 0))} measured={int(report.get('measured_bases', 0))}",
            f"  Logical groups   : total={int(report.get('logical_groups_total', 0))} measured={int(report.get('logical_groups_measured', 0))} skipped_zero_strength={int(report.get('logical_groups_skipped_zero_strength', 0))}",
            "  Cohorts:",
        ]
    )
    cohorts = report.get("cohorts") if isinstance(report.get("cohorts"), list) else []
    if cohorts:
        for cohort in cohorts:
            mean_norm = _auto_strength_safe_number(cohort.get("mean_norm"))
            mean_text = "?" if mean_norm is None else f"{mean_norm:.6g}"
            lines.append(
                "    - {group}/{family}: mean={mean} logical={logical} measured_logical={measured_logical} bases={bases} measured_bases={measured_bases} skipped_zero_strength_bases={skipped}".format(
                    group=cohort.get("group"),
                    family=cohort.get("family"),
                    mean=mean_text,
                    logical=int(cohort.get("logical_count", 0)),
                    measured_logical=int(cohort.get("measured_logical_count", 0)),
                    bases=int(cohort.get("base_count", 0)),
                    measured_bases=int(cohort.get("measured_base_count", 0)),
                    skipped=int(cohort.get("skipped_zero_strength_base_count", 0)),
                )
            )
    else:
        lines.append("    - none")

    logical_groups = report.get("logical_groups") if isinstance(report.get("logical_groups"), list) else []
    boosts, pullbacks, neutral = _auto_strength_report_split_groups(logical_groups)

    lines.append("  Strongest boosts:")
    if boosts:
        for item in boosts[:6]:
            lines.append(_auto_strength_report_line_for_group(item))
        remaining = len(boosts) - min(6, len(boosts))
        if remaining > 0:
            lines.append(f"    - ... {remaining} more boost groups in JSON report")
    else:
        lines.append("    - none")

    lines.append("  Strongest pullbacks:")
    if pullbacks:
        for item in pullbacks[:6]:
            lines.append(_auto_strength_report_line_for_group(item))
        remaining = len(pullbacks) - min(6, len(pullbacks))
        if remaining > 0:
            lines.append(f"    - ... {remaining} more pullback groups in JSON report")
    else:
        lines.append("    - none")

    if neutral:
        lines.append("  Near global:")
        for item in neutral[:6]:
            lines.append(_auto_strength_report_line_for_group(item))
        remaining = len(neutral) - min(6, len(neutral))
        if remaining > 0:
            lines.append(f"    - ... {remaining} more near-global groups in JSON report")

    return "\n".join(lines)


def _build_auto_strength_stack_text_report(stack_report: Dict[str, Any]) -> str:
    rows = stack_report.get("rows") if isinstance(stack_report.get("rows"), list) else []
    analyzed = sum(1 for row in rows if row.get("status") == "analyzed" and isinstance(row.get("report"), dict))
    lines = [
        "DoRA Power LoRA Loader — auto-strength analysis report",
        f"Node auto-strength enabled : {bool(stack_report.get('auto_strength_enabled', False))}",
        f"Requested device           : {stack_report.get('auto_strength_device', 'auto')}",
        f"Ratio window              : {float(stack_report.get('ratio_floor', 0.0)):.4f} .. {float(stack_report.get('ratio_ceiling', 0.0)):.4f}",
        f"Rows total/analyzed       : {len(rows)}/{analyzed}",
    ]
    if not rows:
        lines.append("No active LoRA rows were processed.")
        return "\n".join(lines)

    lines.append("")
    for idx, row in enumerate(rows):
        if idx > 0:
            lines.append("-" * 88)
        lines.append(_build_auto_strength_row_text_report(row))
    return "\n".join(lines)

class DoraPowerLoraLoader:
    """
    Power LoRA Loader-style stack + DoRA/Flux2 key-fix loader.

    - Accepts a dynamic number of LoRAs (single node, like rgthree Power Lora Loader)
    - Fixes Flux/Flux2 naming mismatches (e.g. `.linear` vs `.lin`, time guidance embed naming)
    - Uses ComfyUI's core DoRA implementation (comfy.lora.load_lora)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
            },
            # Dynamic/stack inputs injected by the JS UI (and/or by other frontends).
            "optional": FlexibleOptionalInputType(
                any_type,
                data={
                    # Flux2 modulation handling
                    "broadcast_modulations": ("BOOLEAN", {"default": True}),
                    "broadcast_include_dora_scale": ("BOOLEAN", {"default": False}),

                    # DoRA decompose debugging (node-adjustable)
                    "dora_decompose_debug": ("BOOLEAN", {"default": False}),
                    "dora_decompose_debug_n": ("INT", {"default": 30, "min": 0, "max": 500, "step": 1}),
                    "dora_decompose_debug_stack_depth": ("INT", {"default": 10, "min": 2, "max": 64, "step": 1}),
                    # Slice-aware magnitude fix for offset/sliced patches (recommended ON for Flux2)
                    "dora_slice_fix": ("BOOLEAN", {"default": True}),
                    "dora_adaln_swap_fix": ("BOOLEAN", {"default": True}),
                    # Z-Image Turbo / Lumina2 architecture-aware normalization
                    "zimage_lumina2_compat": ("BOOLEAN", {"default": True}),

                    # Optional per-base auto-strength redistribution
                    "auto_strength_enabled": ("BOOLEAN", {"default": False}),
                    "auto_strength_device": (["auto", "cpu", "gpu"], {"default": "gpu"}),
                    "auto_strength_ratio_floor": ("FLOAT", {"default": _AUTO_STRENGTH_RATIO_FLOOR, "min": 0.0, "max": 16.0, "step": 0.01}),
                    "auto_strength_ratio_ceiling": ("FLOAT", {"default": _AUTO_STRENGTH_RATIO_CEILING, "min": 0.0, "max": 16.0, "step": 0.01}),
                },
            ),
            "hidden": {},
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "auto_strength_report_json", "analysis_report")
    FUNCTION = "load_loras"
    CATEGORY = "loaders"

    def _load_one(
        self,
        model,
        clip,
        lora_name: str,
        strength_model: float,
        strength_clip: float,
        verbose: bool,
        log_unloaded_keys: bool,
        broadcast_auto_scale: bool,
        broadcast_scale: float,
        broadcast_modulations: bool,
        broadcast_include_dora_scale: bool,
        model_state_dict: Optional[Dict[str, Any]],
        model_sd_keys: Optional[Set[str]],
        model_sd_list: Optional[List[str]],
        clip_state_dict: Optional[Dict[str, Any]],
        clip_sd_keys: Optional[Set[str]],
        clip_sd_list: Optional[List[str]],
        analysis_load_device: Any,
        zimage_lumina2_compat: bool,
        auto_strength_enabled: bool,
        auto_strength_device: str,
        auto_strength_ratio_floor: float,
        auto_strength_ratio_ceiling: float,
    ):
        auto_strength_report: Optional[Dict[str, Any]] = None
        lora_path = folder_paths.get_full_path("loras", lora_name)
        if not lora_path:
            raise FileNotFoundError(f"LoRA not found: {lora_name}")

        # Load and normalize formats.
        try:
            lora_sd_raw = comfy.utils.load_torch_file(lora_path, safe_load=True)
        except TypeError:
            # Older ComfyUI builds may not expose safe_load kwarg
            lora_sd_raw = comfy.utils.load_torch_file(lora_path)

        # RAW stats (before convert_lora) – tells us if training/export already produced zero lora_up.
        raw_up_n, raw_up_z, raw_up_mx, raw_up_dt = _suffix_tensor_stats(lora_sd_raw, ".lora_up.weight")
        raw_dn_n, raw_dn_z, raw_dn_mx, raw_dn_dt = _suffix_tensor_stats(lora_sd_raw, ".lora_down.weight")
        if verbose and (raw_up_n or raw_dn_n):
            _LOG.info(
                "[DoRA Power LoRA Loader] %s (raw): .lora_up n=%d zero=%d max|x|=%g dtypes=%s | "
                ".lora_down n=%d zero=%d max|x|=%g dtypes=%s",
                lora_name,
                raw_up_n,
                raw_up_z,
                raw_up_mx,
                raw_up_dt,
                raw_dn_n,
                raw_dn_z,
                raw_dn_mx,
                raw_dn_dt,
            )

        if raw_up_n and raw_up_mx == 0.0:
            _LOG.warning(
                "[DoRA Power LoRA Loader] %s (raw): ALL lora_up matrices are zero. "
                "This LoRA has no direction update, so lora_diff will be 0 and it will barely change the image. "
                "This is a training/export issue, not a loader issue.",
                lora_name,
            )

        # Convert (needed for some formats)
        lora_sd_conv = comfy.lora_convert.convert_lora(lora_sd_raw)

        # CONVERTED stats – tells us if convert_lora is zeroing lora_up.
        conv_up_n, conv_up_z, conv_up_mx, conv_up_dt = _suffix_tensor_stats(lora_sd_conv, ".lora_up.weight")
        conv_dn_n, conv_dn_z, conv_dn_mx, conv_dn_dt = _suffix_tensor_stats(lora_sd_conv, ".lora_down.weight")
        if verbose and (conv_up_n or conv_dn_n):
            _LOG.info(
                "[DoRA Power LoRA Loader] %s (converted): .lora_up n=%d zero=%d max|x|=%g dtypes=%s | "
                ".lora_down n=%d zero=%d max|x|=%g dtypes=%s",
                lora_name,
                conv_up_n,
                conv_up_z,
                conv_up_mx,
                conv_up_dt,
                conv_dn_n,
                conv_dn_z,
                conv_dn_mx,
                conv_dn_dt,
            )

        # If conversion killed lora_up, bypass conversion for this file.
        if raw_up_n and raw_up_mx > 0.0 and conv_up_n == raw_up_n and conv_up_mx == 0.0:
            _LOG.warning(
                "[DoRA Power LoRA Loader] %s: convert_lora appears to zero lora_up (raw max|x|=%g -> converted max|x|=0). "
                "Bypassing convert_lora for this file.",
                lora_name,
                raw_up_mx,
            )
            # Reload fresh to avoid in-place conversion side effects.
            try:
                lora_sd = comfy.utils.load_torch_file(lora_path, safe_load=True)
            except TypeError:
                lora_sd = comfy.utils.load_torch_file(lora_path)
        else:
            lora_sd = lora_sd_conv

        _normalize_diffusers_dora_magnitude_keys(lora_sd, verbose=verbose)

        if verbose:
            n_dora = sum(1 for k in lora_sd.keys() if str(k).endswith(".dora_scale"))
            _LOG.info("[DoRA Power LoRA Loader] %s: dora_scale keys=%d total_keys=%d", lora_name, n_dora, len(lora_sd))

        _log_lora_tensor_health(lora_name, lora_sd, verbose=verbose)
        _log_lora_direction_stats(lora_name, lora_sd, verbose=verbose)

        # Start with standard ComfyUI key map.
        key_map: Dict[str, Any] = {}
        if model is not None:
            key_map = comfy.lora.model_lora_keys_unet(model.model, key_map)
        if clip is not None:
            key_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, key_map)

        if zimage_lumina2_compat and model is not None:
            _apply_zimage_lumina2_compat(
                lora_sd=lora_sd,
                model=model,
                model_sd_keys=model_sd_keys,
                key_map=key_map,
                verbose=verbose,
            )

        # Flux2/OneTrainer DoRA compat: rewrite + broadcast missing modules into keys ComfyUI maps.
        # This fixes cases where critical dora_scale/lora_up/down tensors never map/load.
        auto_strength_logical_groups: Dict[str, Tuple[str, float]] = {}
        if model is not None:
            _apply_flux2_onetrainer_dora_compat(
                lora_sd=lora_sd,
                model=model,
                model_sd_keys=model_sd_keys,
                key_map=key_map,
                verbose=verbose,
                broadcast_auto_scale=broadcast_auto_scale,
                broadcast_scale=broadcast_scale,
                broadcast_modulations=broadcast_modulations,
                broadcast_include_dora_scale=broadcast_include_dora_scale,
                auto_strength_logical_groups=auto_strength_logical_groups,
            )

        # Extract base module names from file keys (after compat rewrites/broadcast).
        lora_bases = _extract_lora_bases(lora_sd.keys())
        if verbose:
            _LOG.info("[DoRA Power LoRA Loader] %s: bases in file: %s", lora_name, len(lora_bases))

        # Extend map with missing bases from the file (includes Flux/Flux2 rewrites in _candidate_base_variants()).
        added, unresolved = _extend_key_map_with_dynamic_matches(
            key_map=key_map,
            lora_bases=lora_bases,
            model_sd_keys=model_sd_keys,
            model_sd_list=model_sd_list,
            clip_sd_keys=clip_sd_keys,
            clip_sd_list=clip_sd_list,
            verbose=verbose,
        )

        if verbose:
            _LOG.info(
                "[DoRA Power LoRA Loader] %s: dynamic mappings added: %s, unresolved: %s",
                lora_name,
                added,
                len(unresolved),
            )

        _fix_onetrainer_output_axis_dora_mats(
            lora_sd=lora_sd,
            key_map=key_map,
            model_state_dict=model_state_dict,
            clip_state_dict=clip_state_dict,
            verbose=verbose,
        )
        _log_lora_direction_stats(lora_name + " (post-fix)", lora_sd, verbose=verbose)

        if auto_strength_enabled and (abs(float(strength_model)) > _AUTO_STRENGTH_EPS or abs(float(strength_clip)) > _AUTO_STRENGTH_EPS):
            base_strengths, auto_strength_report = _auto_strength_analyze_base_targets(
                lora_sd=lora_sd,
                lora_bases=lora_bases,
                key_map=key_map,
                model_state_dict=model_state_dict,
                clip_state_dict=clip_state_dict,
                analysis_device_mode=auto_strength_device,
                analysis_load_device=analysis_load_device,
                strength_model=strength_model,
                strength_clip=strength_clip,
                ratio_floor=auto_strength_ratio_floor,
                ratio_ceiling=auto_strength_ratio_ceiling,
                logical_groups=auto_strength_logical_groups,
                verbose=verbose,
            )
            base_ratios = _auto_strength_targets_to_ratios(
                base_strengths=base_strengths,
                key_map=key_map,
                model_state_dict=model_state_dict,
                clip_state_dict=clip_state_dict,
                strength_model=strength_model,
                strength_clip=strength_clip,
            )
            lora_sd, _ = _apply_base_strength_ratios(lora_sd, base_ratios)
            _log_lora_direction_stats(lora_name + " (post-auto-strength)", lora_sd, verbose=verbose)

        # Load patches (DoRA handling remains in comfy.lora internals).
        try:
            loaded = comfy.lora.load_lora(lora_sd, key_map, log_missing=log_unloaded_keys)
        except TypeError:
            # Fallbacks for older/variant signatures
            try:
                loaded = comfy.lora.load_lora(lora_sd, key_map, log_unloaded_keys)
            except TypeError:
                loaded = comfy.lora.load_lora(lora_sd, key_map)

        _log_loaded_tensor_health(lora_name, loaded, verbose=verbose)

        # Apply patches to provided model/clip (already cloned by caller).
        applied_m = []
        applied_c = []
        patch_strength_model = strength_model
        patch_strength_clip = strength_clip

        if model is not None:
            try:
                applied_m = model.add_patches(loaded, patch_strength_model) or []
            except Exception:
                model.add_patches(loaded, patch_strength_model)
        if clip is not None:
            try:
                applied_c = clip.add_patches(loaded, patch_strength_clip) or []
            except Exception:
                clip.add_patches(loaded, patch_strength_clip)

        if verbose:
            def _n(x):
                try:
                    return len(x)
                except Exception:
                    return 0

            _LOG.info(
                "[DoRA Power LoRA Loader] %s: patches=%s applied(model)=%s applied(clip)=%s strengths(m/c)=%s/%s",
                lora_name,
                _n(loaded),
                _n(applied_m),
                _n(applied_c),
                patch_strength_model,
                patch_strength_clip,
            )
            if isinstance(applied_m, list) and applied_m:
                _LOG.info("[DoRA Power LoRA Loader] %s: sample applied(model) keys: %s", lora_name, applied_m[:10])
            if isinstance(applied_c, list) and applied_c:
                _LOG.info("[DoRA Power LoRA Loader] %s: sample applied(clip) keys: %s", lora_name, applied_c[:10])

        return model, clip, auto_strength_report

    def load_loras(self, model, clip, **kwargs):
        # Global controls (provided by JS UI; also safe if absent)
        stack_enabled = bool(kwargs.get("stack_enabled", True))
        verbose = bool(kwargs.get("verbose", False))
        log_unloaded_keys = bool(kwargs.get("log_unloaded_keys", False))
        broadcast_auto_scale = bool(kwargs.get("broadcast_auto_scale", True))
        broadcast_modulations = bool(kwargs.get("broadcast_modulations", True))
        broadcast_include_dora_scale = bool(kwargs.get("broadcast_include_dora_scale", False))
        try:
            broadcast_scale = float(kwargs.get("broadcast_scale", 1.0))
        except Exception:
            broadcast_scale = 1.0

        # DoRA decompose debug controls (node-adjustable)
        dora_dbg = bool(kwargs.get("dora_decompose_debug", False))
        try:
            dora_dbg_n = int(kwargs.get("dora_decompose_debug_n", 30))
        except Exception:
            dora_dbg_n = 30
        try:
            dora_dbg_stack = int(kwargs.get("dora_decompose_debug_stack_depth", 10))
        except Exception:
            dora_dbg_stack = 10
        dora_slice_fix = bool(kwargs.get("dora_slice_fix", True))
        dora_adaln_swap_fix = bool(kwargs.get("dora_adaln_swap_fix", True))
        zimage_lumina2_compat = bool(kwargs.get("zimage_lumina2_compat", True))
        auto_strength_enabled = bool(kwargs.get("auto_strength_enabled", False))
        auto_strength_device = _normalize_auto_strength_device(kwargs.get("auto_strength_device", "gpu"))
        try:
            auto_strength_ratio_floor = float(kwargs.get("auto_strength_ratio_floor", _AUTO_STRENGTH_RATIO_FLOOR))
        except Exception:
            auto_strength_ratio_floor = _AUTO_STRENGTH_RATIO_FLOOR
        try:
            auto_strength_ratio_ceiling = float(kwargs.get("auto_strength_ratio_ceiling", _AUTO_STRENGTH_RATIO_CEILING))
        except Exception:
            auto_strength_ratio_ceiling = _AUTO_STRENGTH_RATIO_CEILING
        if auto_strength_ratio_ceiling < auto_strength_ratio_floor:
            auto_strength_ratio_floor, auto_strength_ratio_ceiling = auto_strength_ratio_ceiling, auto_strength_ratio_floor

        _set_dora_decomp_cfg(
            dbg=dora_dbg,
            dbg_n=dora_dbg_n,
            dbg_stack=dora_dbg_stack,
            slice_fix=dora_slice_fix,
            adaln_swap_fix=dora_adaln_swap_fix,
        )

        report_rows: List[Dict[str, Any]] = []

        if not stack_enabled:
            stack_report = {
                "schema": 1,
                "kind": "dora_power_lora_auto_strength_stack_report",
                "auto_strength_enabled": auto_strength_enabled,
                "auto_strength_device": auto_strength_device,
                "ratio_floor": auto_strength_ratio_floor,
                "ratio_ceiling": auto_strength_ratio_ceiling,
                "rows": report_rows,
            }
            report_json = _auto_strength_json_dumps(stack_report, pretty=True)
            report_text = _build_auto_strength_stack_text_report(stack_report)
            return {
                "result": (model, clip, report_json, report_text),
                "ui": {
                    "auto_strength_report_json": (report_json,),
                    "analysis_report": (report_text,),
                },
            }

        entries = _parse_lora_stack_kwargs(kwargs)
        if not entries:
            stack_report = {
                "schema": 1,
                "kind": "dora_power_lora_auto_strength_stack_report",
                "auto_strength_enabled": auto_strength_enabled,
                "auto_strength_device": auto_strength_device,
                "ratio_floor": auto_strength_ratio_floor,
                "ratio_ceiling": auto_strength_ratio_ceiling,
                "rows": report_rows,
            }
            report_json = _auto_strength_json_dumps(stack_report, pretty=True)
            report_text = _build_auto_strength_stack_text_report(stack_report)
            return {
                "result": (model, clip, report_json, report_text),
                "ui": {
                    "auto_strength_report_json": (report_json,),
                    "analysis_report": (report_text,),
                },
            }

        # Clone once, then apply multiple loras onto the same patched instances.
        new_model = model.clone() if model is not None else None
        new_clip = clip.clone() if clip is not None else None

        # Prepare state_dict key sets/lists once for dynamic matching.
        model_sd_keys = model_sd_list = None
        clip_sd_keys = clip_sd_list = None
        model_state_dict = None
        clip_state_dict = None

        if new_model is not None:
            model_state_dict = new_model.model.state_dict()
            model_sd_list = list(model_state_dict.keys())
            model_sd_keys = set(model_sd_list)

        if new_clip is not None:
            clip_state_dict = new_clip.cond_stage_model.state_dict()
            clip_sd_list = list(clip_state_dict.keys())
            clip_sd_keys = set(clip_sd_list)

        analysis_load_device = _auto_strength_get_analysis_load_device(new_model, new_clip)
        if auto_strength_enabled and auto_strength_device == "gpu":
            resolved_analysis_device = _torch_device_or_none(analysis_load_device)
            if resolved_analysis_device is None or resolved_analysis_device.type == "cpu":
                _LOG.warning(
                    "[DoRA Power LoRA Loader] auto-strength: requested analysis device 'gpu' but no usable accelerator load_device was found; falling back to cpu"
                )

        for row_index, e in enumerate(entries):
            lora_name = e.get("lora")
            row_info: Dict[str, Any] = {
                "row_index": row_index,
                "enabled": bool(e.get("on", True)),
                "lora_name": str(lora_name or "None"),
            }
            if not lora_name or lora_name in ("None", "NONE"):
                row_info.update({
                    "status": "empty",
                    "status_detail": "No LoRA selected.",
                    "strength_model": 0.0,
                    "strength_clip": 0.0,
                })
                report_rows.append(row_info)
                continue
            if not e.get("on", True):
                continue
            sm = float(e.get("strength_model", 0.0))
            sc = float(e.get("strength_clip", sm))
            row_info.update({
                "strength_model": sm,
                "strength_clip": sc,
            })
            if abs(sm) <= _AUTO_STRENGTH_EPS and abs(sc) <= _AUTO_STRENGTH_EPS:
                row_info.update({
                    "status": "zero_strength",
                    "status_detail": "Both model and clip strengths are zero or below analysis epsilon.",
                })
                report_rows.append(row_info)
                continue

            new_model, new_clip, auto_strength_report = self._load_one(
                new_model,
                new_clip,
                lora_name=lora_name,
                strength_model=sm,
                strength_clip=sc,
                verbose=verbose,
                log_unloaded_keys=log_unloaded_keys,
                broadcast_auto_scale=broadcast_auto_scale,
                broadcast_scale=broadcast_scale,
                broadcast_modulations=broadcast_modulations,
                broadcast_include_dora_scale=broadcast_include_dora_scale,
                model_state_dict=model_state_dict,
                model_sd_keys=model_sd_keys,
                model_sd_list=model_sd_list,
                clip_state_dict=clip_state_dict,
                clip_sd_keys=clip_sd_keys,
                clip_sd_list=clip_sd_list,
                analysis_load_device=analysis_load_device,
                zimage_lumina2_compat=zimage_lumina2_compat,
                auto_strength_enabled=auto_strength_enabled,
                auto_strength_device=auto_strength_device,
                auto_strength_ratio_floor=auto_strength_ratio_floor,
                auto_strength_ratio_ceiling=auto_strength_ratio_ceiling,
            )
            did_analyze = isinstance(auto_strength_report, dict)
            if did_analyze:
                status = "analyzed"
                detail = "Auto-strength report generated."
            elif auto_strength_enabled:
                status = "auto_strength_skipped"
                detail = "Auto-strength was enabled, but no analysis report was generated."
            else:
                status = "applied_without_auto_strength"
                detail = "LoRA applied without auto-strength analysis."

            row_info.update(
                {
                    "status": status,
                    "status_detail": detail,
                    "report": auto_strength_report if did_analyze else None,
                }
            )
            report_rows.append(row_info)

        stack_report = {
            "schema": 1,
            "kind": "dora_power_lora_auto_strength_stack_report",
            "auto_strength_enabled": auto_strength_enabled,
            "auto_strength_device": auto_strength_device,
            "ratio_floor": auto_strength_ratio_floor,
            "ratio_ceiling": auto_strength_ratio_ceiling,
            "rows": report_rows,
        }
        report_json = _auto_strength_json_dumps(stack_report, pretty=True)
        report_text = _build_auto_strength_stack_text_report(stack_report)
        return {
            "result": (new_model, new_clip, report_json, report_text),
            "ui": {
                "auto_strength_report_json": (report_json,),
                "analysis_report": (report_text,),
            },
        }
