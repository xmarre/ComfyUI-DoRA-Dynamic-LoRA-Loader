import logging
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
    _LOG.warning("[DoRA Power LoRA Loader] patched ComfyUI weight_decompose for correct DoRA normalization (norm(V) + slice fix).")

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
        except Exception:
            sig = None
            has_intermediate = False
            idx_intermediate = -1

        def calculate_weight_fixed(self, *args, **kwargs):
            a = list(args)

            if has_intermediate and idx_intermediate >= 0:
                if len(a) > idx_intermediate:
                    a[idx_intermediate] = torch.float32
                if len(a) <= idx_intermediate:
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


def _state_dict_looks_quantized(sd: Dict[str, Any]) -> bool:
    """Heuristic: any int8/uint8/float8 tensors in state_dict => quantized/mixed precision build."""
    float8_dtypes = []
    for name in (
        "float8_e4m3fn",
        "float8_e5m2",
        "float8_e4m3fnuz",
        "float8_e5m2fnuz",
    ):
        dt = getattr(torch, name, None)
        if dt is not None:
            float8_dtypes.append(dt)

    for v in sd.values():
        if not isinstance(v, torch.Tensor):
            continue
        if v.dtype in (torch.int8, torch.uint8):
            return True
        if float8_dtypes and v.dtype in tuple(float8_dtypes):
            return True
    return False


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
    key_map: Optional[Dict[str, str]] = None,
    verbose: bool = False,
    broadcast_auto_scale: bool = True,
    broadcast_scale: float = 1.0,
    broadcast_modulations: bool = True,
    broadcast_include_dora_scale: bool = False,
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
    key_map: Dict[str, str],
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
                },
            ),
            "hidden": {},
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("MODEL", "CLIP")
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
        model_is_quantized: bool,
        model_sd_keys: Optional[Set[str]],
        model_sd_list: Optional[List[str]],
        clip_sd_keys: Optional[Set[str]],
        clip_sd_list: Optional[List[str]],
    ):
        lora_path = folder_paths.get_full_path("loras", lora_name)
        if not lora_path:
            raise FileNotFoundError(f"LoRA not found: {lora_name}")

        # Load and normalize formats.
        try:
            lora_sd = comfy.utils.load_torch_file(lora_path, safe_load=True)
        except TypeError:
            # Older ComfyUI builds may not expose safe_load kwarg
            lora_sd = comfy.utils.load_torch_file(lora_path)
        lora_sd = comfy.lora_convert.convert_lora(lora_sd)

        if verbose:
            n_dora = sum(1 for k in lora_sd.keys() if str(k).endswith(".dora_scale"))
            _LOG.info("[DoRA Power LoRA Loader] %s: dora_scale keys=%d total_keys=%d", lora_name, n_dora, len(lora_sd))

        _log_lora_tensor_health(lora_name, lora_sd, verbose=verbose)

        # If your Flux2 base is quantized/mixed-precision, DoRA can produce NaNs in some ComfyUI builds
        # (pink/magenta decode). Key-mapping can be 100% correct and you'll still get pink.
        # We can't safely dequantize here without depending on ComfyUI internals, so we emit a clear warning.
        if model_is_quantized and any(str(k).endswith(".dora_scale") for k in lora_sd.keys()):
            _LOG.warning(
                "[DoRA Power LoRA Loader] %s: quantized/mixed-precision base model detected; DoRA may produce NaNs (pink). "
                "If this happens, test with an FP16 base (non-quantized) to confirm.",
                lora_name,
            )

        # Start with standard ComfyUI key map.
        key_map: Dict[str, str] = {}
        if model is not None:
            key_map = comfy.lora.model_lora_keys_unet(model.model, key_map)
        if clip is not None:
            key_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, key_map)

        # Flux2/OneTrainer DoRA compat: rewrite + broadcast missing modules into keys ComfyUI maps.
        # This fixes cases where critical dora_scale/lora_up/down tensors never map/load.
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
        if model is not None:
            try:
                applied_m = model.add_patches(loaded, strength_model) or []
            except Exception:
                model.add_patches(loaded, strength_model)
        if clip is not None:
            try:
                applied_c = clip.add_patches(loaded, strength_clip) or []
            except Exception:
                clip.add_patches(loaded, strength_clip)

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
                strength_model,
                strength_clip,
            )
            if isinstance(applied_m, list) and applied_m:
                _LOG.info("[DoRA Power LoRA Loader] %s: sample applied(model) keys: %s", lora_name, applied_m[:10])
            if isinstance(applied_c, list) and applied_c:
                _LOG.info("[DoRA Power LoRA Loader] %s: sample applied(clip) keys: %s", lora_name, applied_c[:10])

        return model, clip

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
        _set_dora_decomp_cfg(
            dbg=dora_dbg,
            dbg_n=dora_dbg_n,
            dbg_stack=dora_dbg_stack,
            slice_fix=dora_slice_fix,
            adaln_swap_fix=dora_adaln_swap_fix,
        )

        if not stack_enabled:
            return (model, clip)

        entries = _parse_lora_stack_kwargs(kwargs)
        if not entries:
            return (model, clip)

        # Clone once, then apply multiple loras onto the same patched instances.
        new_model = model.clone() if model is not None else None
        new_clip = clip.clone() if clip is not None else None

        # Prepare state_dict key sets/lists once for dynamic matching.
        model_sd_keys = model_sd_list = None
        clip_sd_keys = clip_sd_list = None
        model_is_quantized = False

        if new_model is not None:
            model_state_dict = new_model.model.state_dict()
            model_sd_list = list(model_state_dict.keys())
            model_sd_keys = set(model_sd_list)
            model_is_quantized = _state_dict_looks_quantized(model_state_dict)
            if model_is_quantized:
                _LOG.warning(
                    "[DoRA Power LoRA Loader] quantized/mixed-precision Flux/Flux2 detected in UNet state_dict; "
                    "DoRA LoRAs can still map correctly but may output pink if the DoRA math path is unstable on quantized weights."
                )

        if new_clip is not None:
            clip_state_dict = new_clip.cond_stage_model.state_dict()
            clip_sd_list = list(clip_state_dict.keys())
            clip_sd_keys = set(clip_sd_list)

        for e in entries:
            lora_name = e.get("lora")
            if not lora_name or lora_name in ("None", "NONE"):
                continue
            if not e.get("on", True):
                continue
            sm = float(e.get("strength_model", 0.0))
            sc = float(e.get("strength_clip", sm))
            if sm == 0.0 and sc == 0.0:
                continue
            new_model, new_clip = self._load_one(
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
                model_is_quantized=model_is_quantized,
                model_sd_keys=model_sd_keys,
                model_sd_list=model_sd_list,
                clip_sd_keys=clip_sd_keys,
                clip_sd_list=clip_sd_list,
            )

        return (new_model, new_clip)
