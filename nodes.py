import logging
import json
import math
import random
import os
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


_STATE_SEED_MIN = -1125899906842624
_STATE_SEED_MAX = 1125899906842624
_STATE_SEED_SPECIALS = {-1, -2, -3}


def _state_manager_new_random_seed() -> int:
    return random.SystemRandom().randint(1, _STATE_SEED_MAX)


def _coerce_state_manager_seed(value: Any, fallback: int = 0) -> int:
    try:
        if isinstance(value, bool):
            raise TypeError("boolean is not a seed")
        if isinstance(value, float) and not math.isfinite(value):
            raise TypeError("non-finite float is not a seed")
        if isinstance(value, str) and not value.strip():
            raise TypeError("blank string is not a seed")
        n = int(float(value))
    except Exception:
        try:
            n = int(fallback)
        except Exception:
            n = 0
    return max(_STATE_SEED_MIN, min(_STATE_SEED_MAX, n))

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



# --------------------------------------------------------------------------------------
# DoRA state manager payload helpers
# --------------------------------------------------------------------------------------

_DORA_STATE_MANAGER_SCHEMA_VERSION = 2
_DORA_STATE_KIND = "dora_state_manager_state"
_DORA_LORA_STACK_KIND = "dora_lora_stack"
_DORA_STATE_SETTINGS_KIND = "dora_state_settings"
_STATE_MANAGER_CONTROL_KIND = "state_manager_control"
_DORA_STATE_LOADER_GLOBAL_KEYS: Set[str] = {
    "stack_enabled",
    "verbose",
    "log_unloaded_keys",
    "broadcast_auto_scale",
    "broadcast_modulations",
    "broadcast_include_dora_scale",
    "broadcast_scale",
    "dora_decompose_debug",
    "dora_decompose_debug_n",
    "dora_decompose_debug_stack_depth",
    "dora_slice_fix",
    "dora_adaln_swap_fix",
    "zimage_lumina2_compat",
    "auto_strength_enabled",
    "auto_strength_device",
    "auto_strength_ratio_floor",
    "auto_strength_ratio_ceiling",
}


def _state_manager_make_id(prefix: str, index: int) -> str:
    return f"{prefix}_{index + 1}"


def _state_manager_default_state() -> Dict[str, Any]:
    return {
        "version": _DORA_STATE_MANAGER_SCHEMA_VERSION,
        "characters": [
            {
                "id": "default_character",
                "name": "Default Character",
                "thumbnail": {},
                "loader_stacks": [
                    {
                        "slot": "default",
                        "label": "Default loader",
                        "loras": [],
                        "loader_globals": {},
                    }
                ],
                # Legacy/default stack mirror. Kept so older workflows and older JS still load.
                "loras": [],
                "loader_globals": {},
                "prompts": [
                    {
                        "id": "default_prompt",
                        "name": "Default Prompt",
                        "positive": "",
                        "negative": "",
                        "text_boxes": [
                            {"role": "positive", "slot": "default", "label": "Default positive", "text": ""},
                            {"role": "negative", "slot": "default", "label": "Default negative", "text": ""},
                        ],
                        "settings": {},
                        "reference_image": {},
                        "fileimage_prefix": "",
                    }
                ],
            }
        ],
    }


def _safe_json_load(raw: Any, fallback: Any) -> Any:
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return fallback
        try:
            return json.loads(text)
        except Exception:
            return fallback
    if isinstance(raw, dict):
        return raw
    return fallback


def _clean_state_id(value: Any, fallback: str) -> str:
    text = str(value or "").strip()
    if not text:
        return fallback
    text = re.sub(r"[^A-Za-z0-9_.:-]+", "_", text).strip("_")
    return text or fallback


def _state_manager_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
        return default
    return bool(value)


def _normalize_manager_lora_row(row: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(row, dict):
        return None
    name = str(row.get("name", row.get("lora", row.get("lora_name", "None"))) or "None").strip()
    if not name:
        name = "None"
    try:
        strength_model = float(row.get("strength_model", row.get("strengthModel", row.get("strength", 1.0))))
    except Exception:
        strength_model = 1.0
    try:
        strength_clip = float(row.get("strength_clip", row.get("strengthClip", row.get("strengthTwo", strength_model))))
    except Exception:
        strength_clip = strength_model
    return {
        "enabled": _state_manager_bool(row.get("enabled", row.get("on", True)), default=True),
        "name": name,
        "strength_model": strength_model,
        "strength_clip": strength_clip,
    }


def _normalize_manager_settings(settings: Any) -> Dict[str, Any]:
    if isinstance(settings, dict):
        return settings
    parsed = _safe_json_load(settings, {})
    return parsed if isinstance(parsed, dict) else {}


def _normalize_manager_text_role(value: Any, fallback: str = "generic") -> str:
    text = str(value or fallback).strip().lower()
    if text in {"positive", "pos"} or "positive" in text:
        return "positive"
    if text in {"negative", "neg"} or "negative" in text:
        return "negative"
    if text == "generic":
        return "generic"
    return fallback


def _is_manager_text_role_key(value: Any) -> bool:
    return _normalize_manager_text_role(value, "") in {"positive", "negative", "generic"}


def _clean_text_slot(value: Any, fallback: str = "default") -> str:
    return _clean_state_id(value, fallback)


def _text_box_key(role: Any, slot: Any) -> str:
    return f"{_normalize_manager_text_role(role)}::{_clean_text_slot(slot)}"


def _default_manager_text_box(role: str, slot: str = "default", text: Any = "") -> Dict[str, Any]:
    role_name = _normalize_manager_text_role(role)
    slot_name = _clean_text_slot(slot)
    return {
        "role": role_name,
        "slot": slot_name,
        "label": f"{role_name.title()} {slot_name}",
        "text": str(text or ""),
    }


def _raw_manager_text_boxes(prompt: Dict[str, Any]) -> List[Any]:
    raw = prompt.get("text_boxes", prompt.get("textBoxes", prompt.get("prompt_boxes", prompt.get("promptBoxes", []))))
    if isinstance(raw, list):
        return list(raw)
    if isinstance(raw, dict):
        out: List[Any] = []
        for key, value in raw.items():
            is_role_key = _is_manager_text_role_key(key)
            if isinstance(value, dict):
                merged = dict(value)
                if is_role_key and "role" not in merged:
                    merged["role"] = key
                elif not is_role_key:
                    merged.setdefault("slot", key)
                out.append(merged)
            else:
                if is_role_key:
                    out.append({"role": key, "text": value})
                else:
                    out.append({"slot": key, "text": value})
        return out
    return []


def _normalize_manager_text_box(raw: Any, index: int = 0) -> Optional[Dict[str, Any]]:
    if raw is None:
        return None
    src = raw if isinstance(raw, dict) else {"text": raw}
    role = _normalize_manager_text_role(src.get("role", src.get("kind", src.get("type", "generic"))))
    slot = _clean_text_slot(src.get("slot", src.get("id", src.get("name", ""))), f"text_{index + 1}" if role == "generic" else "default")
    return {
        "role": role,
        "slot": slot,
        "label": str(src.get("label", src.get("name", f"{role} {slot}")) or f"{role} {slot}").strip() or f"{role} {slot}",
        "text": str(src.get("text", src.get("value", src.get("prompt", ""))) or ""),
    }


def _normalize_manager_text_boxes(prompt: Dict[str, Any]) -> List[Dict[str, Any]]:
    legacy_positive = str(prompt.get("positive", prompt.get("positive_prompt", "")) or "")
    legacy_negative = str(prompt.get("negative", prompt.get("negative_prompt", "")) or "")
    raw_boxes = _raw_manager_text_boxes(prompt)
    boxes: List[Dict[str, Any]] = []
    used: Set[str] = set()

    for index, raw in enumerate(raw_boxes):
        normalized = _normalize_manager_text_box(raw, index)
        if normalized is None:
            continue
        base_slot = normalized["slot"]
        slot = base_slot
        suffix = 2
        while _text_box_key(normalized["role"], slot) in used:
            slot = f"{base_slot}_{suffix}"
            suffix += 1
        normalized["slot"] = slot
        used.add(_text_box_key(normalized["role"], slot))
        boxes.append(normalized)

    def upsert_legacy(role: str, text: str) -> None:
        key = _text_box_key(role, "default")
        existing = next((box for box in boxes if _text_box_key(box.get("role"), box.get("slot")) == key), None)
        if existing is not None:
            if text and not existing.get("text"):
                existing["text"] = text
            return
        if text or not raw_boxes:
            boxes.append(_default_manager_text_box(role, "default", text))

    upsert_legacy("positive", legacy_positive)
    upsert_legacy("negative", legacy_negative)
    return boxes


def _pick_manager_text_box(prompt: Dict[str, Any], role: str, slot: str = "default") -> Optional[Dict[str, Any]]:
    boxes = _normalize_manager_text_boxes(prompt)
    role_name = _normalize_manager_text_role(role)
    slot_name = _clean_text_slot(slot)
    for box in boxes:
        if box.get("role") == role_name and _clean_text_slot(box.get("slot")) == slot_name:
            return box
    for box in boxes:
        if box.get("role") == role_name and _clean_text_slot(box.get("slot")) == "default":
            return box
    for box in boxes:
        if box.get("role") == role_name:
            return box
    return None


def _normalize_manager_thumbnail(thumbnail: Any) -> Dict[str, Any]:
    if isinstance(thumbnail, dict):
        filename = str(thumbnail.get("filename", "")).strip()
        subfolder = str(thumbnail.get("subfolder", "")).strip()
        type_name = str(thumbnail.get("type", "input")).strip() or "input"
        url = str(thumbnail.get("url", "")).strip()
        if filename:
            return {"filename": filename, "subfolder": subfolder, "type": type_name}
        if url:
            return {"url": url}
        return {}
    text = str(thumbnail or "").strip()
    if text:
        return {"url": text}
    return {}


def _normalize_runtime_lora_stack(raw: Any) -> Optional[Dict[str, Any]]:
    payload = _safe_json_load(raw, None)
    if not isinstance(payload, dict):
        return None

    # Accept either the explicit stack output from DoRA Power LoRA Loader or a resolved
    # manager payload. The latter makes manager chaining and old workflows less brittle.
    kind = payload.get("kind")
    if kind not in {_DORA_LORA_STACK_KIND, _DORA_STATE_KIND}:
        return None

    rows_in = payload.get("loras", payload.get("rows", []))
    rows: List[Dict[str, Any]] = []
    if isinstance(rows_in, list):
        for row in rows_in:
            normalized = _normalize_manager_lora_row(row)
            if normalized is not None:
                rows.append(normalized)

    return {
        "version": _DORA_STATE_MANAGER_SCHEMA_VERSION,
        "kind": _DORA_LORA_STACK_KIND,
        "loras": rows,
        "loader_globals": _normalize_manager_loader_globals(payload.get("loader_globals", payload.get("globals", {}))),
    }


def _manager_rows_to_lora_entries(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for row in rows:
        normalized = _normalize_manager_lora_row(row)
        if normalized is None:
            continue
        name = normalized.get("name", "None")
        if name in ("", "None", "NONE"):
            continue
        entries.append({
            "on": bool(normalized.get("enabled", True)),
            "lora": name,
            "strength_model": float(normalized.get("strength_model", 1.0)),
            "strength_clip": float(normalized.get("strength_clip", normalized.get("strength_model", 1.0))),
        })
    return entries


def _lora_entries_to_manager_rows(entries: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        rows.append({
            "enabled": bool(entry.get("on", entry.get("enabled", True))),
            "name": str(entry.get("lora", entry.get("name", "None")) or "None"),
            "strength_model": float(entry.get("strength_model", entry.get("strength", 1.0))),
            "strength_clip": float(entry.get("strength_clip", entry.get("strengthTwo", entry.get("strength_model", entry.get("strength", 1.0))))),
        })
    return rows


def _build_lora_stack_payload(entries: Iterable[Dict[str, Any]], loader_globals: Dict[str, Any], state_slot: Any = "default", label: Any = None) -> Dict[str, Any]:
    slot = _clean_loader_slot(state_slot, "default")
    return {
        "version": _DORA_STATE_MANAGER_SCHEMA_VERSION,
        "kind": _DORA_LORA_STACK_KIND,
        "slot": slot,
        "label": str(label or slot),
        "loras": _lora_entries_to_manager_rows(entries),
        "loader_globals": _normalize_manager_loader_globals(loader_globals),
    }




def _clean_loader_slot(value: Any, fallback: str = "default") -> str:
    text = str(value or "").strip()
    if not text:
        text = fallback
    text = re.sub(r"[^A-Za-z0-9_.:-]+", "_", text).strip("_")
    return text or fallback


def _normalize_manager_loader_stack(stack: Any, index: int = 0) -> Optional[Dict[str, Any]]:
    if not isinstance(stack, dict):
        return None
    slot = _clean_loader_slot(stack.get("slot", stack.get("id", stack.get("name", ""))), f"loader_{index + 1}")
    label = str(stack.get("label", stack.get("name", slot)) or slot).strip() or slot
    rows_in = stack.get("loras", stack.get("rows", []))
    loras: List[Dict[str, Any]] = []
    if isinstance(rows_in, list):
        for row in rows_in:
            normalized = _normalize_manager_lora_row(row)
            if normalized is not None:
                loras.append(normalized)
    return {
        "slot": slot,
        "label": label,
        "loras": loras,
        "loader_globals": _normalize_manager_loader_globals(stack.get("loader_globals", stack.get("globals", {}))),
    }


def _normalize_manager_legacy_loader_globals(character: Dict[str, Any]) -> Dict[str, Any]:
    globals_in = character.get("globals")
    loader_globals = character.get("loader_globals")
    merged: Dict[str, Any] = {}
    if isinstance(globals_in, dict):
        merged.update(globals_in)
    if isinstance(loader_globals, dict):
        merged.update(loader_globals)
    return _normalize_manager_loader_globals(merged)


def _normalize_manager_loader_stacks(character: Dict[str, Any]) -> List[Dict[str, Any]]:
    stacks_in = character.get("loader_stacks")
    raw_stacks: List[Any] = []
    if isinstance(stacks_in, list):
        raw_stacks = stacks_in
    elif isinstance(stacks_in, dict):
        for slot, value in stacks_in.items():
            if isinstance(value, dict):
                merged = dict(value)
                merged.setdefault("slot", slot)
                raw_stacks.append(merged)

    stacks: List[Dict[str, Any]] = []
    used_slots: Set[str] = set()
    for index, stack in enumerate(raw_stacks):
        normalized = _normalize_manager_loader_stack(stack, index)
        if normalized is None:
            continue
        base_slot = normalized["slot"]
        slot = base_slot
        suffix = 2
        while slot in used_slots:
            slot = f"{base_slot}_{suffix}"
            suffix += 1
        used_slots.add(slot)
        normalized["slot"] = slot
        stacks.append(normalized)

    legacy_loader_globals = _normalize_manager_legacy_loader_globals(character)

    # Legacy migration: previous versions stored one character-level LoRA stack.
    if not stacks:
        rows_in = character.get("loras")
        legacy_rows: List[Dict[str, Any]] = []
        if isinstance(rows_in, list):
            for row in rows_in:
                normalized = _normalize_manager_lora_row(row)
                if normalized is not None:
                    legacy_rows.append(normalized)
        stacks.append({
            "slot": "default",
            "label": "Default loader",
            "loras": legacy_rows,
            "loader_globals": legacy_loader_globals,
        })
    elif legacy_loader_globals:
        # Older State Manager saves can contain loader_stacks plus the actual loader
        # globals only on the legacy character mirror. Preserve those globals for the
        # default stack instead of normalizing them away.
        default_stack = next(
            (stack for stack in stacks if isinstance(stack, dict) and _clean_loader_slot(stack.get("slot", ""), "default") == "default"),
            stacks[0],
        )
        if isinstance(default_stack, dict) and not default_stack.get("loader_globals"):
            default_stack["loader_globals"] = legacy_loader_globals

    return stacks


def _pick_loader_stack(loader_stacks: Any, slot: Any = "default") -> Dict[str, Any]:
    stacks = loader_stacks if isinstance(loader_stacks, list) else []
    if not stacks:
        return {"slot": "default", "label": "Default loader", "loras": [], "loader_globals": {}}
    wanted = _clean_loader_slot(slot, "default")
    for stack in stacks:
        if isinstance(stack, dict) and _clean_loader_slot(stack.get("slot", ""), "default") == wanted:
            return stack
    for stack in stacks:
        if isinstance(stack, dict) and _clean_loader_slot(stack.get("slot", ""), "default") == "default":
            return stack
    return stacks[0]


def _normalize_manager_loader_globals(globals_in: Any) -> Dict[str, Any]:
    if not isinstance(globals_in, dict):
        return {}
    out: Dict[str, Any] = {}
    for key in _DORA_STATE_LOADER_GLOBAL_KEYS:
        if key not in globals_in:
            continue
        value = globals_in.get(key)
        if key in {
            "stack_enabled",
            "verbose",
            "log_unloaded_keys",
            "broadcast_auto_scale",
            "broadcast_modulations",
            "broadcast_include_dora_scale",
            "dora_decompose_debug",
            "dora_slice_fix",
            "dora_adaln_swap_fix",
            "zimage_lumina2_compat",
            "auto_strength_enabled",
        }:
            out[key] = _state_manager_bool(value)
        elif key in {"dora_decompose_debug_n", "dora_decompose_debug_stack_depth"}:
            try:
                out[key] = int(value)
            except Exception:
                continue
        elif key == "auto_strength_device":
            out[key] = _normalize_auto_strength_device(value)
        else:
            try:
                out[key] = float(value)
            except Exception:
                continue
    return out


def _normalize_manager_prompt(prompt: Any, index: int) -> Optional[Dict[str, Any]]:
    if not isinstance(prompt, dict):
        return None
    prompt_id = _clean_state_id(prompt.get("id"), _state_manager_make_id("prompt", index))
    name = str(prompt.get("name") or f"Prompt {index + 1}").strip() or f"Prompt {index + 1}"
    settings = _normalize_settings_with_canonical_seed(prompt.get("settings", {}))
    reference_image = _normalize_manager_thumbnail(prompt.get("reference_image", prompt.get("referenceImage", prompt.get("prompt_image", prompt.get("image", {})))))
    fileimage_prefix = str(prompt.get("fileimage_prefix", prompt.get("filename_prefix", prompt.get("file_image_prefix", ""))) or "").strip()
    text_boxes = _normalize_manager_text_boxes(prompt)
    positive_box = next((box for box in text_boxes if box.get("role") == "positive" and box.get("slot") == "default"), None) or next((box for box in text_boxes if box.get("role") == "positive"), None)
    negative_box = next((box for box in text_boxes if box.get("role") == "negative" and box.get("slot") == "default"), None) or next((box for box in text_boxes if box.get("role") == "negative"), None)
    return {
        "id": prompt_id,
        "name": name,
        "positive": str(positive_box.get("text", "") if positive_box else prompt.get("positive", prompt.get("positive_prompt", "")) or ""),
        "negative": str(negative_box.get("text", "") if negative_box else prompt.get("negative", prompt.get("negative_prompt", "")) or ""),
        "text_boxes": text_boxes,
        "settings": settings,
        "reference_image": reference_image,
        "fileimage_prefix": fileimage_prefix,
    }


def _normalize_state_manager_state(raw: Any) -> Dict[str, Any]:
    parsed = _safe_json_load(raw, _state_manager_default_state())
    if not isinstance(parsed, dict):
        parsed = _state_manager_default_state()

    characters_in = parsed.get("characters")
    if not isinstance(characters_in, list):
        characters_in = []

    characters: List[Dict[str, Any]] = []
    used_ids: Set[str] = set()
    for char_index, char in enumerate(characters_in):
        if not isinstance(char, dict):
            continue
        base_id = _clean_state_id(char.get("id"), _state_manager_make_id("character", char_index))
        char_id = base_id
        suffix = 2
        while char_id in used_ids:
            char_id = f"{base_id}_{suffix}"
            suffix += 1
        used_ids.add(char_id)

        name = str(char.get("name") or f"Character {char_index + 1}").strip() or f"Character {char_index + 1}"
        loras_in = char.get("loras")
        loras = []
        if isinstance(loras_in, list):
            for row in loras_in:
                normalized = _normalize_manager_lora_row(row)
                if normalized is not None:
                    loras.append(normalized)

        loader_stacks = _normalize_manager_loader_stacks(char)
        default_loader_stack = _pick_loader_stack(loader_stacks, "default")

        prompts_in = char.get("prompts")
        prompts: List[Dict[str, Any]] = []
        if isinstance(prompts_in, list):
            used_prompt_ids: Set[str] = set()
            for prompt_index, prompt in enumerate(prompts_in):
                normalized_prompt = _normalize_manager_prompt(prompt, prompt_index)
                if normalized_prompt is None:
                    continue
                prompt_base_id = normalized_prompt["id"]
                prompt_id = prompt_base_id
                prompt_suffix = 2
                while prompt_id in used_prompt_ids:
                    prompt_id = f"{prompt_base_id}_{prompt_suffix}"
                    prompt_suffix += 1
                used_prompt_ids.add(prompt_id)
                normalized_prompt["id"] = prompt_id
                prompts.append(normalized_prompt)
        if not prompts:
            prompts = _state_manager_default_state()["characters"][0]["prompts"]

        characters.append(
            {
                "id": char_id,
                "name": name,
                "thumbnail": _normalize_manager_thumbnail(char.get("thumbnail", {})),
                "loader_stacks": loader_stacks,
                # Legacy/default stack mirror for older graphs and older consumers.
                "loras": default_loader_stack.get("loras", loras),
                "loader_globals": default_loader_stack.get("loader_globals", _normalize_manager_loader_globals(char.get("loader_globals", char.get("globals", {})))),
                "prompts": prompts,
            }
        )

    if not characters:
        characters = _state_manager_default_state()["characters"]

    return {
        "version": _DORA_STATE_MANAGER_SCHEMA_VERSION,
        "characters": characters,
    }


def _pick_state_manager_character(state: Dict[str, Any], selected_character_id: Any) -> Dict[str, Any]:
    selected = str(selected_character_id or "").strip()
    characters = state.get("characters") if isinstance(state.get("characters"), list) else []
    for char in characters:
        if isinstance(char, dict) and char.get("id") == selected:
            return char
    return characters[0]


def _pick_state_manager_prompt(character: Dict[str, Any], selected_prompt_id: Any) -> Dict[str, Any]:
    selected = str(selected_prompt_id or "").strip()
    prompts = character.get("prompts") if isinstance(character.get("prompts"), list) else []
    for prompt in prompts:
        if isinstance(prompt, dict) and prompt.get("id") == selected:
            return prompt
    return prompts[0] if prompts else _state_manager_default_state()["characters"][0]["prompts"][0]


def _resolve_dora_state_payload(
    state_json: Any,
    selected_character_id: Any,
    selected_prompt_id: Any,
) -> Dict[str, Any]:
    """Resolve the selected saved state without reading any downstream runtime inputs.

    The state manager is intentionally an execution-time source. Save/load/apply behavior
    is handled by the frontend as explicit graph editing; runtime execution must stay
    acyclic.
    """
    state = _normalize_state_manager_state(state_json)
    character = _pick_state_manager_character(state, selected_character_id)
    prompt = _pick_state_manager_prompt(character, selected_prompt_id)

    settings = _normalize_settings_with_canonical_seed(prompt.get("settings", {}))
    loader_stacks = _normalize_manager_loader_stacks(character)
    default_stack = _pick_loader_stack(loader_stacks, "default")
    loras = default_stack.get("loras", character.get("loras", []))
    loader_globals = default_stack.get("loader_globals", character.get("loader_globals", {}))
    text_boxes = _normalize_manager_text_boxes(prompt)
    positive_box = _pick_manager_text_box(prompt, "positive", "default")
    negative_box = _pick_manager_text_box(prompt, "negative", "default")
    positive = str(positive_box.get("text", "") if positive_box else prompt.get("positive", "") or "")
    negative = str(negative_box.get("text", "") if negative_box else prompt.get("negative", "") or "")

    return {
        "version": _DORA_STATE_MANAGER_SCHEMA_VERSION,
        "kind": _DORA_STATE_KIND,
        "character": {
            "id": character.get("id", ""),
            "name": character.get("name", ""),
            "thumbnail": character.get("thumbnail", {}),
        },
        "prompt": {
            "id": prompt.get("id", ""),
            "name": prompt.get("name", ""),
            "reference_image": prompt.get("reference_image", {}),
            "fileimage_prefix": str(prompt.get("fileimage_prefix", "") or ""),
        },
        "loader_stacks": loader_stacks,
        "loras": loras,
        "loader_globals": loader_globals,
        "settings": settings,
        "text_boxes": text_boxes,
        "positive_prompt": positive,
        "negative_prompt": negative,
        "reference_image": prompt.get("reference_image", {}),
        "fileimage_prefix": str(prompt.get("fileimage_prefix", "") or ""),
    }


def _build_state_settings_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    settings = _normalize_settings_with_canonical_seed(payload.get("settings", {}))
    seed = _extract_seed_from_settings(settings)
    return {
        "version": _DORA_STATE_MANAGER_SCHEMA_VERSION,
        "kind": _DORA_STATE_SETTINGS_KIND,
        "character": payload.get("character") if isinstance(payload.get("character"), dict) else {},
        "prompt": payload.get("prompt") if isinstance(payload.get("prompt"), dict) else {},
        "settings": settings,
        "seed": seed,
    }


def _queued_runtime_state_from_ui_state(ui_state_json: Any) -> Optional[Dict[str, Any]]:
    parsed = _safe_json_load(ui_state_json, {})
    if not isinstance(parsed, dict):
        return None
    raw = parsed.get("__dsm_queued_runtime_state")
    return _normalize_runtime_dora_state_payload(raw)


def _settings_with_runtime_seed(settings: Any, seed: int) -> Dict[str, Any]:
    normalized = dict(_normalize_manager_settings(settings))
    normalized["seed"] = seed
    rgthree_seed = normalized.get("rgthree_seed")
    if isinstance(rgthree_seed, dict):
        rgthree_seed = dict(rgthree_seed)
        rgthree_seed["seed"] = seed
        widgets = rgthree_seed.get("widgets")
        if isinstance(widgets, dict):
            widgets = dict(widgets)
            for key in ("seed", "noise_seed", "value"):
                if key in widgets:
                    widgets[key] = seed
            rgthree_seed["widgets"] = widgets
        normalized["rgthree_seed"] = rgthree_seed
    nodes = normalized.get("nodes")
    if isinstance(nodes, list):
        updated_nodes = []
        for node in nodes:
            if not isinstance(node, dict):
                updated_nodes.append(node)
                continue
            updated_node = dict(node)
            widgets = updated_node.get("widgets")
            if isinstance(widgets, dict):
                widgets = dict(widgets)
                for key in ("seed", "noise_seed", "value"):
                    if key in widgets:
                        widgets[key] = seed
                updated_node["widgets"] = widgets
            if updated_node.get("is_seed_node") or updated_node.get("seed") is not None or updated_node.get("seed_widgets"):
                updated_node["seed"] = seed
                seed_widgets = updated_node.get("seed_widgets")
                if isinstance(seed_widgets, dict):
                    updated_node["seed_widgets"] = {key: seed for key in seed_widgets}
            updated_nodes.append(updated_node)
        normalized["nodes"] = updated_nodes
    return normalized


def _resolve_state_manager_runtime_seed(payload: Dict[str, Any]) -> Dict[str, Any]:
    normalized = _normalize_runtime_dora_state_payload(payload) or dict(payload)
    settings = _normalize_settings_with_canonical_seed(normalized.get("settings", {}))
    seed = _extract_seed_from_settings(settings)
    if seed in _STATE_SEED_SPECIALS:
        settings = _settings_with_runtime_seed(settings, _state_manager_new_random_seed())
    normalized["settings"] = settings
    return normalized


def _build_state_control_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    # Preferred State Manager edge. It remains usable as an editor relationship for
    # Save/Load connected, and it now also carries the resolved selected state at
    # runtime. This keeps state_control-only graphs consistent with direct dora_state
    # and text output wiring during queued prompt/character wildcarding.
    normalized = _normalize_runtime_dora_state_payload(payload) or dict(payload)
    character = normalized.get("character") if isinstance(normalized.get("character"), dict) else {}
    prompt = normalized.get("prompt") if isinstance(normalized.get("prompt"), dict) else {}
    return {
        "version": _DORA_STATE_MANAGER_SCHEMA_VERSION,
        "kind": _STATE_MANAGER_CONTROL_KIND,
        "state": normalized,
        "character": character,
        "prompt": prompt,
        "selected_character_id": character.get("id", ""),
        "selected_prompt_id": prompt.get("id", ""),
        "loader_stacks": normalized.get("loader_stacks", []),
        "loras": normalized.get("loras", []),
        "loader_globals": normalized.get("loader_globals", {}),
        "settings": normalized.get("settings", {}),
        "text_boxes": normalized.get("text_boxes", []),
        "positive_prompt": normalized.get("positive_prompt", ""),
        "negative_prompt": normalized.get("negative_prompt", ""),
        "reference_image": normalized.get("reference_image", {}),
        "fileimage_prefix": normalized.get("fileimage_prefix", ""),
    }




def _state_manager_blank_image() -> torch.Tensor:
    return torch.zeros((1, 1, 1, 3), dtype=torch.float32)


def _state_manager_thumbnail_path(thumbnail: Any) -> Optional[str]:
    info = _normalize_manager_thumbnail(thumbnail)
    filename = str(info.get("filename", "")).strip()
    if not filename:
        return None
    subfolder = str(info.get("subfolder", "")).strip()
    type_name = str(info.get("type", "input")).strip().lower() or "input"
    if type_name == "input":
        root = folder_paths.get_input_directory()
    elif type_name == "output":
        root = folder_paths.get_output_directory()
    elif type_name == "temp":
        root = folder_paths.get_temp_directory()
    else:
        root = folder_paths.get_input_directory()
    root_abs = os.path.realpath(root)
    path = os.path.realpath(os.path.join(root_abs, subfolder, filename))
    try:
        common = os.path.commonpath([root_abs, path])
    except Exception:
        return None
    if common != root_abs:
        return None
    if not os.path.isfile(path):
        return None
    return path


def _try_load_state_manager_image_from_info(image_info: Any, label: str = "image") -> Optional[torch.Tensor]:
    path = _state_manager_thumbnail_path(image_info)
    if not path:
        return None
    try:
        import numpy as np
        from PIL import Image, ImageOps

        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")
            arr = np.asarray(img, dtype=np.float32) / 255.0
        if arr.ndim != 3 or arr.shape[-1] != 3:
            return None
        return torch.from_numpy(arr)[None, ...]
    except Exception as exc:
        _LOG.warning("[State Manager] failed to load %s %r: %s", label, path, exc)
        return None


def _load_state_manager_image_from_info(image_info: Any, label: str = "image") -> torch.Tensor:
    image = _try_load_state_manager_image_from_info(image_info, label)
    if image is not None:
        return image
    return _state_manager_blank_image()


def _load_state_manager_character_image(character: Dict[str, Any]) -> torch.Tensor:
    return _load_state_manager_image_from_info(character.get("thumbnail", {}) if isinstance(character, dict) else {}, "character image")


def _load_state_manager_prompt_or_character_image(character: Dict[str, Any], prompt: Dict[str, Any]) -> torch.Tensor:
    prompt_image = prompt.get("reference_image", {}) if isinstance(prompt, dict) else {}
    prompt_tensor = _try_load_state_manager_image_from_info(prompt_image, "prompt reference image")
    if prompt_tensor is not None:
        return prompt_tensor
    return _load_state_manager_character_image(character)


def _extract_seed_from_settings(settings: Any, fallback: int = 0) -> int:
    settings_dict = _normalize_manager_settings(settings)

    def _coerce(value: Any) -> Optional[int]:
        try:
            if isinstance(value, bool):
                return None
            if isinstance(value, float) and not math.isfinite(value):
                return None
            if isinstance(value, str) and not value.strip():
                return None
            return _coerce_state_manager_seed(value, fallback)
        except Exception:
            return None

    for key in ("seed", "noise_seed", "rgthree_seed"):
        value = settings_dict.get(key)
        if isinstance(value, dict):
            for nested_key in ("seed", "value", "noise_seed"):
                coerced = _coerce(value.get(nested_key))
                if coerced is not None:
                    return coerced
            widgets = value.get("widgets")
            if isinstance(widgets, dict):
                for nested_key in ("seed", "noise_seed", "value"):
                    coerced = _coerce(widgets.get(nested_key))
                    if coerced is not None:
                        return coerced
        else:
            coerced = _coerce(value)
            if coerced is not None:
                return coerced

    nodes = settings_dict.get("nodes")
    if isinstance(nodes, list):
        for node in nodes:
            if not isinstance(node, dict):
                continue
            widgets = node.get("widgets")
            if not isinstance(widgets, dict):
                continue
            for key in ("seed", "noise_seed", "value"):
                coerced = _coerce(widgets.get(key))
                if coerced is not None:
                    return coerced

    return _coerce_state_manager_seed(fallback, 0)


def _normalize_settings_with_canonical_seed(settings: Any) -> Dict[str, Any]:
    settings_dict = dict(_normalize_manager_settings(settings))
    settings_dict["seed"] = _extract_seed_from_settings(settings_dict)
    return settings_dict

def _normalize_runtime_dora_state_payload(raw: Any) -> Optional[Dict[str, Any]]:
    payload = _safe_json_load(raw, None)
    if not isinstance(payload, dict):
        return None

    # State Manager's STATE_MANAGER_CONTROL output is the preferred graph edge for
    # save/load-only wiring. It also needs to carry the selected queued state at
    # execution time so State Text Box, State Seed, and DoRA Loader nodes can all
    # follow queued prompt/character wildcarding without direct dora_state/text links.
    kind = payload.get("kind")
    if kind == _STATE_MANAGER_CONTROL_KIND and isinstance(payload.get("state"), dict):
        payload = payload.get("state")
        kind = payload.get("kind") if isinstance(payload, dict) else None
    elif kind == _STATE_MANAGER_CONTROL_KIND:
        payload = dict(payload)
        payload["kind"] = _DORA_STATE_KIND
        kind = _DORA_STATE_KIND

    if kind != _DORA_STATE_KIND:
        return None

    loader_stacks = _normalize_manager_loader_stacks(payload)
    default_stack = _pick_loader_stack(loader_stacks, "default")
    loras = default_stack.get("loras", [])
    globals_out = default_stack.get("loader_globals", _normalize_manager_loader_globals(payload.get("loader_globals", payload.get("globals", {}))))
    text_boxes = _normalize_manager_text_boxes(payload)
    positive_box = _pick_manager_text_box(payload, "positive", "default")
    negative_box = _pick_manager_text_box(payload, "negative", "default")
    return {
        "version": _DORA_STATE_MANAGER_SCHEMA_VERSION,
        "kind": _DORA_STATE_KIND,
        "character": payload.get("character") if isinstance(payload.get("character"), dict) else {},
        "prompt": payload.get("prompt") if isinstance(payload.get("prompt"), dict) else {},
        "loader_stacks": loader_stacks,
        "loras": loras,
        "loader_globals": globals_out,
        "settings": _normalize_settings_with_canonical_seed(payload.get("settings", {})),
        "text_boxes": text_boxes,
        "positive_prompt": str(positive_box.get("text", "") if positive_box else payload.get("positive_prompt", "") or ""),
        "negative_prompt": str(negative_box.get("text", "") if negative_box else payload.get("negative_prompt", "") or ""),
        "reference_image": _normalize_manager_thumbnail(payload.get("reference_image", payload.get("prompt", {}).get("reference_image", {}) if isinstance(payload.get("prompt"), dict) else {})),
        "fileimage_prefix": str(payload.get("fileimage_prefix", payload.get("prompt", {}).get("fileimage_prefix", "") if isinstance(payload.get("prompt"), dict) else "") or ""),
    }


def _state_payload_text_for_box(state_payload: Optional[Dict[str, Any]], role: Any, slot: Any = "default") -> Optional[str]:
    if not isinstance(state_payload, dict):
        return None
    role_name = _normalize_manager_text_role(role)
    slot_name = _clean_text_slot(slot, "default")
    box = _pick_manager_text_box(state_payload, role_name, slot_name)
    if isinstance(box, dict):
        return str(box.get("text", "") or "")
    if role_name == "positive" and "positive_prompt" in state_payload:
        return str(state_payload.get("positive_prompt") or "")
    if role_name == "negative" and "negative_prompt" in state_payload:
        return str(state_payload.get("negative_prompt") or "")
    return None


def _state_payload_seed(state_payload: Optional[Dict[str, Any]]) -> Optional[int]:
    if not isinstance(state_payload, dict):
        return None
    if "settings" not in state_payload:
        return None
    return _extract_seed_from_settings(state_payload.get("settings", {}), None)


def _select_state_loader_stack(state_payload: Optional[Dict[str, Any]], state_slot: Any = "default") -> Optional[Dict[str, Any]]:
    if not isinstance(state_payload, dict):
        return None
    stacks = state_payload.get("loader_stacks")
    if isinstance(stacks, list) and stacks:
        wanted = _clean_loader_slot(state_slot, "default")
        for stack in stacks:
            if isinstance(stack, dict) and _clean_loader_slot(stack.get("slot", ""), "default") == wanted:
                return stack
        return None
    if isinstance(state_payload.get("loras"), list):
        return {
            "slot": "default",
            "label": "Default loader",
            "loras": state_payload.get("loras", []),
            "loader_globals": _normalize_manager_loader_globals(state_payload.get("loader_globals", {})),
        }
    return None


def _parse_dora_state_lora_entries(state_payload: Optional[Dict[str, Any]], state_slot: Any = "default") -> Optional[List[Dict[str, Any]]]:
    stack = _select_state_loader_stack(state_payload, state_slot)
    if not isinstance(stack, dict):
        return None
    rows = stack.get("loras")
    if not isinstance(rows, list):
        return None
    return _manager_rows_to_lora_entries(rows)


def _state_payload_get_loader_global(state_payload: Optional[Dict[str, Any]], key: str, fallback: Any, state_slot: Any = "default") -> Any:
    stack = _select_state_loader_stack(state_payload, state_slot)
    if isinstance(stack, dict):
        globals_in = stack.get("loader_globals")
        if isinstance(globals_in, dict) and key in globals_in:
            return globals_in[key]
    stacks = state_payload.get("loader_stacks") if isinstance(state_payload, dict) else None
    if isinstance(stacks, list) and stacks:
        return fallback
    if isinstance(state_payload, dict):
        globals_in = state_payload.get("loader_globals")
        if isinstance(globals_in, dict) and key in globals_in:
            return globals_in[key]
    return fallback


def _loader_cache_lora_file_signature(lora_name: Any) -> Dict[str, Any]:
    name = str(lora_name or "None")
    out: Dict[str, Any] = {"name": name}
    if not name or name in {"None", "NONE"}:
        return out
    try:
        path = folder_paths.get_full_path("loras", name)
    except Exception:
        path = None
    if not path:
        out["missing"] = True
        return out
    try:
        st = os.stat(path)
        out["size"] = int(st.st_size)
        out["mtime_ns"] = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1000000000)))
    except Exception:
        out["path"] = str(path)
    return out


def _loader_cache_float(value: Any, fallback: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        out = float(fallback)
    return out if math.isfinite(out) else float(fallback)


def _loader_cache_entries(entries: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        enabled = bool(entry.get("on", entry.get("enabled", True)))
        if not enabled:
            continue
        name = str(entry.get("lora", entry.get("name", "None")) or "None")
        if name in {"", "None", "NONE"}:
            continue
        sm = _loader_cache_float(entry.get("strength_model", entry.get("strength", 0.0)), 0.0)
        sc = _loader_cache_float(entry.get("strength_clip", entry.get("strengthTwo", sm)), sm)
        out.append({
            "on": True,
            "lora": name,
            "strength_model": sm,
            "strength_clip": sc,
            "file": _loader_cache_lora_file_signature(name),
        })
    return out


def _loader_cache_globals(kwargs: Dict[str, Any], state_payload: Optional[Dict[str, Any]], state_slot: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in sorted(_DORA_STATE_LOADER_GLOBAL_KEYS):
        if key == "stack_enabled":
            fallback = kwargs.get(key, True)
        elif key in {"broadcast_auto_scale", "broadcast_modulations", "dora_slice_fix", "dora_adaln_swap_fix", "zimage_lumina2_compat"}:
            fallback = kwargs.get(key, True)
        elif key == "auto_strength_device":
            fallback = kwargs.get(key, "gpu")
        elif key == "auto_strength_ratio_floor":
            fallback = kwargs.get(key, _AUTO_STRENGTH_RATIO_FLOOR)
        elif key == "auto_strength_ratio_ceiling":
            fallback = kwargs.get(key, _AUTO_STRENGTH_RATIO_CEILING)
        elif key == "broadcast_scale":
            fallback = kwargs.get(key, 1.0)
        elif key in {"dora_decompose_debug_n"}:
            fallback = kwargs.get(key, 30)
        elif key in {"dora_decompose_debug_stack_depth"}:
            fallback = kwargs.get(key, 10)
        else:
            fallback = kwargs.get(key, False)
        value = _state_payload_get_loader_global(state_payload, key, fallback, state_slot) if state_payload is not None else fallback
        if key in {
            "stack_enabled",
            "verbose",
            "log_unloaded_keys",
            "broadcast_auto_scale",
            "broadcast_modulations",
            "broadcast_include_dora_scale",
            "dora_decompose_debug",
            "dora_slice_fix",
            "dora_adaln_swap_fix",
            "zimage_lumina2_compat",
            "auto_strength_enabled",
        }:
            out[key] = _state_manager_bool(value)
        elif key in {"dora_decompose_debug_n", "dora_decompose_debug_stack_depth"}:
            try:
                out[key] = int(value)
            except Exception:
                out[key] = int(fallback)
        elif key == "auto_strength_device":
            out[key] = _normalize_auto_strength_device(value)
        else:
            out[key] = _loader_cache_float(value, _loader_cache_float(fallback, 0.0))
    return out


def _dora_loader_cache_key_from_inputs(model: Any, clip: Any, kwargs: Dict[str, Any]) -> str:
    state_slot = _clean_loader_slot(kwargs.get("state_slot", "default"), "default")
    state_payload = _normalize_runtime_dora_state_payload(kwargs.get("dora_state"))
    if state_payload is None:
        state_payload = _normalize_runtime_dora_state_payload(kwargs.get("state_control"))

    entries = _parse_dora_state_lora_entries(state_payload, state_slot) if state_payload is not None else None
    if entries is None:
        entries = _parse_lora_stack_kwargs(kwargs)

    payload = {
        "schema": 2,
        "kind": "dora_power_lora_loader_cache_key",
        "model_identity": id(model) if model is not None else None,
        "clip_identity": id(clip) if clip is not None else None,
        "state_slot": state_slot,
        "loras": _loader_cache_entries(entries),
        "loader_globals": _loader_cache_globals(kwargs, state_payload, state_slot),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)

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
]

_SCALEABLE_SUFFIXES = (
    ".alpha",
    ".diff",
    ".diff_b",
    ".set_weight",
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

_RESHAPE_WEIGHT_MAX_DIMS = 8


def _canonical_reshape_weight_dim(value: Any) -> Optional[int]:
    """Return a positive integer dimension when value is safe shape metadata."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float) and math.isfinite(value) and value > 0:
        rounded = round(value)
        if abs(value - rounded) <= 1e-6:
            return int(rounded)
    return None


def _sanitize_reshape_weight_metadata(lora_sd: Dict[str, Any], lora_name: str = "", verbose: bool = False) -> int:
    """
    Keep .reshape_weight keys only when they are small shape metadata.

    ComfyUI's LoRAAdapter.load() treats <base>.reshape_weight as a shape list and
    calls .tolist() on it while constructing the adapter. If a malformed/exported
    LoRA stores a real tensor under that suffix, the direct .tolist() can allocate a
    huge nested Python object or crash in native code before Python can raise a
    normal exception. The loader does not support .reshape_weight as an independent
    weight patch, so invalid entries are report/debug data at best and must not be
    passed into comfy.lora.load_lora(...).
    """
    removed = 0
    for key in list(lora_sd.keys()):
        if not str(key).endswith(".reshape_weight"):
            continue
        value = lora_sd.get(key)
        valid = False
        canonical_vals = None
        reason = "unsupported value"

        try:
            if isinstance(value, torch.Tensor):
                numel = int(value.numel())
                ndim = int(value.ndim)
                if numel <= 0:
                    reason = "empty tensor"
                elif numel > _RESHAPE_WEIGHT_MAX_DIMS or ndim > 1:
                    reason = f"not shape metadata (shape={tuple(value.shape)} numel={numel})"
                else:
                    vals = value.detach().cpu().flatten().tolist()
                    if 0 < len(vals) <= _RESHAPE_WEIGHT_MAX_DIMS:
                        canonical_vals = [_canonical_reshape_weight_dim(v) for v in vals]
                        valid = all(v is not None for v in canonical_vals)
                    if not valid:
                        reason = f"non-integer shape values ({vals!r})"
            elif isinstance(value, (list, tuple)):
                vals = list(value)
                if 0 < len(vals) <= _RESHAPE_WEIGHT_MAX_DIMS:
                    canonical_vals = [_canonical_reshape_weight_dim(v) for v in vals]
                    valid = all(v is not None for v in canonical_vals)
                if not valid:
                    reason = f"invalid shape list ({vals!r})"
            else:
                reason = f"unsupported type {type(value).__name__}"
        except (RuntimeError, TypeError, ValueError, OverflowError) as exc:
            valid = False
            reason = str(exc)

        if valid:
            try:
                lora_sd[key] = torch.tensor(canonical_vals, dtype=torch.int64)
                continue
            except (RuntimeError, TypeError, ValueError, OverflowError) as exc:
                reason = str(exc)

        lora_sd.pop(key, None)
        removed += 1
        _LOG.warning(
            "[DoRA Power LoRA Loader] %s: dropping unsafe %s before comfy.lora.load_lora: %s",
            lora_name or "LoRA",
            key,
            reason,
        )

    if verbose and removed:
        _LOG.info("[DoRA Power LoRA Loader] %s: dropped %d unsafe reshape_weight metadata entries", lora_name, removed)
    return removed


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


_AUTO_STRENGTH_JSON_MAX_DEPTH = 32
_AUTO_STRENGTH_JSON_MAX_ITEMS = 2048
_AUTO_STRENGTH_JSON_MAX_STRING_CHARS = 32768


def _auto_strength_json_truncate_string(value: str) -> str:
    if len(value) <= _AUTO_STRENGTH_JSON_MAX_STRING_CHARS:
        return value
    omitted = len(value) - _AUTO_STRENGTH_JSON_MAX_STRING_CHARS
    return value[:_AUTO_STRENGTH_JSON_MAX_STRING_CHARS] + f"... <truncated {omitted} chars>"


def _auto_strength_json_safe(
    value: Any,
    *,
    _depth: int = 0,
    _seen: Optional[Set[int]] = None,
) -> Any:
    """
    Convert auto-strength diagnostics to a bounded, JSON-safe structure.

    This is intentionally report-only. It must not mutate or approximate the LoRA
    tensors used for actual loading. Diagnostic serialization should degrade to a
    bounded report instead of crashing generation.
    """
    if _seen is None:
        _seen = set()

    if _depth > _AUTO_STRENGTH_JSON_MAX_DEPTH:
        return "<max-depth>"

    if value is None or isinstance(value, bool):
        return value

    if isinstance(value, int) and not isinstance(value, bool):
        return value

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, str):
        return _auto_strength_json_truncate_string(value)

    if isinstance(value, (bytes, bytearray)):
        try:
            return _auto_strength_json_truncate_string(value.decode("utf-8", errors="replace"))
        except Exception:
            return _auto_strength_json_truncate_string(repr(value))

    if isinstance(value, torch.Tensor):
        try:
            return {
                "__type__": "torch.Tensor",
                "shape": [int(x) for x in value.shape],
                "dtype": str(value.dtype),
                "device": str(value.device),
            }
        except Exception:
            return {"__type__": "torch.Tensor"}

    if isinstance(value, torch.device):
        return str(value)

    if isinstance(value, torch.dtype):
        return str(value)

    if isinstance(value, Mapping):
        object_id = id(value)
        if object_id in _seen:
            return "<cycle>"
        _seen.add(object_id)
        try:
            out: Dict[str, Any] = {}
            omitted = 0
            for index, item in enumerate(value.items()):
                if index >= _AUTO_STRENGTH_JSON_MAX_ITEMS:
                    try:
                        omitted = max(0, len(value) - _AUTO_STRENGTH_JSON_MAX_ITEMS)
                    except Exception:
                        omitted = 1
                    break
                try:
                    key, item_value = item
                except Exception:
                    continue
                try:
                    key_text = str(key)
                except Exception:
                    key_text = repr(key)
                key_text = _auto_strength_json_truncate_string(key_text)
                out[key_text] = _auto_strength_json_safe(item_value, _depth=_depth + 1, _seen=_seen)
            if omitted:
                out["__auto_strength_report_truncated__"] = True
                out["__omitted_items__"] = omitted
            return out
        finally:
            _seen.discard(object_id)

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        object_id = id(value)
        if object_id in _seen:
            return "<cycle>"
        _seen.add(object_id)
        try:
            out: List[Any] = []
            limit = min(len(value), _AUTO_STRENGTH_JSON_MAX_ITEMS)
            for index in range(limit):
                out.append(_auto_strength_json_safe(value[index], _depth=_depth + 1, _seen=_seen))
            omitted = max(0, len(value) - limit)
            if omitted:
                out.append({"__auto_strength_report_truncated__": True, "__omitted_items__": omitted})
            return out
        except Exception:
            return _auto_strength_json_truncate_string(repr(value))
        finally:
            _seen.discard(object_id)

    try:
        if hasattr(value, "item") and callable(value.item):
            return _auto_strength_json_safe(value.item(), _depth=_depth + 1, _seen=_seen)
    except Exception:
        pass

    try:
        return _auto_strength_json_truncate_string(str(value))
    except Exception:
        return "<unserializable>"


def _auto_strength_json_dumps(value: Any, *, pretty: bool = False) -> str:
    kwargs = {"ensure_ascii": False, "sort_keys": False}
    if pretty:
        kwargs["indent"] = 2
    else:
        kwargs["separators"] = (",", ":")
    safe_value = _auto_strength_json_safe(value)
    try:
        return json.dumps(safe_value, allow_nan=False, **kwargs)
    except Exception as exc:
        _LOG.warning("[DoRA Power LoRA Loader] failed to serialize auto-strength report JSON: %s", exc)
        fallback = {
            "schema": 1,
            "kind": "dora_power_lora_auto_strength_stack_report",
            "serialization_error": str(exc),
        }
        return json.dumps(fallback, allow_nan=False, **kwargs)


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


def _auto_strength_slice_destination_weight(weight: Any, sl: Optional[Tuple[int, int, int]]) -> Optional[torch.Tensor]:
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
    return _auto_strength_slice_destination_weight(weight, sl)


def _auto_strength_get_effective_destination_weight(
    base: str,
    key_map: Dict[str, Any],
    model_state_dict: Optional[Dict[str, Any]],
    clip_state_dict: Optional[Dict[str, Any]],
    current_model: Any = None,
    current_clip: Any = None,
    analysis_device: Optional[torch.device] = None,
) -> Optional[torch.Tensor]:
    """
    Return the destination weight currently seen by this loader invocation.

    Invariant: stacked DoRA auto-strength must measure against the actual destination
    weight that the current row will modify. Later rows on the same branch therefore
    need to see earlier rows already present in the patcher, not just the pristine
    state_dict snapshot captured before the row loop.
    """
    dest, sl = _unwrap_key_map_target(key_map.get(base))
    if not dest:
        return None

    patcher = None
    if model_state_dict is not None and dest in model_state_dict:
        if hasattr(current_model, "patch_weight_to_device"):
            patcher = current_model
    elif clip_state_dict is not None and dest in clip_state_dict:
        clip_patcher = getattr(current_clip, "patcher", None)
        if hasattr(clip_patcher, "patch_weight_to_device"):
            patcher = clip_patcher

    if patcher is not None:
        try:
            patches = getattr(patcher, "patches", None)
            if isinstance(patches, dict) and dest in patches:
                weight = patcher.patch_weight_to_device(dest, device_to=analysis_device, return_weight=True)
                sliced = _auto_strength_slice_destination_weight(weight, sl)
                if isinstance(sliced, torch.Tensor):
                    return sliced
        except Exception:
            pass

    return _auto_strength_get_destination_weight(base, key_map, model_state_dict, clip_state_dict)


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
    current_model: Any = None,
    current_clip: Any = None,
) -> Optional[float]:
    """
    Return a comparable magnitude score for a single base's update.

    Supported cases:
      - standard LoRA / DoRA low-rank pairs / direct deltas: RMS(delta)
      - DoRA low-rank pairs: RMS(actual post-normalization DoRA update)
      - direct delta tensors (.diff / .diff_b / .set_weight)

    Invariant: scores must be comparable across destination tensor sizes. Using raw
    Frobenius norms violates that invariant because ||ΔW||_F scales with sqrt(numel),
    which over-boosts smaller spatial layers in mixed architectures like SDXL. We
    therefore compare RMS update magnitude (scaled Frobenius / sqrt(numel)).

    Returns None when the base has no measurable linear delta representation.
    """
    prefix = base + "."
    direct_norms: List[float] = []
    dora_scale = lora_sd.get(base + ".dora_scale")
    has_dora = isinstance(dora_scale, torch.Tensor)
    dest_weight = None
    if has_dora:
        dest_weight = _auto_strength_get_effective_destination_weight(
            base,
            key_map,
            model_state_dict,
            clip_state_dict,
            current_model=current_model,
            current_clip=current_clip,
            analysis_device=analysis_device,
        )

    for suffix in (".diff", ".diff_b", ".set_weight"):
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
    current_model: Any = None,
    current_clip: Any = None,
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
            current_model=current_model,
            current_clip=current_clip,
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
            current_model=current_model,
            current_clip=current_clip,
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
    current_model: Any = None,
    current_clip: Any = None,
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
            current_model=current_model,
            current_clip=current_clip,
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
    current_model: Any = None,
    current_clip: Any = None,
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
        current_model=current_model,
        current_clip=current_clip,
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
            elif ks.endswith((".diff", ".diff_b", ".set_weight")):
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


class StateManager:
    """
    Workflow-serialized preset manager for character LoRA stacks, prompt templates,
    settings, and seed state. Runtime execution is source-only and acyclic: capture/load
    actions are explicit frontend graph edits, not execution-time input feedback.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "state_json": (
                    "STRING",
                    {
                        "default": json.dumps(_state_manager_default_state(), separators=(",", ":")),
                        "multiline": True,
                    },
                ),
                "ui_state_json": (
                    "STRING",
                    {
                        "default": json.dumps({"version": _DORA_STATE_MANAGER_SCHEMA_VERSION}, separators=(",", ":")),
                        "multiline": False,
                    },
                ),
                "selected_character_id": ("STRING", {"default": "default_character", "multiline": False}),
                "selected_prompt_id": ("STRING", {"default": "default_prompt", "multiline": False}),
            }
        }

    RETURN_TYPES = ("DORA_STATE", "STRING", "STRING", "STRING", "DORA_LORA_STACK", "DORA_STATE_SETTINGS", "INT", "STATE_MANAGER_CONTROL", "IMAGE", "STRING")
    RETURN_NAMES = (
        "dora_state",
        "positive_prompt_template",
        "negative_prompt_template",
        "settings_json",
        "selected_lora_stack",
        "state_settings",
        "seed",
        "state_control",
        "character_image",
        "fileimage_prefix",
    )
    FUNCTION = "resolve_state"
    CATEGORY = "state managers"

    @classmethod
    def IS_CHANGED(
        cls,
        state_json: Any,
        ui_state_json: Any = "",
        selected_character_id: Any = "",
        selected_prompt_id: Any = "",
    ):
        payload = {
            "state_json": state_json,
            "ui_state_json": ui_state_json,
            "selected_character_id": selected_character_id,
            "selected_prompt_id": selected_prompt_id,
        }
        resolved = _queued_runtime_state_from_ui_state(ui_state_json) or _resolve_dora_state_payload(
            state_json,
            selected_character_id,
            selected_prompt_id,
        )
        if _extract_seed_from_settings(resolved.get("settings", {})) in _STATE_SEED_SPECIALS:
            payload["runtime_seed_nonce"] = _state_manager_new_random_seed()
        return json.dumps(payload, sort_keys=True, default=str)

    @classmethod
    def VALIDATE_INPUTS(
        cls,
        state_json: Any,
        ui_state_json: Any = "",
        selected_character_id: Any = "",
        selected_prompt_id: Any = "",
    ):
        del ui_state_json
        state = _normalize_state_manager_state(state_json)
        character = _pick_state_manager_character(state, selected_character_id)
        prompt = _pick_state_manager_prompt(character, selected_prompt_id)
        if not character:
            return "State Manager: no character states are available."
        if not prompt:
            return "State Manager: no prompt states are available for the selected character."
        return True

    def resolve_state(
        self,
        state_json: Any,
        ui_state_json: Any = "",
        selected_character_id: Any = "",
        selected_prompt_id: Any = "",
    ):
        runtime_payload = _queued_runtime_state_from_ui_state(ui_state_json)
        if runtime_payload is not None:
            payload = _resolve_state_manager_runtime_seed(runtime_payload)
            state = _normalize_state_manager_state(state_json)
            character = _pick_state_manager_character(
                state,
                (payload.get("character") or {}).get("id", selected_character_id) if isinstance(payload.get("character"), dict) else selected_character_id,
            )
            prompt = _pick_state_manager_prompt(
                character,
                (payload.get("prompt") or {}).get("id", selected_prompt_id) if isinstance(payload.get("prompt"), dict) else selected_prompt_id,
            )
        else:
            payload = _resolve_dora_state_payload(
                state_json,
                selected_character_id,
                selected_prompt_id,
            )
            payload = _resolve_state_manager_runtime_seed(payload)
            state = _normalize_state_manager_state(state_json)
            character = _pick_state_manager_character(state, selected_character_id)
            prompt = _pick_state_manager_prompt(character, selected_prompt_id)
        settings = _normalize_settings_with_canonical_seed(payload.get("settings", {}))
        settings_json = json.dumps(settings, ensure_ascii=False, sort_keys=True, indent=2)
        lora_stack_payload = _build_lora_stack_payload(
            _manager_rows_to_lora_entries(payload.get("loras", [])),
            payload.get("loader_globals", {}),
        )
        state_settings_payload = _build_state_settings_payload(payload)
        state_control_payload = _build_state_control_payload(payload)
        seed = _extract_seed_from_settings(settings)
        character_image = _load_state_manager_prompt_or_character_image(character, prompt)
        return (
            payload,
            payload.get("positive_prompt", ""),
            payload.get("negative_prompt", ""),
            settings_json,
            lora_stack_payload,
            state_settings_payload,
            seed,
            state_control_payload,
            character_image,
            payload.get("fileimage_prefix", ""),
        )


class StateManagerTextBox:
    """Editable prompt/text node controlled by State Manager save/load actions.

    The state_control input gives the State Manager frontend a safe non-STRING
    edge to discover this node for Save connected / Load connected and carries
    the resolved text at runtime when connected.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "role": (["positive", "negative", "generic"], {"default": "positive"}),
                "text": ("STRING", {"default": "", "multiline": True}),
                "state_slot": ("STRING", {"default": "default", "multiline": False}),
            },
            "optional": {
                "state_control": ("STATE_MANAGER_CONTROL",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "emit"
    CATEGORY = "state managers"

    @classmethod
    def IS_CHANGED(cls, role: Any, text: Any = "", state_slot: Any = "default", state_control: Any = None):
        state_payload = _normalize_runtime_dora_state_payload(state_control)
        controlled_text = _state_payload_text_for_box(state_payload, role, state_slot)
        effective_text = str(controlled_text if controlled_text is not None else (text or ""))
        return json.dumps(
            {
                "role": str(role),
                "state_slot": str(state_slot),
                "text": effective_text,
                "controlled_character_id": (state_payload.get("character", {}) or {}).get("id", "") if isinstance(state_payload, dict) else "",
                "controlled_prompt_id": (state_payload.get("prompt", {}) or {}).get("id", "") if isinstance(state_payload, dict) else "",
            },
            ensure_ascii=False,
            sort_keys=True,
        )

    def emit(self, role: Any, text: Any = "", state_slot: Any = "default", state_control: Any = None):
        state_payload = _normalize_runtime_dora_state_payload(state_control)
        controlled_text = _state_payload_text_for_box(state_payload, role, state_slot)
        if controlled_text is not None:
            return (controlled_text,)
        return (str(text or ""),)


class StateManagerSeed:
    """Editable seed node controlled by State Manager save/load actions.

    The frontend resolves rgthree-style special seed values before queueing from
    the graph UI. The backend keeps the same special values working for API or
    partial-queue paths that send them through unchanged.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": (
                    "INT",
                    {
                        "default": -1,
                        "min": _STATE_SEED_MIN,
                        "max": _STATE_SEED_MAX,
                        "step": 1,
                    },
                ),
            },
            "optional": {
                "state_control": ("STATE_MANAGER_CONTROL",),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("SEED",)
    FUNCTION = "emit"
    CATEGORY = "state managers"

    @classmethod
    def IS_CHANGED(
        cls,
        seed: Any,
        state_control: Any = None,
        prompt: Any = None,
        extra_pnginfo: Any = None,
        unique_id: Any = None,
        control_after_generate: Any = None,
    ):
        del prompt, extra_pnginfo, unique_id, control_after_generate
        state_payload = _normalize_runtime_dora_state_payload(state_control)
        controlled_seed = _state_payload_seed(state_payload)
        seed_i = _coerce_state_manager_seed(controlled_seed if controlled_seed is not None else seed, -1)
        if seed_i in _STATE_SEED_SPECIALS:
            return _state_manager_new_random_seed()
        return json.dumps(
            {
                "seed": seed_i,
                "controlled_character_id": (state_payload.get("character", {}) or {}).get("id", "") if isinstance(state_payload, dict) else "",
                "controlled_prompt_id": (state_payload.get("prompt", {}) or {}).get("id", "") if isinstance(state_payload, dict) else "",
            },
            ensure_ascii=False,
            sort_keys=True,
        )

    def emit(
        self,
        seed: Any,
        state_control: Any = None,
        prompt: Any = None,
        extra_pnginfo: Any = None,
        unique_id: Any = None,
        control_after_generate: Any = None,
    ):
        del control_after_generate
        state_payload = _normalize_runtime_dora_state_payload(state_control)
        controlled_seed = _state_payload_seed(state_payload)
        original_seed = _coerce_state_manager_seed(controlled_seed if controlled_seed is not None else seed, -1)
        seed_i = original_seed
        if seed_i in _STATE_SEED_SPECIALS:
            if seed_i in (-2, -3):
                _LOG.warning(
                    "[State Manager Seed] cannot %s the last seed server-side; using a new random seed.",
                    "increment" if seed_i == -2 else "decrement",
                )
            seed_i = _state_manager_new_random_seed()
            self._update_metadata_seed(original_seed, seed_i, prompt, extra_pnginfo, unique_id)
        return (seed_i,)

    @staticmethod
    def _update_metadata_seed(original_seed: int, seed: int, prompt: Any, extra_pnginfo: Any, unique_id: Any) -> None:
        if unique_id is None:
            return
        uid = str(unique_id)

        try:
            prompt_node = prompt.get(uid) if isinstance(prompt, dict) else None
            inputs = prompt_node.get("inputs") if isinstance(prompt_node, dict) else None
            if isinstance(inputs, dict) and inputs.get("seed") == original_seed:
                inputs["seed"] = seed
        except Exception:
            pass

        try:
            workflow = extra_pnginfo.get("workflow") if isinstance(extra_pnginfo, dict) else None
            nodes = workflow.get("nodes") if isinstance(workflow, dict) else None
            if not isinstance(nodes, list):
                return
            for node in nodes:
                if not isinstance(node, dict) or str(node.get("id")) != uid:
                    continue
                values = node.get("widgets_values")
                if isinstance(values, list):
                    for index, value in enumerate(values):
                        if value == original_seed:
                            values[index] = seed
                break
        except Exception:
            pass


# Backward-compatible class alias for workflows created with the earlier name.
DoraStateManager = StateManager


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
                    # Optional runtime state-manager input. When connected, this overrides local LoRA rows
                    # and any saved loader-global settings in the manager payload.
                    # Prefer state_control for save/load-only graph-editor workflows.
                    "dora_state": ("DORA_STATE",),

                    # Editor-only relationship used by State Manager Save/Load connected.
                    # Ignored by backend execution.
                    "state_control": ("STATE_MANAGER_CONTROL",),
                    "state_slot": ("STRING", {"default": "default", "multiline": False}),

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

    RETURN_TYPES = ("MODEL", "CLIP", "STRING", "STRING", "DORA_LORA_STACK")
    RETURN_NAMES = ("MODEL", "CLIP", "auto_strength_report_json", "analysis_report", "lora_stack")
    HAS_INTERMEDIATE_OUTPUT = True
    FUNCTION = "load_loras"
    CATEGORY = "loaders"

    @classmethod
    def IS_CHANGED(cls, model: Any = None, clip: Any = None, **kwargs):
        return _dora_loader_cache_key_from_inputs(model, clip, kwargs)

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

        _sanitize_reshape_weight_metadata(lora_sd, lora_name=lora_name, verbose=verbose)

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
                current_model=model,
                current_clip=clip,
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
        # state_control is the preferred State Manager relationship edge. When it
        # carries a runtime payload, use it as the fallback source for LoRA rows
        # and loader-global settings so state_control-only graphs follow queued
        # character/prompt wildcarding. A direct dora_state input still wins.
        state_slot = _clean_loader_slot(kwargs.get("state_slot", "default"), "default")
        state_payload = _normalize_runtime_dora_state_payload(kwargs.get("dora_state"))
        if state_payload is None:
            state_payload = _normalize_runtime_dora_state_payload(kwargs.get("state_control"))
        kwargs.pop("state_control", None)

        # Global controls (provided by JS UI; optionally overridden by DoRA State Manager)
        stack_enabled = bool(_state_payload_get_loader_global(state_payload, "stack_enabled", kwargs.get("stack_enabled", True), state_slot))
        verbose = bool(_state_payload_get_loader_global(state_payload, "verbose", kwargs.get("verbose", False), state_slot))
        log_unloaded_keys = bool(_state_payload_get_loader_global(state_payload, "log_unloaded_keys", kwargs.get("log_unloaded_keys", False), state_slot))
        broadcast_auto_scale = bool(_state_payload_get_loader_global(state_payload, "broadcast_auto_scale", kwargs.get("broadcast_auto_scale", True), state_slot))
        broadcast_modulations = bool(_state_payload_get_loader_global(state_payload, "broadcast_modulations", kwargs.get("broadcast_modulations", True), state_slot))
        broadcast_include_dora_scale = bool(_state_payload_get_loader_global(state_payload, "broadcast_include_dora_scale", kwargs.get("broadcast_include_dora_scale", False), state_slot))
        try:
            broadcast_scale = float(_state_payload_get_loader_global(state_payload, "broadcast_scale", kwargs.get("broadcast_scale", 1.0), state_slot))
        except Exception:
            broadcast_scale = 1.0

        # DoRA decompose debug controls (node-adjustable / state-manager overrideable)
        dora_dbg = bool(_state_payload_get_loader_global(state_payload, "dora_decompose_debug", kwargs.get("dora_decompose_debug", False), state_slot))
        try:
            dora_dbg_n = int(_state_payload_get_loader_global(state_payload, "dora_decompose_debug_n", kwargs.get("dora_decompose_debug_n", 30), state_slot))
        except Exception:
            dora_dbg_n = 30
        try:
            dora_dbg_stack = int(_state_payload_get_loader_global(state_payload, "dora_decompose_debug_stack_depth", kwargs.get("dora_decompose_debug_stack_depth", 10), state_slot))
        except Exception:
            dora_dbg_stack = 10
        dora_slice_fix = bool(_state_payload_get_loader_global(state_payload, "dora_slice_fix", kwargs.get("dora_slice_fix", True), state_slot))
        dora_adaln_swap_fix = bool(_state_payload_get_loader_global(state_payload, "dora_adaln_swap_fix", kwargs.get("dora_adaln_swap_fix", True), state_slot))
        zimage_lumina2_compat = bool(_state_payload_get_loader_global(state_payload, "zimage_lumina2_compat", kwargs.get("zimage_lumina2_compat", True), state_slot))
        auto_strength_enabled = bool(_state_payload_get_loader_global(state_payload, "auto_strength_enabled", kwargs.get("auto_strength_enabled", False), state_slot))
        auto_strength_device = _normalize_auto_strength_device(_state_payload_get_loader_global(state_payload, "auto_strength_device", kwargs.get("auto_strength_device", "gpu"), state_slot))
        try:
            auto_strength_ratio_floor = float(_state_payload_get_loader_global(state_payload, "auto_strength_ratio_floor", kwargs.get("auto_strength_ratio_floor", _AUTO_STRENGTH_RATIO_FLOOR), state_slot))
        except Exception:
            auto_strength_ratio_floor = _AUTO_STRENGTH_RATIO_FLOOR
        try:
            auto_strength_ratio_ceiling = float(_state_payload_get_loader_global(state_payload, "auto_strength_ratio_ceiling", kwargs.get("auto_strength_ratio_ceiling", _AUTO_STRENGTH_RATIO_CEILING), state_slot))
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

        entries = _parse_dora_state_lora_entries(state_payload, state_slot) if state_payload is not None else None
        if entries is None:
            entries = _parse_lora_stack_kwargs(kwargs)

        loader_globals_payload = {
            "stack_enabled": stack_enabled,
            "verbose": verbose,
            "log_unloaded_keys": log_unloaded_keys,
            "broadcast_auto_scale": broadcast_auto_scale,
            "broadcast_modulations": broadcast_modulations,
            "broadcast_include_dora_scale": broadcast_include_dora_scale,
            "broadcast_scale": broadcast_scale,
            "dora_decompose_debug": dora_dbg,
            "dora_decompose_debug_n": dora_dbg_n,
            "dora_decompose_debug_stack_depth": dora_dbg_stack,
            "dora_slice_fix": dora_slice_fix,
            "dora_adaln_swap_fix": dora_adaln_swap_fix,
            "zimage_lumina2_compat": zimage_lumina2_compat,
            "auto_strength_enabled": auto_strength_enabled,
            "auto_strength_device": auto_strength_device,
            "auto_strength_ratio_floor": auto_strength_ratio_floor,
            "auto_strength_ratio_ceiling": auto_strength_ratio_ceiling,
        }
        lora_stack_payload = _build_lora_stack_payload(entries, loader_globals_payload, state_slot)

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
                "result": (model, clip, report_json, report_text, lora_stack_payload),
                "ui": {
                    "auto_strength_report_json": (report_json,),
                    "analysis_report": (report_text,),
                },
            }

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
                "result": (model, clip, report_json, report_text, lora_stack_payload),
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
            "result": (new_model, new_clip, report_json, report_text, lora_stack_payload),
            "ui": {
                "auto_strength_report_json": (report_json,),
                "analysis_report": (report_text,),
            },
        }
