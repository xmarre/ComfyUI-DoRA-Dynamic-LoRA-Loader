"""Opt-in runtime/bypass LoRA mode for the DoRA Power LoRA Loader.

The normal ComfyUI LoRA path materializes patched model weights. On very large
HIGH_VRAM models that can retain a second full copy of every LoRA-targeted base
weight. Runtime bypass mode keeps the base weights untouched and evaluates
standard LoRA adapters in the module forward pass instead.

Safety is deliberate: ComfyUI's bypass LoRA implementation does not implement
DoRA magnitude normalization, so this mode refuses DoRA and other adapter forms
that cannot be represented exactly by the supported runtime path.
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple

import comfy.patcher_extension
import comfy.weight_adapter
import comfy.utils
import folder_paths

from . import nodes as _base


_LOG = logging.getLogger(__name__)
_RUNTIME_INPUT = "runtime_bypass_lora"
_RUNTIME_INJECTION_PREFIX = "dora_runtime_bypass_lora"


class RuntimeBypassUnsupportedError(RuntimeError):
    """Raised when runtime mode would change the mathematical meaning of a LoRA."""


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off", ""}:
            return False
        return default
    return bool(value)


def _is_dora_key(key: Any) -> bool:
    text = str(key).lower()
    return (
        "dora_scale" in text
        or "lora_magnitude_vector" in text
        or ".w_norm" in text
        or ".b_norm" in text
    )


def _read_lora_key_names(path: str) -> List[str]:
    """Read only key names where possible; avoid materializing safetensor payloads."""
    if str(path).lower().endswith(".safetensors"):
        try:
            from safetensors import safe_open

            with safe_open(path, framework="pt", device="cpu") as f:
                return [str(k) for k in f.keys()]
        except Exception as exc:
            _LOG.debug(
                "[DoRA Power LoRA Loader] runtime bypass: safetensors key-only scan failed for %s (%r); falling back to Comfy loader.",
                path,
                exc,
            )

    try:
        data = comfy.utils.load_torch_file(path, safe_load=True)
    except TypeError:
        data = comfy.utils.load_torch_file(path)
    try:
        return [str(k) for k in data.keys()]
    finally:
        del data


def _raw_dora_keys(lora_name: str) -> List[str]:
    path = folder_paths.get_full_path("loras", lora_name)
    if not path:
        raise FileNotFoundError(f"LoRA not found: {lora_name}")
    return [key for key in _read_lora_key_names(path) if _is_dora_key(key)]


def _patch_target(raw_key: Any) -> Tuple[str, Any, Any]:
    if isinstance(raw_key, str):
        return raw_key, None, None
    if isinstance(raw_key, tuple) and raw_key:
        key = raw_key[0]
        offset = raw_key[1] if len(raw_key) > 1 else None
        function = raw_key[2] if len(raw_key) > 2 else None
        return str(key), offset, function
    return str(raw_key), None, None


def _resolve_module(root: Any, weight_key: str) -> Any:
    if not weight_key.endswith(".weight"):
        raise RuntimeBypassUnsupportedError(
            f"Runtime bypass only supports adapter weight targets; got {weight_key!r}."
        )

    module_key = weight_key[: -len(".weight")]
    module = root
    for part in module_key.split("."):
        try:
            module = getattr(module, part)
            continue
        except (AttributeError, TypeError):
            pass

        try:
            module = module[part]
            continue
        except (KeyError, IndexError, TypeError, AttributeError):
            pass

        if part.isdigit():
            try:
                module = module[int(part)]
                continue
            except (KeyError, IndexError, TypeError, AttributeError):
                pass

        raise RuntimeBypassUnsupportedError(
            f"Runtime bypass could not resolve module {module_key!r} for weight {weight_key!r}."
        )
    return module


def _validate_lora_adapter(adapter: Any, lora_name: str, raw_key: Any) -> None:
    if not isinstance(adapter, comfy.weight_adapter.LoRAAdapter):
        raise RuntimeBypassUnsupportedError(
            "Runtime bypass currently supports standard LoRA adapters only. "
            f"{lora_name!r} produced {type(adapter).__name__} for {raw_key!r}. "
            "Disable Runtime bypass LoRA for this file."
        )

    weights = getattr(adapter, "weights", None)
    if not isinstance(weights, (tuple, list)):
        raise RuntimeBypassUnsupportedError(
            f"Runtime bypass could not inspect the LoRA adapter for {lora_name!r} at {raw_key!r}."
        )

    # ComfyUI LoRAAdapter weights are:
    # (up, down, alpha, mid, dora_scale, reshape)
    dora_scale = weights[4] if len(weights) > 4 else None
    reshape = weights[5] if len(weights) > 5 else None
    if dora_scale is not None:
        raise RuntimeBypassUnsupportedError(
            "Runtime bypass LoRA is enabled, but "
            f"{lora_name!r} contains DoRA magnitude scaling at {raw_key!r}. "
            "ComfyUI's runtime bypass path does not apply DoRA magnitude normalization, "
            "so running it would silently change the result. Disable Runtime bypass LoRA for this DoRA."
        )
    if reshape is not None:
        raise RuntimeBypassUnsupportedError(
            "Runtime bypass does not currently support LoRA reshape metadata. "
            f"{lora_name!r} uses it at {raw_key!r}. Disable Runtime bypass LoRA for this file."
        )


def _make_stacked_injection(
    root: Any,
    adapters: Iterable[Dict[str, Any]],
) -> Tuple[List[Any], int]:
    """Build one injection whose same-module hooks unwind in strict reverse order."""
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in adapters:
        grouped[str(item["key"])].append(item)

    hooks: List[Any] = []
    for weight_key, items in grouped.items():
        module = _resolve_module(root, weight_key)
        if not hasattr(module, "weight"):
            raise RuntimeBypassUnsupportedError(
                f"Runtime bypass target {weight_key!r} resolved to {type(module).__name__}, which has no weight attribute."
            )
        for item in items:
            hook = comfy.weight_adapter.BypassForwardHook(
                module,
                item["adapter"],
                multiplier=float(item["strength"]),
            )
            hooks.append(hook)

    active_hooks: List[Any] = []

    def inject_all(_model_patcher):
        # ModelPatcher normally prevents duplicate injection, but keep this
        # injection idempotent when exercised directly as well.
        if active_hooks:
            return
        try:
            for hook in hooks:
                hook.inject()
                active_hooks.append(hook)
        except Exception:
            while active_hooks:
                hook = active_hooks.pop()
                try:
                    hook.eject()
                except Exception:
                    _LOG.exception(
                        "[DoRA Power LoRA Loader] runtime bypass: rollback eject failed."
                    )
            raise

    def eject_all(_model_patcher):
        # Multiple LoRAs on one module form nested forward wrappers. They MUST be
        # removed in reverse order or an older wrapper can be restored accidentally.
        first_error = None
        while active_hooks:
            hook = active_hooks.pop()
            try:
                hook.eject()
            except Exception as exc:
                _LOG.exception(
                    "[DoRA Power LoRA Loader] runtime bypass: hook eject failed."
                )
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    injection = comfy.patcher_extension.PatcherInjection(
        inject=inject_all,
        eject=eject_all,
    )
    return [injection], len(hooks)


def _instance_override(obj: Any, name: str, replacement: Any):
    """Install an instance-only method override and return a restoration callback."""
    d = getattr(obj, "__dict__", None)
    had_instance_value = isinstance(d, dict) and name in d
    previous_instance_value = d.get(name) if had_instance_value else None
    setattr(obj, name, replacement)

    def restore():
        if had_instance_value:
            setattr(obj, name, previous_instance_value)
        else:
            try:
                delattr(obj, name)
            except AttributeError:
                pass

    return restore


class RuntimeBypassDoraPowerLoraLoader(_base.DoraPowerLoraLoader):
    """DoRA Power LoRA Loader with an opt-in low-VRAM standard-LoRA runtime path."""

    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        optional = inputs.get("optional")
        if optional is not None:
            optional[_RUNTIME_INPUT] = (
                "BOOLEAN",
                {
                    "default": False,
                    "tooltip": (
                        "Apply standard LoRAs in the forward pass instead of materializing patched model weights. "
                        "Greatly reduces persistent VRAM on very large HIGH_VRAM models. DoRA and unsupported "
                        "adapter/offset forms are refused rather than approximated."
                    ),
                },
            )
        return inputs

    @classmethod
    def IS_CHANGED(cls, model: Any = None, clip: Any = None, **kwargs):
        # The base loader's cache key intentionally only includes its known globals.
        # Add this mode explicitly so toggling it always invalidates the node output.
        base_key = super().IS_CHANGED(model=model, clip=clip, **kwargs)
        return json.dumps(
            {
                "base": base_key,
                _RUNTIME_INPUT: _as_bool(kwargs.get(_RUNTIME_INPUT, False)),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def _runtime_preflight(self, lora_name: str) -> None:
        cache = getattr(self, "_runtime_dora_scan_cache", None)
        if cache is None:
            cache = {}
            self._runtime_dora_scan_cache = cache

        path = folder_paths.get_full_path("loras", lora_name)
        if not path:
            raise FileNotFoundError(f"LoRA not found: {lora_name}")
        try:
            st = os.stat(path)
            signature = (str(path), int(st.st_size), int(getattr(st, "st_mtime_ns", 0)))
        except OSError:
            signature = (str(path), None, None)

        if signature not in cache:
            if len(cache) >= 32:
                cache.pop(next(iter(cache)))
            cache[signature] = _raw_dora_keys(lora_name)

        dora_keys = cache[signature]
        if dora_keys:
            examples = ", ".join(dora_keys[:3])
            suffix = "" if len(dora_keys) <= 3 else f" (+{len(dora_keys) - 3} more)"
            raise RuntimeBypassUnsupportedError(
                "Runtime bypass LoRA is enabled, but "
                f"{lora_name!r} is a DoRA/magnitude LoRA ({examples}{suffix}). "
                "ComfyUI's bypass adapter math does not implement DoRA magnitude normalization. "
                "Disable Runtime bypass LoRA for this file; it will not be silently approximated."
            )

    def _capture_add_patches(
        self,
        target: str,
        state_dict_keys: set[str],
        original_add_patches: Any,
        lora_name: str,
    ):
        capture = self._runtime_bypass_capture[target]

        def add_patches_runtime(patches, *args, **kwargs):
            strength_patch = kwargs.get("strength_patch", args[0] if args else 1.0)
            strength_model = kwargs.get("strength_model", args[1] if len(args) > 1 else 1.0)
            try:
                strength_patch_f = float(strength_patch)
                strength_model_f = float(strength_model)
            except Exception as exc:
                raise RuntimeBypassUnsupportedError(
                    f"Runtime bypass received non-scalar patch strengths for {lora_name!r}."
                ) from exc

            if abs(strength_model_f - 1.0) > 1e-12:
                raise RuntimeBypassUnsupportedError(
                    "Runtime bypass cannot reproduce ModelPatcher strength_model scaling. "
                    f"{lora_name!r} requested strength_model={strength_model_f}."
                )

            regular: Dict[Any, Any] = {}
            applied: List[Any] = []

            for raw_key, patch_data in patches.items():
                key, offset, function = _patch_target(raw_key)
                if key not in state_dict_keys:
                    continue

                if isinstance(patch_data, comfy.weight_adapter.WeightAdapterBase):
                    _validate_lora_adapter(patch_data, lora_name, raw_key)
                    if offset is not None or function is not None:
                        raise RuntimeBypassUnsupportedError(
                            "Runtime bypass does not currently support sliced/offset or transformed LoRA targets. "
                            f"{lora_name!r} uses one at {raw_key!r}. Disable Runtime bypass LoRA for this file."
                        )
                    capture.append(
                        {
                            "key": key,
                            "adapter": patch_data,
                            "strength": strength_patch_f,
                            "lora_name": lora_name,
                        }
                    )
                    applied.append(raw_key)
                else:
                    # Match ComfyUI's own bypass loader: non-adapter patches retain
                    # their normal materialized semantics. Standard LoRA files should
                    # overwhelmingly/entirely take the adapter path above.
                    regular[raw_key] = patch_data

            if regular:
                result = original_add_patches(regular, *args, **kwargs)
                if result:
                    applied.extend(result)
                _LOG.warning(
                    "[DoRA Power LoRA Loader] runtime bypass: %s contains %d non-adapter patch(es) for %s; "
                    "those patches still use normal materialized weight patching.",
                    lora_name,
                    len(regular),
                    target,
                )

            return applied

        return add_patches_runtime

    def _load_one(self, model, clip, *args, **kwargs):
        if not getattr(self, "_runtime_bypass_active", False):
            return super()._load_one(model, clip, *args, **kwargs)

        lora_name = str(kwargs.get("lora_name") or "")
        if not lora_name:
            raise RuntimeBypassUnsupportedError("Runtime bypass could not determine the active LoRA name.")

        self._runtime_preflight(lora_name)

        restores = []
        try:
            if model is not None:
                model_keys = set(model.model.state_dict().keys())
                original = model.add_patches
                restores.append(
                    _instance_override(
                        model,
                        "add_patches",
                        self._capture_add_patches("model", model_keys, original, lora_name),
                    )
                )

            if clip is not None:
                clip_keys = set(clip.cond_stage_model.state_dict().keys())
                original = clip.add_patches
                restores.append(
                    _instance_override(
                        clip,
                        "add_patches",
                        self._capture_add_patches("clip", clip_keys, original, lora_name),
                    )
                )

            return super()._load_one(model, clip, *args, **kwargs)
        finally:
            for restore in reversed(restores):
                restore()

    def load_loras(self, model, clip, **kwargs):
        runtime_enabled = _as_bool(kwargs.get(_RUNTIME_INPUT, False))
        if not runtime_enabled:
            return super().load_loras(model, clip, **kwargs)

        if not hasattr(comfy.weight_adapter, "BypassForwardHook"):
            raise RuntimeError(
                "Runtime bypass LoRA requires a ComfyUI build with weight_adapter.BypassForwardHook support. "
                "Disable Runtime bypass LoRA or update ComfyUI."
            )

        self._runtime_bypass_active = True
        self._runtime_bypass_capture = {"model": [], "clip": []}
        self._runtime_dora_scan_cache = {}

        try:
            output = super().load_loras(model, clip, **kwargs)
            captured = bool(
                self._runtime_bypass_capture["model"]
                or self._runtime_bypass_capture["clip"]
            )
            if not isinstance(output, dict) or "result" not in output:
                if captured:
                    raise RuntimeBypassUnsupportedError(
                        "Runtime bypass captured LoRA adapters, but the loader output shape is unsupported, "
                        "so the adapters cannot be injected. Disable Runtime bypass LoRA."
                    )
                return output

            result = output["result"]
            if not isinstance(result, (tuple, list)):
                if captured:
                    raise RuntimeBypassUnsupportedError(
                        "Runtime bypass captured LoRA adapters, but the loader result is not a sequence, "
                        "so the adapters cannot be injected. Disable Runtime bypass LoRA."
                    )
                return output

            new_model = result[0] if len(result) > 0 else None
            new_clip = result[1] if len(result) > 1 else None

            model_count = 0
            clip_count = 0
            injection_key = f"{_RUNTIME_INJECTION_PREFIX}_{id(self)}"

            model_adapters = self._runtime_bypass_capture["model"]
            if new_model is not None and model_adapters:
                injections, model_count = _make_stacked_injection(new_model.model, model_adapters)
                new_model.set_injections(injection_key, injections)

            clip_adapters = self._runtime_bypass_capture["clip"]
            if new_clip is not None and clip_adapters:
                injections, clip_count = _make_stacked_injection(new_clip.cond_stage_model, clip_adapters)
                new_clip.patcher.set_injections(injection_key, injections)

            _LOG.info(
                "[DoRA Power LoRA Loader] Runtime bypass LoRA enabled: model_hooks=%d clip_hooks=%d. "
                "Base weights remain unmaterialized for these standard-LoRA adapters.",
                model_count,
                clip_count,
            )

            return output
        finally:
            self._runtime_bypass_active = False
            self._runtime_bypass_capture = {"model": [], "clip": []}
            self._runtime_dora_scan_cache = {}
