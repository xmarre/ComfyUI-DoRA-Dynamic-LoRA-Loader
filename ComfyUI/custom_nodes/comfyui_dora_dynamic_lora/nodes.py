import logging
from typing import Dict, Iterable, List, Optional, Set, Tuple

import comfy.lora
import comfy.lora_convert
import comfy.utils
import folder_paths

_LOG = logging.getLogger(__name__)

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
                _LOG.info("[DoRA Dynamic LoRA Loader] map: %s -> %s", base, found)
        else:
            unresolved.append(base)
            if verbose:
                _LOG.warning("[DoRA Dynamic LoRA Loader] unresolved LoRA base: %s", base)

    return added, unresolved


class DoraDynamicLoraLoader:
    """
    Drop-in LoRA loader that:
    - keeps standard ComfyUI key_map generation,
    - adds dynamic suffix-based mapping for bases in the LoRA file,
    - then uses comfy.lora.load_lora so DoRA works as core implements it.
    """

    @classmethod
    def INPUT_TYPES(cls):
        loras = ["None"] + folder_paths.get_filename_list("loras")
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_name": (loras,),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
                "verbose": ("BOOLEAN", {"default": False}),
                "log_unloaded_keys": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load"
    CATEGORY = "loaders"

    def load(self, model, clip, lora_name, strength_model, strength_clip, verbose=False, log_unloaded_keys=False):
        if lora_name in (None, "None", ""):
            return (model, clip)

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

        # Extract base module names from file keys.
        lora_bases = _extract_lora_bases(lora_sd.keys())
        if verbose:
            _LOG.info("[DoRA Dynamic LoRA Loader] bases in file: %s", len(lora_bases))

        # Start with standard ComfyUI key map.
        key_map: Dict[str, str] = {}
        if model is not None:
            key_map = comfy.lora.model_lora_keys_unet(model.model, key_map)
        if clip is not None:
            key_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, key_map)

        # Prepare state_dict key sets/lists for dynamic matching.
        model_sd_keys = model_sd_list = None
        clip_sd_keys = clip_sd_list = None

        if model is not None:
            model_state_dict = model.model.state_dict()
            model_sd_list = list(model_state_dict.keys())
            model_sd_keys = set(model_sd_list)

        if clip is not None:
            clip_state_dict = clip.cond_stage_model.state_dict()
            clip_sd_list = list(clip_state_dict.keys())
            clip_sd_keys = set(clip_sd_list)

        # Extend map with missing bases from the file.
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
                "[DoRA Dynamic LoRA Loader] dynamic mappings added: %s, unresolved: %s",
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

        # Apply patches to clones (same pattern as comfy.sd.load_lora_for_models).
        new_model = model.clone() if model is not None else None
        new_clip = clip.clone() if clip is not None else None

        loaded_model_keys = set(new_model.add_patches(loaded, strength_model)) if new_model is not None else set()
        loaded_clip_keys = set(new_clip.add_patches(loaded, strength_clip)) if new_clip is not None else set()

        # Optional warning for patches that did not land anywhere.
        if verbose:
            for key in loaded.keys():
                if (key not in loaded_model_keys) and (key not in loaded_clip_keys):
                    _LOG.warning("[DoRA Dynamic LoRA Loader] patch not applied to model/clip: %s", key)

        return (new_model, new_clip)
