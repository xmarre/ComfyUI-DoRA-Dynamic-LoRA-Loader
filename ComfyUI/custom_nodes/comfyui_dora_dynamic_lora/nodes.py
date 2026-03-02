import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import comfy.lora
import comfy.lora_convert
import comfy.utils
import folder_paths

_LOG = logging.getLogger(__name__)

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


def _clone_base_block(lora_sd: Dict[str, Any], from_base: str, to_base: str) -> int:
    """
    Clone all entries under from_base.* to to_base.* (by prefix), preserving suffixes.
    Returns number of keys created.
    """
    created = 0
    prefix = from_base + "."
    keys = list(lora_sd.keys())
    for k in keys:
        if not k.startswith(prefix):
            continue
        nk = to_base + k[len(from_base) :]
        if nk in lora_sd:
            continue
        lora_sd[nk] = lora_sd[k]
        created += 1
    return created


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
    verbose: bool = False,
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

    # Determine block counts
    depth, depth_single = _get_unet_config_counts(model)
    if depth == 0 and depth_single == 0:
        depth, depth_single = _infer_flux_block_counts(model_sd_keys)
    if verbose:
        _LOG.info("[DoRA Power LoRA Loader] flux2 compat: inferred blocks: double=%s single=%s", depth, depth_single)

    # 2) Broadcast global modulations if present
    # Source bases from OneTrainer export (the ones you logged as missing).
    src_img = "transformer.double_stream_modulation_img.linear"
    src_txt = "transformer.double_stream_modulation_txt.linear"
    src_single = "transformer.single_stream_modulation.linear"

    # Only broadcast if those bases exist in the LoRA file.
    have_img = any(k.startswith(src_img + ".") for k in lora_sd.keys())
    have_txt = any(k.startswith(src_txt + ".") for k in lora_sd.keys())
    have_single = any(k.startswith(src_single + ".") for k in lora_sd.keys())

    if have_img and depth > 0:
        created = 0
        for i in range(depth):
            dst = f"transformer.transformer_blocks.{i}.norm1.linear"
            created += _clone_base_block(lora_sd, src_img, dst)
        if verbose:
            _LOG.info(
                "[DoRA Power LoRA Loader] flux2 compat: broadcast %s -> transformer_blocks.*.norm1.linear (keys=%s)",
                src_img,
                created,
            )
        # prevent always-missing spam for the original, unmapped keys
        _delete_prefix_keys(lora_sd, src_img + ".")

    if have_txt and depth > 0:
        created = 0
        for i in range(depth):
            dst = f"transformer.transformer_blocks.{i}.norm1_context.linear"
            created += _clone_base_block(lora_sd, src_txt, dst)
        if verbose:
            _LOG.info(
                "[DoRA Power LoRA Loader] flux2 compat: broadcast %s -> transformer_blocks.*.norm1_context.linear (keys=%s)",
                src_txt,
                created,
            )
        _delete_prefix_keys(lora_sd, src_txt + ".")

    if have_single and depth_single > 0:
        created = 0
        for i in range(depth_single):
            dst = f"transformer.single_transformer_blocks.{i}.norm.linear"
            created += _clone_base_block(lora_sd, src_single, dst)
        if verbose:
            _LOG.info(
                "[DoRA Power LoRA Loader] flux2 compat: broadcast %s -> single_transformer_blocks.*.norm.linear (keys=%s)",
                src_single,
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
            "optional": FlexibleOptionalInputType(any_type),
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

        # Flux2/OneTrainer DoRA compat: rewrite + broadcast missing modules into keys ComfyUI maps.
        # This fixes cases where critical dora_scale/lora_up/down tensors never map/load.
        if model is not None:
            _apply_flux2_onetrainer_dora_compat(
                lora_sd=lora_sd,
                model=model,
                model_sd_keys=model_sd_keys,
                verbose=verbose,
            )

        # Extract base module names from file keys.
        lora_bases = _extract_lora_bases(lora_sd.keys())
        if verbose:
            _LOG.info("[DoRA Power LoRA Loader] %s: bases in file: %s", lora_name, len(lora_bases))

        # Start with standard ComfyUI key map.
        key_map: Dict[str, str] = {}
        if model is not None:
            key_map = comfy.lora.model_lora_keys_unet(model.model, key_map)
        if clip is not None:
            key_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, key_map)

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

        # Apply patches to provided model/clip (already cloned by caller).
        if model is not None:
            model.add_patches(loaded, strength_model)
        if clip is not None:
            clip.add_patches(loaded, strength_clip)

        return model, clip

    def load_loras(self, model, clip, **kwargs):
        # Global controls (provided by JS UI; also safe if absent)
        stack_enabled = bool(kwargs.get("stack_enabled", True))
        verbose = bool(kwargs.get("verbose", False))
        log_unloaded_keys = bool(kwargs.get("log_unloaded_keys", False))

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

        if new_model is not None:
            model_state_dict = new_model.model.state_dict()
            model_sd_list = list(model_state_dict.keys())
            model_sd_keys = set(model_sd_list)

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
                model_sd_keys=model_sd_keys,
                model_sd_list=model_sd_list,
                clip_sd_keys=clip_sd_keys,
                clip_sd_list=clip_sd_list,
            )

        return (new_model, new_clip)
