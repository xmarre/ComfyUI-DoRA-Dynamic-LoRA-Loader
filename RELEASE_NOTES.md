# DoRA Dynamic LoRA Loader v1.0.39

Adds an opt-in runtime LoRA path for reducing the large persistent VRAM cost of materializing standard LoRA patches on very large HIGH_VRAM models, together with proper CI and automated release infrastructure.

## Runtime bypass LoRA

- Adds `Runtime bypass LoRA (low VRAM; standard LoRA only)` to the DoRA Power LoRA Loader.
- Keeps the option disabled by default so existing workflows retain the current materialized-weight behavior.
- Applies supported standard LoRA adapters in the module forward pass rather than permanently materializing LoRA-patched base weights.
- Supports multiple stacked LoRAs targeting the same module and removes nested forward wrappers in reverse order during ejection.
- Keeps LoRA strength as a runtime adapter multiplier, avoiding a full base-weight rematerialization solely because the strength changes.

## Correctness safeguards

Runtime bypass intentionally fails closed whenever ComfyUI's bypass path cannot reproduce the original patch semantics exactly.

It refuses:

- DoRA magnitude scaling (`dora_scale`, PEFT/Diffusers `lora_magnitude_vector`, `w_norm`, `b_norm`)
- non-LoRA weight-adapter types
- reshape metadata
- sliced, offset, or transformed adapter targets
- non-default `strength_model` patch semantics

DoRA is checked from raw file keys before application and again from the constructed adapter object. Unsupported files produce an explicit error instead of silently changing their mathematical meaning.

## Testing and release automation

- Adds a GitHub Actions test matrix against pinned ComfyUI v0.29.2, v0.30.2, and v0.31.1 commits.
- Compiles the Python sources on every tested ComfyUI version.
- Checks every frontend JavaScript file with `node --check`.
- Builds the package wheel in CI.
- Adds runtime-bypass tests covering default-off behavior, cache invalidation, DoRA/reshape/offset rejection, non-adapter fallback behavior, standard adapter capture without materialization, stacked-LoRA additive equivalence, and clean forward restoration.
- Adds an automated GitHub release workflow gated on a successful `tests` run from `main`.
- GitHub releases contain a versioned repository ZIP and `SHA256SUMS` manifest and use this file as the release notes.
- Aligns Comfy Registry publishing with the pinned action revisions used by Spectrum MiniMax H3.

## Compatibility

The runtime bypass option is opt-in. With it disabled, the loader continues through the existing LoRA/DoRA path. State Manager behavior, existing loader globals, Flux/Flux2 compatibility handling, ZiT/Lumina2 normalization, DoRA fixes, and auto-strength remain on their existing path.
