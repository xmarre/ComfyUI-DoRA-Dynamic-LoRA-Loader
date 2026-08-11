# DoRA Dynamic LoRA Loader v1.0.39

This release adds an opt-in runtime LoRA path that avoids the large persistent VRAM cost of materializing standard LoRA patches on very large HIGH_VRAM models. It also adds reproducible CI, Comfy Registry packaging exclusions, and automated GitHub release infrastructure.

## Runtime bypass LoRA

- Adds `Runtime bypass LoRA (low VRAM)` to the DoRA Power LoRA Loader.
- Keeps the option **OFF by default**, so existing workflows retain the previous materialized-weight behavior.
- For supported standard LoRAs, runtime bypass preserves the intended LoRA transformation while changing how it is executed: the normal path evaluates `(W + ΔW)x`, while runtime bypass keeps `W` untouched and evaluates `Wx + ΔWx` during the forward pass.
- Avoids retaining a full materialized LoRA-patched copy of all targeted base weights.
- Supports multiple stacked LoRAs on the same module with state-aware, reverse-order hook removal.
- Strength changes update the runtime adapter multiplier instead of requiring the full base-weight set to be rematerialized.
- Non-adapter patches retain normal ComfyUI materialized patch semantics and are reported in the log.

## Correctness safeguards

Runtime bypass deliberately fails closed whenever the supported forward-pass path cannot reproduce the normal patch semantics.

It refuses:

- DoRA magnitude scaling (`dora_scale`, PEFT/Diffusers `lora_magnitude_vector`, `w_norm`, `b_norm`)
- non-LoRA weight-adapter types
- reshape metadata
- sliced, offset, or transformed adapter targets
- non-default `strength_model` patch semantics

DoRA is checked from raw file keys before application and again from the constructed adapter object. Unsupported files produce an explicit error instead of silently changing their mathematical meaning.

The injection lifecycle is also state-aware: partial injection rollback, repeated ejection, and stacked wrappers cannot restore a stale module `forward`. Runtime mode also fails closed if adapters were captured but the parent loader returns an output shape that cannot accept the resulting injections.

## MiniMax-H3 VRAM validation

The original HIGH_VRAM trace showed 208 LoRA-targeted MiniMax-H3 weights retaining exactly one additional BF16-sized live allocation each, totaling about **38,220 MiB** of extra live VRAM before inference.

With runtime bypass enabled in the tested MiniMax-H3 workflow:

- that model-sized materialized LoRA duplication disappeared;
- repeated LoRA strength changes peaked in the low-70-GiB range and returned to the same settled live-allocation baseline afterward instead of accumulating on each change;
- a Turbo LoRA remained visibly effective in runtime mode.

Small floating-point differences from the merged path are still possible because the operations are executed separately and may follow a different dtype/order path.

## Testing and release automation

- Adds package/wheel validation, Python compilation, and frontend JavaScript syntax checks.
- Adds compatibility/runtime-bypass tests against pinned ComfyUI v0.29.2, v0.30.2, and v0.31.1 revisions with explicit CPU PyTorch-family pins and version reporting.
- Tests cover default-off behavior, cache invalidation, bounded multi-LoRA preflight caching, DoRA/reshape/offset rejection, non-adapter fallback behavior, standard adapter capture without materialization, stacked-LoRA additive equivalence, repeated hook injection/ejection, cloned-patcher interception end to end, and fail-closed unsupported loader output.
- Adds `.comfyignore` so development-only `.github/` and `tests/` content is not shipped in Comfy Registry archives.
- Adds an automated GitHub release workflow gated on a successful `tests` run from `main`.
- GitHub releases contain a versioned repository ZIP and `SHA256SUMS` manifest and use this file as the release notes.
- Existing release tags are accepted only for an exact same-commit rerun; a conflicting existing tag fails the release rather than silently reporting success.
- Aligns Comfy Registry publishing with the pinned action revisions used by Spectrum MiniMax H3.

## Compatibility

Runtime bypass is opt-in. With it disabled, the loader continues through the existing LoRA/DoRA path. State Manager behavior, existing loader globals, Flux/Flux2 compatibility handling, ZiT/Lumina2 normalization, DoRA fixes, and auto-strength remain on their existing path.
