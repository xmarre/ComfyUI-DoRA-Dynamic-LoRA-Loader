# comfyui_dora_dynamic_lora

Custom ComfyUI node that loads **DoRA LoRAs** (and regular LoRAs) when the LoRA file’s module keys
don’t match ComfyUI’s built-in key map (common with **OneTrainer / Flux** training exports).

ComfyUI already supports DoRA math (it reads `*.dora_scale` and applies weight decomposition).
This node does **not** re-implement DoRA — it fixes **key mapping** so those tensors actually load.

## Main symptom this fixes (Flux.2 Klein 9B etc.)

When the LoRA weights fail to map/load, a common visible result is a **pink/magenta output image** (or otherwise
obviously broken output), even though the workflow “runs”.

Logs usually include lots of lines like:

`lora key not loaded: transformer....linear.dora_scale`

Meaning: the DoRA/LoRA tensors exist in the file, but ComfyUI can’t map them to real model weights.

## What it does

1. Loads the LoRA file and normalizes formats via `comfy.lora_convert.convert_lora`.
2. Builds ComfyUI’s standard `key_map` (so all built-in mapping logic stays intact).
3. Extracts “base module names” from the LoRA file (e.g. `...linear` before `.lora_up.weight`, `.dora_scale`, etc.).
4. For bases missing in `key_map`, it dynamically matches them to `model/clip.state_dict()` keys by suffix.
5. Calls ComfyUI’s `comfy.lora.load_lora(...)` to produce patches and applies them to cloned model/clip.

## Install

Place this folder here:

`ComfyUI/custom_nodes/comfyui_dora_dynamic_lora/`

Then restart ComfyUI.

## Node

**DoRA Dynamic LoRA Loader (fix OneTrainer/Flux keys)**  
Category: `loaders`

Inputs:
- `model`, `clip`
- `lora_name`
- `strength_model`, `strength_clip`
- `verbose`: logs created mappings + unresolved bases
- `log_unloaded_keys`: forwards missing-key logging to the core loader (useful for debugging)

## Notes / limitations

- The dynamic matcher is best-effort. If multiple model weights share the same suffix, it picks the shortest match
  and prefers keys containing `diffusion_model.`.
- If you suspect wrong mapping, enable `verbose` and inspect the `map: <base> -> <weight>` logs.

## Troubleshooting

- Still seeing “lora key not loaded”:
  - enable `verbose = true` and `log_unloaded_keys = true`
  - check which bases are unresolved in logs
  - if the unresolved bases have a consistent prefix mismatch, extend `_candidate_base_variants()` in `nodes.py`.
