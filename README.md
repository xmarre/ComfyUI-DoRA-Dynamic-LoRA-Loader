# comfyui_dora_dynamic_lora

Custom ComfyUI node that loads and stacks **regular LoRAs and DoRA LoRAs**, with additional Flux/Flux2 + OneTrainer
compatibility and DoRA stability fixes.

This repo contains two distinct parts:

1) **A Power LoRA Loader-style node** (multiple LoRAs in one node, per-LoRA strengths).
2) **Targeted ComfyUI patches and transforms** needed for Flux/Flux2 DoRA LoRAs to apply correctly and avoid known
   failure modes.

## Confirmed adaLN swap-scale alignment fix (Flux2 DoRA)

A known Flux2 DoRA failure mode is fixed by aligning DoRA’s magnitude vector (`dora_scale`) with the same permutation
ComfyUI applies to the LoRA delta for **adaLN_modulation** weights.

Implementation:

- This repo patches `comfy.weight_adapter.base.weight_decompose`.
- When ComfyUI applies a `swap_scale_shift` function to the delta (used for adaLN_modulation weights), the patch
  applies that *same* transform to `dora_scale` before computing the DoRA scaling.
- Node toggle: **“DoRA adaLN swap_scale_shift fix”** (`dora_adaln_swap_fix`, default **ON**).

## Other fixes and compatibility layers

### 1) Correct DoRA normalization (norm(V) in fp32)

This repo patches `comfy.weight_adapter.base.weight_decompose` to:

- Perform DoRA math in **fp32**.
- Normalize using the norm of the **updated weight** `V = W + Δ` (where `Δ` is the LoRA delta after applying
  `alpha`), rather than normalizing against the base weight.

### 2) Slice-aware `dora_scale` for sliced/offset patches (Flux2 qkv)

Flux/Flux2 key maps can include **sliced targets** (e.g. qkv packed weights). In those cases, ComfyUI applies LoRA to
only a slice of a weight tensor. This repo’s `weight_decompose` patch includes an optional **slice fix** that slices
`dora_scale` to the matching offset/length when possible.

Node toggle: **“DoRA slice-fix for offset patches (Flux2)”** (`dora_slice_fix`, default **ON**).

### 3) Force fp32 intermediates when building `lora_diff`

This repo patches `comfy.weight_adapter.lora.*.calculate_weight()` to force `intermediate_dtype=torch.float32`.
This is specifically to avoid mixed-precision paths flushing very small intermediate products to zero while building
`lora_diff`.

### 4) OneTrainer “Apply on output axis (DoRA only)” direction-matrix fix

Some OneTrainer exports store the direction matrices (`lora_up`/`lora_down`, or `lora_A`/`lora_B`) in a layout that
does not match the destination weight (swapped and/or transposed). This repo compares the shapes against the mapped
destination weight and applies one of the following fixes when it matches a known pattern:

- swap `up` and `down`
- transpose one or both matrices

This fix runs automatically when a base has `*.dora_scale` and corresponding direction matrices.

### 5) Flux2 / OneTrainer key compatibility transforms

Before mapping/loading, the loader may transform the LoRA state dict:

- Rename `transformer.time_guidance_embed.*` → `transformer.time_text_embed.*` (only if the target prefix is not
  already present).
- Broadcast OneTrainer’s **global modulation** LoRAs onto the **per-block** keys ComfyUI maps, using the current
  model’s `key_map` to discover the actual targets.

Broadcast controls:

- **Broadcast OneTrainer modulation LoRAs** (`broadcast_modulations`, default **ON**)
- **Include DoRA dora_scale in broadcast** (`broadcast_include_dora_scale`, default **OFF**)
- **Auto-scale broadcast** (`broadcast_auto_scale`, default **ON**) — divides `broadcast_scale` by the number of
  broadcast targets.
- **Broadcast scale** (`broadcast_scale`, default `1.0`)

### 6) Dynamic key mapping (suffix matching + `.linear → .lin`)

After building ComfyUI’s standard key map via:

- `comfy.lora.model_lora_keys_unet(...)`
- `comfy.lora.model_lora_keys_clip(...)`

…this node extends it for any base modules present in the LoRA file but missing from the map. It matches bases to
`model.state_dict()` / `clip.state_dict()` keys by suffix, including these built-in variants:

- stripping common prefixes (`diffusion_model.`, `model.`, `transformer.`, etc.)
- rewriting Flux naming differences: `.linear` ↔ `.lin`

If multiple candidates match, it picks the shortest match and prefers candidates containing `diffusion_model.`.

### 7) `convert_lora` bypass when it zeroes direction matrices

The loader normally runs `comfy.lora_convert.convert_lora(...)`. It also computes stats on direction matrices before
and after conversion. If conversion turns a non-zero set of `.lora_up.weight` tensors into all zeros, it reloads the
file and bypasses conversion for that LoRA.

### 8) Diagnostics: NaN/Inf checks + quantization warnings

The loader emits warnings when:

- The LoRA file contains NaN/Inf tensors.
- The loaded patches contain NaN/Inf tensors.
- A quantized/mixed-precision base model is detected in the UNet `state_dict()` and the LoRA contains DoRA tensors
  (`*.dora_scale`).

## Install

Copy the folder into:

`ComfyUI/custom_nodes/comfyui_dora_dynamic_lora/`

Restart ComfyUI.

## Node

**DoRA Power LoRA Loader**  
Category: `loaders`

### Per-LoRA rows

Each row has:

- Enabled toggle
- LoRA name (dropdown; loaded from `/dora_dynamic_lora/loras`)
- Weight (applies to Model/CLIP)

### Global options

- Stack Enabled
- Verbose
- Log Unloaded Keys
- Broadcast OneTrainer modulation LoRAs
- Include DoRA dora_scale in broadcast
- Auto-scale broadcast
- Broadcast scale
- DoRA slice-fix for offset patches (Flux2)
- DoRA adaLN swap_scale_shift fix
- DoRA decompose debug logs
- DoRA debug lines
- DoRA debug stack depth

## How it applies LoRAs (high level)

For each enabled row:

1) Load the LoRA file (safe_load when supported).
2) Optionally bypass `convert_lora` if it zeroes direction matrices.
3) Apply Flux2/OneTrainer compatibility transforms (rename + optional broadcast).
4) Build ComfyUI key map for UNet and CLIP, then extend it with dynamic suffix matches.
5) Apply OneTrainer output-axis direction-matrix fix (when applicable).
6) Call `comfy.lora.load_lora(...)` and apply patches via `model.add_patches(...)` / `clip.add_patches(...)`.

## Important implementation detail

This custom node **monkey-patches** ComfyUI internals at import time:

- `comfy.weight_adapter.base.weight_decompose`
- `comfy.weight_adapter.lora.*.calculate_weight` (classes that expose it)

These patches affect DoRA/LoRA application in the running ComfyUI process, not only this node.

## Troubleshooting

- Flux2 DoRA instability:
  - ensure **DoRA adaLN swap_scale_shift fix** is enabled (`dora_adaln_swap_fix`)
  - check logs for NaN/Inf warnings (LoRA tensors or loaded patches)
- LoRA loads but has almost no effect:
  - in verbose mode, the loader warns if **all** `.lora_up` direction matrices are zero in the file (training/export
    issue, not loader)
- Suspected mapping problems:
  - enable **Verbose** and **Log Unloaded Keys** and inspect:
    - `map: <base> -> <weight>` lines
    - `unresolved LoRA base:` lines
