# ComfyUI-DoRA-Dynamic-LoRA-Loader

Custom ComfyUI node that loads and stacks **regular LoRAs and DoRA LoRAs**, with additional **Flux / Flux2 + Diffusers/PEFT + OneTrainer compatibility**, **Z-Image Turbo / Lumina2 attention-format compatibility**, optional **auto-strength redistribution**, and multiple **DoRA correctness / stability fixes**.

This repo contains two distinct parts:

1. **A Power LoRA Loader-style node**
   - multiple LoRAs in one node
   - per-LoRA strengths
   - stacked application in one place

2. **Targeted ComfyUI patches and transforms**
   - fixes and compatibility layers needed for Flux / Flux2 DoRA LoRAs to load and apply correctly
   - protection against several known failure modes

Auto-strength support in this loader was inspired by [Comfyui-flux2klein-Lora-loader](https://github.com/capitan01R/Comfyui-flux2klein-Lora-loader) and [Comfyui-ZiT-Lora-loader](https://github.com/capitan01R/Comfyui-ZiT-Lora-loader).

This implementation was reworked for the unified DoRA + standard LoRA path in this loader, including Flux.2 Klein and ZiT/Lumina2 compatibility handling.

---

## Auto-strength

This node includes optional **auto-strength** redistribution for loaded LoRAs / DoRAs.

When enabled, the loader:

- measures a comparable per-base update magnitude
- computes a per-base target relative to the mean of similar mapped destinations
- converts those absolute targets into **redistribution ratios**
- bakes only that **ratio** into the LoRA tensors before loading

### Important implementation detail

The loader intentionally preserves the caller's normal outer **Model / CLIP patch strength** path.

That means auto-strength adjusts only the **relative balance between bases**, while the row's normal weight still controls the final overall strength.

This is especially important for **DoRA**: the outer strength is part of ComfyUI's normal post-normalization application path, so baking the full absolute target directly into the tensors would not be equivalent.

If:

- `auto_strength_ratio_floor = 1.0`
- `auto_strength_ratio_ceiling = 1.0`

then enabling auto-strength is a true no-op.

### Current auto-strength behavior

- compares mapped bases using a normalized magnitude score
- keeps Flux / Flux2 compat-broadcasted logical sources from being over-counted during measurement
- preserves the normal outer patch strength during final application
- is intended to redistribute relative base strength, not replace the row's overall weight

---

## AdaLN swap-scale alignment fix (Flux2 DoRA)

A known Flux2 DoRA failure mode is fixed by aligning DoRA’s magnitude vector (`dora_scale`) with the same permutation ComfyUI applies to the LoRA delta for **adaLN_modulation** weights.

### Implementation

This repo patches `comfy.weight_adapter.base.weight_decompose`.

When ComfyUI applies a `swap_scale_shift` transform to the LoRA delta for adaLN-related weights, this patch applies that **same transform** to `dora_scale` before computing the DoRA scaling term.

**Node toggle:** `DoRA adaLN swap_scale_shift fix` (`dora_adaln_swap_fix`, default **ON**)

---

## Other fixes and compatibility layers

### 1) Correct DoRA normalization (`norm(V)` in fp32)

This repo patches `comfy.weight_adapter.base.weight_decompose` to:

- perform DoRA math in **fp32**
- normalize using the norm of the **updated weight** `V = W + delta` (where `delta` is the LoRA delta after applying `alpha`)
- reshape `dora_scale` onto the active normalization axis before division so non-square targets do not broadcast incorrectly

This is both more stable and more faithful to DoRA’s intended magnitude handling.

---

### 2) Slice-aware `dora_scale` for sliced / offset patches (Flux2 qkv)

Flux / Flux2 key maps can include **sliced targets** such as packed qkv weights. In those cases, ComfyUI applies the LoRA patch to only a slice of a larger tensor.

This repo’s `weight_decompose` patch includes an optional **slice fix** that slices `dora_scale` to the matching offset / length when possible, so the DoRA magnitude vector stays aligned with the actual patched slice.

**Node toggle:** `DoRA slice-fix for offset patches (Flux2)` (`dora_slice_fix`, default **ON**)

---

### 3) Force fp32 intermediates when building `lora_diff`

This repo patches `comfy.weight_adapter.lora.*.calculate_weight()` to force:

- `intermediate_dtype=torch.float32`

This is specifically to avoid mixed-precision paths flushing very small intermediate products to zero while building `lora_diff`.

---

### 4) Direction-matrix orientation fix for Diffusers / PEFT FLUX2 DoRA and compatible exports

Some Flux / Flux2 DoRA exports use Diffusers / PEFT-style direction matrices where:

- `.lora_B.*` is the **up** matrix
- `.lora_A.*` is the **down** matrix

If those are interpreted with the wrong orientation in later compatibility paths, the loader can end up swapping already-correct matrices into the wrong layout, which then produces shape errors on mapped Flux2 targets such as:

- `single_blocks.*.linear1.weight`
- `single_blocks.*.linear2.weight`
- fused qkv / proj targets

This repo centralizes the directional suffix-pair semantics and uses the corrected orientation consistently in the relevant compatibility paths, so Diffusers / PEFT FLUX2 DoRA exports are not “fixed” into an invalid matrix layout.

This directly addresses failure patterns such as:

- `mat1 and mat2 shapes cannot be multiplied`
- `shape '[6144, 6144]' is invalid for input of size 1024`

---

### 5) Output-axis direction-matrix fix for known DoRA export layouts

Some DoRA exports store the direction matrices (`lora_up` / `lora_down`, or `lora_A` / `lora_B`) in a layout that does not match the destination weight. Depending on the export, they may be swapped and/or transposed relative to what ComfyUI expects.

This repo compares those matrix shapes against the mapped destination weight and applies one of the following fixes when a known pattern is detected:

- swap `up` and `down`
- transpose one matrix
- transpose both matrices

This fix runs automatically when a base has `*.dora_scale` and matching direction matrices.

---

### 6) Diffusers / PEFT DoRA magnitude-vector compatibility (`lora_magnitude_vector` → `dora_scale`)

Some Diffusers / PEFT DoRA exports store the DoRA magnitude tensor under:

- `*.lora_magnitude_vector`
- `*.lora_magnitude_vector.weight`
- `*.lora_magnitude_vector.default`
- `*.lora_magnitude_vector.default.weight`
- `*.lora_magnitude_vector.default_0`
- `*.lora_magnitude_vector.default_0.weight`

ComfyUI-style loading expects the equivalent tensor under:

- `*.dora_scale`

Before mapping / loading, this repo normalizes those Diffusers / PEFT-style DoRA magnitude keys into Comfy-style `dora_scale` keys.

Without this step, the LoRA direction matrices may load while the DoRA magnitude vectors remain behind as unloaded keys, which means the file is **not** being applied as full DoRA.

This directly fixes the common log pattern:

- `lora key not loaded: ...lora_magnitude_vector`

---

### 7) Flux2 / OneTrainer key compatibility transforms

Before mapping / loading, the loader may transform the LoRA state dict:

- rename `transformer.time_guidance_embed.*` → `transformer.time_text_embed.*`  
  only if the target prefix is not already present
- broadcast OneTrainer’s **global modulation** LoRAs onto the **per-block** keys ComfyUI actually maps, using the live model’s `key_map` to discover real targets

#### Broadcast controls

- `Broadcast OneTrainer modulation LoRAs` (`broadcast_modulations`, default **ON**)
- `Include DoRA dora_scale in broadcast` (`broadcast_include_dora_scale`, default **OFF**)
- `Auto-scale broadcast` (`broadcast_auto_scale`, default **ON**)  
  divides `broadcast_scale` by the number of broadcast targets
- `Broadcast scale` (`broadcast_scale`, default `1.0`)

#### Auto-strength interaction

For compat-broadcasted Flux / Flux2 sources, auto-strength measures the **logical source group** rather than treating every synthetic broadcast clone as a separate weak layer.

That prevents a single broadcasted source from skewing target computation just because the loader expanded it into multiple real mapped bases.

---

### 8) Dynamic key mapping (suffix matching + `.linear` ↔ `.lin`)

After building ComfyUI’s standard key map via:

- `comfy.lora.model_lora_keys_unet(...)`
- `comfy.lora.model_lora_keys_clip(...)`

…the node extends that map for base modules present in the LoRA file but missing from the standard map.

It matches bases against `model.state_dict()` / `clip.state_dict()` keys by suffix, including these built-in variants:

- stripping common prefixes such as:
  - `diffusion_model.`
  - `model.`
  - `transformer.`
- rewriting Flux naming differences:
  - `.linear` ↔ `.lin`

If multiple candidates match, it picks the shortest match and prefers candidates containing `diffusion_model.`.

---

### 9) Z-Image Turbo / Lumina2 architecture-aware attention compatibility

Before mapping / loading, the loader can normalize ZiT / Lumina2 LoRAs into the model’s native fused-attention form.

#### What it does

- detects Lumina2 / Z-Image-style models by class name and/or live `state_dict()` structure
- adds exact ZiT / Lumina2 key-map aliases, including:
  - `transformer.*`
  - `base_model.model.*`
  - bare bases
  - `lora_unet_*`
  - `lycoris_*`
- normalizes common export spelling variants:
  - `attention.to.q` → `attention.to_q`
  - `attention.to.k` → `attention.to_k`
  - `attention.to.v` → `attention.to_v`
  - `attention.to.out.0` → `attention.to_out.0`
- fuses split attention Q / K / V LoRAs:
  - `attention.to_q.*`
  - `attention.to_k.*`
  - `attention.to_v.*`
  into native `attention.qkv.*`
- remaps `attention.to_out.0.*` → `attention.out.*`

#### Important implementation detail

The Q / K / V fusion is done as an **exact larger-rank LoRA**, not by naïvely concatenating both matrices.

Per-component `alpha` values are absorbed into the fused `up` matrix before building the block-diagonal fused adapter, and the fused adapter is then emitted with `alpha = 1`.

Compatible per-output auxiliary tensors such as:

- `dora_scale`
- `diff`
- `w_norm`

are concatenated along the output dimension when all three components are present and shape-compatible.

**Node toggle:** `ZiT/Lumina2 auto-fix (QKV fuse + out remap)` (`zimage_lumina2_compat`, default **ON**)

---

### 10) `convert_lora` bypass when it zeroes direction matrices

The loader normally runs:

- `comfy.lora_convert.convert_lora(...)`

It also computes stats on direction matrices before and after conversion. If conversion turns a non-zero set of direction matrices into all zeros, the loader reloads the file and bypasses conversion for that LoRA.

This is meant to protect against destructive conversion paths on certain exports.

---

### 11) Diagnostics: NaN / Inf checks + quantization warnings

The loader emits warnings when:

- the LoRA file contains NaN / Inf tensors
- the loaded patches contain NaN / Inf tensors
- a quantized or mixed-precision base model is detected in the UNet `state_dict()` and the LoRA contains DoRA tensors (`*.dora_scale`)

---

## Install

### Option 1: Manual install

Copy this repository folder into:

`ComfyUI/custom_nodes/ComfyUI-DoRA-Dynamic-LoRA-Loader/`

Then restart ComfyUI.

### Option 2: ComfyUI Manager

Install it through **ComfyUI Manager** by searching for:

`ComfyUI-DoRA-Dynamic-LoRA-Loader`

Then restart ComfyUI after installation or update.

---

## Node

**DoRA Power LoRA Loader**  
Category: `loaders`

### Per-LoRA rows

Each row has:

- enabled toggle
- LoRA name dropdown  
  loaded from `/dora_dynamic_lora/loras`
- weight  
  applied to both Model and CLIP

### Global options

- Stack Enabled
- Verbose
- Log Unloaded Keys
- Auto-strength enabled
- Auto-strength ratio floor
- Auto-strength ratio ceiling
- Broadcast OneTrainer modulation LoRAs
- Include DoRA dora_scale in broadcast
- Auto-scale broadcast
- Broadcast scale
- DoRA slice-fix for offset patches (Flux2)
- DoRA adaLN swap_scale_shift fix
- Auto-strength analysis device (`auto` / `cpu` / `gpu`)
- ZiT/Lumina2 auto-fix (QKV fuse + out remap)
- DoRA decompose debug logs
- DoRA debug lines
- DoRA debug stack depth

---

## How it applies LoRAs

For each enabled row:

1. load the LoRA file (`safe_load` when supported)
2. optionally bypass `convert_lora` if it zeroes direction matrices
3. build ComfyUI key maps for UNet and CLIP
4. optionally apply ZiT / Lumina2 attention normalization  
   qkv fuse + `to_out.0` remap + exact key aliases
5. apply Flux2 / OneTrainer compatibility transforms  
   rename + optional broadcast
6. normalize Diffusers / PEFT DoRA magnitude keys  
   `lora_magnitude_vector` → `dora_scale`
7. extend the key map with dynamic suffix matches
8. apply direction-matrix compatibility fixes when applicable
9. if enabled, compute per-base auto-strength redistribution ratios  
   and bake only those ratios into the LoRA tensors
10. call `comfy.lora.load_lora(...)`
11. apply patches via `model.add_patches(...)` / `clip.add_patches(...)`  
    using the normal outer Model / CLIP strengths

---

## Important implementation detail

This custom node **monkey-patches** ComfyUI internals at import time:

- `comfy.weight_adapter.base.weight_decompose`
- `comfy.weight_adapter.lora.*.calculate_weight`  
  for classes that expose it

These patches affect DoRA / LoRA application in the running ComfyUI process, not only this node.

---

## Troubleshooting

### Flux2 DoRA instability

- ensure `DoRA adaLN swap_scale_shift fix` (`dora_adaln_swap_fix`) is enabled
- check logs for NaN / Inf warnings in:
  - LoRA tensors
  - loaded patches

### `lora_magnitude_vector` keys show as unloaded

- this indicates a Diffusers / PEFT DoRA export format
- current versions of this repo normalize those keys into `dora_scale` before loading
- if you still see them after updating, enable:
  - `Verbose`
  - `Log Unloaded Keys`

### Shape errors on Flux2 targets

If you see errors such as:

- `mat1 and mat2 shapes cannot be multiplied`
- `shape '[6144, 6144]' is invalid for input of size 1024`

that usually points to a direction-matrix layout / orientation mismatch in the export or a compatibility path interpreting Diffusers-style `.lora_A` / `.lora_B` pairs incorrectly.

Current versions of this repo include compatibility handling for that path. If you still see these errors after updating, enable:

- `Verbose`
- `Log Unloaded Keys`

and inspect:

- `OneTrainer output-axis DoRA mat-fix: checked=... fixed=...`
- `patches=... applied(model)=...`
- the first few `ERROR lora ...` lines

### LoRA loads but has almost no effect

- in verbose mode, the loader warns if **all** direction matrices are zero in the file
- that usually points to a training / export issue rather than a loader issue

### Auto-strength changes the output in unexpected ways

Auto-strength is meant to redistribute relative base strength, not replace the row’s normal overall weight.

In current versions of this repo:

- `auto_strength_ratio_floor = 1.0`
- `auto_strength_ratio_ceiling = 1.0`

should behave like auto-strength disabled.

If it does not, that points to a loader bug rather than “strong settings”.

### Suspected mapping problems

Enable **Verbose** and **Log Unloaded Keys** and inspect:

- `map: <base> -> <weight>`
- `unresolved LoRA base:`
- unloaded key logs

---

## Notes

This repo is meant for cases where plain ComfyUI LoRA loading is not enough, especially for:

- Flux / Flux2 DoRA LoRAs
- OneTrainer DoRA exports
- Diffusers / PEFT DoRA exports
- Z-Image Turbo / Lumina2 attention-format LoRAs
