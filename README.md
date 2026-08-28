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

## Runtime bypass adapters (low VRAM)

The loader includes an optional **Runtime bypass LoRA (low VRAM)** mode for supported standard LoRA and plain LoKr adapters.

The word **bypass** refers to bypassing **weight materialization**, not bypassing or weakening the LoRA effect. For a normal additive LoRA, the regular merged path evaluates a layer with `W + ΔW`, while the runtime path keeps `W` untouched and evaluates the equivalent low-rank contribution during the forward pass:

```text
regular: (W + ΔW)x
runtime: Wx + ΔWx
```

For supported standard LoRAs these are mathematically equivalent. Small floating-point differences can still occur because the operations are executed in a different order/dtype path, and runtime bypass adds the low-rank operations on every forward instead of paying the full merge/materialization cost up front.

This follows ComfyUI's own bypass adapter design: ComfyUI describes it as loading LoRA **without modifying base model weights** and injecting the LoRA computation into the forward pass (`base_forward(x) + lora_path(x)`).

### Why it saves so much VRAM

Normal materialized patching may need to retain the original model weights while also holding the fully patched replacement weights. On very large HIGH_VRAM models this can mean tens of GiB of additional live VRAM even when the LoRA file itself is only a few hundred MiB.

Runtime bypass keeps the base model weights unchanged and stores/evaluates the low-rank adapter tensors directly, avoiding that full patched-model copy for supported adapters.

### Plain LoKr support

Plain LoKr adapters use ComfyUI's native LoKr forward-bypass implementation when it is available. Direct-factor, decomposed, and mixed LoRA + LoKr stacks are supported. The runtime-only adapter metadata is normalized where current ComfyUI's LoKr materialization and bypass paths use different alpha/rank conventions; the source adapter remains unchanged.

Older ComfyUI revisions without a real `LoKrAdapter.h()` implementation reject LoKr runtime bypass with an explicit update/disable message. DoRA-LoKr magnitude scaling remains unsupported because ComfyUI's LoKr bypass path does not implement DoRA normalization.

### Important limitation: DoRA is not runtime-bypassed

Current ComfyUI bypass LoRA math implements the additive LoRA path, but not DoRA magnitude normalization/rescaling. Treating a DoRA as ordinary bypass LoRA would therefore change its mathematics.

This loader deliberately **fails closed** when runtime bypass encounters DoRA or another unsupported adapter form. It does not silently approximate it. Disable Runtime bypass LoRA for those files and use the normal materialized path instead.

Runtime bypass currently refuses:

- DoRA / magnitude-vector LoRAs (`dora_scale`, Diffusers/PEFT `lora_magnitude_vector`, `w_norm`, `b_norm`)
- DoRA-LoKr and adapter types other than standard LoRA/plain LoKr
- LoRA reshape metadata
- sliced / offset / transformed adapter targets
- non-default `strength_model` patch semantics

The option is **OFF by default**, so existing workflows retain the previous materialized LoRA/DoRA behavior unless you explicitly enable it.

---

## Auto-strength

This node includes optional **auto-strength** redistribution for loaded LoRAs / DoRAs.

When enabled, the loader:

- measures a comparable per-base update magnitude
- groups repeated transformer projections by mapped destination role, exact tensor shape, and slice identity
- detects isolated per-base magnitude anomalies against comparable logical sources
- converts those absolute targets into **redistribution ratios**
- bakes only that **ratio** into the LoRA tensors before loading

### Important implementation detail

The loader intentionally preserves the caller’s normal outer **Model / CLIP patch strength** path.

That means auto-strength adjusts only the **relative balance between bases**, while the row’s normal weight still controls the final overall strength.

This is especially important for **DoRA**: the outer strength is part of ComfyUI’s normal post-normalization application path, so baking the full absolute target directly into the tensors would not be equivalent.

If:

- `auto_strength_ratio_floor = 1.0`
- `auto_strength_ratio_ceiling = 1.0`

then enabling auto-strength is a true no-op.

### Current auto-strength behavior

- scores ordinary LoRA with absolute `RMS(ΔW)` after LoRA alpha/rank scaling
- scores DoRA with the RMS of its actual post-normalization weight update
- keeps model and CLIP, tensor families, repeated-block projection roles, tensor shapes, and sliced destinations in separate cohorts
- infers projection roles structurally from mapped destination paths such as repeated `blocks` / `layers` containers; it does not contain a MiniMax-H3 projection-name table
- leaves unclassifiable linear destinations at their normal global strength instead of pooling unrelated linear layers
- requires at least five measured logical sources before a linear role cohort can correct anomalies
- models repeated-block linear roles in log space with a log-median center and MAD-based robust dispersion
- leaves members inside that expected distribution at ratio `1.0`; the cohort center is not an automatic target
- suppresses correction when more than one candidate is detected, preserving coherent depth-dependent or multimodal regimes
- preserves the existing arithmetic-mean reference for non-linear tensor families such as convolution cohorts
- keeps Flux / Flux2 compat-broadcasted logical sources from being over-counted during measurement
- preserves the normal outer patch strength during final application
- is intended to redistribute relative base strength, not replace the row's overall weight
- `auto` resolves to CPU-safe analysis
- `gpu` is the explicit accelerator path
- default node UI state is `gpu`

Ordinary LoRA scoring deliberately remains independent of the destination weight's
numeric values. A base-relative score such as `RMS(ΔW) / RMS(W0)` would make the same
LoRA redistribute differently across base checkpoints and can make quantized storage
representations part of the result. Role-aware cohorts address the cross-projection
scale problem, while the outlier gate preserves ordinary within-role variation and
checkpoint-independent standard-LoRA analysis. DoRA is
the exception because its update is defined by normalization against the live effective
destination weight; its existing post-normalization measurement therefore remains
base-dependent by design.

### Performance note

Auto-strength still does **extra loader-time compute**, especially for:

- **initial load / first generation**
- workflows with **multiple loader nodes**
- **high-rank DoRAs**
- large backbones such as **Flux / Flux2**

However, current versions are **much faster than the earlier CPU-bound analysis path** when you choose the explicit GPU analysis mode.

The loader now supports an **auto-strength analysis device** option:

- `auto` — uses CPU-safe analysis
- `cpu` — forces analysis to CPU
- `gpu` — prefers the model/CLIP accelerator load device and falls back to CPU if needed

In practice this means the expensive analysis pass can still run on the GPU/accelerator for faster load-time measurement, but `auto` stays on the CPU-safe path.

So the practical tradeoff is now:

- still **higher loader-time compute** than auto-strength disabled
- `gpu` can be **much faster than the old CPU-only analysis path**
- while keeping the stronger quality / accuracy gains from the more faithful redistribution and DoRA application path

If you want the lowest overhead, disable auto-strength.  
If you want strong layer-aware balancing with the faster accelerator path, keep it enabled and use `gpu`.

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
- Runtime bypass LoRA (low VRAM)
- Verbose
- Log Unloaded Keys
- Auto-strength enabled
- Auto-strength analysis device
- Auto-strength ratio floor
- Auto-strength ratio ceiling
- Broadcast OneTrainer modulation LoRAs
- Include DoRA dora_scale in broadcast
- Auto-scale broadcast
- Broadcast scale
- DoRA slice-fix for offset patches (Flux2)
- DoRA adaLN swap_scale_shift fix
- Auto-strength analysis device (`auto` / `cpu` / `gpu`, default **GPU**; `auto` = CPU-safe analysis)
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
   on the selected analysis device and bake only those ratios into the LoRA tensors
10. call `comfy.lora.load_lora(...)`
11. apply supported standard LoRA adapters either:
    - through the normal `model.add_patches(...)` / `clip.add_patches(...)` materialized path when Runtime bypass LoRA is OFF, or
    - through runtime forward-pass adapter injection when Runtime bypass LoRA is ON

---

## Important implementation detail

This custom node **monkey-patches** ComfyUI internals at import time:

- `comfy.weight_adapter.base.weight_decompose`
- `comfy.weight_adapter.lora.*.calculate_weight`  
  for classes that expose it

These patches affect DoRA / LoRA application in the running ComfyUI process, not only this node.

---

## Troubleshooting

### Runtime bypass rejects an adapter

Runtime bypass only takes adapters for which the supported forward-pass path preserves the normal patch semantics. In particular, current ComfyUI bypass LoRA does **not** implement DoRA magnitude normalization.

If the loader reports DoRA/magnitude-vector, reshape, offset/transform, an older ComfyUI build without LoKr bypass math, or another unsupported adapter form, disable **Runtime bypass LoRA (low VRAM)** for that file.

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

### Auto-strength is slower than disabled

That is still expected to some degree.

Auto-strength adds extra analysis work during loader execution, and the cost can still scale with:

- number of loader nodes
- number of enabled LoRAs
- adapter rank
- model size
- DoRA usage

However, current versions can run that analysis on GPU / accelerator for larger measurements, which makes it **much faster than the earlier CPU-bound path** on supported setups.

If you want the best balance of speed and quality, use:

- `Auto-strength enabled = ON`
- `Auto-strength analysis device = auto`

If you want to force the old safest path, use:

- `Auto-strength analysis device = cpu`

If you want to prefer GPU / accelerator analysis explicitly, use:

- `Auto-strength analysis device = gpu`

When `gpu` is selected but no usable accelerator load device is available, the loader falls back to CPU and logs a warning.

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

---

## State Manager

The **State Manager** uses a persistent user library for reusable characters and prompt presets. Workflows made with the older **DoRA State Manager** node name remain supported through a legacy alias.

The library is stored under the active ComfyUI user directory:

```text
<ComfyUI user directory>/<active user id>/dora_state_manager/state-library.json
```

Ordinary workflow JSON contains only the selected character/prompt UUID binding and workflow-specific queue options. Character names, prompt text, LoRA stacks, settings, thumbnails, reference-image metadata, and filename prefixes are not embedded in workflow JSON.

The manager separates **saved state** from **runtime execution**:

- execution outputs are source-only and acyclic
- save/load/apply actions are explicit frontend graph edits
- `state_control` links are editor/control-only and do not replace prompt text, seed values, or LoRA rows during execution
- no prompt, LoRA stack, settings, or seed value is captured through runtime feedback inputs or wildcard feedback loops

Use it to save:

- character / LoRA combinations across multiple separate DoRA loader nodes
- per-loader DoRA settings such as auto-strength and compatibility toggles
- multiple positive / negative prompt **templates** per character
- arbitrary downstream node widget snapshots
- seed state, including rgthree-style seed nodes
- one character image / thumbnail reference; the UI preview is CSS-scaled, while the `character_image` output loads the original uploaded image file

### Library-first UI

The State Manager opens with the saved-state library as its largest pane:

- **All presets** shows prompt presets from every character in one dense, scrollable grid.
- **Characters** switches to the character/LoRA-state library.
- Search matches character names, preset names, `fileimage_prefix` values, and saved LoRA filenames.
- The result count updates while typing, and the search is applied in place without dirtying the workflow.
- Library scroll position and search text survive selection changes.
- Preset, character, LoRA, settings/seed, and queue editing live in tabs in the detail pane.
- Resizing the node gives the library the additional space; the collection has no fixed-height cap.

The storage redesign keeps node class names, widget order, selection widgets, outputs, connected save/load behavior, and the legacy **DoRA State Manager** alias. Existing schema-v3 embedded libraries are imported into persistent storage on first load. Migration is fingerprinted and idempotent, preserves valid collision-free UUIDs, remaps unsafe/colliding IDs, and never overwrites an unrelated local preset.

If a workflow references UUIDs that are unavailable on the current machine, the node reports that the selected preset is unavailable. It does not fall back to another local character. A deliberately empty/default node continues to use an ephemeral built-in **Default Character** until it is edited into a persistent character.

### Library portability and recovery

- **Export character** downloads only the selected character and its prompt presets.
- **Export library** downloads a versioned backup of the entire persistent library.
- **Import** accepts character exports, library exports, and legacy State Manager exports.
- Library writes use a lock, optimistic revisions, a flushed temporary file, atomic `os.replace()`, and directory fsync where supported.
- Malformed storage is quarantined as `state-library.json.corrupt-<timestamp>` instead of being overwritten.
- A stale write from another browser tab is rejected. Use **Reload library** to load the current revision before editing again.

The old workflow/node-keyed browser backup is no longer used or read. Browser-local UI state cannot repopulate a sanitized workflow with private presets.

Presets that exist only in an old browser backup require an explicit transition: run v1.0.40, open the workflow so its backup is restored, export or save the restored state, then update and import/migrate it. The redesign does not delete old browser entries, so this recovery remains possible by temporarily returning to v1.0.40.

### Basic wiring

Recommended connected save/load wiring:

1. Add **State Manager**.
2. Add **State Text Box** nodes for editable positive and negative prompt templates.
3. Add **State Seed** for an editable seed value.
4. Add one or more **DoRA Power LoRA Loader** nodes.
5. Give every DoRA loader a unique **State slot** value, for example `face`, `outfit`, `style`, `refiner`.
6. Connect `state_control` from the manager to each helper node's optional `state_control` input, including every DoRA loader you want managed.
7. Connect `text` from each State Text Box into the prompt/wildcard path.
8. Connect `seed` from State Seed only to the sampler seed input.
9. Configure LoRA rows on each DoRA Power LoRA Loader as usual.

The manager should not receive processed wildcard, prompt-generation, or other runtime text outputs. Store the wildcard/template prompt in State Text Box, then let your wildcard node expand it downstream.

Correct shape:

```text
State Manager.state_control
  -> State Text Box.state_control

State Text Box.text
  -> Wildcards Processor
      -> CLIP Text Encode
```

Avoid feedback shapes such as:

```text
Wildcards Processor output
  -> State Manager input
  -> Wildcards Processor input
```

The state manager no longer exposes runtime capture inputs, so this cycle is not part of the node design. Do not wire wildcard or prompt-processing output back into State Manager.

The older runtime-output path is still available for workflows that intentionally want the manager to drive execution:

```text
State Manager.dora_state -> DoRA Power LoRA Loader.dora_state
DoRA loader State slot selects which saved loader stack to use
State Manager.positive_prompt_template -> prompt/wildcard path
State Manager.negative_prompt_template -> prompt/wildcard path
```

Do not use that runtime path as the replacement for connected save/load. For editable prompt and seed values that survive normal execution, use `state_control` plus State Text Box / State Seed.


### Multiple DoRA loaders

A character now stores `loader_stacks`, not only one flat `loras` list. Each stack has a stable `slot`. The DoRA loader exposes a **State slot** widget; Save/Load connected matches by that slot.

Example:

```text
State Manager.state_control -> DoRA Loader A.state_control  # State slot: face
State Manager.state_control -> DoRA Loader B.state_control  # State slot: outfit
State Manager.state_control -> DoRA Loader C.state_control  # State slot: style
State Manager.state_control -> DoRA Loader D.state_control  # State slot: refiner
```

Click **Save connected** to capture all four stacks separately. Click **Load connected** to push each saved stack back into the matching loader.

### Save / load / apply workflow

The UI has four graph-editing actions:

- **Save connected** — captures state from nodes connected downstream of `state_control`.
- **Load connected** — pushes the selected character/preset into nodes connected downstream of `state_control`.
- **Save selected** — captures state from selected graph nodes.
- **Apply selected** — pushes the selected character/preset into selected graph nodes.

Selecting a character tile or prompt preset only changes the manager selection. It does not mutate other nodes until **Load connected** or **Apply selected** is clicked.

For older workflows that do not have `state_control` links, the frontend keeps a compatibility fallback that can inspect legacy runtime output links. Treat that as migration support, not the canonical wiring for new graphs.

### Pattern 1: deterministic downstream settings

The manager outputs:

- `settings_json` — selected prompt preset settings as JSON text
- `state_settings` — typed `DORA_STATE_SETTINGS` payload
- `seed` — integer seed extracted from saved settings / rgthree seed snapshots
- `state_control` — typed `STATE_MANAGER_CONTROL` payload used only by the frontend to discover connected nodes for Save connected / Load connected
- `character_image` — original uploaded character image as an `IMAGE` tensor

Future or external nodes can add optional inputs for `settings_json`, `state_settings`, or `seed` to consume saved state deterministically during execution. `state_control` is reserved for editor save/load association.

### Pattern 2: frontend apply/capture

For existing nodes that do not have settings inputs, use the UI actions:

- save selected/connected AutoGuidance or Scale-Locked Guidance nodes into the prompt preset
- save selected/connected rgthree seed nodes into the prompt preset
- load/apply the preset back into those graph nodes later

The settings snapshot stores widget values by node identity and class/title fallback. This is intended for controlled workflows where the same downstream nodes remain in the graph.

### Outputs

- `dora_state` — typed `DORA_STATE` payload for **DoRA Power LoRA Loader**
- `positive_prompt_template` — selected prompt preset's positive template text
- `negative_prompt_template` — selected prompt preset's negative template text
- `settings_json` — selected prompt preset settings serialized as JSON
- `selected_lora_stack` — selected character's LoRA stack as a typed `DORA_LORA_STACK` payload
- `state_settings` — typed `DORA_STATE_SETTINGS` payload for future compatible nodes
- `seed` — integer seed extracted from saved settings / rgthree seed snapshots
- `state_control` — typed `STATE_MANAGER_CONTROL` payload for editor/control-only save/load association
- `character_image` — original uploaded character image as an `IMAGE` tensor. The manager uploads the original image to ComfyUI's input folder and only scales it visually in the browser.

### Persistence and payload behavior

The manager state is stored in the workflow through hidden backend widgets:

- `state_json`
- `ui_state_json`
- `selected_character_id`
- `selected_prompt_id`

The `DORA_STATE` payload contains the selected character identity, selected prompt identity, saved LoRA rows, saved loader-global overrides, prompt templates, and prompt settings. The DoRA loader accepts this typed payload; invalid or missing `dora_state` data falls back to the loader's existing local row-widget parsing.
