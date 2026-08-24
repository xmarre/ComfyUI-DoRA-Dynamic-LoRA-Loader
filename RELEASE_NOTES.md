# DoRA Dynamic LoRA Loader v1.0.42

This hotfix makes State Manager prompt-preset and character deletion persist correctly when removing the final stored entry.

## State Manager deletion fix

- Removes the selected prompt and selects its adjacent preset when the character has other prompts.
- Removes a character when its sole prompt is deleted, matching the persisted character schema's requirement that a stored character contain at least one prompt.
- Allows the final stored character to be deleted and shows the existing non-persistent default placeholder for an empty library.
- Treats ephemeral missing-selection placeholders as already unavailable, repairs the workflow binding to the nearest durable selection, and sends no storage write for that repair.
- Uses the same tested deletion helpers from the preset library, prompt editor, and character library controls.
- Adds regression coverage for multi-preset deletion, sole-preset deletion, final-character deletion, and stale prompt/character placeholders.

# DoRA Dynamic LoRA Loader v1.0.41

This release moves State Manager presets into a backend-authoritative per-user library and adds mathematically equivalent runtime bypass support for plain LoKr adapters on compatible ComfyUI revisions.

## Plain LoKr runtime bypass

- Adds runtime bypass support for plain direct-factor, decomposed, and mixed LoRA + LoKr adapter stacks when ComfyUI provides native LoKr bypass math.
- Preserves current ComfyUI materialized LoKr semantics across direct-factor alpha values and unequal decomposed ranks through runtime-only adapter copies.
- Keeps source adapters unchanged and continues to reject DoRA-LoKr, sliced/offset/transformed targets, reshape LoRA, and unsupported adapter forms.
- Rejects LoKr runtime mode explicitly on older ComfyUI revisions whose `LoKrAdapter` lacks forward-bypass math.
- Defers deterministic runtime validation failures across the base loader's retry boundary and commits captured adapters transactionally, preventing partial or duplicate hooks after a failed materialized-patch attempt.
- Adds parity, mixed-stack, rejection, retry, rollback, and hook-restoration coverage against a pinned current ComfyUI revision.

## State Manager persistent library architecture

- Moves reusable characters, prompt presets, LoRA stacks, settings, thumbnails, reference metadata, and filename prefixes into a backend-authoritative library under the ComfyUI user directory.
- Limits workflow serialization to UUID bindings and workflow-specific queue configuration.
- Removes workflow/node-keyed `localStorage` library backups and their automatic restore path. This is a breaking change for presets that exist only in a browser backup: before updating, open the workflow with v1.0.40, let the backup restore, then save/export it. After updating, the old browser entry remains untouched and can still be recovered by temporarily returning to v1.0.40 and exporting the restored state.
- Adds atomic, revisioned storage with UUID validation, locking, corruption quarantine, explicit missing-preset errors, and stale-write rejection.
- Adds idempotent migration for legacy embedded schema-v3 libraries plus explicit character/library import and export.
- Keeps runtime outputs, connected save/load/apply, text boxes, seeds, multiple loaders, queue wildcarding, and the legacy node alias compatible.
- Keeps queue-time library values transient and out of the queued workflow copy.
- Scopes backend libraries to the active ComfyUI user, including multi-user installations.

# DoRA Dynamic LoRA Loader v1.0.40

This release fixes DoRA Power LoRA Loader settings appearing to reset when switching away from a ComfyUI workflow tab and returning to it on newer store-backed ComfyUI frontends.

## Workflow-tab state restoration fix

The affected loader could restore its saved workflow state correctly while still displaying bootstrap/default widget values after graph reconstruction. This was most visible with settings such as:

- `auto_strength_enabled`
- `auto_strength_device`
- `auto_strength_ratio_floor`
- `auto_strength_ratio_ceiling`
- other loader-global controls and the loader state slot

A browser lifecycle trace confirmed that edited values were present in the outgoing workflow serialization and arrived intact in the returning node's `configure()` call. The canonical `properties.dora_power_lora` state and live `_doraGlobals` were also correct after restore. Only the visible standard widget facade remained at its defaults.

## Root cause

Current ComfyUI frontends back standard widget values with `WidgetValueStore`. Widgets are registered by graph ID, node ID, widget name, and type.

The DoRA Power LoRA Loader dynamically rebuilds its widgets. A freshly created node first builds bootstrap widgets, then workflow `configure()` restores the real loader state and rebuilds the same widget names. On the newer frontend, recreating a same-name/same-type widget can reconnect it to the already-registered bootstrap store entry instead of adopting the restored `addWidget(..., initialValue)` value.

That produced the misleading state where the loader internally held the correct workflow values but the UI continued to show defaults.

## Fix

- Keeps `properties.dora_power_lora` as the authoritative workflow representation for the loader.
- Keeps `_doraRows` / `_doraGlobals` as the loader-owned live state.
- Disables generic LiteGraph widget workflow serialization for this dynamic loader so `widgets_values` / `widgets_values_named` cannot become competing persistent state stores.
- Synchronizes dynamically recreated known loader widgets back to the canonical property-backed value after frontend widget registration.
- Performs one deterministic post-`configure()` reconciliation of attached known widgets.
- Does not use timeout-based repair loops.
- Keeps `widgets_values_named` and positional `widgets_values` only as migration inputs when canonical state is absent.
- Preserves legacy workflow compatibility, partial reconfiguration, null-state recovery, LoRA rows, and state-slot persistence.

## Validation

The final fix was verified in the real browser reproduction that previously failed repeatedly: after changing non-default auto-strength values, switching to another workflow, and switching back, the settings now remain correct.

Regression coverage includes:

- canonical loader state surviving a workflow-tab round trip;
- bootstrap defaults never overriding incoming canonical state;
- canonical state beating conflicting stale named-widget data;
- store-backed same-name widget reconstruction being forced back to canonical floor, ceiling, device, and state-slot values;
- stale widget facades not overwriting loader-owned state during serialization;
- legacy named/positional migration;
- partial configure and `dora_power_lora: null` recovery.

The existing compatibility/runtime test matrix remains green against pinned ComfyUI v0.29.2, v0.30.2, and v0.31.1 revisions.

## Compatibility

No workflow migration is required. Existing workflows continue to use the same node and state format. The widget-store synchronization is narrowly scoped to the DoRA Power LoRA Loader; on older LiteGraph/frontends without the newer store behavior it reduces to assigning the widget the canonical value it was already intended to have.
