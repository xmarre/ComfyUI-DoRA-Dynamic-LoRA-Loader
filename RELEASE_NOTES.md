# DoRA Dynamic LoRA Loader v1.0.47

This release fixes State Manager character/preset selection persistence across a real ComfyUI restart. A saved non-default selection now restores as the same persistent character/preset before any State Manager interaction, so its reference set and managed prompt routing remain attached to the intended preset after startup.

## Restart selection persistence

- Treats the hidden selection widgets that ComfyUI already restored before `onConfigure()` as the startup authority instead of independently preferring a potentially stale `widgets_values_named` shadow.
- Uses the normal workflow selection mirror only to recover the old stale-default signature; an explicit non-default configured selection remains authoritative.
- Prevents constructor `default_character/default_prompt` placeholders from becoming persistent authority merely because an asynchronous library load finishes first.
- Starts loaded-workflow library hydration from the `loadedGraphNode` lifecycle barrier while keeping a separate deferred path for newly created unsaved nodes.
- Preserves an established authoritative selection during serialization even if transient widget state briefly contains constructor defaults.

## Workflow persistence and distribution-safe mode

- Captures State Manager DOM selection changes with a bounded ComfyUI change transaction when workflow-serialized state actually changes. This closes the event-ordering gap where the global `mouseup` snapshot could occur before the tile `click` updated the selected UUIDs.
- Does not force whole-graph snapshots for backend-only library edits such as prompt text changes.
- Distribution-safe workflows continue to serialize `default_character/default_prompt` and omit the private workflow UUID mirror.
- In distribution-safe mode, only the selected character/prompt UUID pair is kept browser-locally behind an opaque binding for same-browser restart recovery; character data, prompts, settings, LoRAs, images, references, and library contents remain excluded.

## Compatibility and invariants

- The backend-authoritative State Manager library format and revision semantics are unchanged.
- Managed prompt-document v5 transport, queue snapshot semantics, request-user scoping, queue-write drain, and `DSM_UNSAVED_MANAGED_TEXT` protection are unchanged.
- Missing UUIDs still do not silently select an unrelated local character.
- Existing normal workflows require no library migration.

## Validation

- PR #82 remains a single implementation/release commit over v1.0.46.
- The full package/frontend, Impact wildcard, ComfyUI compatibility/runtime, and managed State Manager → Impact → Continuum test matrix is green for the implementation.
- Real ComfyUI process-restart acceptance passed: the intended non-default character/preset restored without State Manager interaction, with the complete seven-reference state and associated prompt routing intact.

# DoRA Dynamic LoRA Loader v1.0.46

This release hardens State Manager prompt persistence and queue transport, and adds the managed prompt-document contract used by H3 Continuum and other verified STRING consumers.

## Managed prompt-document transport

- Adds State Manager managed-prompt contract v5 with atomic persistent text + `prompt_document` writes while keeping the v4 text-only write path available for compatibility.
- Persists explicit Fixed/List/Timeline intent and geometry; missing legacy metadata is represented only by a request-local `inherit` sidecar and is never inferred back into persistent storage.
- Freezes the selected persistent preset at queue time, verifies handler ordering, and binds the authoritative State Manager Text Box through ImpactWildcardProcessor to the intended downstream consumer.
- Keeps explicit persisted documents authoritative and never promotes stale Impact `populated_text` or other downstream mirrors into persistent State Manager authority.

## Queue/persistence synchronization

- Tracks the active State Manager library write and drains pending/in-flight persistence before queue materialization.
- Aborts queue submission when a persistent write is blocked or fails instead of launching with stale data.
- Fails fast with `DSM_UNSAVED_MANAGED_TEXT` when a connected local Text Box is non-empty but the selected persistent managed text is still empty.
- Preserves unrelated library data, revision protection, request-user scoping, and existing State Manager/loader synchronization behavior.

## Packaging and compatibility

- Ships the new `state_manager_prompt_bridge.py` and `state_manager_prompt_document.py` modules in the Comfy package.
- Existing presets without prompt-document metadata remain loadable. When paired with H3 Continuum 3.4.3, historical exact chunk separators such as `[0-7s]` / `[7-14s]` can be validated request-locally without rewriting the preset.
- Impact/Core remain compatibility inputs; no Impact or ComfyUI Core patch is required.

## Validation

- Exact-head Actions run 35423724177 completed successfully.
- Production validation confirmed the managed queue receipt, Impact edge receipt, and consumer sidecar carried the same non-empty 6680-character prompt, the same snapshot revision, and the same SHA-256 before H3 Continuum execution.

# DoRA Dynamic LoRA Loader v1.0.45

This release fixes split state between the State Manager and connected DoRA Power LoRA Loader nodes so the visible loader configuration, saved preset, and runtime state remain consistent.

## Bidirectional State Manager / loader synchronization

- Synchronizes loader-owned edits back into the selected State Manager loader stack, including LoRA rows, model/CLIP strengths, Auto-strength, analysis device, ratio bounds, and persisted loader globals.
- Pushes State Manager loader edits and character/preset changes into connected loaders immediately instead of leaving the visible loader stale until execution.
- Restores connected loader state after persistent-library refreshes, imports, and write rollbacks.
- Uses the actually connected loader State slot in the State Manager settings panel instead of an unrelated empty `default` stack.
- Exposes both Auto-strength ratio floor and ceiling for each managed loader stack and normalizes their bounds with the same semantics as the loader.
- Preserves State-slot identity across renames and rejects collisions without overwriting another saved stack.
- On workflow load, keeps durable saved presets authoritative while preserving a configured loader when the State Manager is still on its unsaved default placeholder.

## Synchronization hardening

- Suppresses reverse notifications for State Manager-originated loader updates to prevent feedback loops.
- Coalesces continuous loader edits with a short trailing debounce so weight dragging does not generate a burst of persistent-library writes.
- Keeps State-slot changes immediate so identity changes are not delayed behind the debounce.
- Tracks and cancels deferred post-load loader synchronization when a State Manager node is removed.
- Removes a dead slot-realignment branch found during review and adds regression coverage for the corrected lifecycle and synchronization paths.

## Validation

- Frontend/package regression suite passes.
- Compatibility and runtime-bypass suites pass against pinned ComfyUI v0.29.2, v0.30.2, v0.31.1, and the 2026-08-21 revision.
- Existing workflows keep the same node and persisted state formats; no workflow migration is required.

# DoRA Dynamic LoRA Loader v1.0.44

This hotfix restores LoRA-row interaction on current ComfyUI frontend main when the displayed rows survive a dynamic loader rebuild.

## Stable custom-widget lifetime

- Keeps each existing LoRA row widget object stable across the loader's immediate build, asynchronous LoRA-list refresh, Add LoRA rebuilds, and other same-name UI rebuilds.
- Rebinds a reused row widget to the newly sanitized canonical row state instead of leaving the frontend bound to an orphaned row object.
- Keeps the auto-strength visualization widget identity stable for the same frontend lifecycle.
- Prevents the v1.0.43 prototype-compatibility preparation from running again after a widget has already been normalized by the frontend.
- Adds a regression that captures the row object before the asynchronous refresh, requires the same object afterward, verifies a click updates canonical loader state, and verifies Add LoRA creates the next row without replacing the existing row object.

## Root cause

Current Vue legacy-widget rendering binds the live custom-widget object when its component mounts and keeps that object while the stable WidgetId/type render key remains unchanged. The loader rebuilt its UI after fetching the LoRA list and created a new `LORA_1` object under the same widget identity. The frontend therefore continued drawing and dispatching pointer events to the old object while the node's live widget list and canonical row state belonged to the replacement object.

# DoRA Dynamic LoRA Loader v1.0.43

This hotfix restores the loaded LoRA row controls on current ComfyUI frontend master while preserving compatibility with older frontends.

## Custom-widget rendering compatibility

- Preserves the DoRA row and auto-strength report widgets' drawing, sizing, pointer, picker, editing, and serialization hooks when the frontend normalizes custom widgets into its store-backed legacy adapter.
- Keeps the existing class instances and state model on classic and older frontends.
- Adds a regression harness that reproduces the current frontend's prototype-replacement behavior and verifies that loaded LoRA names and strengths still render and remain interactive.

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
