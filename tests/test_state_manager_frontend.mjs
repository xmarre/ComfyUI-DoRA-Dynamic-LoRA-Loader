import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";


async function loadStateManagerHelpers() {
  const sourceUrl = new URL("../web/dora_state_manager.js", import.meta.url);
  let source = await readFile(sourceUrl, "utf8");
  source = source
    .replace('import { app } from "../../scripts/app.js";', "let capturedExtension = null; const app = { registerExtension(value) { capturedExtension = value; }, graph: { extra: {} } };")
    .replace('import { api } from "../../scripts/api.js";', "const api = { fetchApi() { throw new Error('not used'); }, apiURL(value) { return value; } };")
    .replace('import "../../scripts/domWidget.js";', "");
  source += `\nexport { capturedExtension, defaultBinding, defaultState, deletePromptPreset, deleteStateCharacter, makeId, materializeEditedDefault, mergeScheduledLibraryUpdate, persistentCharacters, serializeBinding, serializeWorkflowUiState, serializeQueuedUiStateOverride, parseLegacyEmbeddedState, stateLibraryClient, stateViewForSelection, syncCharacterLoaderStacksToConnectedNodes, syncConnectedLoaderStateIntoManager, synchronizeConnectedLoadersAfterLibraryLoad, restoreNodeAndConnectedLoadersFromLibrary, normalizeLoaderGlobals, pickPrimarySettingsLoaderStack, refreshAllNodesFromLibrary };\n`;
  const encoded = Buffer.from(source, "utf8").toString("base64");
  return import(`data:text/javascript;base64,${encoded}#${Date.now()}-${Math.random()}`);
}


function privateCharacter(id, name, promptText) {
  return {
    id,
    name,
    prompts: [{ id: `${id}-prompt`, name: "Private preset", positive: promptText }],
  };
}


test("workflow binding contains IDs/configuration only and no private library payload", async () => {
  const helpers = await loadStateManagerHelpers();
  const binding = JSON.parse(helpers.serializeBinding());
  assert.deepEqual(binding, { version: 1, kind: "dora_state_manager_binding" });
  const serialized = JSON.stringify(binding);
  for (const secret of ["Private Character", "private prompt", "private.safetensors", "thumbnail", "reference_image", "loader_stacks"]) {
    assert.equal(serialized.includes(secret), false);
  }
});


test("the installed onSerialize hook scrubs private widget and property payloads", async () => {
  const helpers = await loadStateManagerHelpers();
  class StateManagerNode {
    onSerialize(output) {
      output.widgets_values = this.widgets.map((widget) => widget.value);
      output.widgets_values_named = Object.fromEntries(this.widgets.map((widget) => [widget.name, widget.value]));
      output.properties = { ...this.properties };
    }
  }
  StateManagerNode.comfyClass = "State Manager";
  await helpers.capturedExtension.beforeRegisterNodeDef(StateManagerNode, {
    name: "State Manager",
    input: { required: {} },
  });
  const privateState = {
    version: 3,
    characters: [privateCharacter("private-character", "Private Character", "private prompt text")],
  };
  const node = new StateManagerNode();
  node.properties = {
    dora_state_manager: privateState,
    dora_state_manager_backup_node_uid: "private-backup-id",
  };
  node.widgets = [
    { name: "state_json", value: JSON.stringify(privateState) },
    { name: "ui_state_json", value: JSON.stringify({ status: "Private Character", panel: "character" }) },
    { name: "selected_character_id", value: "private-character" },
    { name: "selected_prompt_id", value: "private-character-prompt" },
  ];
  node.__dsm = { state: privateState, uiState: { status: "Private Character", panel: "character" } };

  const output = {};
  node.onSerialize(output);
  const serialized = JSON.stringify(output);
  assert.equal(serialized.includes("Private Character"), false);
  assert.equal(serialized.includes("private prompt text"), false);
  assert.equal(Object.prototype.hasOwnProperty.call(output.properties, "dora_state_manager"), false);
  assert.deepEqual(JSON.parse(output.widgets_values[0]), helpers.defaultBinding());
  assert.deepEqual(JSON.parse(output.widgets_values[1]), {
    version: 2,
    queue_prompt_wildcard: false,
    queue_character_wildcard: false,
    queue_randomize_saved_seed: false,
    queue_character_ids: [],
  });
});


test("failed legacy migration remains serialized for a lossless retry", async () => {
  const helpers = await loadStateManagerHelpers();
  class StateManagerNode {
    onSerialize(output) {
      output.widgets_values = this.widgets.map((widget) => widget.value);
      output.widgets_values_named = Object.fromEntries(this.widgets.map((widget) => [widget.name, widget.value]));
      output.properties = {};
    }
  }
  StateManagerNode.comfyClass = "State Manager";
  await helpers.capturedExtension.beforeRegisterNodeDef(StateManagerNode, {
    name: "State Manager",
    input: { required: {} },
  });
  const legacy = { version: 3, characters: [privateCharacter("legacy", "Legacy", "recover me")] };
  const node = new StateManagerNode();
  node.properties = {};
  node.widgets = [
    { name: "state_json", value: helpers.serializeBinding() },
    { name: "ui_state_json", value: helpers.serializeWorkflowUiState({}) },
    { name: "selected_character_id", value: "legacy" },
    { name: "selected_prompt_id", value: "legacy-prompt" },
  ];
  node.__dsm = { state: helpers.defaultState(), uiState: {} };
  node.__dsmPendingLegacyState = legacy;
  const output = {};
  node.onSerialize(output);
  const preserved = JSON.parse(output.widgets_values[0]);
  assert.equal(preserved.characters[0].id, "legacy");
  assert.equal(preserved.characters[0].prompts[0].positive, "recover me");
  assert.deepEqual(JSON.parse(node.widgets[0].value), helpers.defaultBinding());
});


test("an untouched ephemeral default is never materialized by selection or queue UI updates", async () => {
  const helpers = await loadStateManagerHelpers();
  const result = helpers.materializeEditedDefault(
    helpers.defaultState(),
    "default_character",
    "default_prompt",
  );
  assert.equal(result.characterId, "default_character");
  assert.equal(result.promptId, "default_prompt");
  assert.equal(result.state.characters[0].id, "default_character");
});


test("materializing an edited default preserves the selected newly-created prompt", async () => {
  const helpers = await loadStateManagerHelpers();
  const state = helpers.defaultState();
  state.characters[0].prompts.push({
    ...structuredClone(state.characters[0].prompts[0]),
    id: "draft_prompt",
    name: "New preset",
  });
  const result = helpers.materializeEditedDefault(state, "default_character", "draft_prompt");
  assert.notEqual(result.characterId, "default_character");
  assert.equal(result.promptId, result.state.characters[0].prompts[1].id);
});


test("new and duplicated presets receive collision-resistant UUID bindings", async () => {
  const helpers = await loadStateManagerHelpers();
  const ids = new Set(Array.from({ length: 100 }, () => helpers.makeId("prompt")));
  assert.equal(ids.size, 100);
  for (const id of ids) assert.match(id, /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i);
});


test("deleting the sole preset removes its character from persistent storage", async () => {
  const helpers = await loadStateManagerHelpers();
  const state = {
    version: 3,
    characters: [privateCharacter("character-a", "A", "A0")],
  };
  const result = helpers.deletePromptPreset(state, "character-a", "character-a-prompt");
  assert.equal(result.deleted, true);
  assert.equal(result.removedCharacter, true);
  assert.equal(result.characterId, "default_character");
  assert.equal(result.promptId, "default_prompt");
  assert.deepEqual(helpers.persistentCharacters(result.state), []);
});


test("deleting one of several presets keeps the character and selects its neighbor", async () => {
  const helpers = await loadStateManagerHelpers();
  const character = privateCharacter("character-a", "A", "A0");
  character.prompts.push({
    ...structuredClone(character.prompts[0]),
    id: "character-a-prompt-2",
    name: "Second preset",
    positive: "A1",
  });
  const result = helpers.deletePromptPreset(
    { version: 3, characters: [character] },
    "character-a",
    "character-a-prompt",
  );
  assert.equal(result.deleted, true);
  assert.equal(result.removedCharacter, false);
  assert.equal(result.characterId, "character-a");
  assert.equal(result.promptId, "character-a-prompt-2");
  assert.deepEqual(
    helpers.persistentCharacters(result.state)[0].prompts.map((prompt) => prompt.id),
    ["character-a-prompt-2"],
  );
});


test("deleting the final character leaves an empty persistent library", async () => {
  const helpers = await loadStateManagerHelpers();
  const result = helpers.deleteStateCharacter(
    { version: 3, characters: [privateCharacter("character-a", "A", "A0")] },
    "character-a",
  );
  assert.equal(result.deleted, true);
  assert.equal(result.characterId, "default_character");
  assert.equal(result.promptId, "default_prompt");
  assert.deepEqual(helpers.persistentCharacters(result.state), []);
});


test("deleting ephemeral stale selections repairs bindings without reporting stored deletions", async () => {
  const helpers = await loadStateManagerHelpers();
  const character = privateCharacter("character-a", "A", "A0");
  helpers.stateLibraryClient.state = { version: 2, characters: [character] };
  const promptState = helpers.stateViewForSelection("character-a", "deleted-prompt");
  const promptResult = helpers.deletePromptPreset(promptState, "character-a", "deleted-prompt");
  assert.equal(promptResult.deleted, false);
  assert.equal(promptResult.characterId, "character-a");
  assert.equal(promptResult.promptId, "character-a-prompt");
  assert.deepEqual(
    helpers.persistentCharacters(promptResult.state)[0].prompts.map((prompt) => prompt.id),
    ["character-a-prompt"],
  );

  const characterState = helpers.stateViewForSelection("deleted-character", "deleted-prompt");
  const characterResult = helpers.deleteStateCharacter(characterState, "deleted-character");
  assert.equal(characterResult.deleted, false);
  assert.equal(characterResult.characterId, "character-a");
  assert.equal(characterResult.promptId, "character-a-prompt");
  assert.deepEqual(
    helpers.persistentCharacters(characterResult.state).map((item) => item.id),
    ["character-a"],
  );
});


test("disjoint character edits rebase without losing either manager's change", async () => {
  const helpers = await loadStateManagerHelpers();
  const base = [
    privateCharacter("character-a", "A", "A0"),
    privateCharacter("character-b", "B", "B0"),
  ];
  const desired = structuredClone(base);
  desired[1].prompts[0].positive = "B1";
  const current = structuredClone(base);
  current[0].prompts[0].positive = "A1";
  const merged = helpers.mergeScheduledLibraryUpdate(base, desired, current);
  assert.equal(merged.conflict, null);
  assert.equal(merged.characters[0].prompts[0].positive, "A1");
  assert.equal(merged.characters[1].prompts[0].positive, "B1");
});


test("same-character concurrent edits are surfaced instead of overwritten", async () => {
  const helpers = await loadStateManagerHelpers();
  const base = [privateCharacter("character-a", "A", "A0")];
  const desired = structuredClone(base);
  desired[0].prompts[0].positive = "A from manager two";
  const current = structuredClone(base);
  current[0].prompts[0].positive = "A from manager one";
  const merged = helpers.mergeScheduledLibraryUpdate(base, desired, current);
  assert.equal(merged.conflict, "character-a");
  assert.equal(merged.characters[0].prompts[0].positive, "A from manager one");
});


test("a stale prompt binding creates one ephemeral prompt without duplicating its character", async () => {
  const helpers = await loadStateManagerHelpers();
  const character = privateCharacter("character-a", "A", "A0");
  helpers.stateLibraryClient.state = { version: 2, characters: [character] };
  const state = helpers.stateViewForSelection("character-a", "deleted-prompt");
  assert.equal(state.characters.filter((entry) => entry.id === "character-a").length, 1);
  const missing = state.characters[0].prompts.find((prompt) => prompt.id === "deleted-prompt");
  assert.equal(missing.__dsm_ephemeral, true);
  assert.equal(helpers.persistentCharacters(state)[0].prompts.some((prompt) => prompt.id === "deleted-prompt"), false);
});


test("workflow UI serialization drops disposable status and panel state", async () => {
  const helpers = await loadStateManagerHelpers();
  const serialized = helpers.serializeWorkflowUiState({
    panel: "character",
    status: "Editing Private Character",
    queue_prompt_wildcard: true,
    queue_character_wildcard: true,
    queue_randomize_saved_seed: false,
    queue_character_ids: ["9aa1ddfd-a018-4f42-9ca5-e8c05d558729"],
  });
  const parsed = JSON.parse(serialized);
  assert.equal(Object.prototype.hasOwnProperty.call(parsed, "status"), false);
  assert.equal(Object.prototype.hasOwnProperty.call(parsed, "panel"), false);
  assert.equal(Object.prototype.hasOwnProperty.call(parsed, "__dsm_library_user_id"), false);
  assert.deepEqual(parsed.queue_character_ids, ["9aa1ddfd-a018-4f42-9ca5-e8c05d558729"]);
});


test("queued manager override carries a runtime seed and selection metadata only", async () => {
  const helpers = await loadStateManagerHelpers();
  const serialized = helpers.serializeQueuedUiStateOverride(
    { status: "Private Character", queue_randomize_saved_seed: true },
    "8e7dd506-439d-4040-b5ba-d9e258259abc",
    "0a4f988a-4f17-4df6-9d2f-5f0042e9306b",
    1234,
    0,
    2,
  );
  const parsed = JSON.parse(serialized);
  assert.equal(parsed.__dsm_runtime_seed, 1234);
  assert.equal(parsed.__dsm_library_user_id, "default");
  assert.equal(parsed.__dsm_queued_runtime_character_id, "8e7dd506-439d-4040-b5ba-d9e258259abc");
  assert.equal(parsed.__dsm_queued_runtime_prompt_id, "0a4f988a-4f17-4df6-9d2f-5f0042e9306b");
  assert.equal(Object.prototype.hasOwnProperty.call(parsed, "__dsm_queued_runtime_state"), false);
  assert.equal(serialized.includes("Private Character"), false);
});


test("legacy embedded state remains detectable for controlled migration", async () => {
  const helpers = await loadStateManagerHelpers();
  const legacy = { version: 3, characters: [{ id: "legacy", name: "Legacy" }] };
  assert.deepEqual(helpers.parseLegacyEmbeddedState(JSON.stringify(legacy)), legacy);
  assert.equal(helpers.parseLegacyEmbeddedState(helpers.serializeBinding()), null);
});


test("browser persistence code cannot resurrect a private library", async () => {
  const source = await readFile(new URL("../web/dora_state_manager.js", import.meta.url), "utf8");
  assert.equal(source.includes("localStorage"), false);
  assert.equal(source.includes("tryRestoreStateBackup"), false);
  assert.equal(source.includes("writeStateBackup"), false);
  assert.equal(source.includes("dora_state_manager_backup_workflow_id"), true, "legacy metadata should only appear in the serialization scrubber");
  assert.match(source, /delete app\.graph\.extra\.dora_state_manager_backup_workflow_id/);
  assert.equal(source.includes("setWidgetValue(widgets.uiStateWidget, serializeUiState"), false);
  const stashIndex = source.indexOf("node.__dsmPendingLegacyState = structuredCloneCompat(embeddedLegacy)");
  const scrubIndex = source.indexOf("setWidgetValue(currentWidgets.stateWidget, serializeBinding())");
  const successIndex = source.indexOf("delete node.__dsmPendingLegacyState;");
  assert.ok(stashIndex >= 0 && scrubIndex > stashIndex && successIndex > scrubIndex);
  assert.match(source, /if \(loaded\) \{[\s\S]*delete node\.__dsmPendingLegacyState;/);
  assert.match(source, /this\.__dsmPendingLegacyState\s*\?\s*serializeState\(this\.__dsmPendingLegacyState\)/);
});


test("queued library values never synchronize into the workflow copy", async () => {
  const source = await readFile(new URL("../web/dora_state_manager.js", import.meta.url), "utf8");
  assert.equal(/syncWidget:\s*true/.test(source), false);
  assert.equal((source.match(/syncWidget\s*=\s*true/g) || []).length, 1);
  assert.equal(source.includes("__dsm_queued_runtime_state"), false);
});


test("blocked writes clear pending work and restore the persisted view", async () => {
  const source = await readFile(new URL("../web/dora_state_manager.js", import.meta.url), "utf8");
  assert.match(source, /function blockLibraryWrites[\s\S]*stateLibraryClient\.pending = \[\]/);
  assert.match(source, /function blockLibraryWrites[\s\S]*restoreNodeAndConnectedLoadersFromLibrary\(node/);
  assert.match(source, /if \(stateLibraryClient\.blocked\) \{[\s\S]*restoreNodeAndConnectedLoadersFromLibrary\(node/);
});


function makeConnectedManagerAndLoader(helpers, character) {
  const manager = {
    id: 1,
    comfyClass: "State Manager",
    outputs: [{ name: "state_control", links: [100] }],
    properties: {},
    widgets: [
      { name: "state_json", value: helpers.serializeBinding() },
      { name: "ui_state_json", value: helpers.serializeWorkflowUiState({}) },
      { name: "selected_character_id", value: character.id },
      { name: "selected_prompt_id", value: character.prompts[0].id },
    ],
    __dsm: {
      state: { version: 2, characters: [structuredClone(character)] },
      uiState: {},
    },
    setDirtyCanvas() {},
  };
  const loader = {
    id: 2,
    comfyClass: "DoRA Power LoRA Loader",
    title: "DoRA Power LoRA Loader",
    inputs: [{ name: "state_control" }],
    properties: { dora_state_slot: "default" },
    widgets: [{ name: "state_slot", value: "default" }],
    setDirtyCanvas() {},
  };
  const graph = {
    links: {
      100: { origin_id: 1, origin_slot: 0, target_id: 2, target_slot: 0 },
    },
    getNodeById(id) {
      if (id === manager.id) return manager;
      if (id === loader.id) return loader;
      return null;
    },
    change() {},
  };
  manager.graph = graph;
  loader.graph = graph;
  return { manager, loader };
}


test("connected loader edits replace the selected State Manager stack immediately", async () => {
  const helpers = await loadStateManagerHelpers();
  const character = privateCharacter("character-a", "A", "A0");
  character.loader_stacks = [{
    slot: "default",
    label: "Default loader",
    loras: [{
      enabled: true,
      name: "old.safetensors",
      strength_model: 1.0,
      strength_clip: 1.0,
    }],
    loader_globals: {
      auto_strength_enabled: false,
      auto_strength_device: "gpu",
      auto_strength_ratio_floor: 0.3,
      auto_strength_ratio_ceiling: 1.5,
    },
  }];
  character.loras = structuredClone(character.loader_stacks[0].loras);
  character.loader_globals = structuredClone(character.loader_stacks[0].loader_globals);
  const { manager, loader } = makeConnectedManagerAndLoader(helpers, character);

  const previousLoaderApi = globalThis.__doraPowerLoraLoaderApi;
  globalThis.__doraPowerLoraLoaderApi = {
    getSlot() {
      return "default";
    },
    getState() {
      return {
        slot: "default",
        label: "Default loader",
        rows: [{
          enabled: true,
          name: "h3-character.safetensors",
          strengthModel: 0.82,
          strengthClip: 0.61,
        }],
        globals: {
          auto_strength_enabled: true,
          auto_strength_device: "cpu",
          auto_strength_ratio_floor: 0.44,
          auto_strength_ratio_ceiling: 1.91,
        },
      };
    },
  };

  try {
    const changed = helpers.syncConnectedLoaderStateIntoManager(
      manager,
      loader,
      { persist: false, render: false, dirty: false },
    );
    assert.equal(changed, 1);

    const savedCharacter = manager.__dsm.state.characters.find((item) => item.id === "character-a");
    const savedStack = savedCharacter.loader_stacks.find((item) => item.slot === "default");
    assert.equal(savedStack.loader_globals.auto_strength_enabled, true);
    assert.equal(savedStack.loader_globals.auto_strength_device, "cpu");
    assert.equal(savedStack.loader_globals.auto_strength_ratio_floor, 0.44);
    assert.equal(savedStack.loader_globals.auto_strength_ratio_ceiling, 1.91);
    assert.equal(savedStack.loras[0].name, "h3-character.safetensors");
    assert.equal(savedStack.loras[0].strength_model, 0.82);
    assert.equal(savedStack.loras[0].strength_clip, 0.61);
  } finally {
    if (previousLoaderApi === undefined) delete globalThis.__doraPowerLoraLoaderApi;
    else globalThis.__doraPowerLoraLoaderApi = previousLoaderApi;
  }
});


test("State Manager loader edits are pushed into the matching loader without a sync loop", async () => {
  const helpers = await loadStateManagerHelpers();
  const character = privateCharacter("character-a", "A", "A0");
  character.loader_stacks = [{
    slot: "default",
    label: "Default loader",
    loras: [{
      enabled: true,
      name: "saved.safetensors",
      strength_model: 0.73,
      strength_clip: 0.52,
    }],
    loader_globals: {
      auto_strength_enabled: true,
      auto_strength_device: "gpu",
      auto_strength_ratio_floor: 0.41,
      auto_strength_ratio_ceiling: 1.77,
    },
  }];
  character.loras = structuredClone(character.loader_stacks[0].loras);
  character.loader_globals = structuredClone(character.loader_stacks[0].loader_globals);
  const { manager, loader } = makeConnectedManagerAndLoader(helpers, character);

  const calls = [];
  const previousLoaderApi = globalThis.__doraPowerLoraLoaderApi;
  globalThis.__doraPowerLoraLoaderApi = {
    getSlot() {
      return "default";
    },
    setSlot(node, slot, options) {
      calls.push({ kind: "slot", node, slot, options });
      node.properties.dora_state_slot = slot;
      return slot;
    },
    setState(node, payload, options) {
      calls.push({ kind: "state", node, payload: structuredClone(payload), options });
      return true;
    },
  };

  try {
    const changed = helpers.syncCharacterLoaderStacksToConnectedNodes(manager, character, "default");
    assert.equal(changed, 1);
    const stateCall = calls.find((call) => call.kind === "state");
    assert.ok(stateCall);
    assert.equal(stateCall.payload.loader_globals.auto_strength_enabled, true);
    assert.equal(stateCall.payload.loader_globals.auto_strength_ratio_floor, 0.41);
    assert.equal(stateCall.payload.loras[0].strength_model, 0.73);
    assert.deepEqual(stateCall.options, { notifyStateManager: false });

    assert.equal(
      calls.some((call) => call.kind === "slot"),
      false,
      "same-slot State Manager edits must not rebuild the loader just to rewrite its slot"
    );
  } finally {
    if (previousLoaderApi === undefined) delete globalThis.__doraPowerLoraLoaderApi;
    else globalThis.__doraPowerLoraLoaderApi = previousLoaderApi;
  }
});


test("loader-side State slot rename moves the saved stack identity instead of duplicating it", async () => {
  const helpers = await loadStateManagerHelpers();
  const character = privateCharacter("character-a", "A", "A0");
  character.loader_stacks = [{
    slot: "default",
    label: "Default loader",
    loras: [{ enabled: true, name: "old.safetensors", strength_model: 1.0, strength_clip: 1.0 }],
    loader_globals: { auto_strength_enabled: false },
  }];
  character.loras = structuredClone(character.loader_stacks[0].loras);
  character.loader_globals = structuredClone(character.loader_stacks[0].loader_globals);
  const { manager, loader } = makeConnectedManagerAndLoader(helpers, character);
  loader.properties.dora_state_slot = "style";
  loader.widgets[0].value = "style";

  const previousLoaderApi = globalThis.__doraPowerLoraLoaderApi;
  globalThis.__doraPowerLoraLoaderApi = {
    getSlot(node) {
      return node.properties.dora_state_slot;
    },
    getState(node) {
      return {
        slot: node.properties.dora_state_slot,
        label: "Style",
        rows: [{
          enabled: true,
          name: "style.safetensors",
          strengthModel: 0.66,
          strengthClip: 0.55,
        }],
        globals: {
          auto_strength_enabled: true,
          auto_strength_device: "gpu",
          auto_strength_ratio_floor: 0.4,
          auto_strength_ratio_ceiling: 1.8,
        },
      };
    },
  };

  try {
    const changed = helpers.syncConnectedLoaderStateIntoManager(
      manager,
      loader,
      { persist: false, render: false, dirty: false, previousSlot: "default" },
    );
    assert.equal(changed, 1);
    const savedCharacter = manager.__dsm.state.characters.find((item) => item.id === "character-a");
    assert.equal(savedCharacter.loader_stacks.length, 1);
    assert.equal(savedCharacter.loader_stacks[0].slot, "style");
    assert.equal(savedCharacter.loader_stacks[0].loras[0].name, "style.safetensors");
    assert.equal(savedCharacter.loader_stacks[0].loader_globals.auto_strength_enabled, true);
    assert.equal(savedCharacter.loader_stacks.some((item) => item.slot === "default"), false);
  } finally {
    if (previousLoaderApi === undefined) delete globalThis.__doraPowerLoraLoaderApi;
    else globalThis.__doraPowerLoraLoaderApi = previousLoaderApi;
  }
});


test("loader-side State slot collision is visibly reverted without overwriting either saved stack", async () => {
  const helpers = await loadStateManagerHelpers();
  const character = privateCharacter("character-a", "A", "A0");
  character.loader_stacks = [
    {
      slot: "default",
      label: "Default loader",
      loras: [{ enabled: true, name: "default-old.safetensors", strength_model: 1.0, strength_clip: 1.0 }],
      loader_globals: { auto_strength_enabled: false },
    },
    {
      slot: "style",
      label: "Existing style",
      loras: [{ enabled: true, name: "existing-style.safetensors", strength_model: 0.4, strength_clip: 0.4 }],
      loader_globals: { auto_strength_enabled: false },
    },
  ];
  character.loras = structuredClone(character.loader_stacks[0].loras);
  character.loader_globals = structuredClone(character.loader_stacks[0].loader_globals);
  const { manager, loader } = makeConnectedManagerAndLoader(helpers, character);
  loader.properties.dora_state_slot = "style";
  loader.widgets[0].value = "style";

  const setSlotCalls = [];
  const previousLoaderApi = globalThis.__doraPowerLoraLoaderApi;
  globalThis.__doraPowerLoraLoaderApi = {
    getSlot(node) {
      return node.properties.dora_state_slot;
    },
    getState(node) {
      return {
        slot: node.properties.dora_state_slot,
        label: "Renamed loader",
        rows: [{
          enabled: true,
          name: "live-loader.safetensors",
          strengthModel: 0.77,
          strengthClip: 0.68,
        }],
        globals: { auto_strength_enabled: true },
      };
    },
    setSlot(node, slot, options) {
      setSlotCalls.push({ slot, options });
      node.properties.dora_state_slot = slot;
      node.widgets[0].value = slot;
      return slot;
    },
  };

  try {
    const changed = helpers.syncConnectedLoaderStateIntoManager(
      manager,
      loader,
      { persist: false, render: false, dirty: false, previousSlot: "default" },
    );
    assert.equal(changed, 1);
    assert.equal(loader.properties.dora_state_slot, "default");
    assert.deepEqual(setSlotCalls, [{
      slot: "default",
      options: { notifyStateManager: false },
    }]);

    const savedCharacter = manager.__dsm.state.characters.find((item) => item.id === "character-a");
    const defaultStack = savedCharacter.loader_stacks.find((item) => item.slot === "default");
    const styleStack = savedCharacter.loader_stacks.find((item) => item.slot === "style");
    assert.equal(defaultStack.loras[0].name, "live-loader.safetensors");
    assert.equal(defaultStack.loader_globals.auto_strength_enabled, true);
    assert.equal(styleStack.loras[0].name, "existing-style.safetensors");
    assert.equal(savedCharacter.loader_stacks.length, 2);
  } finally {
    if (previousLoaderApi === undefined) delete globalThis.__doraPowerLoraLoaderApi;
    else globalThis.__doraPowerLoraLoaderApi = previousLoaderApi;
  }
});


test("first load of a durable preset applies saved loader state instead of overwriting the preset", async () => {
  const helpers = await loadStateManagerHelpers();
  const character = privateCharacter("character-a", "Saved Character", "A0");
  character.loader_stacks = [{
    slot: "default",
    label: "Default loader",
    loras: [{
      enabled: true,
      name: "saved-character.safetensors",
      strength_model: 0.58,
      strength_clip: 0.49,
    }],
    loader_globals: {
      auto_strength_enabled: true,
      auto_strength_device: "cpu",
      auto_strength_ratio_floor: 0.46,
      auto_strength_ratio_ceiling: 1.88,
    },
  }];
  character.loras = structuredClone(character.loader_stacks[0].loras);
  character.loader_globals = structuredClone(character.loader_stacks[0].loader_globals);
  const { manager, loader } = makeConnectedManagerAndLoader(helpers, character);

  const calls = [];
  const previousLoaderApi = globalThis.__doraPowerLoraLoaderApi;
  globalThis.__doraPowerLoraLoaderApi = {
    getSlot() {
      return "default";
    },
    setSlot(node, slot, options) {
      calls.push({ kind: "slot", slot, options });
      node.properties.dora_state_slot = slot;
      return slot;
    },
    setState(node, payload, options) {
      calls.push({ kind: "state", payload: structuredClone(payload), options });
      return true;
    },
  };

  try {
    const before = structuredClone(manager.__dsm.state);
    const changed = helpers.synchronizeConnectedLoadersAfterLibraryLoad(manager);
    assert.equal(changed, 1);
    assert.deepEqual(manager.__dsm.state, before, "opening a saved preset must not rewrite its library state");

    const stateCall = calls.find((call) => call.kind === "state");
    assert.ok(stateCall);
    assert.equal(stateCall.payload.loras[0].name, "saved-character.safetensors");
    assert.equal(stateCall.payload.loader_globals.auto_strength_enabled, true);
    assert.equal(stateCall.payload.loader_globals.auto_strength_device, "cpu");
    assert.equal(stateCall.payload.loader_globals.auto_strength_ratio_floor, 0.46);
    assert.deepEqual(stateCall.options, { notifyStateManager: false });
  } finally {
    if (previousLoaderApi === undefined) delete globalThis.__doraPowerLoraLoaderApi;
    else globalThis.__doraPowerLoraLoaderApi = previousLoaderApi;
  }
});


test("library rollback restores the persisted loader globals into the visible connected loader", async () => {
  const helpers = await loadStateManagerHelpers();
  const persisted = privateCharacter("character-a", "A", "A0");
  persisted.loader_stacks = [{
    slot: "default",
    label: "Default loader",
    loras: [{
      enabled: true,
      name: "persisted.safetensors",
      strength_model: 0.62,
      strength_clip: 0.51,
    }],
    loader_globals: {
      auto_strength_enabled: false,
      auto_strength_device: "gpu",
      auto_strength_ratio_floor: 0.3,
      auto_strength_ratio_ceiling: 1.5,
    },
  }];
  persisted.loras = structuredClone(persisted.loader_stacks[0].loras);
  persisted.loader_globals = structuredClone(persisted.loader_stacks[0].loader_globals);
  helpers.stateLibraryClient.state = { version: 2, characters: [structuredClone(persisted)] };

  const unpersisted = structuredClone(persisted);
  unpersisted.loader_stacks[0].loras[0].name = "unpersisted.safetensors";
  unpersisted.loader_stacks[0].loader_globals.auto_strength_enabled = true;
  unpersisted.loader_globals = structuredClone(unpersisted.loader_stacks[0].loader_globals);
  unpersisted.loras = structuredClone(unpersisted.loader_stacks[0].loras);
  const { manager, loader } = makeConnectedManagerAndLoader(helpers, unpersisted);

  const calls = [];
  const previousLoaderApi = globalThis.__doraPowerLoraLoaderApi;
  const previousRaf = globalThis.requestAnimationFrame;
  globalThis.requestAnimationFrame = () => 1;
  globalThis.__doraPowerLoraLoaderApi = {
    getSlot() {
      return "default";
    },
    setSlot(node, slot, options) {
      calls.push({ kind: "slot", slot, options });
      node.properties.dora_state_slot = slot;
      return slot;
    },
    setState(node, payload, options) {
      calls.push({ kind: "state", payload: structuredClone(payload), options });
      return true;
    },
  };

  try {
    const changed = helpers.restoreNodeAndConnectedLoadersFromLibrary(manager, {
      status: "Library write rejected.",
    });
    assert.equal(changed, 1);

    const restoredCharacter = manager.__dsm.state.characters.find((item) => item.id === "character-a");
    assert.equal(restoredCharacter.loader_stacks[0].loras[0].name, "persisted.safetensors");
    assert.equal(restoredCharacter.loader_stacks[0].loader_globals.auto_strength_enabled, false);

    const stateCall = calls.find((call) => call.kind === "state");
    assert.ok(stateCall);
    assert.equal(stateCall.payload.loras[0].name, "persisted.safetensors");
    assert.equal(stateCall.payload.loader_globals.auto_strength_enabled, false);
    assert.deepEqual(stateCall.options, { notifyStateManager: false });
  } finally {
    if (previousLoaderApi === undefined) delete globalThis.__doraPowerLoraLoaderApi;
    else globalThis.__doraPowerLoraLoaderApi = previousLoaderApi;
    if (previousRaf === undefined) delete globalThis.requestAnimationFrame;
    else globalThis.requestAnimationFrame = previousRaf;
  }
});


test("untouched default loader state does not materialize a persistent preset on first load", async () => {
  const helpers = await loadStateManagerHelpers();
  const character = helpers.defaultState().characters[0];
  const { manager } = makeConnectedManagerAndLoader(helpers, character);

  const previousLoaderApi = globalThis.__doraPowerLoraLoaderApi;
  globalThis.__doraPowerLoraLoaderApi = {
    getSlot() {
      return "default";
    },
    getState() {
      return {
        slot: "default",
        label: "Default loader",
        rows: [{ enabled: true, name: "None", strengthModel: 1.0, strengthClip: 1.0 }],
        globals: {
          stack_enabled: true,
          verbose: false,
          log_unloaded_keys: false,
          broadcast_auto_scale: true,
          broadcast_modulations: true,
          broadcast_include_dora_scale: false,
          broadcast_scale: 1.0,
          dora_decompose_debug: false,
          dora_decompose_debug_n: 30,
          dora_decompose_debug_stack_depth: 10,
          dora_slice_fix: true,
          dora_adaln_swap_fix: true,
          zimage_lumina2_compat: true,
          auto_strength_enabled: false,
          auto_strength_device: "gpu",
          auto_strength_ratio_floor: 0.30,
          auto_strength_ratio_ceiling: 1.50,
        },
      };
    },
  };

  try {
    const changed = helpers.synchronizeConnectedLoadersAfterLibraryLoad(manager);
    assert.equal(changed, 0);
    assert.equal(manager.__dsm.state.characters[0].id, "default_character");
    assert.deepEqual(helpers.persistentCharacters(manager.__dsm.state), []);
  } finally {
    if (previousLoaderApi === undefined) delete globalThis.__doraPowerLoraLoaderApi;
    else globalThis.__doraPowerLoraLoaderApi = previousLoaderApi;
  }
});


test("State Manager canonicalizes auto-strength ratio bounds with loader semantics", async () => {
  const helpers = await loadStateManagerHelpers();
  assert.deepEqual(
    helpers.normalizeLoaderGlobals({
      auto_strength_ratio_floor: 2.0,
      auto_strength_ratio_ceiling: 0.5,
    }),
    {
      auto_strength_ratio_floor: 0.5,
      auto_strength_ratio_ceiling: 2.0,
    },
  );
  assert.deepEqual(
    helpers.normalizeLoaderGlobals({
      auto_strength_ratio_ceiling: 0.2,
    }),
    {
      auto_strength_ratio_floor: 0.2,
      auto_strength_ratio_ceiling: 0.3,
    },
  );
});


test("capturing a named connected loader replaces the unused sole default stack", async () => {
  const helpers = await loadStateManagerHelpers();
  const character = privateCharacter("character-a", "A", "A0");
  character.loader_stacks = [{
    slot: "default",
    label: "Default loader",
    loras: [],
    loader_globals: {},
  }];
  character.loras = [];
  character.loader_globals = {};
  const { manager, loader } = makeConnectedManagerAndLoader(helpers, character);
  loader.properties.dora_state_slot = "loader_207";
  loader.widgets[0].value = "loader_207";

  const previousLoaderApi = globalThis.__doraPowerLoraLoaderApi;
  globalThis.__doraPowerLoraLoaderApi = {
    getSlot(node) {
      return node.properties.dora_state_slot;
    },
    getState(node) {
      return {
        slot: node.properties.dora_state_slot,
        label: "DoRA loader",
        rows: [{
          enabled: true,
          name: "h3.safetensors",
          strengthModel: 0.9,
          strengthClip: 0.9,
        }],
        globals: { auto_strength_enabled: true },
      };
    },
  };

  try {
    const changed = helpers.syncConnectedLoaderStateIntoManager(
      manager,
      loader,
      { persist: false, render: false, dirty: false },
    );
    assert.equal(changed, 1);
    const savedCharacter = manager.__dsm.state.characters.find((item) => item.id === "character-a");
    assert.equal(savedCharacter.loader_stacks.length, 1);
    assert.equal(savedCharacter.loader_stacks[0].slot, "loader_207");
    assert.equal(savedCharacter.loader_stacks[0].loader_globals.auto_strength_enabled, true);
    assert.equal(savedCharacter.loader_stacks[0].loras[0].name, "h3.safetensors");
  } finally {
    if (previousLoaderApi === undefined) delete globalThis.__doraPowerLoraLoaderApi;
    else globalThis.__doraPowerLoraLoaderApi = previousLoaderApi;
  }
});


test("State Manager settings select the connected loader stack instead of an unused default stack", async () => {
  const helpers = await loadStateManagerHelpers();
  const character = privateCharacter("character-a", "A", "A0");
  character.loader_stacks = [
    {
      slot: "default",
      label: "Unused default",
      loras: [],
      loader_globals: { auto_strength_enabled: false },
    },
    {
      slot: "loader_207",
      label: "Connected loader",
      loras: [{ enabled: true, name: "h3.safetensors", strength_model: 1.0, strength_clip: 1.0 }],
      loader_globals: { auto_strength_enabled: true },
    },
  ];
  character.loras = [];
  character.loader_globals = {};
  const { manager, loader } = makeConnectedManagerAndLoader(helpers, character);
  loader.properties.dora_state_slot = "loader_207";
  loader.widgets[0].value = "loader_207";

  const previousLoaderApi = globalThis.__doraPowerLoraLoaderApi;
  globalThis.__doraPowerLoraLoaderApi = {
    getSlot(node) {
      return node.properties.dora_state_slot;
    },
  };

  try {
    const selected = helpers.pickPrimarySettingsLoaderStack(manager, character);
    assert.equal(selected.slot, "loader_207");
    assert.equal(selected.loader_globals.auto_strength_enabled, true);
  } finally {
    if (previousLoaderApi === undefined) delete globalThis.__doraPowerLoraLoaderApi;
    else globalThis.__doraPowerLoraLoaderApi = previousLoaderApi;
  }
});


test("persistent library refresh updates connected loaders and can skip the writer node", async () => {
  const helpers = await loadStateManagerHelpers();
  const persisted = privateCharacter("character-a", "A", "A0");
  persisted.loader_stacks = [{
    slot: "default",
    label: "Default loader",
    loras: [{ enabled: true, name: "persisted.safetensors", strength_model: 0.7, strength_clip: 0.6 }],
    loader_globals: { auto_strength_enabled: true },
  }];
  persisted.loras = structuredClone(persisted.loader_stacks[0].loras);
  persisted.loader_globals = structuredClone(persisted.loader_stacks[0].loader_globals);
  helpers.stateLibraryClient.state = { version: 2, characters: [structuredClone(persisted)] };

  const { manager } = makeConnectedManagerAndLoader(helpers, persisted);
  helpers.stateLibraryClient.nodes.add(manager);

  const calls = [];
  const previousLoaderApi = globalThis.__doraPowerLoraLoaderApi;
  const previousRaf = globalThis.requestAnimationFrame;
  globalThis.requestAnimationFrame = () => 1;
  globalThis.__doraPowerLoraLoaderApi = {
    getSlot() {
      return "default";
    },
    setState(node, payload, options) {
      calls.push({ payload: structuredClone(payload), options });
      return true;
    },
  };

  try {
    helpers.refreshAllNodesFromLibrary({ syncLoaders: true });
    assert.equal(calls.length, 1);
    assert.equal(calls[0].payload.loras[0].name, "persisted.safetensors");
    assert.equal(calls[0].payload.loader_globals.auto_strength_enabled, true);
    assert.deepEqual(calls[0].options, { notifyStateManager: false });

    calls.length = 0;
    helpers.refreshAllNodesFromLibrary({ syncLoaders: true, skipLoaderSyncNode: manager });
    assert.equal(calls.length, 0);
  } finally {
    helpers.stateLibraryClient.nodes.delete(manager);
    if (previousLoaderApi === undefined) delete globalThis.__doraPowerLoraLoaderApi;
    else globalThis.__doraPowerLoraLoaderApi = previousLoaderApi;
    if (previousRaf === undefined) delete globalThis.requestAnimationFrame;
    else globalThis.requestAnimationFrame = previousRaf;
  }
});


test("post-load loader synchronization frame is tracked and canceled on State Manager removal", async () => {
  const source = await readFile(new URL("../web/dora_state_manager.js", import.meta.url), "utf8");
  assert.match(source, /ctx\.postLoadSyncFrame\s*=\s*requestAnimationFrame/);
  assert.match(source, /if \(ctx\.postLoadSyncFrame\) \{[\s\S]*cancelAnimationFrame\(ctx\.postLoadSyncFrame\)[\s\S]*ctx\.postLoadSyncFrame\s*=\s*0/);
  assert.match(source, /if \(!stateLibraryClient\.nodes\.has\(node\)\) return;/);
});


test("applying a saved loader stack does not contain the dead self-comparison slot guard", async () => {
  const source = await readFile(new URL("../web/dora_state_manager.js", import.meta.url), "utf8");
  assert.equal(
    source.includes('normalizeLoaderSlot(getDoraLoaderSlot(targetNode), "default") !== slot'),
    false,
  );
});
