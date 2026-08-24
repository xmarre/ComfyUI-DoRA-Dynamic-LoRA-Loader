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
  source += `\nexport { capturedExtension, defaultBinding, defaultState, deletePromptPreset, deleteStateCharacter, makeId, materializeEditedDefault, mergeScheduledLibraryUpdate, persistentCharacters, serializeBinding, serializeWorkflowUiState, serializeQueuedUiStateOverride, parseLegacyEmbeddedState, stateLibraryClient, stateViewForSelection };\n`;
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


test("deleting an ephemeral stale preset repairs selection without reporting a stored deletion", async () => {
  const helpers = await loadStateManagerHelpers();
  const character = privateCharacter("character-a", "A", "A0");
  helpers.stateLibraryClient.state = { version: 2, characters: [character] };
  const state = helpers.stateViewForSelection("character-a", "deleted-prompt");
  const result = helpers.deletePromptPreset(state, "character-a", "deleted-prompt");
  assert.equal(result.deleted, false);
  assert.equal(result.characterId, "character-a");
  assert.equal(result.promptId, "character-a-prompt");
  assert.deepEqual(
    helpers.persistentCharacters(result.state)[0].prompts.map((prompt) => prompt.id),
    ["character-a-prompt"],
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
  const successIndex = source.indexOf("if (loaded) delete node.__dsmPendingLegacyState");
  assert.ok(stashIndex >= 0 && scrubIndex > stashIndex && successIndex > scrubIndex);
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
  assert.match(source, /if \(stateLibraryClient\.blocked\) \{[\s\S]*refreshNodeFromLibrary\(node/);
});
