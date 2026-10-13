import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

// Real DoRA loader, loader persistence, and State Manager code driven through
// the workflow lifecycle: open (create → configure → loadedGraphNode), leave
// (serialize → remove), reopen, save, apply, and queue.

const LOADER_CLASS = "DoRA Power LoRA Loader";

async function importPatched(relativePath, replacements, exports = "") {
  let source = await readFile(new URL(relativePath, import.meta.url), "utf8");
  for (const [from, to] of replacements) source = source.replace(from, to);
  source += exports;
  const encoded = Buffer.from(source, "utf8").toString("base64");
  return import(`data:text/javascript;base64,${encoded}#${Date.now()}-${Math.random()}`);
}

async function loadExtensions() {
  const loaderExtensions = [];
  globalThis.__ownershipTestApp = {
    canvas: { canvas: {} },
    graph: null,
    registerExtension(value) {
      loaderExtensions.push(value);
    },
  };
  globalThis.__ownershipTestApi = { addEventListener() {} };
  globalThis.LiteGraph = {};
  globalThis.fetch = async () => ({
    ok: true,
    async json() {
      return ["face.safetensors", "style.safetensors", "preset.safetensors", "workflow-b.safetensors"];
    },
  });
  const appImport = 'import { app } from "../../scripts/app.js";';
  const appGlobal = "const app = globalThis.__ownershipTestApp;";
  await importPatched("../web/dora_power_lora_loader.js", [
    [appImport, appGlobal],
    ['import { api } from "../../scripts/api.js";', "const api = globalThis.__ownershipTestApi;"],
  ]);
  await importPatched("../web/dora_power_lora_persistence.js", [[appImport, appGlobal]]);

  const manager = await importPatched(
    "../web/dora_state_manager.js",
    [
      [appImport, "let capturedExtension = null; const app = { registerExtension(value) { capturedExtension = value; }, graph: { extra: {} } };"],
      ['import { api } from "../../scripts/api.js";', "const api = { fetchApi(...args) { return globalThis.__dsmTestFetchApi(...args); }, apiURL(value) { return value; } };"],
      ['import "../../scripts/domWidget.js";', ""],
    ],
    "\nexport { app, capturedExtension, initializeNode, stateLibraryClient, serializeBinding, serializeWorkflowUiState, mutatePromptForStateManagers, saveLoadersIntoCharacter, syncCharacterLoaderStacksToConnectedNodes, updateState, flushPendingLibraryWrites, reloadStateLibrary };\n",
  );

  class LoaderNode {
    constructor() {
      this.id = -1;
      this.type = LOADER_CLASS;
      this.comfyClass = LOADER_CLASS;
      this.title = LOADER_CLASS;
      this.min_size = [0, 0];
      this.size = [360, 200];
      this.properties = {};
      this.widgets = [];
      this.inputs = [{ name: "state_control", link: null }];
    }

    addWidget(type, name, value, callback, options = {}) {
      const widget = { callback, name, options, type, value };
      this.widgets.push(widget);
      return widget;
    }

    addCustomWidget(widget) {
      this.widgets.push(widget);
      return widget;
    }

    computeSize() {
      return [this.size[0], 40 + this.widgets.length * 20];
    }

    configure(info) {
      for (const [key, value] of Object.entries(info?.properties || {})) this.properties[key] = value;
    }

    setDirtyCanvas() {}
  }
  LoaderNode.comfyClass = LOADER_CLASS;
  for (const extension of loaderExtensions) {
    await extension.beforeRegisterNodeDef?.(LoaderNode, { name: LOADER_CLASS, display_name: LOADER_CLASS });
  }
  return { manager, LoaderNode };
}

const presetCharacter = () => ({
  id: "character-a",
  name: "Character A",
  loader_stacks: [{
    slot: "face",
    label: "Face",
    loras: [{ enabled: false, name: "preset.safetensors", strength_model: 0.4, strength_clip: 0.3 }],
    loader_globals: { stack_enabled: true, auto_strength_enabled: false },
  }],
  prompts: [{ id: "prompt-a", name: "Prompt A", positive: "saved", negative: "", settings: {} }],
});

function loaderProperties(slot, rows, globals) {
  return { dora_state_slot: slot, dora_power_lora: { rows, globals } };
}

const WORKFLOW_A = {
  loader: loaderProperties(
    "face",
    [
      { enabled: true, name: "face.safetensors", strengthModel: 0.85, strengthClip: 0.65 },
      { enabled: false, name: "style.safetensors", strengthModel: 1.2, strengthClip: 1.1 },
    ],
    { stack_enabled: true, auto_strength_enabled: true, auto_strength_device: "cpu", auto_strength_ratio_floor: 0.4 },
  ),
};
const WORKFLOW_B = {
  loader: loaderProperties(
    "face",
    [{ enabled: true, name: "workflow-b.safetensors", strengthModel: 0.55, strengthClip: 0.45 }],
    { stack_enabled: false, verbose: true },
  ),
};

async function settle(frames) {
  for (let round = 0; round < 6; round += 1) {
    await new Promise((resolve) => setTimeout(resolve, 0));
    const pending = [...frames.values()];
    frames.clear();
    for (const callback of pending) callback();
  }
}

function makeHarness({ manager, LoaderNode }, library = { revision: 1, characters: [presetCharacter()], putOutcomes: [] }) {
  const frames = new Map();
  let nextFrame = 1;
  globalThis.requestAnimationFrame = (callback) => {
    const id = nextFrame++;
    frames.set(id, callback);
    return id;
  };
  globalThis.cancelAnimationFrame = (id) => frames.delete(id);

  const writes = [];
  const snapshot = () => ({ version: 2, revision: library.revision, characters: structuredClone(library.characters), user_id: "default" });
  const respond = (ok, status, body) => ({ ok, status, async json() { return body; } });
  globalThis.__dsmTestFetchApi = async (path, options = {}) => {
    const method = options.method || "GET";
    if (method === "GET") return respond(true, 200, snapshot());
    writes.push({ path, method });
    const outcome = library.putOutcomes.shift() || "ok";
    if (typeof outcome === "object" && outcome.deferred) await outcome.deferred;
    if (outcome === "conflict" || outcome?.then === "conflict") return respond(false, 409, { error: "revision conflict" });
    library.characters = JSON.parse(options.body).characters;
    library.revision += 1;
    return respond(true, 200, snapshot());
  };

  async function open(workflow) {
    const loader = new LoaderNode();
    loader.onNodeCreated?.();
    loader.id = 2;
    loader.configure({ properties: structuredClone(workflow.loader) });
    loader.inputs[0].link = 100;

    const managerNode = {
      id: 1,
      type: "State Manager",
      comfyClass: "State Manager",
      inputs: [],
      outputs: [{ name: "state_control", links: [100] }],
      widgets: [
        { name: "state_json", value: manager.serializeBinding() },
        { name: "ui_state_json", value: manager.serializeWorkflowUiState({}) },
        { name: "selected_character_id", value: "character-a" },
        { name: "selected_prompt_id", value: "prompt-a" },
      ],
      properties: {},
      size: [820, 720],
      __dsm: { state: null, uiState: null, renderFrame: 0 },
      setSize() {},
      setDirtyCanvas() {},
    };
    const graph = {
      _nodes: [managerNode, loader],
      links: { 100: { origin_id: 1, origin_slot: 0, target_id: 2, target_slot: 0 } },
      getNodeById(id) {
        return graph._nodes.find((node) => node.id === id) || null;
      },
      change() {},
    };
    managerNode.graph = graph;
    loader.graph = graph;
    manager.app.graph = graph;

    manager.initializeNode(managerNode, {});
    managerNode.onConfigure({
      widgets_values: managerNode.widgets.map((widget) => widget.value),
      properties: {},
    });
    manager.capturedExtension.loadedGraphNode(managerNode);
    await settle(frames);
    return { managerNode, loader };
  }

  function leave({ managerNode, loader }) {
    const serialized = { properties: structuredClone(loader.properties) };
    loader.onSerialize(serialized);
    managerNode.onRemoved?.();
    loader.onRemoved?.();
    return { loader: JSON.parse(JSON.stringify(serialized.properties)) };
  }

  return { open, leave, library, writes, frames };
}

const savedSlots = (characters, id = "character-a") =>
  characters.find((character) => character.id === id).loader_stacks.map((stack) => stack.slot);

// Same path as the Save connected loaders button: explicit save into the
// selected character, then the State Manager's library write.
async function saveConnectedLoaders(manager, tab, characterId = "character-a") {
  const { managerNode, loader } = tab;
  const state = structuredClone(managerNode.__dsm.state);
  const character = state.characters.find((item) => item.id === characterId);
  const result = manager.saveLoadersIntoCharacter(character, [loader]);
  manager.updateState(managerNode, state, managerNode.__dsm.uiState, {
    characterId,
    promptId: character.prompts[0].id,
    renameCommits: result.renameCommits,
    render: false,
  });
  await manager.flushPendingLibraryWrites().catch(() => {});
  return result;
}

const ownState = (loader) => ({
  slot: loader.properties.dora_state_slot,
  rows: loader.properties.dora_power_lora.rows,
  globals: loader.properties.dora_power_lora.globals,
});

function queuedLoaderInputs(manager, managerNode, loader, uiState) {
  managerNode.widgets[1].value = manager.serializeWorkflowUiState(uiState);
  managerNode.__dsm.uiState = uiState;
  const loaderInputs = { state_control: ["1", 7] };
  for (const widget of loader.widgets) {
    if (/^LORA_\d+$/.test(widget.name)) loaderInputs[widget.name] = widget.serializeValue();
  }
  const promptPayload = {
    output: {
      "1": { class_type: "StateManager", inputs: { ui_state_json: managerNode.widgets[1].value } },
      "2": { class_type: "DoraPowerLoraLoader", inputs: loaderInputs },
    },
    workflow: { nodes: [] },
  };
  manager.mutatePromptForStateManagers(promptPayload, 0, 1);
  return promptPayload.output["2"].inputs;
}

test("switching between workflows bound to the same preset keeps each loader's own state", async () => {
  const extensions = await loadExtensions();
  const { manager } = extensions;
  const harness = makeHarness(extensions);

  let tab = await harness.open(WORKFLOW_A);
  const expectedA = structuredClone(ownState(tab.loader));
  assert.deepEqual(expectedA.rows, WORKFLOW_A.loader.dora_power_lora.rows);
  const savedA = harness.leave(tab);

  tab = await harness.open(WORKFLOW_B);
  const expectedB = structuredClone(ownState(tab.loader));
  assert.deepEqual(expectedB.rows, WORKFLOW_B.loader.dora_power_lora.rows);
  assert.equal(expectedB.globals.stack_enabled, false);
  const savedB = harness.leave(tab);

  tab = await harness.open(savedA);
  assert.deepEqual(ownState(tab.loader), expectedA, "workflow A reopened with its own rows, globals, and slot");
  harness.leave(tab);

  tab = await harness.open(savedB);
  assert.deepEqual(ownState(tab.loader), expectedB, "workflow B reopened with its own rows, globals, and slot");

  assert.deepEqual(harness.writes, [], "opening and switching workflows never writes the shared library");
  assert.deepEqual(harness.library.characters, [presetCharacter()]);

  const ownRows = (inputs) => Object.keys(inputs).filter((key) => key.startsWith("LORA_")).map((key) => inputs[key].lora);
  for (const uiState of [{}, { queue_prompt_wildcard: true }, { queue_randomize_saved_seed: true }]) {
    const inputs = queuedLoaderInputs(manager, tab.managerNode, tab.loader, uiState);
    assert.deepEqual(inputs.state_control, ["1", 7], JSON.stringify(uiState));
    assert.deepEqual(ownRows(inputs), ["workflow-b.safetensors"], JSON.stringify(uiState));
  }
  const wildcard = queuedLoaderInputs(manager, tab.managerNode, tab.loader, {
    queue_character_wildcard: true,
    queue_character_ids: ["character-a"],
  });
  assert.equal(wildcard.state_control.loader_override, true);
  assert.deepEqual(wildcard.state_control.loras.map((row) => row.name), ["preset.safetensors"]);
  harness.leave(tab);
});

const renameSlot = (loader, slot) => loader.widgets.find((widget) => widget.name === "state_slot").callback(slot);
const renameOf = (loader) => loader.properties.dora_state_slot_rename;
const savedStack = (characters, slot, id = "character-a") =>
  characters.find((character) => character.id === id).loader_stacks.find((stack) => stack.slot === slot);
const rowNames = (stack) => stack.loras.map((row) => row.name);

test("a renamed loader's stack moves only after the library write commits, including after a rejected write", async () => {
  const extensions = await loadExtensions();
  const { manager } = extensions;
  const harness = makeHarness(extensions);
  const tab = await harness.open(WORKFLOW_A);

  renameSlot(tab.loader, "style");
  renameSlot(tab.loader, "hair");
  assert.equal(renameOf(tab.loader).from, "face", "the rename keeps the last saved slot across further renames");

  harness.library.putOutcomes.push("conflict");
  await saveConnectedLoaders(manager, tab);
  assert.equal(manager.stateLibraryClient.blocked, true);
  assert.equal(renameOf(tab.loader)?.from, "face", "a rejected write keeps the rename pending");
  assert.deepEqual(savedSlots(harness.library.characters), ["face"]);

  const writesBeforeBlockedSave = harness.writes.length;
  await saveConnectedLoaders(manager, tab);
  assert.equal(harness.writes.length, writesBeforeBlockedSave, "a save while writes are blocked sends nothing");
  assert.equal(renameOf(tab.loader)?.from, "face", "a blocked save keeps the rename pending");

  await manager.reloadStateLibrary(tab.managerNode);
  await saveConnectedLoaders(manager, tab);
  assert.deepEqual(savedSlots(harness.library.characters), ["hair"], "the retried save moves the stack instead of duplicating it");
  assert.deepEqual(rowNames(savedStack(harness.library.characters, "hair")), ["face.safetensors", "style.safetensors"]);
  assert.equal(renameOf(tab.loader), undefined, "the committed write acknowledges the rename");

  await saveConnectedLoaders(manager, tab);
  assert.deepEqual(savedSlots(harness.library.characters), ["hair"]);

  renameSlot(tab.loader, "face");
  renameSlot(tab.loader, "hair");
  assert.equal(renameOf(tab.loader), undefined, "renaming back to the saved slot clears the rename");

  const character = structuredClone(harness.library.characters[0]);
  tab.loader.properties.dora_power_lora.rows[0].strengthModel = 0.1;
  assert.equal(manager.syncCharacterLoaderStacksToConnectedNodes(tab.managerNode, character), 1);
  assert.equal(tab.loader.properties.dora_power_lora.rows[0].strengthModel, 0.85, "applying the preset uses the moved stack");

  const wildcard = queuedLoaderInputs(manager, tab.managerNode, tab.loader, {
    queue_character_wildcard: true,
    queue_character_ids: ["character-a"],
  });
  assert.equal(wildcard.state_control.loader_override, true);
  assert.equal(wildcard.state_control.loader_stacks[0].slot, "hair");
  assert.deepEqual(wildcard.state_control.loras.map((row) => row.name), ["face.safetensors", "style.safetensors"]);
  harness.leave(tab);
});

test("a rename onto an occupied slot keeps both stacks even when the write is rejected", async () => {
  const extensions = await loadExtensions();
  const { manager } = extensions;
  const harness = makeHarness(extensions);
  harness.library.characters[0].loader_stacks.push({
    slot: "outfit",
    label: "Outfit",
    loras: [{ enabled: true, name: "outfit.safetensors", strength_model: 1, strength_clip: 1 }],
    loader_globals: {},
  });
  const tab = await harness.open(WORKFLOW_A);

  renameSlot(tab.loader, "outfit");
  harness.library.putOutcomes.push("conflict");
  const result = await saveConnectedLoaders(manager, tab);
  assert.equal(result.notes.length, 1);
  assert.equal(tab.loader.properties.dora_state_slot, "face", "the loader returns to its saved slot");
  assert.equal(renameOf(tab.loader)?.from, "face", "the rejected write leaves the rename unacknowledged");
  assert.deepEqual(savedSlots(harness.library.characters), ["face", "outfit"]);
  assert.deepEqual(rowNames(savedStack(harness.library.characters, "face")), ["preset.safetensors"]);

  await manager.reloadStateLibrary(tab.managerNode);
  await saveConnectedLoaders(manager, tab);
  assert.deepEqual(savedSlots(harness.library.characters), ["face", "outfit"]);
  assert.deepEqual(rowNames(savedStack(harness.library.characters, "face")), ["face.safetensors", "style.safetensors"]);
  assert.deepEqual(rowNames(savedStack(harness.library.characters, "outfit")), ["outfit.safetensors"]);
  assert.equal(renameOf(tab.loader), undefined);
  harness.leave(tab);
});

const editFirstRow = (loader, strengthModel) => {
  const rows = structuredClone(loader.properties.dora_power_lora.rows);
  rows[0].strengthModel = strengthModel;
  loader.properties.dora_power_lora.rows = rows;
  loader._doraRows = rows;
};

test("a rename saved while its tab is closed resolves from the workflow alone, in the same session and after a restart", async () => {
  for (const restart of [false, true]) {
    let extensions = await loadExtensions();
    let harness = makeHarness(extensions);
    let tab = await harness.open(WORKFLOW_A);

    renameSlot(tab.loader, "hair");
    let release;
    harness.library.putOutcomes.push({ deferred: new Promise((resolve) => { release = resolve; }) });
    const pendingSave = saveConnectedLoaders(extensions.manager, tab);
    const savedA = harness.leave(tab);
    assert.deepEqual(
      [savedA.loader.dora_state_slot_rename.from, savedA.loader.dora_state_slot_rename.savedTo],
      ["face", "hair"],
      "the tab was serialized before the write committed",
    );

    tab = await harness.open(WORKFLOW_B);
    release();
    await pendingSave;
    assert.deepEqual(savedSlots(harness.library.characters), ["hair"]);
    harness.leave(tab);

    if (restart) {
      // Fresh module instances: nothing from the earlier session survives
      // except the server library and the serialized workflow.
      extensions = await loadExtensions();
      harness = makeHarness(extensions, harness.library);
    }
    tab = await harness.open(savedA);
    editFirstRow(tab.loader, 0.33);
    await saveConnectedLoaders(extensions.manager, tab);
    const label = restart ? "after restart" : "same session";
    assert.deepEqual(savedSlots(harness.library.characters), ["hair"], `${label}: no duplicate and no resurrected slot`);
    assert.equal(savedStack(harness.library.characters, "hair").loras[0].strength_model, 0.33, `${label}: the edit is saved`);
    assert.equal(renameOf(tab.loader), undefined, `${label}: the committed save acknowledges the rename`);
    harness.leave(tab);
  }
});

test("overlapping saves of a loader renamed twice persist exactly one stack", async () => {
  for (const firstWrite of ["ok", "conflict"]) {
    const extensions = await loadExtensions();
    const { manager } = extensions;
    const harness = makeHarness(extensions);
    const tab = await harness.open(WORKFLOW_A);

    renameSlot(tab.loader, "hair");
    let release;
    const held = new Promise((resolve) => { release = resolve; });
    harness.library.putOutcomes.push({ deferred: held, then: firstWrite });
    const firstSave = saveConnectedLoaders(manager, tab);
    renameSlot(tab.loader, "style");
    const secondSave = saveConnectedLoaders(manager, tab);
    release();
    await Promise.all([firstSave, secondSave]);

    if (firstWrite === "ok") {
      assert.deepEqual(savedSlots(harness.library.characters), ["style"], "the second save moves the first save's stack");
      assert.equal(renameOf(tab.loader), undefined, "both acknowledgments match the writes that held them");
    } else {
      assert.deepEqual(savedSlots(harness.library.characters), ["face"], "a rejected first write drops the queued second write");
      assert.equal(renameOf(tab.loader)?.from, "face");
      await manager.reloadStateLibrary(tab.managerNode);
      await saveConnectedLoaders(manager, tab);
      assert.deepEqual(savedSlots(harness.library.characters), ["style"]);
      assert.equal(renameOf(tab.loader), undefined);
    }
    assert.deepEqual(rowNames(savedStack(harness.library.characters, "style")), ["face.safetensors", "style.safetensors"]);
    harness.leave(tab);
  }
});

test("a rename onto an identical saved stack removes the stale source", async () => {
  const extensions = await loadExtensions();
  const { manager } = extensions;
  const harness = makeHarness(extensions);
  const tab = await harness.open(WORKFLOW_A);
  const own = tab.loader.properties.dora_power_lora;
  harness.library.characters[0].loader_stacks.push({
    slot: "hair",
    label: "Hair",
    loras: own.rows.map((row) => ({
      enabled: row.enabled,
      name: row.name,
      strength_model: row.strengthModel,
      strength_clip: row.strengthClip,
    })),
    loader_globals: structuredClone(own.globals),
  });
  await manager.reloadStateLibrary(tab.managerNode);
  const hairBefore = structuredClone(savedStack(harness.library.characters, "hair").loras);

  renameSlot(tab.loader, "hair");
  const result = await saveConnectedLoaders(manager, tab);
  assert.deepEqual(result.notes, []);
  assert.deepEqual(savedSlots(harness.library.characters), ["hair"]);
  assert.deepEqual(savedStack(harness.library.characters, "hair").loras, hairBefore);
  assert.equal(renameOf(tab.loader), undefined);
  harness.leave(tab);
});

test("a renamed loader never overwrites a different stack when its previous slot is not in the preset", async () => {
  const extensions = await loadExtensions();
  const { manager } = extensions;
  const harness = makeHarness(extensions);
  const characterB = {
    id: "character-b",
    name: "Character B",
    loader_stacks: [{
      slot: "style",
      label: "Style",
      loras: [{ enabled: true, name: "saved-style.safetensors", strength_model: 0.9, strength_clip: 0.9 }],
      loader_globals: {},
    }],
    prompts: [{ id: "prompt-b", name: "Prompt B", positive: "b", negative: "", settings: {} }],
  };
  harness.library.characters[0].loader_stacks.push(structuredClone(characterB.loader_stacks[0]));
  harness.library.characters.push(characterB);
  const savedStacks = () => harness.library.characters.map((character) => [
    character.id,
    character.loader_stacks.map((stack) => [stack.slot, rowNames(stack)]),
  ]);
  const stacksBefore = savedStacks();
  const workflow = structuredClone(WORKFLOW_A);
  workflow.loader.dora_state_slot = "new";
  const tab = await harness.open(workflow);
  const rowsBefore = structuredClone(tab.loader.properties.dora_power_lora.rows);

  renameSlot(tab.loader, "style");
  let result = await saveConnectedLoaders(manager, tab);
  assert.equal(result.saved, 0);
  assert.match(result.notes[0], /Not saved/);
  assert.deepEqual(savedStacks(), stacksBefore, "the occupied stack is untouched");
  assert.equal(tab.loader.properties.dora_state_slot, "style");
  assert.deepEqual(tab.loader.properties.dora_power_lora.rows, rowsBefore);
  assert.equal(renameOf(tab.loader)?.from, "new", "a rejected save keeps the rename");

  // Select another preset (as prompt selection does, without applying stacks)
  // and save into it: the rename still cannot claim its occupied slot.
  const state = structuredClone(tab.managerNode.__dsm.state);
  manager.updateState(tab.managerNode, state, tab.managerNode.__dsm.uiState, {
    characterId: "character-b",
    promptId: "prompt-b",
    render: false,
  });
  result = await saveConnectedLoaders(manager, tab, "character-b");
  assert.equal(result.saved, 0);
  assert.deepEqual(savedStacks(), stacksBefore);
  assert.equal(renameOf(tab.loader)?.from, "new");
  harness.leave(tab);
});
