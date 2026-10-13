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
    "\nexport { app, capturedExtension, initializeNode, stateLibraryClient, serializeBinding, serializeWorkflowUiState, mutatePromptForStateManagers, saveLoadersIntoCharacter, syncCharacterLoaderStacksToConnectedNodes };\n",
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

function makeHarness({ manager, LoaderNode }) {
  const frames = new Map();
  let nextFrame = 1;
  globalThis.requestAnimationFrame = (callback) => {
    const id = nextFrame++;
    frames.set(id, callback);
    return id;
  };
  globalThis.cancelAnimationFrame = (id) => frames.delete(id);

  const library = { revision: 1, characters: [presetCharacter()] };
  const writes = [];
  globalThis.__dsmTestFetchApi = async (path, options = {}) => {
    const method = options.method || "GET";
    if (method !== "GET") writes.push({ path, method });
    return {
      ok: true,
      async json() {
        return { version: 2, revision: library.revision, characters: structuredClone(library.characters), user_id: "default" };
      },
    };
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

test("saving a renamed loader moves its stack, and an occupied slot is never overwritten", async () => {
  const extensions = await loadExtensions();
  const { manager } = extensions;
  const harness = makeHarness(extensions);
  const tab = await harness.open(WORKFLOW_A);
  const character = presetCharacter();
  character.loader_stacks.push({
    slot: "outfit",
    label: "Outfit",
    loras: [{ enabled: true, name: "outfit.safetensors", strength_model: 1, strength_clip: 1 }],
    loader_globals: {},
  });
  const slotWidget = tab.loader.widgets.find((widget) => widget.name === "state_slot");
  const slots = () => character.loader_stacks.map((stack) => stack.slot);

  slotWidget.callback("style");
  slotWidget.callback("hair");
  assert.equal(tab.loader.properties.dora_state_slot_previous, "face", "lineage keeps the last saved slot across renames");
  let result = manager.saveLoadersIntoCharacter(character, [tab.loader]);
  assert.deepEqual(result.notes, []);
  assert.deepEqual(slots(), ["hair", "outfit"], "rename to a new slot moves the saved stack");
  assert.deepEqual(character.loader_stacks[0].loras.map((row) => row.name), ["face.safetensors", "style.safetensors"]);
  assert.equal(tab.loader.properties.dora_state_slot_previous, undefined);

  slotWidget.callback("outfit");
  result = manager.saveLoadersIntoCharacter(character, [tab.loader]);
  assert.equal(result.notes.length, 1);
  assert.deepEqual(slots(), ["hair", "outfit"], "rename onto an occupied slot keeps both stacks");
  assert.deepEqual(character.loader_stacks[1].loras.map((row) => row.name), ["outfit.safetensors"]);
  assert.equal(tab.loader.properties.dora_state_slot, "hair", "the loader stays on its previous slot");
  assert.equal(tab.loader.properties.dora_state_slot_previous, undefined);

  slotWidget.callback("face");
  slotWidget.callback("hair");
  assert.equal(tab.loader.properties.dora_state_slot_previous, undefined, "renaming back to the saved slot clears lineage");

  tab.loader.properties.dora_power_lora.rows[0].strengthModel = 0.1;
  assert.equal(manager.syncCharacterLoaderStacksToConnectedNodes(tab.managerNode, character), 1);
  assert.equal(tab.loader.properties.dora_power_lora.rows[0].strengthModel, 0.85, "applying the preset uses the moved stack");

  manager.stateLibraryClient.state = { version: 2, characters: [character] };
  tab.managerNode.__dsm.state = { version: 2, characters: [character] };
  const wildcard = queuedLoaderInputs(manager, tab.managerNode, tab.loader, {
    queue_character_wildcard: true,
    queue_character_ids: ["character-a"],
  });
  assert.equal(wildcard.state_control.loader_override, true);
  assert.equal(wildcard.state_control.loader_stacks[0].slot, "hair");
  assert.deepEqual(wildcard.state_control.loras.map((row) => row.name), ["face.safetensors", "style.safetensors"]);
  harness.leave(tab);
});
