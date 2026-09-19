import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";


async function loadStateManagerHelpers() {
  const sourceUrl = new URL("../web/dora_state_manager.js", import.meta.url);
  let source = await readFile(sourceUrl, "utf8");
  source = source
    .replace('import { app } from "../../scripts/app.js";', "let capturedExtension = null; const app = { registerExtension(value) { capturedExtension = value; }, graph: { extra: {} } };")
    .replace('import { api } from "../../scripts/api.js";', "const api = { fetchApi(...args) { if (typeof globalThis.__dsmTestFetchApi === 'function') return globalThis.__dsmTestFetchApi(...args); throw new Error('not used'); }, apiURL(value) { return value; } };")
    .replace('import "../../scripts/domWidget.js";', "");
  source += `\nexport { app, capturedExtension, defaultBinding, defaultState, deletePromptPreset, deleteStateCharacter, makeId, materializeEditedDefault, mergeScheduledLibraryUpdate, persistentCharacters, serializeBinding, serializeWorkflowUiState, serializeQueuedUiStateOverride, parseLegacyEmbeddedState, normalizeSelectionIdentity, readSelectionMirror, readLocalSelection, writeLocalSelection, writeSelectionMirror, configuredSelectionIdentity, selectionResolutionForLibraryLoad, selectionIdentityForLibraryLoad, authoritativeSelectionIdentity, rememberAuthoritativeSelection, captureStateManagerWorkflowState, updateState, initializeNode, stateLibraryClient, stateViewForSelection, syncCharacterLoaderStacksToConnectedNodes, syncConnectedLoaderStateIntoManager, synchronizeConnectedLoadersAfterLibraryLoad, restoreNodeAndConnectedLoadersFromLibrary, normalizeLoaderGlobals, pickPrimarySettingsLoaderStack, refreshAllNodesFromLibrary, normalizePromptDocument, previewPromptTransportText, requestLogicalTimelineSkeleton, updateManagedStateTextBox, updateManagedPromptDocument, mutatePromptForStateManagers, writePendingLibrary, flushPendingLibraryWrites, validateManagedQueueTextState, prepareStateManagerQueuePayload };\n`;
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


test("startup selection follows the values ComfyUI restored before onConfigure", async () => {
  const helpers = await loadStateManagerHelpers();
  const node = {
    widgets: [
      { name: "state_json", value: helpers.serializeBinding() },
      { name: "ui_state_json", value: helpers.serializeWorkflowUiState({}) },
      { name: "selected_character_id", value: "character-a" },
      { name: "selected_prompt_id", value: "prompt-a" },
    ],
    properties: {
      dora_state_manager_selection_v1: {
        version: 1,
        character_id: "stale-character",
        prompt_id: "stale-prompt",
      },
    },
  };
  const serialized = {
    widgets_values: [
      helpers.serializeBinding(),
      helpers.serializeWorkflowUiState({}),
      "character-a",
      "prompt-a",
    ],
    properties: structuredClone(node.properties),
  };

  assert.deepEqual(
    helpers.selectionIdentityForLibraryLoad(node, serialized),
    { characterId: "character-a", promptId: "prompt-a" },
  );
});


test("startup selection does not let a stale named shadow override ComfyUI's restored positional widgets", async () => {
  const helpers = await loadStateManagerHelpers();
  const node = {
    widgets: [
      { name: "state_json", value: helpers.serializeBinding() },
      { name: "ui_state_json", value: helpers.serializeWorkflowUiState({}) },
      { name: "selected_character_id", value: "character-a" },
      { name: "selected_prompt_id", value: "prompt-a" },
    ],
    properties: {},
  };
  const serialized = {
    widgets_values: [
      helpers.serializeBinding(),
      helpers.serializeWorkflowUiState({}),
      "character-a",
      "prompt-a",
    ],
    widgets_values_named: {
      selected_character_id: "default_character",
      selected_prompt_id: "default_prompt",
    },
    properties: {},
  };

  assert.deepEqual(
    helpers.selectionIdentityForLibraryLoad(node, serialized),
    { characterId: "character-a", promptId: "prompt-a" },
  );
});


test("startup selection follows named restoration when ComfyUI already applied it", async () => {
  const helpers = await loadStateManagerHelpers();
  const node = {
    widgets: [
      { name: "state_json", value: helpers.serializeBinding() },
      { name: "ui_state_json", value: helpers.serializeWorkflowUiState({}) },
      { name: "selected_character_id", value: "character-named" },
      { name: "selected_prompt_id", value: "prompt-named" },
    ],
    properties: {},
  };
  const serialized = {
    widgets_values: [
      helpers.serializeBinding(),
      helpers.serializeWorkflowUiState({}),
      "character-positional",
      "prompt-positional",
    ],
    widgets_values_named: {
      selected_character_id: "character-named",
      selected_prompt_id: "prompt-named",
    },
    properties: {},
  };

  assert.deepEqual(
    helpers.selectionIdentityForLibraryLoad(node, serialized),
    { characterId: "character-named", promptId: "prompt-named" },
  );
});


test("startup selection mirror rescues a configured preset when live widgets are still defaults", async () => {
  const helpers = await loadStateManagerHelpers();
  const node = {
    widgets: [
      { name: "state_json", value: helpers.serializeBinding() },
      { name: "ui_state_json", value: helpers.serializeWorkflowUiState({}) },
      { name: "selected_character_id", value: "default_character" },
      { name: "selected_prompt_id", value: "default_prompt" },
    ],
    properties: {
      dora_state_manager_selection_v1: {
        version: 1,
        character_id: "character-a",
        prompt_id: "prompt-a",
      },
    },
  };

  assert.deepEqual(
    helpers.selectionIdentityForLibraryLoad(node),
    { characterId: "character-a", promptId: "prompt-a" },
  );
});


test("nondefault mirror repairs serialized defaults left by the startup race", async () => {
  const helpers = await loadStateManagerHelpers();
  const node = {
    widgets: [
      { name: "state_json", value: helpers.serializeBinding() },
      { name: "ui_state_json", value: helpers.serializeWorkflowUiState({}) },
      { name: "selected_character_id", value: "default_character" },
      { name: "selected_prompt_id", value: "default_prompt" },
    ],
    properties: {
      dora_state_manager_selection_v1: {
        version: 1,
        character_id: "character-a",
        prompt_id: "prompt-a",
      },
    },
  };
  const serialized = {
    widgets_values_named: {
      selected_character_id: "default_character",
      selected_prompt_id: "default_prompt",
    },
    properties: structuredClone(node.properties),
  };

  assert.deepEqual(
    helpers.selectionIdentityForLibraryLoad(node, serialized),
    { characterId: "character-a", promptId: "prompt-a" },
  );
});

test("distribution-safe startup migrates the PR82 workflow mirror before local binding exists", async () => {
  const helpers = await loadStateManagerHelpers();
  const previousStorage = globalThis.localStorage;
  const storage = new Map();
  globalThis.localStorage = {
    getItem(key) { return storage.has(key) ? storage.get(key) : null; },
    setItem(key, value) { storage.set(key, String(value)); },
    removeItem(key) { storage.delete(key); },
  };
  const node = {
    id: 42,
    widgets: [
      { name: "state_json", value: helpers.serializeBinding() },
      { name: "ui_state_json", value: helpers.serializeWorkflowUiState({}) },
      { name: "selected_character_id", value: "default_character" },
      { name: "selected_prompt_id", value: "default_prompt" },
    ],
    properties: {
      dora_state_manager_distribution_safe_serialization: true,
      dora_state_manager_selection_v1: {
        version: 1,
        character_id: "character-a",
        prompt_id: "prompt-a",
      },
    },
  };
  const serialized = {
    id: 42,
    widgets_values_named: {
      selected_character_id: "default_character",
      selected_prompt_id: "default_prompt",
    },
    properties: structuredClone(node.properties),
  };

  try {
    assert.deepEqual(
      helpers.selectionIdentityForLibraryLoad(node, serialized),
      { characterId: "character-a", promptId: "prompt-a" },
    );
  } finally {
    if (previousStorage === undefined) delete globalThis.localStorage;
    else globalThis.localStorage = previousStorage;
  }
});


test("distribution-safe startup restores only this browser's local selection", async () => {
  const helpers = await loadStateManagerHelpers();
  const previousStorage = globalThis.localStorage;
  const storage = new Map();
  globalThis.localStorage = {
    getItem(key) { return storage.has(key) ? storage.get(key) : null; },
    setItem(key, value) { storage.set(key, String(value)); },
    removeItem(key) { storage.delete(key); },
  };
  const node = {
    id: 42,
    widgets: [
      { name: "state_json", value: helpers.serializeBinding() },
      { name: "ui_state_json", value: helpers.serializeWorkflowUiState({}) },
      { name: "selected_character_id", value: "character-a" },
      { name: "selected_prompt_id", value: "prompt-a" },
    ],
    properties: {
      dora_state_manager_distribution_safe_serialization: true,
      dora_state_manager_local_selection_binding_v1: "opaque-binding",
    },
  };

  try {
    helpers.writeSelectionMirror(node, "character-a", "prompt-a");
    assert.equal(node.properties.dora_state_manager_selection_v1, undefined);
    assert.deepEqual(
      helpers.readLocalSelection(node),
      { characterId: "character-a", promptId: "prompt-a" },
    );

    const serialized = {
      id: 42,
      widgets_values_named: {
        selected_character_id: "default_character",
        selected_prompt_id: "default_prompt",
      },
      properties: structuredClone(node.properties),
    };
    assert.deepEqual(
      helpers.selectionIdentityForLibraryLoad(node, serialized),
      { characterId: "character-a", promptId: "prompt-a" },
    );

    storage.clear();
    // A different browser/machine first restores the distribution-safe workflow
    // placeholders into the live widgets before onConfigure runs.
    node.widgets[2].value = "default_character";
    node.widgets[3].value = "default_prompt";
    assert.deepEqual(
      helpers.selectionIdentityForLibraryLoad(node, serialized),
      { characterId: "default_character", promptId: "default_prompt" },
    );
  } finally {
    if (previousStorage === undefined) delete globalThis.localStorage;
    else globalThis.localStorage = previousStorage;
  }
});


test("selection mirror is updated with the selected persistent preset", async () => {
  const helpers = await loadStateManagerHelpers();
  const node = { properties: {} };
  helpers.writeSelectionMirror(node, "character-a", "prompt-a");
  assert.deepEqual(node.properties.dora_state_manager_selection_v1, {
    version: 1,
    character_id: "character-a",
    prompt_id: "prompt-a",
  });
});


test("State Manager DOM mutations capture only workflow-relevant post-click state", async () => {
  const helpers = await loadStateManagerHelpers();
  const character = privateCharacter("character-a", "Character A", "saved");
  const promptId = character.prompts[0].id;
  const widgets = [
    { name: "state_json", value: helpers.serializeBinding() },
    { name: "ui_state_json", value: helpers.serializeWorkflowUiState({}) },
    { name: "selected_character_id", value: "default_character" },
    { name: "selected_prompt_id", value: "default_prompt" },
  ];
  const snapshots = [{
    characterId: widgets[2].value,
    promptId: widgets[3].value,
    mirror: null,
  }];
  const previousCanvas = helpers.app.canvas;
  helpers.app.canvas = {
    emitBeforeChange() {},
    emitAfterChange() {
      snapshots.push({
        characterId: widgets[2].value,
        promptId: widgets[3].value,
        mirror: structuredClone(node.properties.dora_state_manager_selection_v1),
      });
    },
  };
  const node = {
    widgets,
    properties: {},
    setDirtyCanvas() {},
    graph: { change() {} },
  };

  try {
    helpers.updateState(
      node,
      { version: 3, characters: [character] },
      {},
      {
        characterId: character.id,
        promptId,
        persist: false,
        render: false,
      },
    );
  } finally {
    helpers.app.canvas = previousCanvas;
  }

  // ComfyUI's global mouseup capture occurs before the DOM click handler. The
  // explicit transaction emitted by updateState() must therefore produce a
  // second snapshot after the selected UUIDs and mirror have been updated.
  assert.deepEqual(snapshots[0], {
    characterId: "default_character",
    promptId: "default_prompt",
    mirror: null,
  });
  assert.deepEqual(snapshots[1], {
    characterId: "character-a",
    promptId,
    mirror: {
      version: 1,
      character_id: "character-a",
      prompt_id: promptId,
    },
  });

  // Persistent library edits are backend-authoritative and are not serialized
  // into the workflow. They must not force a whole-graph snapshot on each DOM
  // input event (for example while typing in a prompt textarea).
  character.name = "Character A renamed";
  helpers.updateState(
    node,
    { version: 3, characters: [character] },
    {},
    {
      characterId: character.id,
      promptId,
      persist: false,
      render: false,
    },
  );
  assert.equal(snapshots.length, 2);
});


test("State Manager workflow hydration uses the loadedGraphNode lifecycle barrier", async () => {
  const source = await readFile(new URL("../web/dora_state_manager.js", import.meta.url), "utf8");
  const initializeIndex = source.indexOf("function initializeNode");
  const queueIndex = source.indexOf("function queueSessionTotalFromArguments");
  const block = source.slice(initializeIndex, queueIndex);
  assert.match(block, /__dsmScheduleNewNodeLibraryLoad/);
  assert.match(block, /onConfigure"[\s\S]*selectionResolutionForLibraryLoad/);
  assert.equal(block.includes("Promise.resolve().then(() => load(serializedNode))"), false);
  assert.match(
    source,
    /loadedGraphNode\(node\)[\s\S]*__dsmLoadedGraphSeen\s*=\s*true[\s\S]*__dsmLoadConfiguredLibrary/,
  );
});

test("late workflow configure cannot be poisoned by an already-completed default startup fetch", async () => {
  const helpers = await loadStateManagerHelpers();
  const previousRaf = globalThis.requestAnimationFrame;
  const previousCancel = globalThis.cancelAnimationFrame;
  const frames = new Map();
  let nextFrame = 1;
  globalThis.requestAnimationFrame = (callback) => {
    const id = nextFrame++;
    frames.set(id, callback);
    return id;
  };
  globalThis.cancelAnimationFrame = (id) => {
    frames.delete(id);
  };

  const savedCharacter = {
    id: "character-a",
    name: "Character A",
    prompts: [{
      id: "prompt-a",
      name: "Prompt A",
      positive: "saved",
      negative: "",
      text_boxes: [
        { role: "positive", slot: "default", label: "Default positive", text: "saved" },
        { role: "negative", slot: "default", label: "Default negative", text: "" },
      ],
      settings: {},
    }],
  };
  globalThis.__dsmTestFetchApi = async () => ({
    ok: true,
    async json() {
      return {
        version: 2,
        revision: 9,
        characters: [structuredClone(savedCharacter)],
        user_id: "default",
      };
    },
  });

  const widgets = [
    { name: "state_json", value: helpers.serializeBinding() },
    { name: "ui_state_json", value: helpers.serializeWorkflowUiState({}) },
    { name: "selected_character_id", value: "default_character" },
    { name: "selected_prompt_id", value: "default_prompt" },
  ];
  const node = {
    id: 42,
    type: "State Manager",
    comfyClass: "State Manager",
    widgets,
    properties: {},
    size: [820, 720],
    __dsm: {
      state: null,
      uiState: null,
      renderFrame: 0,
      postLoadSyncFrame: 0,
    },
    setSize() {},
    setDirtyCanvas() {},
    graph: { change() {} },
  };

  try {
    helpers.initializeNode(node, {});

    // Force the old failing ordering: allow an unconfigured default fetch to
    // finish before ComfyUI supplies the workflow payload. It may render the
    // built-in default, but it must not make that placeholder authoritative.
    node.__dsmLoadConfiguredLibrary();
    await new Promise((resolve) => setTimeout(resolve, 0));

    assert.equal(widgets[2].value, "default_character");
    assert.equal(widgets[3].value, "default_prompt");
    assert.equal(helpers.authoritativeSelectionIdentity(node), null);
    assert.equal(node.properties.dora_state_manager_selection_v1, undefined);

    const serialized = {
      widgets_values: [
        helpers.serializeBinding(),
        helpers.serializeWorkflowUiState({}),
        "character-a",
        "prompt-a",
      ],
      widgets_values_named: {
        state_json: helpers.serializeBinding(),
        ui_state_json: helpers.serializeWorkflowUiState({}),
        selected_character_id: "character-a",
        selected_prompt_id: "prompt-a",
      },
      properties: {},
    };

    // Real LGraphNode.configure() restores widgets before it invokes
    // onConfigure(). Reproduce that ordering instead of leaving constructor
    // defaults live while passing a contradictory serialized payload.
    widgets[2].value = "character-a";
    widgets[3].value = "prompt-a";
    node.onConfigure(serialized);

    // Selection restoration is synchronous at configure time. No API response
    // or animation frame may be required to stop post-load serialization from
    // seeing constructor defaults.
    assert.equal(widgets[2].value, "character-a");
    assert.equal(widgets[3].value, "prompt-a");
    assert.deepEqual(
      helpers.authoritativeSelectionIdentity(node),
      { characterId: "character-a", promptId: "prompt-a" },
    );

    helpers.capturedExtension.loadedGraphNode(node);
    await new Promise((resolve) => setTimeout(resolve, 0));

    assert.equal(widgets[2].value, "character-a");
    assert.equal(widgets[3].value, "prompt-a");
    assert.equal(node.__dsmLibraryHydrated, true);
    assert.equal(node.__dsm.state.characters[0].id, "character-a");
    assert.equal(node.__dsm.state.characters[0].prompts[0].id, "prompt-a");
  } finally {
    helpers.stateLibraryClient.nodes.delete(node);
    delete globalThis.__dsmTestFetchApi;
    if (previousRaf === undefined) delete globalThis.requestAnimationFrame;
    else globalThis.requestAnimationFrame = previousRaf;
    if (previousCancel === undefined) delete globalThis.cancelAnimationFrame;
    else globalThis.cancelAnimationFrame = previousCancel;
  }
});

test("managed State Manager text integration updates the authoritative selected prompt", async () => {
  const helpers = await loadStateManagerHelpers();
  assert.equal(globalThis.__doraStateManagerPromptApi?.contract_version, 5);
  assert.deepEqual(
    [...(globalThis.__doraStateManagerPromptApi?.capabilities || [])],
    [
      "authoritative_persistent_text_v1",
      "impact_wildcard_queue_bridge_v1",
      "backend_impact_prompt_bridge_v1",
      "backend_persistent_text_write_v1",
      "prompt_document_v1",
    ],
  );
  assert.equal(typeof globalThis.__doraStateManagerPromptApi?.setTextBox, "function");
  assert.equal(typeof globalThis.__doraStateManagerPromptApi?.setPromptDocument, "function");
  const state = {
    version: 3,
    characters: [{
      id: "character-a",
      name: "Character A",
      prompts: [{
        id: "prompt-a",
        name: "Prompt A",
        positive: "old prompt",
        negative: "",
        text_boxes: [
          { role: "positive", slot: "default", label: "Default positive", text: "old prompt" },
          { role: "negative", slot: "default", label: "Default negative", text: "" },
        ],
        settings: {},
      }],
    }],
  };
  const nodes = new Map();
  const graph = {
    links: {
      12: { origin_id: 1, origin_slot: 0, target_id: 2, target_slot: 0 },
    },
    getNodeById(id) { return nodes.get(id) || null; },
    change() {},
  };
  const manager = {
    id: 1,
    type: "State Manager",
    comfyClass: "State Manager",
    outputs: [{ name: "state_control", type: "STATE_MANAGER_CONTROL", links: [12] }],
    widgets: [
      { name: "state_json", value: helpers.serializeBinding() },
      { name: "ui_state_json", value: helpers.serializeWorkflowUiState({}) },
      { name: "selected_character_id", value: "character-a" },
      { name: "selected_prompt_id", value: "prompt-a" },
    ],
    properties: {},
    __dsm: { state, uiState: {}, renderFrame: 0 },
    graph,
  };
  const textWidget = { name: "text", value: "old prompt" };
  const textNode = {
    id: 2,
    type: "State Manager Text Box",
    comfyClass: "State Manager Text Box",
    inputs: [{ name: "state_control", type: "STATE_MANAGER_CONTROL", link: 12 }],
    outputs: [{ name: "text", type: "STRING", links: [] }],
    widgets: [
      { name: "role", value: "positive" },
      textWidget,
      { name: "state_slot", value: "default" },
    ],
    graph,
  };
  nodes.set(1, manager);
  nodes.set(2, textNode);

  const timeline = "Global.\n\n[0-7s]\nOne.\n\n[7-14s]\nTwo.";
  const result = await helpers.updateManagedStateTextBox(
    manager,
    textNode,
    timeline,
    { persist: false, render: false },
  );

  assert.equal(result.status, "updated");
  assert.equal(result.role, "positive");
  assert.equal(result.slot, "default");
  assert.equal(textWidget.value, timeline);
  const selected = manager.__dsm.state.characters
    .find((character) => character.id === "character-a")
    .prompts.find((prompt) => prompt.id === "prompt-a");
  assert.equal(selected.positive, timeline);
  assert.equal(
    selected.text_boxes.find((box) => box.role === "positive" && box.slot === "default").text,
    timeline,
  );
});


test("managed State Manager external write is server-confirmed before reporting success", async () => {
  const helpers = await loadStateManagerHelpers();
  const timeline = "Global.\n\n[0-7s]\nOne.\n\n[7-14s]\nTwo.";
  const state = {
    version: 3,
    characters: [{
      id: "character-a",
      name: "Character A",
      prompts: [{
        id: "prompt-a",
        name: "Prompt A",
        positive: "old prompt",
        negative: "",
        text_boxes: [
          { role: "positive", slot: "default", label: "Default positive", text: "old prompt" },
          { role: "negative", slot: "default", label: "Default negative", text: "" },
        ],
        settings: {},
      }],
    }],
  };
  const nodes = new Map();
  const graph = {
    links: {
      12: { origin_id: 1, origin_slot: 0, target_id: 2, target_slot: 0 },
    },
    getNodeById(id) { return nodes.get(id) || null; },
    change() {},
  };
  const manager = {
    id: 1,
    type: "State Manager",
    comfyClass: "State Manager",
    outputs: [{ name: "state_control", type: "STATE_MANAGER_CONTROL", links: [12] }],
    widgets: [
      { name: "state_json", value: helpers.serializeBinding() },
      { name: "ui_state_json", value: helpers.serializeWorkflowUiState({}) },
      { name: "selected_character_id", value: "character-a" },
      { name: "selected_prompt_id", value: "prompt-a" },
    ],
    properties: {},
    __dsm: { state: structuredClone(state), uiState: {}, renderFrame: 0 },
    graph,
  };
  const textWidget = { name: "text", value: "old prompt" };
  const textNode = {
    id: 2,
    type: "State Manager Text Box",
    comfyClass: "State Manager Text Box",
    title: "Sequence Prompt",
    inputs: [{ name: "state_control", type: "STATE_MANAGER_CONTROL", link: 12 }],
    outputs: [{ name: "text", type: "STRING", links: [] }],
    widgets: [
      { name: "role", value: "positive" },
      textWidget,
      { name: "state_slot", value: "default" },
    ],
    graph,
  };
  nodes.set(1, manager);
  nodes.set(2, textNode);

  helpers.stateLibraryClient.state = structuredClone(state);
  helpers.stateLibraryClient.revision = 7;
  helpers.stateLibraryClient.pending = [];
  helpers.stateLibraryClient.writing = false;
  helpers.stateLibraryClient.blocked = false;

  const calls = [];
  const previousRaf = globalThis.requestAnimationFrame;
  globalThis.requestAnimationFrame = () => 1;
  globalThis.__dsmTestFetchApi = async (path, options = {}) => {
    calls.push({ path, options });
    const request = JSON.parse(options.body);
    const persisted = structuredClone(state);
    const prompt = persisted.characters[0].prompts[0];
    prompt.positive = request.text;
    prompt.text_boxes[0] = {
      ...prompt.text_boxes[0],
      label: request.label,
      text: request.text,
    };
    return {
      ok: true,
      async json() {
        return {
          version: 1,
          revision: 8,
          characters: persisted.characters,
          user_id: "default",
        };
      },
    };
  };

  try {
    const result = await globalThis.__doraStateManagerPromptApi.setTextBox(
      manager,
      textNode,
      timeline,
    );
    assert.equal(result.status, "updated");
    assert.equal(result.persistent_verified, true);
    assert.equal(result.library_revision, 8);
    assert.equal(result.contract_version, 4);
    assert.equal(result.write_revision, "backend-write-v1");
    assert.equal(textWidget.value, timeline);
    assert.equal(calls.length, 1);
    assert.equal(
      calls[0].path,
      "/dora_dynamic_lora/state-library/characters/character-a/prompts/prompt-a/text-box",
    );
    assert.equal(calls[0].options.method, "PUT");
    const request = JSON.parse(calls[0].options.body);
    assert.equal(request.expected_revision, 7);
    assert.equal(request.role, "positive");
    assert.equal(request.slot, "default");
    assert.equal(request.label, "Sequence Prompt");
    assert.equal(request.text, timeline);
    assert.equal(
      helpers.stateLibraryClient.state.characters[0].prompts[0].text_boxes[0].text,
      timeline,
    );
  } finally {
    delete globalThis.__dsmTestFetchApi;
    if (previousRaf === undefined) delete globalThis.requestAnimationFrame;
    else globalThis.requestAnimationFrame = previousRaf;
  }
});


test("managed State Manager external write does not mutate local mirrors when backend persistence fails", async () => {
  const helpers = await loadStateManagerHelpers();
  const state = {
    version: 3,
    characters: [{
      id: "character-a",
      name: "Character A",
      prompts: [{
        id: "prompt-a",
        name: "Prompt A",
        positive: "old prompt",
        negative: "",
        text_boxes: [
          { role: "positive", slot: "default", label: "Default positive", text: "old prompt" },
          { role: "negative", slot: "default", label: "Default negative", text: "" },
        ],
        settings: {},
      }],
    }],
  };
  const nodes = new Map();
  const graph = {
    links: { 12: { origin_id: 1, origin_slot: 0, target_id: 2, target_slot: 0 } },
    getNodeById(id) { return nodes.get(id) || null; },
    change() {},
  };
  const manager = {
    id: 1,
    type: "State Manager",
    comfyClass: "State Manager",
    outputs: [{ name: "state_control", type: "STATE_MANAGER_CONTROL", links: [12] }],
    widgets: [
      { name: "state_json", value: helpers.serializeBinding() },
      { name: "ui_state_json", value: helpers.serializeWorkflowUiState({}) },
      { name: "selected_character_id", value: "character-a" },
      { name: "selected_prompt_id", value: "prompt-a" },
    ],
    properties: {},
    __dsm: { state: structuredClone(state), uiState: {}, renderFrame: 0 },
    graph,
  };
  const textWidget = { name: "text", value: "old prompt" };
  const textNode = {
    id: 2,
    type: "State Manager Text Box",
    comfyClass: "State Manager Text Box",
    inputs: [{ name: "state_control", type: "STATE_MANAGER_CONTROL", link: 12 }],
    outputs: [{ name: "text", type: "STRING", links: [] }],
    widgets: [
      { name: "role", value: "positive" },
      textWidget,
      { name: "state_slot", value: "default" },
    ],
    graph,
  };
  nodes.set(1, manager);
  nodes.set(2, textNode);
  helpers.stateLibraryClient.state = structuredClone(state);
  helpers.stateLibraryClient.revision = 3;
  helpers.stateLibraryClient.pending = [];
  helpers.stateLibraryClient.writing = false;
  helpers.stateLibraryClient.blocked = false;

  globalThis.__dsmTestFetchApi = async () => ({
    ok: false,
    status: 409,
    async json() {
      return { error: "revision conflict", code: "revision_conflict" };
    },
  });

  try {
    await assert.rejects(
      globalThis.__doraStateManagerPromptApi.setTextBox(manager, textNode, "new timeline"),
      /revision conflict/,
    );
    assert.equal(textWidget.value, "old prompt");
    assert.equal(manager.__dsm.state.characters[0].prompts[0].positive, "old prompt");
    assert.equal(
      helpers.stateLibraryClient.state.characters[0].prompts[0].text_boxes[0].text,
      "old prompt",
    );
  } finally {
    delete globalThis.__dsmTestFetchApi;
  }
});


test("ordinary queues preserve Impact links and carry request-local persistent identity", async () => {
  const helpers = await loadStateManagerHelpers();
  const timeline = "Global.\n\n[0-7s]\nOne.\n\n[7-14s]\nTwo.";
  helpers.stateLibraryClient.userId = "queue-user";
  const state = {
    version: 3,
    characters: [{
      id: "character-a",
      name: "Character A",
      prompts: [{
        id: "prompt-a",
        name: "Prompt A",
        positive: timeline,
        negative: "",
        text_boxes: [
          { role: "positive", slot: "default", label: "Default positive", text: timeline },
          { role: "negative", slot: "default", label: "Default negative", text: "" },
        ],
        settings: {},
      }],
    }],
  };

  for (const mode of ["fixed", "populate", "reproduce"]) {
    const manager = {
      id: 1,
      type: "State Manager",
      comfyClass: "State Manager",
      inputs: [],
      outputs: [{ name: "state_control", type: "STATE_MANAGER_CONTROL", links: [12] }],
      widgets: [
        { name: "state_json", value: helpers.serializeBinding() },
        { name: "ui_state_json", value: helpers.serializeWorkflowUiState({}) },
        { name: "selected_character_id", value: "character-a" },
        { name: "selected_prompt_id", value: "prompt-a" },
      ],
      __dsm: { state: structuredClone(state), uiState: {} },
    };
    const textNode = {
      id: 2,
      type: "State Manager Text Box",
      comfyClass: "State Manager Text Box",
      inputs: [{ name: "state_control", type: "STATE_MANAGER_CONTROL", link: 12 }],
      outputs: [{ name: "text", type: "STRING", links: [13] }],
      widgets: [
        { name: "role", value: "positive" },
        { name: "text", value: "stale local text" },
        { name: "state_slot", value: "default" },
      ],
    };
    const impact = {
      id: 3,
      type: "ImpactWildcardProcessor",
      comfyClass: "ImpactWildcardProcessor",
      inputs: [{ name: "wildcard_text", type: "STRING", link: 13 }],
      outputs: [{ name: "STRING", type: "STRING", links: [] }],
      widgets: [
        { name: "wildcard_text", value: "stale wildcard" },
        { name: "populated_text", value: "stale populated" },
        { name: "mode", value: mode },
        { name: "seed", value: 123 },
      ],
    };
    const nodes = [manager, textNode, impact];
    const graph = {
      _nodes: nodes,
      links: {
        12: { origin_id: 1, origin_slot: 0, target_id: 2, target_slot: 0 },
        13: { origin_id: 2, origin_slot: 0, target_id: 3, target_slot: 0 },
      },
      getNodeById(id) { return nodes.find((node) => node.id === id) || null; },
      change() {},
    };
    nodes.forEach((node) => { node.graph = graph; });
    helpers.app.graph = graph;

    const promptPayload = {
      output: {
        "1": {
          class_type: "StateManager",
          inputs: {
            state_json: helpers.serializeBinding(),
            ui_state_json: helpers.serializeWorkflowUiState({}),
            selected_character_id: "character-a",
            selected_prompt_id: "prompt-a",
          },
        },
        "2": {
          class_type: "StateManagerTextBox",
          inputs: {
            role: "positive",
            text: "stale local text",
            state_slot: "default",
            state_control: ["1", 7],
          },
        },
        "3": {
          class_type: "ImpactWildcardProcessor",
          inputs: {
            wildcard_text: ["2", 0],
            populated_text: "stale populated",
            mode,
            seed: 123,
          },
        },
      },
      workflow: { nodes: [] },
    };

    const changed = helpers.mutatePromptForStateManagers(promptPayload, 0, 1);
    assert.equal(changed, 2);
    assert.equal(promptPayload.output["2"].inputs.text, timeline);
    assert.deepEqual(promptPayload.output["3"].inputs.wildcard_text, ["2", 0]);
    assert.equal(promptPayload.output["3"].inputs.populated_text, "stale populated");
    assert.equal(promptPayload.output["3"].inputs.mode, mode);

    const queuedUi = JSON.parse(promptPayload.output["1"].inputs.ui_state_json);
    assert.equal(queuedUi.__dsm_library_user_id, "queue-user");
    assert.equal(queuedUi.__dsm_queued_runtime_character_id, "character-a");
    assert.equal(queuedUi.__dsm_queued_runtime_prompt_id, "prompt-a");
    assert.equal(queuedUi.__dsm_frontend_prompt_contract_version, 5);
    assert.equal(queuedUi.__dsm_frontend_prompt_contract_revision, "backend-document-write-v1");
    assert.equal(Object.prototype.hasOwnProperty.call(queuedUi, "__dsm_queued_runtime_nonce"), false);
    assert.equal(Object.prototype.hasOwnProperty.call(JSON.parse(manager.widgets[1].value), "__dsm_library_user_id"), false);
  }
  helpers.stateLibraryClient.userId = "default";
});


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


test("onSerialize never replaces an authoritative workflow selection with transient default widgets", async () => {
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

  const node = new StateManagerNode();
  node.properties = {};
  node.widgets = [
    { name: "state_json", value: helpers.serializeBinding() },
    { name: "ui_state_json", value: helpers.serializeWorkflowUiState({}) },
    { name: "selected_character_id", value: "default_character" },
    { name: "selected_prompt_id", value: "default_prompt" },
  ];
  node.__dsm = { state: helpers.defaultState(), uiState: {} };
  helpers.rememberAuthoritativeSelection(node, "character-a", "prompt-a");

  const output = {};
  node.onSerialize(output);

  assert.equal(node.widgets[2].value, "default_character");
  assert.equal(node.widgets[3].value, "default_prompt");
  assert.equal(output.widgets_values[2], "character-a");
  assert.equal(output.widgets_values[3], "prompt-a");
  assert.equal(output.widgets_values_named.selected_character_id, "character-a");
  assert.equal(output.widgets_values_named.selected_prompt_id, "prompt-a");
  assert.deepEqual(output.properties.dora_state_manager_selection_v1, {
    version: 1,
    character_id: "character-a",
    prompt_id: "prompt-a",
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
  assert.equal(parsed.__dsm_frontend_prompt_contract_version, 5);
  assert.equal(parsed.__dsm_frontend_prompt_contract_revision, "backend-document-write-v1");
  assert.equal(Object.prototype.hasOwnProperty.call(parsed, "__dsm_queued_runtime_state"), false);
  assert.equal(serialized.includes("Private Character"), false);
});


test("legacy embedded state remains detectable for controlled migration", async () => {
  const helpers = await loadStateManagerHelpers();
  const legacy = { version: 3, characters: [{ id: "legacy", name: "Legacy" }] };
  assert.deepEqual(helpers.parseLegacyEmbeddedState(JSON.stringify(legacy)), legacy);
  assert.equal(helpers.parseLegacyEmbeddedState(helpers.serializeBinding()), null);
});


test("browser persistence is selection-only and cannot resurrect a private library", async () => {
  const helpers = await loadStateManagerHelpers();
  const source = await readFile(new URL("../web/dora_state_manager.js", import.meta.url), "utf8");
  assert.equal(source.includes("tryRestoreStateBackup"), false);
  assert.equal(source.includes("writeStateBackup"), false);
  assert.equal(source.includes("dora_state_manager_backup_workflow_id"), true, "legacy metadata should only appear in the serialization scrubber");
  assert.match(source, /delete app\.graph\.extra\.dora_state_manager_backup_workflow_id/);
  assert.equal(source.includes("setWidgetValue(widgets.uiStateWidget, serializeUiState"), false);

  const previousStorage = globalThis.localStorage;
  const storage = new Map();
  globalThis.localStorage = {
    getItem(key) { return storage.has(key) ? storage.get(key) : null; },
    setItem(key, value) { storage.set(key, String(value)); },
    removeItem(key) { storage.delete(key); },
  };
  const node = {
    id: 42,
    properties: {
      dora_state_manager_distribution_safe_serialization: true,
      dora_state_manager_local_selection_binding_v1: "opaque-binding",
    },
  };
  try {
    assert.equal(helpers.writeLocalSelection(node, "character-a", "prompt-a"), true);
    assert.equal(storage.size, 1);
    const [key, raw] = [...storage.entries()][0];
    assert.match(key, /^dora_state_manager_local_selection_v1:opaque-binding:42$/);
    assert.deepEqual(Object.keys(JSON.parse(raw)).sort(), [
      "character_id",
      "prompt_id",
      "version",
    ]);
    assert.deepEqual(JSON.parse(raw), {
      version: 1,
      character_id: "character-a",
      prompt_id: "prompt-a",
    });
    assert.equal(raw.includes("characters"), false);
    assert.equal(raw.includes("text_boxes"), false);
    assert.equal(raw.includes("settings"), false);
    assert.equal(raw.includes("loras"), false);
  } finally {
    if (previousStorage === undefined) delete globalThis.localStorage;
    else globalThis.localStorage = previousStorage;
  }

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


test("v5 setPromptDocument persists exact descriptor while setTextBox remains v4", async () => {
  const helpers = await loadStateManagerHelpers();
  const timeline = "Shared.\n\n[0-5s]\nONE\n\n[5-10s]\nTWO\n\n[10-15s]\nTHREE";
  const descriptor = {
    schema_version: 1,
    format: "timeline",
    routing: "logical_chunks",
    geometry: { chunks: 3, chunk_seconds: "5" },
  };
  const state = {
    version: 3,
    characters: [{
      id: "character-a",
      name: "Character A",
      prompts: [{
        id: "prompt-a",
        name: "Prompt A",
        positive: "old prompt",
        negative: "",
        text_boxes: [
          { role: "positive", slot: "default", label: "Sequence Prompt", text: "old prompt" },
          { role: "negative", slot: "default", label: "Default negative", text: "" },
        ],
        settings: {},
      }],
    }],
  };
  const nodes = new Map();
  const graph = {
    links: { 12: { origin_id: 1, origin_slot: 0, target_id: 2, target_slot: 0 } },
    getNodeById(id) { return nodes.get(id) || null; },
    change() {},
  };
  const manager = {
    id: 1,
    type: "State Manager",
    comfyClass: "State Manager",
    outputs: [{ name: "state_control", type: "STATE_MANAGER_CONTROL", links: [12] }],
    widgets: [
      { name: "state_json", value: helpers.serializeBinding() },
      { name: "ui_state_json", value: helpers.serializeWorkflowUiState({}) },
      { name: "selected_character_id", value: "character-a" },
      { name: "selected_prompt_id", value: "prompt-a" },
    ],
    properties: {},
    __dsm: { state: structuredClone(state), uiState: {}, renderFrame: 0 },
    graph,
  };
  const textWidget = { name: "text", value: "old prompt" };
  const textNode = {
    id: 2,
    type: "State Manager Text Box",
    comfyClass: "State Manager Text Box",
    title: "Sequence Prompt",
    inputs: [{ name: "state_control", type: "STATE_MANAGER_CONTROL", link: 12 }],
    outputs: [{ name: "text", type: "STRING", links: [] }],
    widgets: [
      { name: "role", value: "positive" },
      textWidget,
      { name: "state_slot", value: "default" },
    ],
    graph,
  };
  nodes.set(1, manager);
  nodes.set(2, textNode);

  helpers.stateLibraryClient.state = structuredClone(state);
  helpers.stateLibraryClient.revision = 7;
  helpers.stateLibraryClient.pending = [];
  helpers.stateLibraryClient.writing = false;
  helpers.stateLibraryClient.blocked = false;

  const calls = [];
  const previousRaf = globalThis.requestAnimationFrame;
  globalThis.requestAnimationFrame = () => 1;
  globalThis.__dsmTestFetchApi = async (path, options = {}) => {
    calls.push({ path, options });
    const request = JSON.parse(options.body);
    const persisted = structuredClone(state);
    const box = persisted.characters[0].prompts[0].text_boxes[0];
    box.text = request.text;
    box.label = request.label;
    box.prompt_document = structuredClone(request.prompt_document);
    persisted.characters[0].prompts[0].positive = request.text;
    return {
      ok: true,
      async json() {
        return {
          status: "updated",
          role: request.role,
          slot: request.slot,
          character_id: "character-a",
          prompt_id: "prompt-a",
          text_sha256: "server-hash",
          prompt_document: structuredClone(request.prompt_document),
          library_revision: 8,
          persistent_verified: true,
          contract_version: 5,
          write_revision: "backend-document-write-v1",
          migrated_container_v2: true,
          snapshot: {
            version: 2,
            revision: 8,
            characters: persisted.characters,
            user_id: "default",
          },
          user_id: "default",
        };
      },
    };
  };

  try {
    const receipt = await globalThis.__doraStateManagerPromptApi.setPromptDocument(
      manager,
      textNode,
      { text: timeline, prompt_document: descriptor },
    );
    assert.equal(receipt.contract_version, 5);
    assert.equal(receipt.write_revision, "backend-document-write-v1");
    assert.equal(receipt.persistent_verified, true);
    assert.deepEqual(receipt.prompt_document, descriptor);
    assert.equal(calls.length, 1);
    assert.equal(
      calls[0].path,
      "/dora_dynamic_lora/state-library/characters/character-a/prompts/prompt-a/prompt-document",
    );
    const request = JSON.parse(calls[0].options.body);
    assert.equal(request.expected_revision, 7);
    assert.equal(request.text, timeline);
    assert.deepEqual(request.prompt_document, descriptor);
    assert.equal(textWidget.value, timeline);
    const persistedBox = helpers.stateLibraryClient.state.characters[0].prompts[0].text_boxes[0];
    assert.equal(persistedBox.text, timeline);
    assert.deepEqual(persistedBox.prompt_document, descriptor);
  } finally {
    delete globalThis.__dsmTestFetchApi;
    if (previousRaf === undefined) delete globalThis.requestAnimationFrame;
    else globalThis.requestAnimationFrame = previousRaf;
  }
});


test("ordinary bulk library writes advertise the v5 prompt-document capability", async () => {
  const helpers = await loadStateManagerHelpers();
  const node = { id: 99 };
  const base = [privateCharacter("character-a", "A", "old")];
  const desired = [privateCharacter("character-a", "A", "new")];

  helpers.stateLibraryClient.state = { version: 2, characters: structuredClone(base) };
  helpers.stateLibraryClient.revision = 7;
  helpers.stateLibraryClient.canonical = "__force_write__";
  helpers.stateLibraryClient.pending = [{
    node,
    baseCharacters: structuredClone(base),
    desiredCharacters: structuredClone(desired),
  }];
  helpers.stateLibraryClient.writing = false;
  helpers.stateLibraryClient.blocked = false;
  helpers.stateLibraryClient.lastAppliedNode = null;

  const calls = [];
  globalThis.__dsmTestFetchApi = async (path, options = {}) => {
    calls.push({ path, options });
    const request = JSON.parse(options.body);
    return {
      ok: true,
      async json() {
        return {
          version: 2,
          revision: 8,
          characters: structuredClone(request.characters),
          user_id: "default",
        };
      },
    };
  };

  try {
    await helpers.writePendingLibrary();
  } finally {
    delete globalThis.__dsmTestFetchApi;
  }

  assert.equal(calls.length, 1);
  assert.equal(calls[0].path, "/dora_dynamic_lora/state-library");
  const request = JSON.parse(calls[0].options.body);
  assert.equal(request.expected_revision, 7);
  assert.equal(request.contract_version, 5);
  assert.deepEqual(request.capabilities, ["prompt_document_v1"]);
  assert.equal(request.characters[0].prompts[0].positive, "new");
});


test("frontend prompt tools delegate parser and skeleton work to the backend provider", async () => {
  const helpers = await loadStateManagerHelpers();
  const calls = [];
  globalThis.__dsmTestFetchApi = async (path, options = {}) => {
    calls.push({ path, options });
    const request = JSON.parse(options.body);
    if (path.endsWith("/inspect")) {
      assert.equal(request.text, "[0-5s]\nONE");
      return {
        ok: true,
        async json() {
          return {
            contract_version: 5,
            available: true,
            valid: true,
            classification: "timeline",
            structure: { sections: [{ kind: "time" }] },
          };
        },
      };
    }
    assert.ok(path.endsWith("/logical-skeleton"));
    assert.deepEqual(request, { chunks: 2, chunk_seconds: "5" });
    return {
      ok: true,
      async json() {
        return {
          contract_version: 5,
          available: true,
          supported: true,
          text: "[0-5s]\n\n[5-10s]\n",
        };
      },
    };
  };

  try {
    const preview = await helpers.previewPromptTransportText("[0-5s]\nONE");
    assert.equal(preview.valid, true);
    assert.equal(preview.classification, "timeline");
    const skeleton = await helpers.requestLogicalTimelineSkeleton(2, "5");
    assert.equal(skeleton.text, "[0-5s]\n\n[5-10s]\n");
  } finally {
    delete globalThis.__dsmTestFetchApi;
  }

  assert.equal(
    calls[0].path,
    "/dora_dynamic_lora/state-library/prompt-document-provider/inspect",
  );
  assert.equal(
    calls[1].path,
    "/dora_dynamic_lora/state-library/prompt-document-provider/logical-skeleton",
  );
});


test("frontend prompt-document normalization rejects backend-invalid scalar coercions", async () => {
  const helpers = await loadStateManagerHelpers();
  const valid = {
    schema_version: 1,
    format: "timeline",
    routing: "logical_chunks",
    geometry: { chunks: 2, chunk_seconds: "5.000" },
  };
  assert.deepEqual(helpers.normalizePromptDocument(valid, { preserveFuture: false }), {
    schema_version: 1,
    format: "timeline",
    routing: "logical_chunks",
    geometry: { chunks: 2, chunk_seconds: "5" },
  });

  for (const malformed of [
    { ...valid, schema_version: "1" },
    { ...valid, geometry: { chunks: "2", chunk_seconds: "5" } },
    { ...valid, geometry: { chunks: 2, chunk_seconds: 5 } },
    { ...valid, geometry: { chunks: 2, chunk_seconds: ".5" } },
    { ...valid, geometry: { chunks: 2, chunk_seconds: "05" } },
    { ...valid, geometry: { chunks: 2, chunk_seconds: "5e0" } },
  ]) {
    assert.throws(
      () => helpers.normalizePromptDocument(malformed, { preserveFuture: false }),
      /prompt_document|chunks|chunk_seconds/i,
    );
  }
});


test("frontend text normalization preserves unknown future prompt-document schemas losslessly", async () => {
  const helpers = await loadStateManagerHelpers();
  const future = {
    schema_version: 99,
    new_mode: "future",
    nested: { values: [1, 2, 3] },
  };
  const state = {
    version: 3,
    characters: [{
      id: "character-a",
      name: "A",
      prompts: [{
        id: "prompt-a",
        name: "P",
        positive: "text",
        text_boxes: [{
          role: "positive",
          slot: "default",
          label: "Main",
          text: "text",
          prompt_document: future,
        }],
      }],
    }],
  };
  helpers.stateLibraryClient.state = structuredClone(state);
  const view = helpers.stateViewForSelection("character-a", "prompt-a");
  const box = view.characters[0].prompts[0].text_boxes[0];
  assert.deepEqual(box.prompt_document, future);
});


test("queue preparation waits for an in-flight State Manager library write", async () => {
  const helpers = await loadStateManagerHelpers();
  const timeline = "Global.\n\n[0-7s]\nOne.\n\n[7-14s]\nTwo.";
  const emptyState = {
    version: 3,
    characters: [{
      id: "character-a",
      name: "Character A",
      prompts: [{
        id: "prompt-a",
        name: "Prompt A",
        positive: "",
        negative: "",
        text_boxes: [
          { role: "positive", slot: "default", label: "Default positive", text: "" },
          { role: "negative", slot: "default", label: "Default negative", text: "" },
        ],
        settings: {},
      }],
    }],
  };
  const desiredState = structuredClone(emptyState);
  desiredState.characters[0].prompts[0].positive = timeline;
  desiredState.characters[0].prompts[0].text_boxes[0].text = timeline;

  const manager = {
    id: 1,
    type: "State Manager",
    comfyClass: "State Manager",
    outputs: [{ name: "state_control", type: "STATE_MANAGER_CONTROL", links: [12] }],
    widgets: [
      { name: "state_json", value: helpers.serializeBinding() },
      { name: "ui_state_json", value: helpers.serializeWorkflowUiState({}) },
      { name: "selected_character_id", value: "character-a" },
      { name: "selected_prompt_id", value: "prompt-a" },
    ],
    properties: {},
    __dsm: { state: structuredClone(desiredState), uiState: {}, renderFrame: 0 },
  };
  const textNode = {
    id: 2,
    type: "State Manager Text Box",
    comfyClass: "State Manager Text Box",
    inputs: [{ name: "state_control", type: "STATE_MANAGER_CONTROL", link: 12 }],
    outputs: [{ name: "text", type: "STRING", links: [] }],
    widgets: [
      { name: "role", value: "positive" },
      { name: "text", value: timeline },
      { name: "state_slot", value: "default" },
    ],
  };
  const nodes = [manager, textNode];
  const graph = {
    _nodes: nodes,
    links: {
      12: { origin_id: 1, origin_slot: 0, target_id: 2, target_slot: 0 },
    },
    getNodeById(id) { return nodes.find((node) => node.id === id) || null; },
    change() {},
  };
  nodes.forEach((node) => { node.graph = graph; });
  helpers.app.graph = graph;

  helpers.stateLibraryClient.state = structuredClone(emptyState);
  helpers.stateLibraryClient.revision = 7;
  helpers.stateLibraryClient.canonical = "__force_write__";
  helpers.stateLibraryClient.pending = [{
    node: manager,
    baseCharacters: structuredClone(emptyState.characters),
    desiredCharacters: structuredClone(desiredState.characters),
  }];
  helpers.stateLibraryClient.writing = false;
  helpers.stateLibraryClient.writePromise = null;
  helpers.stateLibraryClient.blocked = false;
  helpers.stateLibraryClient.lastAppliedNode = null;
  helpers.stateLibraryClient.nodes.add(manager);

  let releaseWrite;
  const gate = new Promise((resolve) => { releaseWrite = resolve; });
  const previousRaf = globalThis.requestAnimationFrame;
  globalThis.requestAnimationFrame = () => 1;
  globalThis.__dsmTestFetchApi = async (_path, options = {}) => {
    await gate;
    const request = JSON.parse(options.body);
    return {
      ok: true,
      async json() {
        return {
          version: 2,
          revision: 8,
          characters: structuredClone(request.characters),
          user_id: "default",
        };
      },
    };
  };

  const promptPayload = {
    output: {
      "1": {
        class_type: "StateManager",
        inputs: {
          state_json: helpers.serializeBinding(),
          ui_state_json: helpers.serializeWorkflowUiState({}),
          selected_character_id: "character-a",
          selected_prompt_id: "prompt-a",
        },
      },
      "2": {
        class_type: "StateManagerTextBox",
        inputs: {
          role: "positive",
          text: timeline,
          state_slot: "default",
          state_control: ["1", 7],
        },
      },
    },
    workflow: { nodes: [] },
  };

  try {
    const write = helpers.writePendingLibrary();
    let prepared = false;
    const preparing = helpers.prepareStateManagerQueuePayload(promptPayload, 0, 1)
      .then((value) => { prepared = true; return value; });
    await Promise.resolve();
    assert.equal(prepared, false);
    releaseWrite();
    await write;
    await preparing;

    assert.equal(helpers.stateLibraryClient.revision, 8);
    assert.equal(
      helpers.stateLibraryClient.state.characters[0].prompts[0].text_boxes[0].text,
      timeline,
    );
    assert.equal(promptPayload.output["2"].inputs.text, timeline);
  } finally {
    delete globalThis.__dsmTestFetchApi;
    helpers.stateLibraryClient.nodes.delete(manager);
    if (previousRaf === undefined) delete globalThis.requestAnimationFrame;
    else globalThis.requestAnimationFrame = previousRaf;
  }
});


test("queue preparation blocks a nonempty managed Text Box when persistent text is empty", async () => {
  const helpers = await loadStateManagerHelpers();
  const timeline = "Global.\n\n[0-7s]\nOne.\n\n[7-14s]\nTwo.";
  const state = {
    version: 3,
    characters: [{
      id: "character-a",
      name: "Character A",
      prompts: [{
        id: "prompt-a",
        name: "Prompt A",
        positive: "",
        negative: "",
        text_boxes: [
          { role: "positive", slot: "default", label: "Default positive", text: "" },
          { role: "negative", slot: "default", label: "Default negative", text: "" },
        ],
        settings: {},
      }],
    }],
  };
  const manager = {
    id: 1,
    type: "State Manager",
    comfyClass: "State Manager",
    outputs: [{ name: "state_control", type: "STATE_MANAGER_CONTROL", links: [12] }],
    widgets: [
      { name: "state_json", value: helpers.serializeBinding() },
      { name: "ui_state_json", value: helpers.serializeWorkflowUiState({}) },
      { name: "selected_character_id", value: "character-a" },
      { name: "selected_prompt_id", value: "prompt-a" },
    ],
    __dsm: { state: structuredClone(state), uiState: {} },
  };
  const textNode = {
    id: 2,
    type: "State Manager Text Box",
    comfyClass: "State Manager Text Box",
    inputs: [{ name: "state_control", type: "STATE_MANAGER_CONTROL", link: 12 }],
    outputs: [{ name: "text", type: "STRING", links: [] }],
    widgets: [
      { name: "role", value: "positive" },
      { name: "text", value: timeline },
      { name: "state_slot", value: "default" },
    ],
  };
  const nodes = [manager, textNode];
  const graph = {
    _nodes: nodes,
    links: { 12: { origin_id: 1, origin_slot: 0, target_id: 2, target_slot: 0 } },
    getNodeById(id) { return nodes.find((node) => node.id === id) || null; },
    change() {},
  };
  nodes.forEach((node) => { node.graph = graph; });
  helpers.app.graph = graph;
  helpers.stateLibraryClient.pending = [];
  helpers.stateLibraryClient.writing = false;
  helpers.stateLibraryClient.writePromise = null;
  helpers.stateLibraryClient.blocked = false;

  const promptPayload = {
    output: {
      "1": {
        class_type: "StateManager",
        inputs: {
          state_json: helpers.serializeBinding(),
          ui_state_json: helpers.serializeWorkflowUiState({}),
          selected_character_id: "character-a",
          selected_prompt_id: "prompt-a",
        },
      },
      "2": {
        class_type: "StateManagerTextBox",
        inputs: {
          role: "positive",
          text: timeline,
          state_slot: "default",
          state_control: ["1", 7],
        },
      },
    },
    workflow: { nodes: [] },
  };

  await assert.rejects(
    helpers.prepareStateManagerQueuePayload(promptPayload, 0, 1),
    (error) => {
      assert.equal(error.code, "DSM_UNSAVED_MANAGED_TEXT");
      assert.match(error.message, /persistent preset is empty/i);
      return true;
    },
  );
  assert.equal(promptPayload.output["2"].inputs.text, timeline);
});
