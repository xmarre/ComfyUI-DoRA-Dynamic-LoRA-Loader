import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const NODE_CLASS = "DoRA Power LoRA Loader";
const DEFAULT_STATE = {
  rows: [{ enabled: true, name: "None", strengthModel: 1.0, strengthClip: 1.0 }],
  globals: {
    stack_enabled: true,
    auto_strength_enabled: false,
    auto_strength_device: "gpu",
    auto_strength_ratio_floor: 0.30,
    auto_strength_ratio_ceiling: 1.50,
  },
};

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

async function loadPersistenceExtension() {
  const sourceUrl = new URL("../web/dora_power_lora_persistence.js", import.meta.url);
  let source = await readFile(sourceUrl, "utf8");
  let extension = null;
  globalThis.__doraPersistenceTestApp = {
    registerExtension(value) {
      extension = value;
    },
  };
  source = source.replace(
    'import { app } from "../../scripts/app.js";',
    "const app = globalThis.__doraPersistenceTestApp;"
  );
  const encoded = Buffer.from(source, "utf8").toString("base64");
  await import(`data:text/javascript;base64,${encoded}#${Date.now()}-${Math.random()}`);
  delete globalThis.__doraPersistenceTestApp;
  assert.ok(extension, "persistence extension should register");
  return extension;
}

function makeNodeType() {
  class FakeNode {}
  FakeNode.comfyClass = NODE_CLASS;

  // Model the loader's existing configure contract: a configure call without
  // dora_power_lora or legacy positional values falls back to defaults.
  FakeNode.prototype.configure = function (info) {
    this.lastConfigureInfo = info;
    let state;
    if (info?.properties?.dora_power_lora) {
      state = clone(info.properties.dora_power_lora);
    } else if (Array.isArray(info?.widgets_values) && info.widgets_values.length) {
      state = {
        rows: [{ enabled: true, name: "legacy.safetensors", strengthModel: 0.7, strengthClip: 0.7 }],
        globals: { stack_enabled: true, migrated_from_legacy_widgets: true },
      };
    } else {
      state = clone(DEFAULT_STATE);
    }

    this.properties = { ...(this.properties || {}), ...(info?.properties || {}) };
    this.properties.dora_power_lora = clone(state);
    this._doraRows = clone(state.rows);
    this._doraGlobals = clone(state.globals);
    return this;
  };

  // Model the loader's existing serializer, which reads properties rather than
  // the live row/global objects and leaves named values from current LiteGraph.
  FakeNode.prototype.onSerialize = function (output) {
    output.properties = { ...(output.properties || {}) };
    output.properties.dora_power_lora = clone(this.properties.dora_power_lora);
    output.widgets_values = [];
  };

  return FakeNode;
}

async function patchedNodeType() {
  const extension = await loadPersistenceExtension();
  const NodeType = makeNodeType();
  await extension.beforeRegisterNodeDef(NodeType, { name: NODE_CLASS, display_name: NODE_CLASS });
  return NodeType;
}

function customState(name = "custom.safetensors") {
  return {
    rows: [{ enabled: true, name, strengthModel: 0.85, strengthClip: 0.65 }],
    globals: {
      stack_enabled: false,
      verbose: true,
      broadcast_modulations: false,
      broadcast_scale: 1.75,
      auto_strength_enabled: true,
      auto_strength_device: "cpu",
      auto_strength_ratio_floor: 0.47,
      auto_strength_ratio_ceiling: 1.83,
      dora_slice_fix: false,
      zimage_lumina2_compat: false,
    },
  };
}

function makeNode(NodeType, state = customState()) {
  const node = new NodeType();
  node.properties = {
    dora_power_lora: clone(state),
    dora_state_slot: "loader_42",
  };
  node._doraRows = clone(state.rows);
  node._doraGlobals = clone(state.globals);
  return node;
}

test("partial configure preserves the live loader state", async () => {
  const NodeType = await patchedNodeType();
  const expected = customState();
  const node = makeNode(NodeType, expected);

  node.configure({ size: [420, 640] });

  assert.deepEqual(node.properties.dora_power_lora, expected);
  assert.equal(node.lastConfigureInfo.properties.dora_state_slot, "loader_42");
  assert.equal(node.lastConfigureInfo.widgets_values_named, undefined);
});

test("named widget-only workflow data migrates into authoritative loader state", async () => {
  const NodeType = await patchedNodeType();
  const node = makeNode(NodeType, DEFAULT_STATE);

  node.configure({
    widgets_values_named: {
      LORA_1: { on: true, lora: "named.safetensors", strength: 0.72, strengthTwo: 0.51 },
      stack_enabled: false,
      broadcast_modulations: false,
      broadcast_scale: 2.25,
      auto_strength_enabled: true,
      auto_strength_device: "auto",
      auto_strength_ratio_floor: 0.41,
      auto_strength_ratio_ceiling: 1.91,
      dora_slice_fix: false,
      state_slot: "named_slot",
    },
  });

  const state = node.properties.dora_power_lora;
  assert.deepEqual(state.rows, [
    { enabled: true, name: "named.safetensors", strengthModel: 0.72, strengthClip: 0.51 },
  ]);
  assert.equal(state.globals.stack_enabled, false);
  assert.equal(state.globals.broadcast_scale, 2.25);
  assert.equal(state.globals.auto_strength_enabled, true);
  assert.equal(state.globals.auto_strength_device, "auto");
  assert.equal(state.globals.auto_strength_ratio_floor, 0.41);
  assert.equal(state.globals.auto_strength_ratio_ceiling, 1.91);
  assert.equal(node.lastConfigureInfo.properties.dora_state_slot, "named_slot");
  assert.equal(node.lastConfigureInfo.widgets_values_named, undefined);
});

test("explicit serialized state remains authoritative over named widget shadows", async () => {
  const NodeType = await patchedNodeType();
  const node = makeNode(NodeType, customState("live.safetensors"));
  const serialized = customState("serialized.safetensors");
  serialized.globals.auto_strength_ratio_floor = 0.58;
  serialized.globals.auto_strength_ratio_ceiling = 1.66;

  node.configure({
    properties: { dora_power_lora: clone(serialized), dora_state_slot: "serialized_slot" },
    widgets_values_named: {
      auto_strength_ratio_floor: 0.30,
      auto_strength_ratio_ceiling: 1.50,
      auto_strength_enabled: false,
    },
  });

  assert.deepEqual(node.properties.dora_power_lora, serialized);
  assert.equal(node.lastConfigureInfo.widgets_values_named, undefined);
});

test("legacy positional widget migration is not shadowed by the preservation guard", async () => {
  const NodeType = await patchedNodeType();
  const node = makeNode(NodeType, customState());

  node.configure({ widgets_values: [true, "legacy.safetensors", 0.7, 0.7, true, false, false] });

  assert.equal(node.properties.dora_power_lora.globals.migrated_from_legacy_widgets, true);
  assert.equal(node.properties.dora_power_lora.rows[0].name, "legacy.safetensors");
});

test("serialization snapshots live row/global state and drops named widget shadows", async () => {
  const NodeType = await patchedNodeType();
  const node = makeNode(NodeType, DEFAULT_STATE);
  const live = customState("live-at-save.safetensors");
  node._doraRows = clone(live.rows);
  node._doraGlobals = clone(live.globals);

  const output = {
    properties: { unrelated: "keep" },
    widgets_values_named: {
      auto_strength_ratio_floor: 0.30,
      auto_strength_ratio_ceiling: 1.50,
    },
  };
  node.onSerialize(output);

  assert.deepEqual(output.properties.dora_power_lora, live);
  assert.equal(output.properties.unrelated, "keep");
  assert.equal(output.properties.dora_state_slot, "loader_42");
  assert.deepEqual(output.widgets_values, []);
  assert.equal(Object.prototype.hasOwnProperty.call(output, "widgets_values_named"), false);
});
