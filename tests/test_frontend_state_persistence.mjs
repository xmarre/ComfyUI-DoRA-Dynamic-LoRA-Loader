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

function makeVisibleWidgets(state, slot = "loader_42") {
  const widgets = [
    {
      name: "LORA_1",
      row: clone(state.rows[0]),
      value: null,
      serializeValue() {
        const row = this.row;
        return {
          on: row.enabled,
          lora: row.name,
          strength: row.strengthModel,
          strengthTwo: row.strengthClip,
        };
      },
    },
  ];
  for (const [name, value] of Object.entries(state.globals)) {
    widgets.push({ name, value: clone(value) });
  }
  widgets.push({ name: "state_slot", value: slot });
  return widgets;
}

function makeNamedWidgetSnapshot(state, slot = "loader_42", { includeRow = false } = {}) {
  const named = {};
  if (includeRow) {
    const row = state.rows[0];
    named.LORA_1 = {
      on: row.enabled,
      lora: row.name,
      strength: row.strengthModel,
      strengthTwo: row.strengthClip,
    };
  } else {
    named.LORA_1 = null;
  }
  Object.assign(named, clone(state.globals));
  named.state_slot = slot;
  return named;
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

  // Model the loader bootstrap: a fresh graph node owns default controls and
  // default live state before configure() receives workflow data.
  FakeNode.prototype.onNodeCreated = function () {
    this.serialize_widgets = true;
    this.properties = {
      dora_power_lora: clone(DEFAULT_STATE),
      dora_state_slot: "loader_default",
    };
    this._doraRows = clone(DEFAULT_STATE.rows);
    this._doraGlobals = clone(DEFAULT_STATE.globals);
    this.widgets = makeVisibleWidgets(DEFAULT_STATE, "loader_default");
    return this;
  };

  FakeNode.prototype.configure = function (info) {
    this.lastConfigureInfo = clone(info ?? {});
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
    this.widgets = makeVisibleWidgets(
      state,
      this.properties.dora_state_slot || "loader_default"
    );
    return this;
  };

  FakeNode.prototype.onSerialize = function (output) {
    output.properties = { ...(output.properties || {}) };
    output.properties.dora_power_lora = clone(this.properties.dora_power_lora);
    output.widgets_values = [];
  };

  // Relevant LiteGraph ordering: generic widget values are captured before the
  // onSerialize hook, but only when serialize_widgets remains enabled.
  FakeNode.prototype.serialize = function () {
    const output = { properties: clone(this.properties || {}) };
    if (this.serialize_widgets && Array.isArray(this.widgets)) {
      output.widgets_values = [];
      output.widgets_values_named = {};
      this.widgets.forEach((widget, index) => {
        if (widget.serialize === false) return;
        const value = widget.value ?? null;
        output.widgets_values[index] = clone(value);
        output.widgets_values_named[widget.name] = clone(value);
      });
    }
    this.onSerialize(output);
    return output;
  };

  return FakeNode;
}

async function patchedNodeType() {
  const extension = await loadPersistenceExtension();
  const NodeType = makeNodeType();
  await extension.beforeRegisterNodeDef(NodeType, { name: NODE_CLASS, display_name: NODE_CLASS });
  return NodeType;
}

function makeFreshNode(NodeType) {
  const node = new NodeType();
  node.onNodeCreated();
  return node;
}

function applyUserState(node, state, slot = "loader_42", { staleWidgetFacade = false } = {}) {
  // Model the actual loader callbacks: _doraRows/_doraGlobals are updated and
  // persistNodeState mirrors them into dora_power_lora immediately.
  node._doraRows = clone(state.rows);
  node._doraGlobals = clone(state.globals);
  node.properties.dora_power_lora = clone(state);
  node.properties.dora_state_slot = slot;
  node.widgets = makeVisibleWidgets(staleWidgetFacade ? DEFAULT_STATE : state, staleWidgetFacade ? "loader_default" : slot);
}

test("loader disables generic LiteGraph widget workflow serialization", async () => {
  const NodeType = await patchedNodeType();
  const node = makeFreshNode(NodeType);
  assert.equal(node.serialize_widgets, false);

  node.configure({ properties: { dora_power_lora: customState(), dora_state_slot: "saved_slot" } });
  assert.equal(node.serialize_widgets, false);

  const output = node.serialize();
  assert.equal(Object.prototype.hasOwnProperty.call(output, "widgets_values_named"), false);
  assert.deepEqual(output.widgets_values, []);
});

test("fresh bootstrap defaults never override incoming serialized loader state", async () => {
  const NodeType = await patchedNodeType();
  const saved = customState("saved.safetensors");
  saved.globals.auto_strength_device = "auto";
  saved.globals.auto_strength_ratio_floor = 0.61;
  saved.globals.auto_strength_ratio_ceiling = 1.94;
  const node = makeFreshNode(NodeType);

  node.configure({ properties: { dora_power_lora: clone(saved), dora_state_slot: "saved_slot" } });

  assert.deepEqual(node.properties.dora_power_lora, saved);
  assert.deepEqual(node._doraGlobals, saved.globals);
  assert.equal(node.properties.dora_state_slot, "saved_slot");
});

test("canonical serialized state beats conflicting legacy named defaults", async () => {
  const NodeType = await patchedNodeType();
  const saved = customState("canonical.safetensors");
  saved.globals.auto_strength_device = "auto";
  saved.globals.auto_strength_ratio_floor = 0.64;
  saved.globals.auto_strength_ratio_ceiling = 1.97;
  const node = makeFreshNode(NodeType);

  node.configure({
    properties: { dora_power_lora: clone(saved), dora_state_slot: "canonical_slot" },
    widgets_values_named: makeNamedWidgetSnapshot(DEFAULT_STATE, "stale_named_slot", { includeRow: true }),
  });

  assert.deepEqual(node.properties.dora_power_lora, saved);
  assert.deepEqual(node._doraGlobals, saved.globals);
  assert.equal(node.properties.dora_state_slot, "canonical_slot");
  assert.equal(node.lastConfigureInfo.widgets_values_named, undefined);
});

test("#14897-style one-shot tab round trip preserves loader state", async () => {
  const NodeType = await patchedNodeType();
  const edited = customState("roundtrip.safetensors");
  edited.globals.auto_strength_enabled = true;
  edited.globals.auto_strength_device = "auto";
  edited.globals.auto_strength_ratio_floor = 0.63;
  edited.globals.auto_strength_ratio_ceiling = 1.92;

  const outgoing = makeFreshNode(NodeType);
  applyUserState(outgoing, edited, "roundtrip_slot");

  // PR #14897 relies on this synchronous graph serialization before replacing
  // the canvas rather than a later debounced draft flush.
  const frozenWorkflowNode = outgoing.serialize();
  assert.deepEqual(frozenWorkflowNode.properties.dora_power_lora, edited);
  assert.equal(frozenWorkflowNode.properties.dora_state_slot, "roundtrip_slot");
  assert.equal(Object.prototype.hasOwnProperty.call(frozenWorkflowNode, "widgets_values_named"), false);

  const restored = makeFreshNode(NodeType);
  assert.deepEqual(restored._doraGlobals, DEFAULT_STATE.globals);
  restored.configure(frozenWorkflowNode);

  assert.deepEqual(restored.properties.dora_power_lora, edited);
  assert.deepEqual(restored._doraGlobals, edited.globals);
  assert.deepEqual(restored._doraRows, edited.rows);
  assert.equal(restored.properties.dora_state_slot, "roundtrip_slot");
});

test("stale frontend widget facade cannot overwrite callback-owned live state", async () => {
  const NodeType = await patchedNodeType();
  const edited = customState("live-wins.safetensors");
  edited.globals.auto_strength_device = "auto";
  edited.globals.auto_strength_ratio_floor = 0.72;
  edited.globals.auto_strength_ratio_ceiling = 2.11;
  const node = makeFreshNode(NodeType);

  // This models an alternate/new frontend renderer where the visual/widget
  // facade still exposes bootstrap defaults even though the loader callbacks
  // already updated _doraGlobals and dora_power_lora.
  applyUserState(node, edited, "live_slot", { staleWidgetFacade: true });
  const output = node.serialize();

  assert.deepEqual(output.properties.dora_power_lora, edited);
  assert.deepEqual(node.properties.dora_power_lora, edited);
  assert.equal(output.properties.dora_state_slot, "live_slot");
});

test("named widget workflow data is accepted only when canonical state is absent", async () => {
  const NodeType = await patchedNodeType();
  const migrated = customState("named-migration.safetensors");
  migrated.globals.auto_strength_device = "auto";
  migrated.globals.auto_strength_ratio_floor = 0.52;
  migrated.globals.auto_strength_ratio_ceiling = 1.74;
  const node = makeFreshNode(NodeType);

  node.configure({
    properties: { unrelated: true },
    widgets_values_named: makeNamedWidgetSnapshot(migrated, "named_slot", { includeRow: true }),
  });

  assert.deepEqual(node.properties.dora_power_lora, migrated);
  assert.equal(node.properties.dora_state_slot, "named_slot");
  assert.equal(node.lastConfigureInfo.widgets_values_named, undefined);
});

test("partial configure preserves existing live state", async () => {
  const NodeType = await patchedNodeType();
  const live = customState("partial.safetensors");
  const node = makeFreshNode(NodeType);
  applyUserState(node, live, "partial_slot");

  node.configure({ size: [420, 640] });

  assert.deepEqual(node.properties.dora_power_lora, live);
  assert.equal(node.properties.dora_state_slot, "partial_slot");
});

test("null serialized state is treated as missing and recovered", async () => {
  const NodeType = await patchedNodeType();
  const live = customState("recover-null.safetensors");
  const node = makeFreshNode(NodeType);
  applyUserState(node, live, "recover_slot");

  node.configure({ properties: { dora_power_lora: null } });

  assert.deepEqual(node.properties.dora_power_lora, live);
});

test("legacy positional widget migration remains available", async () => {
  const NodeType = await patchedNodeType();
  const node = makeFreshNode(NodeType);

  node.configure({ widgets_values: [true, "legacy.safetensors", 0.7, 0.7, true, false, false] });

  assert.equal(node.properties.dora_power_lora.globals.migrated_from_legacy_widgets, true);
  assert.equal(node.properties.dora_power_lora.rows[0].name, "legacy.safetensors");
});
