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

  // Model the loader's existing serializer. Current LiteGraph has already
  // populated widgets_values_named before this hook executes.
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

function makeVisibleWidgets(state, slot = "loader_42") {
  const widgets = [
    {
      name: "LORA_1",
      row: clone(state.rows[0]),
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
    // Current LiteGraph stores widget.value in named workflow state. The custom
    // LoRA row has no value property, so it commonly serializes as null here.
    named.LORA_1 = null;
  }
  Object.assign(named, clone(state.globals));
  named.state_slot = slot;
  return named;
}

function makeNode(NodeType, state = customState(), { widgets = true } = {}) {
  const node = new NodeType();
  node.properties = {
    dora_power_lora: clone(state),
    dora_state_slot: "loader_42",
  };
  node._doraRows = clone(state.rows);
  node._doraGlobals = clone(state.globals);
  node.widgets = widgets ? makeVisibleWidgets(state) : [];
  return node;
}

test("partial configure preserves current visible loader state", async () => {
  const NodeType = await patchedNodeType();
  const expected = customState();
  const node = makeNode(NodeType, expected);

  node.configure({ size: [420, 640] });

  assert.deepEqual(node.properties.dora_power_lora, expected);
  assert.equal(node.lastConfigureInfo.properties.dora_state_slot, "loader_42");
  assert.equal(node.lastConfigureInfo.widgets_values_named, undefined);
});

test("named widget-only workflow data migrates into canonical loader state", async () => {
  const NodeType = await patchedNodeType();
  const node = makeNode(NodeType, DEFAULT_STATE);
  const named = customState("named.safetensors");
  named.globals.broadcast_scale = 2.25;
  named.globals.auto_strength_device = "auto";
  named.globals.auto_strength_ratio_floor = 0.41;
  named.globals.auto_strength_ratio_ceiling = 1.91;

  node.configure({
    widgets_values_named: makeNamedWidgetSnapshot(named, "named_slot", { includeRow: true }),
  });

  const state = node.properties.dora_power_lora;
  assert.deepEqual(state.rows, named.rows);
  assert.equal(state.globals.stack_enabled, false);
  assert.equal(state.globals.broadcast_scale, 2.25);
  assert.equal(state.globals.auto_strength_enabled, true);
  assert.equal(state.globals.auto_strength_device, "auto");
  assert.equal(state.globals.auto_strength_ratio_floor, 0.41);
  assert.equal(state.globals.auto_strength_ratio_ceiling, 1.91);
  assert.equal(node.lastConfigureInfo.properties.dora_state_slot, "named_slot");
  assert.equal(node.lastConfigureInfo.widgets_values_named, undefined);
});

test("named globals override stale serialized defaults while preserving serialized LoRA rows", async () => {
  const NodeType = await patchedNodeType();
  const serialized = customState("keep-serialized-row.safetensors");
  serialized.globals = clone(DEFAULT_STATE.globals);
  const node = makeNode(NodeType, DEFAULT_STATE);
  const visible = customState("ignored-named-row.safetensors");
  visible.globals.auto_strength_device = "auto";
  visible.globals.auto_strength_ratio_floor = 0.52;
  visible.globals.auto_strength_ratio_ceiling = 1.74;

  node.configure({
    properties: { dora_power_lora: clone(serialized), dora_state_slot: "serialized_slot" },
    widgets_values_named: makeNamedWidgetSnapshot(visible, "visible_slot", { includeRow: false }),
  });

  const state = node.properties.dora_power_lora;
  assert.deepEqual(state.rows, serialized.rows);
  assert.equal(state.globals.auto_strength_enabled, true);
  assert.equal(state.globals.auto_strength_device, "auto");
  assert.equal(state.globals.auto_strength_ratio_floor, 0.52);
  assert.equal(state.globals.auto_strength_ratio_ceiling, 1.74);
  assert.equal(node.lastConfigureInfo.properties.dora_state_slot, "visible_slot");
});

test("named LoRA rows override stale serialized rows when a row snapshot is available", async () => {
  const NodeType = await patchedNodeType();
  const serialized = customState("stale-row.safetensors");
  const visible = customState("visible-row.safetensors");
  const node = makeNode(NodeType, DEFAULT_STATE);

  node.configure({
    properties: { dora_power_lora: clone(serialized) },
    widgets_values_named: makeNamedWidgetSnapshot(visible, "loader_42", { includeRow: true }),
  });

  assert.deepEqual(node.properties.dora_power_lora.rows, visible.rows);
  assert.equal(node.properties.dora_power_lora.globals.auto_strength_ratio_floor, visible.globals.auto_strength_ratio_floor);
});

test("serialized state stays authoritative when no named widget snapshot exists", async () => {
  const NodeType = await patchedNodeType();
  const node = makeNode(NodeType, customState("live.safetensors"));
  const serialized = customState("serialized.safetensors");
  serialized.globals.auto_strength_ratio_floor = 0.58;
  serialized.globals.auto_strength_ratio_ceiling = 1.66;

  node.configure({
    properties: { dora_power_lora: clone(serialized), dora_state_slot: "serialized_slot" },
  });

  assert.deepEqual(node.properties.dora_power_lora, serialized);
  assert.equal(node.lastConfigureInfo.properties.dora_state_slot, "serialized_slot");
});

test("null serialized state is treated as missing and recovered", async () => {
  const NodeType = await patchedNodeType();
  const live = customState("recover-null.safetensors");
  const node = makeNode(NodeType, live);

  node.configure({ properties: { dora_power_lora: null } });

  assert.deepEqual(node.properties.dora_power_lora, live);
});

test("legacy positional widget migration is not shadowed by the preservation guard", async () => {
  const NodeType = await patchedNodeType();
  const node = makeNode(NodeType, customState());

  node.configure({ widgets_values: [true, "legacy.safetensors", 0.7, 0.7, true, false, false] });

  assert.equal(node.properties.dora_power_lora.globals.migrated_from_legacy_widgets, true);
  assert.equal(node.properties.dora_power_lora.rows[0].name, "legacy.safetensors");
});

test("serialization uses visible widget values over stale live globals and properties", async () => {
  const NodeType = await patchedNodeType();
  const stale = clone(DEFAULT_STATE);
  const visible = customState("visible-at-save.safetensors");
  visible.globals.auto_strength_device = "auto";
  visible.globals.auto_strength_ratio_floor = 0.63;
  visible.globals.auto_strength_ratio_ceiling = 1.92;
  const node = makeNode(NodeType, stale, { widgets: false });
  node.widgets = makeVisibleWidgets(visible, "visible_slot");

  const output = {
    properties: { unrelated: "keep" },
    widgets_values_named: makeNamedWidgetSnapshot(visible, "visible_slot", { includeRow: false }),
  };
  node.onSerialize(output);

  assert.deepEqual(output.properties.dora_power_lora, visible);
  assert.deepEqual(node.properties.dora_power_lora, visible);
  assert.equal(output.properties.unrelated, "keep");
  assert.equal(output.properties.dora_state_slot, "visible_slot");
  assert.equal(node.properties.dora_state_slot, "visible_slot");
  assert.deepEqual(output.widgets_values, []);
  assert.equal(Object.prototype.hasOwnProperty.call(output, "widgets_values_named"), false);
});

test("serialization preserves live LoRA rows when named row value is null", async () => {
  const NodeType = await patchedNodeType();
  const live = customState("keep-live-row.safetensors");
  const node = makeNode(NodeType, live);

  const output = {
    properties: {},
    widgets_values_named: makeNamedWidgetSnapshot(live, "loader_42", { includeRow: false }),
  };
  node.onSerialize(output);

  assert.deepEqual(output.properties.dora_power_lora.rows, live.rows);
  assert.equal(output.properties.dora_power_lora.globals.auto_strength_ratio_floor, live.globals.auto_strength_ratio_floor);
});
