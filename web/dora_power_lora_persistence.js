import { app } from "../../scripts/app.js";

const NODE_CLASS = "DoRA Power LoRA Loader";
const EXT_NAME = "comfyui_dora_dynamic_lora.power_lora_persistence";

const GLOBAL_WIDGET_KEYS = [
  "stack_enabled",
  "verbose",
  "log_unloaded_keys",
  "broadcast_modulations",
  "broadcast_auto_scale",
  "broadcast_scale",
  "broadcast_include_dora_scale",
  "auto_strength_enabled",
  "auto_strength_device",
  "auto_strength_ratio_floor",
  "auto_strength_ratio_ceiling",
  "dora_decompose_debug",
  "dora_decompose_debug_n",
  "dora_decompose_debug_stack_depth",
  "dora_slice_fix",
  "dora_adaln_swap_fix",
  "zimage_lumina2_compat",
];

function cloneJson(value) {
  if (value == null) return value;
  try {
    return JSON.parse(JSON.stringify(value));
  } catch (_) {
    return value;
  }
}

function isObject(value) {
  return !!value && typeof value === "object" && !Array.isArray(value);
}

function normalizeWidgetRow(value) {
  if (!isObject(value)) return null;
  const strengthModel = Number(value.strengthModel ?? value.strength_model ?? value.strength);
  const strengthClip = Number(
    value.strengthClip ??
      value.strength_clip ??
      value.strengthTwo ??
      value.strengthModel ??
      value.strength_model ??
      value.strength
  );
  return {
    enabled: value.enabled !== undefined ? !!value.enabled : (value.on !== undefined ? !!value.on : true),
    name: String(value.name ?? value.lora ?? "None"),
    strengthModel: Number.isFinite(strengthModel) ? strengthModel : 1.0,
    strengthClip: Number.isFinite(strengthClip)
      ? strengthClip
      : (Number.isFinite(strengthModel) ? strengthModel : 1.0),
  };
}

function rowsFromNodeWidgets(node) {
  const rows = [];
  for (const widget of Array.isArray(node?.widgets) ? node.widgets : []) {
    const match = /^LORA_(\d+)$/.exec(String(widget?.name ?? ""));
    if (!match) continue;

    let raw = isObject(widget?.row) ? widget.row : null;
    if (!raw && typeof widget?.serializeValue === "function") {
      try {
        const serialized = widget.serializeValue();
        if (isObject(serialized)) raw = serialized;
      } catch (_) {}
    }

    const row = normalizeWidgetRow(raw);
    if (row) rows.push({ index: Number(match[1]), row });
  }
  rows.sort((a, b) => a.index - b.index);
  return rows.map((entry) => entry.row);
}

function snapshotStoredLiveState(node) {
  const stored = isObject(node?.properties?.dora_power_lora)
    ? node.properties.dora_power_lora
    : null;
  const liveRows = Array.isArray(node?._doraRows) ? node._doraRows : null;
  const liveGlobals = isObject(node?._doraGlobals) ? node._doraGlobals : null;

  if (!liveRows && !liveGlobals) {
    return stored ? cloneJson(stored) : null;
  }

  return cloneJson({
    rows: liveRows ?? (Array.isArray(stored?.rows) ? stored.rows : []),
    globals: liveGlobals ?? (isObject(stored?.globals) ? stored.globals : {}),
  });
}

function snapshotAuthoritativeState(node) {
  const base = snapshotStoredLiveState(node) || { rows: [], globals: {} };
  const widgets = Array.isArray(node?.widgets) ? node.widgets : [];
  const widgetRows = rowsFromNodeWidgets(node);
  const globals = isObject(base.globals) ? { ...base.globals } : {};
  const globalKeys = new Set([...GLOBAL_WIDGET_KEYS, ...Object.keys(globals)]);
  let capturedGlobal = false;

  for (const widget of widgets) {
    const name = String(widget?.name ?? "");
    if (!globalKeys.has(name) || widget?.value === undefined) continue;
    globals[name] = cloneJson(widget.value);
    capturedGlobal = true;
  }

  const rows = widgetRows.length
    ? widgetRows
    : (Array.isArray(base.rows) ? base.rows : []);
  if (!rows.length && !Object.keys(globals).length && !capturedGlobal) return null;
  return cloneJson({ rows, globals });
}

function stateFromNamedWidgetValues(named, baseGlobals = null) {
  if (!isObject(named)) return null;

  const rows = Object.entries(named)
    .map(([key, value]) => {
      const match = /^LORA_(\d+)$/.exec(key);
      const row = match ? normalizeWidgetRow(value) : null;
      return match && row ? { index: Number(match[1]), row } : null;
    })
    .filter(Boolean)
    .sort((a, b) => a.index - b.index)
    .map((entry) => entry.row);

  const globals = {};
  let hasGlobals = false;
  const globalKeys = new Set(GLOBAL_WIDGET_KEYS);
  if (isObject(baseGlobals)) {
    for (const key of Object.keys(baseGlobals)) globalKeys.add(key);
  }
  for (const key of globalKeys) {
    if (!Object.prototype.hasOwnProperty.call(named, key)) continue;
    globals[key] = cloneJson(named[key]);
    hasGlobals = true;
  }

  if (!rows.length && !hasGlobals) return null;
  return { rows, globals };
}

function mergeState(baseState, overlayState) {
  if (!baseState && !overlayState) return null;
  const base = isObject(baseState) ? baseState : {};
  const overlay = isObject(overlayState) ? overlayState : {};
  const baseRows = Array.isArray(base.rows) ? base.rows : [];
  const overlayRows = Array.isArray(overlay.rows) ? overlay.rows : [];
  return cloneJson({
    rows: overlayRows.length ? overlayRows : baseRows,
    globals: {
      ...(isObject(base.globals) ? base.globals : {}),
      ...(isObject(overlay.globals) ? overlay.globals : {}),
    },
  });
}

function snapshotStateSlot(node, named = null) {
  const namedSlot = isObject(named) ? named.state_slot : null;
  if (typeof namedSlot === "string" && namedSlot.trim()) return namedSlot;

  const stateSlotWidget = (Array.isArray(node?.widgets) ? node.widgets : [])
    .find((widget) => widget?.name === "state_slot");
  if (typeof stateSlotWidget?.value === "string" && stateSlotWidget.value.trim()) {
    return stateSlotWidget.value;
  }

  const storedSlot = node?.properties?.dora_state_slot;
  return typeof storedSlot === "string" && storedSlot.trim() ? storedSlot : null;
}

function prepareConfigureInfo(node, info) {
  if (!isObject(info)) return info;

  const next = { ...info };
  const originalProperties = isObject(info.properties) ? info.properties : null;
  const properties = originalProperties ? { ...originalProperties } : {};
  const serializedState = isObject(properties.dora_power_lora)
    ? cloneJson(properties.dora_power_lora)
    : null;
  const hasLegacyWidgetValues = Array.isArray(info.widgets_values) && info.widgets_values.length > 0;

  // Current ComfyUI serializes the actual visible standard widget values into
  // widgets_values_named. Older loader builds could leave dora_power_lora stale
  // even while those widget values were correct. Reconcile that snapshot into
  // the canonical state before removing the duplicate representation.
  const liveState = !serializedState && !hasLegacyWidgetValues
    ? snapshotAuthoritativeState(node)
    : null;
  const baseState = serializedState || liveState;
  const namedState = stateFromNamedWidgetValues(info.widgets_values_named, baseState?.globals);

  if (!hasLegacyWidgetValues || serializedState) {
    const reconciledState = mergeState(baseState, namedState);
    if (reconciledState) properties.dora_power_lora = reconciledState;
  }

  const slot = snapshotStateSlot(node, info.widgets_values_named);
  if (slot) properties.dora_state_slot = slot;

  if (originalProperties || Object.keys(properties).length) next.properties = properties;

  // dora_power_lora is the single source after reconciliation. Leaving named
  // values in the configure payload lets LiteGraph replay values directly into
  // transient widgets without invoking loader callbacks.
  delete next.widgets_values_named;
  return next;
}

function applyLiveStateToSerialization(node, output) {
  if (!isObject(output)) return;

  const visibleState = snapshotAuthoritativeState(node);
  const namedState = stateFromNamedWidgetValues(output.widgets_values_named, visibleState?.globals);
  const state = mergeState(visibleState, namedState);
  output.properties = isObject(output.properties) ? output.properties : {};

  if (state) {
    output.properties.dora_power_lora = state;
    node.properties = isObject(node.properties) ? node.properties : {};
    node.properties.dora_power_lora = cloneJson(state);
  }

  const slot = snapshotStateSlot(node, output.widgets_values_named);
  if (slot) {
    output.properties.dora_state_slot = slot;
    node.properties = isObject(node.properties) ? node.properties : {};
    node.properties.dora_state_slot = slot;
  }

  // Keep the dynamic loader state single-sourced after first reconciling the
  // current LiteGraph widget snapshot above.
  output.widgets_values = [];
  delete output.widgets_values_named;
}

app.registerExtension({
  name: EXT_NAME,
  async beforeRegisterNodeDef(nodeType, nodeData) {
    const nodeName = nodeData?.name ?? "";
    const displayName = nodeData?.display_name ?? "";
    const comfyClass = nodeType?.comfyClass ?? "";
    if (nodeName !== NODE_CLASS && displayName !== NODE_CLASS && comfyClass !== NODE_CLASS) return;
    if (nodeType.prototype.__doraPowerLoraPersistencePatched) return;
    nodeType.prototype.__doraPowerLoraPersistencePatched = true;

    const previousConfigure = nodeType.prototype.configure;
    nodeType.prototype.configure = function (info) {
      const guardedInfo = prepareConfigureInfo(this, info);
      return previousConfigure?.call(this, guardedInfo);
    };

    const previousOnSerialize = nodeType.prototype.onSerialize;
    nodeType.prototype.onSerialize = function (output) {
      const result = previousOnSerialize?.call(this, output);
      applyLiveStateToSerialization(this, output);
      return result;
    };
  },
});
