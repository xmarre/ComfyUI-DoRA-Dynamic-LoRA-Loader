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

function snapshotCanonicalState(node) {
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

function validStateSlot(value) {
  return typeof value === "string" && value.trim() ? value : null;
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

  // Canonical workflow state always wins when present. widgets_values_named is
  // migration-only for old workflows that do not contain dora_power_lora.
  // In particular, never let stale/default named widget shadows overwrite an
  // existing property-backed loader state during graph reconstruction.
  if (serializedState) {
    properties.dora_power_lora = serializedState;
  } else if (!hasLegacyWidgetValues) {
    const liveState = snapshotCanonicalState(node);
    const namedState = stateFromNamedWidgetValues(
      info.widgets_values_named,
      liveState?.globals
    );

    if (namedState) {
      properties.dora_power_lora = cloneJson({
        rows: namedState.rows.length
          ? namedState.rows
          : (Array.isArray(liveState?.rows) ? liveState.rows : []),
        globals: {
          ...(isObject(liveState?.globals) ? liveState.globals : {}),
          ...(isObject(namedState.globals) ? namedState.globals : {}),
        },
      });
    } else if (liveState) {
      properties.dora_power_lora = liveState;
    }
  }

  const incomingSlot = validStateSlot(properties.dora_state_slot);
  const namedSlot = !incomingSlot && !serializedState && isObject(info.widgets_values_named)
    ? validStateSlot(info.widgets_values_named.state_slot)
    : null;
  const slot = incomingSlot || namedSlot || validStateSlot(node?.properties?.dora_state_slot);
  if (slot) properties.dora_state_slot = slot;

  if (originalProperties || Object.keys(properties).length) next.properties = properties;

  // The loader owns its dynamic widget restoration. Do not let LiteGraph replay
  // named values directly into a transient/default widget layout.
  delete next.widgets_values_named;
  return next;
}

function prepareCanonicalSerialization(node) {
  // Every loader control callback updates _doraRows/_doraGlobals and persists
  // them into dora_power_lora. Those live objects are therefore the node-owned
  // authority. Do not read widget.value here: alternate frontend renderers may
  // keep a widget facade whose value lags the callback-owned loader state.
  node.serialize_widgets = false;
  node.properties = isObject(node.properties) ? node.properties : {};

  const state = snapshotCanonicalState(node);
  if (state) node.properties.dora_power_lora = cloneJson(state);

  const slot = validStateSlot(node.properties.dora_state_slot);
  return { state, slot };
}

function finalizeCanonicalSerialization(node, output, prepared) {
  if (!isObject(output)) return;
  output.properties = isObject(output.properties) ? output.properties : {};

  if (prepared.state) {
    output.properties.dora_power_lora = cloneJson(prepared.state);
    node.properties.dora_power_lora = cloneJson(prepared.state);
  }
  if (prepared.slot) {
    output.properties.dora_state_slot = prepared.slot;
    node.properties.dora_state_slot = prepared.slot;
  }

  // Older loader builds intentionally emitted an empty positional list. Keep
  // that harmless compatibility marker, but never emit a second named state.
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

    const previousOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = previousOnNodeCreated?.apply(this, arguments);
      // Workflow persistence is property-backed. Generic LiteGraph widget
      // serialization creates a competing representation and is unnecessary.
      this.serialize_widgets = false;
      return result;
    };

    const previousConfigure = nodeType.prototype.configure;
    nodeType.prototype.configure = function (info) {
      this.serialize_widgets = false;
      const guardedInfo = prepareConfigureInfo(this, info);
      const result = previousConfigure?.call(this, guardedInfo);
      this.serialize_widgets = false;
      return result;
    };

    const previousOnSerialize = nodeType.prototype.onSerialize;
    nodeType.prototype.onSerialize = function (output) {
      const prepared = prepareCanonicalSerialization(this);
      const result = previousOnSerialize?.call(this, output);
      finalizeCanonicalSerialization(this, output, prepared);
      return result;
    };
  },
});
