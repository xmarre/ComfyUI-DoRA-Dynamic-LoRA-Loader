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

function snapshotLiveState(node) {
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

function stateFromNamedWidgetValues(named) {
  if (!isObject(named)) return null;

  const rows = Object.entries(named)
    .map(([key, value]) => {
      const match = /^LORA_(\d+)$/.exec(key);
      if (!match || !isObject(value)) return null;
      const strengthModel = Number(value.strength ?? value.strengthModel);
      const strengthClip = Number(value.strengthTwo ?? value.strengthClip ?? value.strength ?? value.strengthModel);
      return {
        index: Number(match[1]),
        row: {
          enabled: value.on !== undefined ? !!value.on : (value.enabled !== undefined ? !!value.enabled : true),
          name: String(value.lora ?? value.name ?? "None"),
          strengthModel: Number.isFinite(strengthModel) ? strengthModel : 1.0,
          strengthClip: Number.isFinite(strengthClip)
            ? strengthClip
            : (Number.isFinite(strengthModel) ? strengthModel : 1.0),
        },
      };
    })
    .filter(Boolean)
    .sort((a, b) => a.index - b.index)
    .map((entry) => entry.row);

  const globals = {};
  let hasGlobals = false;
  for (const key of GLOBAL_WIDGET_KEYS) {
    if (!Object.prototype.hasOwnProperty.call(named, key)) continue;
    globals[key] = cloneJson(named[key]);
    hasGlobals = true;
  }

  if (!rows.length && !hasGlobals) return null;
  return { rows, globals };
}

function prepareConfigureInfo(node, info) {
  if (!isObject(info)) return info;

  const next = { ...info };
  const originalProperties = isObject(info.properties) ? info.properties : null;
  const properties = originalProperties ? { ...originalProperties } : {};
  const hasSerializedState = Object.prototype.hasOwnProperty.call(properties, "dora_power_lora");
  const hasLegacyWidgetValues = Array.isArray(info.widgets_values) && info.widgets_values.length > 0;
  const namedState = stateFromNamedWidgetValues(info.widgets_values_named);

  if (!hasSerializedState && !hasLegacyWidgetValues) {
    const fallbackState = namedState ?? snapshotLiveState(node);
    if (fallbackState) properties.dora_power_lora = fallbackState;
  }

  if (!Object.prototype.hasOwnProperty.call(properties, "dora_state_slot")) {
    const namedSlot = info.widgets_values_named?.state_slot;
    const currentSlot = node?.properties?.dora_state_slot;
    if (typeof namedSlot === "string" && namedSlot.trim()) {
      properties.dora_state_slot = namedSlot;
    } else if (typeof currentSlot === "string" && currentSlot.trim()) {
      properties.dora_state_slot = currentSlot;
    }
  }

  if (originalProperties || Object.keys(properties).length) next.properties = properties;

  // The loader owns a dynamic widget layout. Current ComfyUI serializes both
  // positional and named widget state, while this loader's authoritative state
  // lives in dora_power_lora. Named values can otherwise be replayed into a
  // transient widget set during configure without updating loader state.
  delete next.widgets_values_named;
  return next;
}

function applyLiveStateToSerialization(node, output) {
  if (!isObject(output)) return;
  const state = snapshotLiveState(node);
  output.properties = isObject(output.properties) ? output.properties : {};
  if (state) output.properties.dora_power_lora = state;

  const slot = node?.properties?.dora_state_slot;
  if (typeof slot === "string" && slot.trim()) output.properties.dora_state_slot = slot;

  // Keep the dynamic loader state single-sourced. Positional values remain an
  // empty compatibility marker for older workflows; named values are omitted.
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
