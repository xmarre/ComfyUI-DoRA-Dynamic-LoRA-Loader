import { app } from "../../scripts/app.js";

const NODE_CLASS = "DoRA Power LoRA Loader";
const EXT_NAME = "comfyui_dora_dynamic_lora.power_lora_persistence";
const TRACE_LIMIT = 240;
const TRACE_STORE_KEY = "__doraPowerLoraPersistenceTrace";
const TRACE_API_KEY = "__doraPowerLoraPersistenceDebug";

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

function isTargetNode(node) {
  return !!node && (
    node.type === NODE_CLASS ||
    node.comfyClass === NODE_CLASS ||
    node.constructor?.comfyClass === NODE_CLASS ||
    node.title === NODE_CLASS
  );
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

function snapshotWidgetFacade(node) {
  const out = {};
  for (const widget of Array.isArray(node?.widgets) ? node.widgets : []) {
    const name = String(widget?.name ?? "");
    if (GLOBAL_WIDGET_KEYS.includes(name) || name === "state_slot") {
      out[name] = cloneJson(widget?.value);
      continue;
    }
    if (!/^LORA_\d+$/.test(name)) continue;
    let row = null;
    if (isObject(widget?.row)) row = normalizeWidgetRow(widget.row);
    if (!row && typeof widget?.serializeValue === "function") {
      try { row = normalizeWidgetRow(widget.serializeValue()); } catch (_) {}
    }
    out[name] = row;
  }
  return out;
}

function getTraceStore() {
  let trace = globalThis[TRACE_STORE_KEY];
  if (!Array.isArray(trace)) {
    trace = [];
    globalThis[TRACE_STORE_KEY] = trace;
  }
  return trace;
}

function traceLifecycle(stage, node, extra = null) {
  if (!isTargetNode(node)) return;
  const trace = getTraceStore();
  const entry = {
    time: new Date().toISOString(),
    stage,
    node_id: node?.id ?? null,
    graph_id: node?.graph?.id ?? null,
    serialize_widgets: node?.serialize_widgets,
    property_state: cloneJson(node?.properties?.dora_power_lora ?? null),
    live_state: cloneJson({
      rows: Array.isArray(node?._doraRows) ? node._doraRows : null,
      globals: isObject(node?._doraGlobals) ? node._doraGlobals : null,
    }),
    state_slot: node?.properties?.dora_state_slot ?? null,
    widget_facade: snapshotWidgetFacade(node),
    extra: cloneJson(extra),
  };
  trace.push(entry);
  if (trace.length > TRACE_LIMIT) trace.splice(0, trace.length - TRACE_LIMIT);
  if (globalThis[TRACE_API_KEY]?.logToConsole === true) {
    console.debug(`[${EXT_NAME}]`, stage, entry);
  }
}

function installTraceApi() {
  const existing = isObject(globalThis[TRACE_API_KEY]) ? globalThis[TRACE_API_KEY] : {};
  globalThis[TRACE_API_KEY] = {
    ...existing,
    logToConsole: existing.logToConsole === true,
    clear() {
      const trace = getTraceStore();
      trace.length = 0;
      return true;
    },
    dump() {
      return JSON.stringify(getTraceStore(), null, 2);
    },
    snapshot() {
      return cloneJson(getTraceStore());
    },
    setConsoleLogging(enabled = true) {
      this.logToConsole = !!enabled;
      return this.logToConsole;
    },
  };
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

function canonicalWidgetValue(node, name, fallback) {
  if (GLOBAL_WIDGET_KEYS.includes(name)) {
    const storedGlobals = isObject(node?.properties?.dora_power_lora?.globals)
      ? node.properties.dora_power_lora.globals
      : null;
    if (storedGlobals && Object.prototype.hasOwnProperty.call(storedGlobals, name)) {
      return cloneJson(storedGlobals[name]);
    }
    if (isObject(node?._doraGlobals) && Object.prototype.hasOwnProperty.call(node._doraGlobals, name)) {
      return cloneJson(node._doraGlobals[name]);
    }
  }
  if (name === "state_slot") {
    return validStateSlot(node?.properties?.dora_state_slot) ?? fallback;
  }
  return fallback;
}

function syncKnownWidgetFacade(node) {
  if (!node) return;
  for (const widget of Array.isArray(node.widgets) ? node.widgets : []) {
    const name = String(widget?.name ?? "");
    if (!GLOBAL_WIDGET_KEYS.includes(name) && name !== "state_slot") continue;
    const desired = canonicalWidgetValue(node, name, widget?.value);
    try { widget.value = cloneJson(desired); } catch (_) {}
  }
}

function installDynamicWidgetValueSync(node) {
  if (!node || node.__doraPowerLoraWidgetValueSyncInstalled) return;
  if (typeof node.addWidget !== "function") return;

  const originalAddWidget = node.addWidget;
  Object.defineProperty(node, "__doraPowerLoraWidgetValueSyncInstalled", {
    value: true,
    configurable: true,
  });

  node.addWidget = function () {
    const widget = originalAddWidget.apply(this, arguments);
    if (!widget) return widget;

    const name = String(arguments[1] ?? widget.name ?? "");
    if (!GLOBAL_WIDGET_KEYS.includes(name) && name !== "state_slot") return widget;

    // Newer ComfyUI frontends keep widget values in WidgetValueStore. Recreating
    // a same-name/same-type widget can therefore return the existing bootstrap
    // state instead of adopting addWidget(..., initialValue). Always push the
    // loader's property-backed canonical value into the registered widget after
    // creation. On older LiteGraph this is just an ordinary value assignment.
    const desired = canonicalWidgetValue(this, name, arguments[2]);
    try { widget.value = cloneJson(desired); } catch (_) {}
    return widget;
  };
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

installTraceApi();

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
      traceLifecycle("onNodeCreated:before", this);
      installDynamicWidgetValueSync(this);
      const result = previousOnNodeCreated?.apply(this, arguments);
      // Workflow persistence is property-backed. Generic LiteGraph widget
      // serialization creates a competing representation and is unnecessary.
      this.serialize_widgets = false;
      traceLifecycle("onNodeCreated:after", this);
      return result;
    };

    const previousConfigure = nodeType.prototype.configure;
    nodeType.prototype.configure = function (info) {
      this.serialize_widgets = false;
      installDynamicWidgetValueSync(this);
      traceLifecycle("configure:incoming", this, {
        incoming_property_state: cloneJson(info?.properties?.dora_power_lora ?? null),
        incoming_state_slot: info?.properties?.dora_state_slot ?? null,
        incoming_widgets_values_named: cloneJson(info?.widgets_values_named ?? null),
        incoming_widgets_values: cloneJson(info?.widgets_values ?? null),
      });
      const guardedInfo = prepareConfigureInfo(this, info);
      traceLifecycle("configure:guarded", this, {
        guarded_property_state: cloneJson(guardedInfo?.properties?.dora_power_lora ?? null),
        guarded_state_slot: guardedInfo?.properties?.dora_state_slot ?? null,
        has_named_values: Object.prototype.hasOwnProperty.call(guardedInfo ?? {}, "widgets_values_named"),
      });
      const result = previousConfigure?.call(this, guardedInfo);
      this.serialize_widgets = false;
      syncKnownWidgetFacade(this);
      traceLifecycle("configure:after", this);
      queueMicrotask(() => traceLifecycle("configure:microtask", this));
      setTimeout(() => traceLifecycle("configure:timeout-250ms", this), 250);
      return result;
    };

    const previousOnSerialize = nodeType.prototype.onSerialize;
    nodeType.prototype.onSerialize = function (output) {
      traceLifecycle("serialize:before", this, {
        output_property_state_before: cloneJson(output?.properties?.dora_power_lora ?? null),
        output_named_before: cloneJson(output?.widgets_values_named ?? null),
      });
      const prepared = prepareCanonicalSerialization(this);
      const result = previousOnSerialize?.call(this, output);
      finalizeCanonicalSerialization(this, output, prepared);
      traceLifecycle("serialize:after", this, {
        output_property_state_after: cloneJson(output?.properties?.dora_power_lora ?? null),
        output_state_slot_after: output?.properties?.dora_state_slot ?? null,
        output_widgets_values_after: cloneJson(output?.widgets_values ?? null),
        has_named_values_after: Object.prototype.hasOwnProperty.call(output ?? {}, "widgets_values_named"),
      });
      return result;
    };
  },
  loadedGraphNode(node) {
    traceLifecycle("loadedGraphNode", node);
    setTimeout(() => traceLifecycle("loadedGraphNode:timeout-500ms", node), 500);
  },
});
