import { app } from "../../scripts/app.js";

const NODE_CLASS = "DoRA Power LoRA Loader";
const EXT_NAME = "comfyui_dora_dynamic_lora.power_lora_loader";
const LORA_API = "/dora_dynamic_lora/loras";

let _cachedLoras = null;
let _cachedLorasPromise = null;

function invalidateLorasCache() {
  _cachedLoras = null;
  _cachedLorasPromise = null;
}

async function fetchLoras() {
  if (_cachedLoras) return _cachedLoras;
  if (_cachedLorasPromise) return _cachedLorasPromise;

  _cachedLorasPromise = (async () => {
    try {
      let loras = null;

      // 1) Preferred: our custom backend endpoint (stable across ComfyUI versions).
      try {
        const r = await fetch(LORA_API, { cache: "no-store" });
        if (r.ok) {
          const j = await r.json();
          if (Array.isArray(j)) loras = j;
        }
      } catch (_) {}

      // 2) Fallback: some builds expose /object_info (all nodes) but not /object_info/<NodeName>
      if (!Array.isArray(loras)) {
        try {
          const r = await fetch("/object_info", { cache: "no-store" });
          if (r.ok) {
            const json = await r.json();
            const nodeInfo = json?.LoraLoader;
            const spec = nodeInfo?.input?.required?.lora_name;
            const values = Array.isArray(spec) ? spec[0] : null;
            if (Array.isArray(values)) loras = values.slice();
          }
        } catch (_) {}
      }

      // Normalize
      const list = Array.isArray(loras) ? loras : [];
      const out = ["None", ...list.filter((x) => x && x !== "None" && x !== "NONE")];
      _cachedLoras = out;
      return out;
    } catch (e) {
      console.warn(`[${EXT_NAME}] Failed to fetch LoRAs`, e);
      // Don't permanently cache failure; allow refresh/retry.
      return ["None"];
    } finally {
      _cachedLorasPromise = null;
    }
  })();

  return _cachedLorasPromise;
}

function removeAllWidgets(node) {
  node.widgets = node.widgets || [];
  node.widgets.length = 0;
}

function makeRowDefaults() {
  return {
    enabled: true,
    name: "None",
    strengthModel: 1.0,
    strengthClip: 1.0,
  };
}

// Back-compat for old workflows that saved widgets_values.
function parseRowsFromWidgetValues(widgetsValues) {
  // old serialized layout:
  //   per-row: enabled, name, strengthModel, strengthClip  (4)
  //   globals: stack_enabled, verbose, log_unloaded_keys   (3)
  const vals = Array.isArray(widgetsValues) ? widgetsValues : [];
  const GLOBALS = 3;
  const PER_ROW = 4;
  const rows = [];

  if (vals.length < GLOBALS) {
    return { rows: [], globals: { stack_enabled: true, verbose: false, log_unloaded_keys: false } };
  }

  const rowsCount = Math.max(0, Math.floor((vals.length - GLOBALS) / PER_ROW));
  let idx = 0;
  for (let i = 0; i < rowsCount; i++) {
    const r = makeRowDefaults();
    r.enabled = Boolean(vals[idx++]);
    r.name = typeof vals[idx] === "string" ? vals[idx] : (vals[idx] ?? "None");
    idx++;
    r.strengthModel = Number(vals[idx++]);
    r.strengthClip = Number(vals[idx++]);
    rows.push(r);
  }
  const globals = {
    stack_enabled: Boolean(vals[idx++]),
    verbose: Boolean(vals[idx++]),
    log_unloaded_keys: Boolean(vals[idx++]),
  };
  return { rows, globals };
}

function defaultState() {
  return {
    rows: [makeRowDefaults()],
    globals: {
      stack_enabled: true,
      verbose: false,
      log_unloaded_keys: false,
      broadcast_auto_scale: true,
      broadcast_scale: 1.0,
    },
  };
}

function sanitizeState(st) {
  const s = st && typeof st === "object" ? st : defaultState();
  const rowsIn = Array.isArray(s.rows) ? s.rows : [];
  const globalsIn = s.globals && typeof s.globals === "object" ? s.globals : {};

  const rows = rowsIn.length
    ? rowsIn.map((r) => ({
        enabled: !!r.enabled,
        name: typeof r.name === "string" ? r.name : "None",
        strengthModel: Number.isFinite(+r.strengthModel) ? +r.strengthModel : 1.0,
        strengthClip: Number.isFinite(+r.strengthClip) ? +r.strengthClip : (Number.isFinite(+r.strengthModel) ? +r.strengthModel : 1.0),
      }))
    : [makeRowDefaults()];

  const globals = {
    stack_enabled: globalsIn.stack_enabled !== undefined ? !!globalsIn.stack_enabled : true,
    verbose: globalsIn.verbose !== undefined ? !!globalsIn.verbose : false,
    log_unloaded_keys: globalsIn.log_unloaded_keys !== undefined ? !!globalsIn.log_unloaded_keys : false,
    broadcast_auto_scale: globalsIn.broadcast_auto_scale !== undefined ? !!globalsIn.broadcast_auto_scale : true,
    broadcast_scale: Number.isFinite(+globalsIn.broadcast_scale) ? +globalsIn.broadcast_scale : 1.0,
  };

  return { rows, globals };
}

function getState(node) {
  node.properties = node.properties || {};
  const st = node.properties.dora_power_lora;
  return sanitizeState(st);
}

function setState(node, st) {
  node.properties = node.properties || {};
  node.properties.dora_power_lora = sanitizeState(st);
}

function setButtonCallback(widget, cb) {
  // ComfyUI/LiteGraph variants differ; force callback into the widget object.
  if (!widget) return;
  widget.callback = cb;
  widget.options = widget.options || {};
  widget.options.callback = cb;
}

function rebuild(node) {
  const st = getState(node);
  const lorasNow = _cachedLoras || ["None"];
  buildUI(node, st, lorasNow);
  fetchLoras().then((loras) => {
    buildUI(node, getState(node), loras);
    node.setDirtyCanvas(true, true);
  });
}

function buildUI(node, state, loraValues) {
  const st = sanitizeState(state);
  setState(node, st);
  removeAllWidgets(node);

  node._doraRows = st.rows;
  node._doraGlobals = st.globals;

  const loras = Array.isArray(loraValues) ? loraValues : ["None"];

  // Refresh button (helps if new LoRAs are added without restarting)
  const wRefresh = node.addWidget("button", "refresh_loras", "↻ Refresh LoRAs");
  setButtonCallback(wRefresh, () => {
    invalidateLorasCache();
    rebuild(node);
  });
  wRefresh.serialize = false;

  // Rows
  node._doraRows.forEach((row, i) => {
    const n = i + 1;

    const wEnabled = node.addWidget("toggle", `lora_${n}_enabled`, !!row.enabled, (v) => {
      row.enabled = !!v;
      setState(node, { rows: node._doraRows, globals: node._doraGlobals });
    });
    wEnabled.label = `LoRA ${n} Enabled`;

    const wName = node.addWidget("combo", `lora_${n}_name`, row.name ?? "None", (v) => (row.name = v), {
      values: loras,
    });
    wName.label = `LoRA ${n}`;
    // ensure state persists
    const _origNameCb = wName.callback;
    wName.callback = (v) => {
      if (_origNameCb) _origNameCb(v);
      setState(node, { rows: node._doraRows, globals: node._doraGlobals });
    };

    const wSm = node.addWidget(
      "number",
      `lora_${n}_strength_model`,
      Number.isFinite(row.strengthModel) ? row.strengthModel : 1.0,
      (v) => {
        row.strengthModel = Number(v);
        setState(node, { rows: node._doraRows, globals: node._doraGlobals });
      },
      { min: -10.0, max: 10.0, step: 0.05 }
    );
    wSm.label = `Strength (Model)`;

    const wSc = node.addWidget(
      "number",
      `lora_${n}_strength_clip`,
      Number.isFinite(row.strengthClip) ? row.strengthClip : (Number.isFinite(row.strengthModel) ? row.strengthModel : 1.0),
      (v) => {
        row.strengthClip = Number(v);
        setState(node, { rows: node._doraRows, globals: node._doraGlobals });
      },
      { min: -10.0, max: 10.0, step: 0.05 }
    );
    wSc.label = `Strength (Clip)`;

    const wRemove = node.addWidget("button", `lora_${n}_remove`, "✖ Remove");
    setButtonCallback(wRemove, () => {
      node._doraRows.splice(i, 1);
      if (!node._doraRows.length) node._doraRows.push(makeRowDefaults());
      setState(node, { rows: node._doraRows, globals: node._doraGlobals });
      // Rebuild with current cached loras (or "None")
      buildUI(node, getState(node), _cachedLoras || ["None"]);
      node.setDirtyCanvas(true, true);
    });
    wRemove.serialize = false;
  });

  // Add row button
  const wAdd = node.addWidget("button", "add_lora", "➕ Add LoRA");
  setButtonCallback(wAdd, () => {
    node._doraRows.push(makeRowDefaults());
    setState(node, { rows: node._doraRows, globals: node._doraGlobals });
    buildUI(node, getState(node), _cachedLoras || ["None"]);
    node.setDirtyCanvas(true, true);
  });
  wAdd.serialize = false;

  // Globals (serialized)
  const wStackEnabled = node.addWidget(
    "toggle",
    "stack_enabled",
    !!node._doraGlobals.stack_enabled,
    (v) => {
      node._doraGlobals.stack_enabled = !!v;
      setState(node, { rows: node._doraRows, globals: node._doraGlobals });
    }
  );
  wStackEnabled.label = "Stack Enabled";

  const wVerbose = node.addWidget("toggle", "verbose", !!node._doraGlobals.verbose, (v) => {
    node._doraGlobals.verbose = !!v;
    setState(node, { rows: node._doraRows, globals: node._doraGlobals });
  });
  wVerbose.label = "Verbose";

  const wLogMissing = node.addWidget(
    "toggle",
    "log_unloaded_keys",
    !!node._doraGlobals.log_unloaded_keys,
    (v) => {
      node._doraGlobals.log_unloaded_keys = !!v;
      setState(node, { rows: node._doraRows, globals: node._doraGlobals });
    }
  );
  wLogMissing.label = "Log Unloaded Keys";

  const wAutoScale = node.addWidget(
    "toggle",
    "broadcast_auto_scale",
    !!node._doraGlobals.broadcast_auto_scale,
    (v) => {
      node._doraGlobals.broadcast_auto_scale = !!v;
      setState(node, { rows: node._doraRows, globals: node._doraGlobals });
    }
  );
  wAutoScale.label = "Auto-scale broadcast (fix pink/NaNs)";

  const wScale = node.addWidget(
    "number",
    "broadcast_scale",
    Number.isFinite(node._doraGlobals.broadcast_scale) ? node._doraGlobals.broadcast_scale : 1.0,
    (v) => {
      node._doraGlobals.broadcast_scale = Number(v);
      setState(node, { rows: node._doraRows, globals: node._doraGlobals });
    },
    { min: 0.0, max: 64.0, step: 0.05 }
  );
  wScale.label = "Broadcast scale";

  // Ensure node is tall enough
  const size = node.computeSize();
  node.size[0] = Math.max(node.size[0], size[0]);
  node.size[1] = Math.max(node.size[1], size[1]);
}

app.registerExtension({
  name: EXT_NAME,
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_CLASS) return;

    const origOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = origOnNodeCreated?.apply(this, arguments);
      rebuild(this);
      return r;
    };

    // Make workflow loading safe:
    // - NEVER let base configure apply widgets_values to our dynamic widget list.
    // - Persist state in node.properties.dora_power_lora.
    const origConfigure = nodeType.prototype.configure;
    nodeType.prototype.configure = function (info) {
      try {
        // 1) Use new persisted state if present
        let st = null;
        if (info?.properties?.dora_power_lora) {
          st = sanitizeState(info.properties.dora_power_lora);
        } else {
          // Back-compat: migrate old widgets_values into properties once.
          if (Array.isArray(info?.widgets_values) && info.widgets_values.length) {
            const parsed = parseRowsFromWidgetValues(info.widgets_values);
            st = sanitizeState({
              rows: parsed.rows,
              globals: {
                ...parsed.globals,
                // new fields default
                broadcast_auto_scale: true,
                broadcast_scale: 1.0,
              },
            });
          } else {
            st = defaultState();
          }
        }
        setState(this, st);

        // 2) Call base configure with widgets_values removed so it cannot abort workflow loading.
        if (origConfigure) {
          const safeInfo = info ? { ...info } : info;
          if (safeInfo) delete safeInfo.widgets_values;
          try {
            origConfigure.call(this, safeInfo);
          } catch (_) {}
        }

        rebuild(this);
      } catch (e) {
        console.warn(`[${EXT_NAME}] configure failed (swallowed to prevent workflow abort)`, e);
        try {
          if (origConfigure) {
            const safeInfo = info ? { ...info } : info;
            if (safeInfo) delete safeInfo.widgets_values;
            origConfigure.call(this, safeInfo);
          }
        } catch (_) {}
      }
      return this;
    };

    // Ensure saves never include widgets_values (avoids "Widget not found" on future loads).
    const origOnSerialize = nodeType.prototype.onSerialize;
    nodeType.prototype.onSerialize = function (o) {
      try {
        if (origOnSerialize) origOnSerialize.call(this, o);
      } catch (_) {}
      o.properties = o.properties || {};
      o.properties.dora_power_lora = getState(this);
      o.widgets_values = [];
    };
  },
});
