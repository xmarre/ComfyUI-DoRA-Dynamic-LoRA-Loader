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
  // legacy serialized layouts:
  //   compact per-row: enabled, name, strengthModel  (3)
  //   old per-row: enabled, name, strengthModel, strengthClip  (4)
  //   globals v1: stack_enabled, verbose, log_unloaded_keys   (3)
  //   globals v2: v1 + dora_decompose_debug, dora_decompose_debug_n,
  //               dora_decompose_debug_stack_depth, dora_slice_fix (7)
  const vals = Array.isArray(widgetsValues) ? widgetsValues : [];

  const parseWithLayout = (PER_ROW, GLOBALS) => {
    if (vals.length < GLOBALS) return null;
    if ((vals.length - GLOBALS) % PER_ROW !== 0) return null;

    const outRows = [];
    let idx = 0;
    const rowsCount = Math.max(0, Math.floor((vals.length - GLOBALS) / PER_ROW));
    for (let i = 0; i < rowsCount; i++) {
      const r = makeRowDefaults();
      r.enabled = Boolean(vals[idx++]);
      const nm = vals[idx++];
      r.name = typeof nm === "string" ? nm : (nm ?? "None");
      const sm = Number(vals[idx++]);
      r.strengthModel = sm;
      if (PER_ROW === 4) {
        const sc = Number(vals[idx++]);
        r.strengthClip = Number.isFinite(sc) ? sc : sm;
      } else {
        r.strengthClip = sm;
      }
      outRows.push(r);
    }

    const globals = {
      stack_enabled: Boolean(vals[idx++]),
      verbose: Boolean(vals[idx++]),
      log_unloaded_keys: Boolean(vals[idx++]),
    };

    if (GLOBALS >= 7) {
      globals.dora_decompose_debug = Boolean(vals[idx++]);
      globals.dora_decompose_debug_n = Number(vals[idx++]);
      globals.dora_decompose_debug_stack_depth = Number(vals[idx++]);
      globals.dora_slice_fix = Boolean(vals[idx++]);
      globals.dora_adaln_swap_fix = true; // legacy widgets_values never had this; default to on
    }

    return { rows: outRows, globals };
  };

  const looksValid = (parsed) => {
    if (!parsed) return false;
    if (parsed.rows && parsed.rows.length) {
      const r0 = parsed.rows[0];
      if (typeof r0.name !== "string") return false;
      if (!Number.isFinite(+r0.strengthModel)) return false;
      if (!Number.isFinite(+r0.strengthClip)) return false;
    }
    return true;
  };

  const layouts = [
    [4, 3],
    [4, 7],
    [3, 3],
    [3, 7],
  ];

  for (const [perRow, globals] of layouts) {
    const parsed = parseWithLayout(perRow, globals);
    if (looksValid(parsed)) return parsed;
  }

  return { rows: [], globals: { stack_enabled: true, verbose: false, log_unloaded_keys: false } };
}

function defaultState() {
  return {
    rows: [makeRowDefaults()],
    globals: {
      stack_enabled: true,
      verbose: false,
      log_unloaded_keys: false,
      broadcast_modulations: true,
      broadcast_auto_scale: true,
      broadcast_scale: 1.0,
      broadcast_include_dora_scale: false,
      dora_decompose_debug: false,
      dora_decompose_debug_n: 30,
      dora_decompose_debug_stack_depth: 10,
      dora_slice_fix: true,
      dora_adaln_swap_fix: true,
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
    broadcast_modulations: globalsIn.broadcast_modulations !== undefined ? !!globalsIn.broadcast_modulations : true,
    broadcast_auto_scale: globalsIn.broadcast_auto_scale !== undefined ? !!globalsIn.broadcast_auto_scale : true,
    broadcast_scale: Number.isFinite(+globalsIn.broadcast_scale) ? +globalsIn.broadcast_scale : 1.0,
    broadcast_include_dora_scale: globalsIn.broadcast_include_dora_scale !== undefined ? !!globalsIn.broadcast_include_dora_scale : false,
    dora_decompose_debug: globalsIn.dora_decompose_debug !== undefined ? !!globalsIn.dora_decompose_debug : false,
    dora_decompose_debug_n: Number.isFinite(+globalsIn.dora_decompose_debug_n) ? Math.max(0, Math.floor(+globalsIn.dora_decompose_debug_n)) : 30,
    dora_decompose_debug_stack_depth: Number.isFinite(+globalsIn.dora_decompose_debug_stack_depth)
      ? Math.max(2, Math.min(64, Math.floor(+globalsIn.dora_decompose_debug_stack_depth)))
      : 10,
    dora_slice_fix: globalsIn.dora_slice_fix !== undefined ? !!globalsIn.dora_slice_fix : true,
    dora_adaln_swap_fix: globalsIn.dora_adaln_swap_fix !== undefined ? !!globalsIn.dora_adaln_swap_fix : true,
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
    wEnabled.label = `LoRA ${n}`;

    const wName = node.addWidget("combo", `lora_${n}_name`, row.name ?? "None", (v) => (row.name = v), {
      values: loras,
    });
    wName.label = `Name`;
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
    wSm.label = `Weight`;

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

  const wBroadcastMods = node.addWidget(
    "toggle",
    "broadcast_modulations",
    !!node._doraGlobals.broadcast_modulations,
    (v) => {
      node._doraGlobals.broadcast_modulations = !!v;
      setState(node, { rows: node._doraRows, globals: node._doraGlobals });
    }
  );
  wBroadcastMods.label = "Broadcast OneTrainer modulation LoRAs";

  const wIncludeDoraScale = node.addWidget(
    "toggle",
    "broadcast_include_dora_scale",
    !!node._doraGlobals.broadcast_include_dora_scale,
    (v) => {
      node._doraGlobals.broadcast_include_dora_scale = !!v;
      setState(node, { rows: node._doraRows, globals: node._doraGlobals });
    }
  );
  wIncludeDoraScale.label = "Include DoRA dora_scale in broadcast";

  const wAutoScale = node.addWidget(
    "toggle",
    "broadcast_auto_scale",
    !!node._doraGlobals.broadcast_auto_scale,
    (v) => {
      node._doraGlobals.broadcast_auto_scale = !!v;
      setState(node, { rows: node._doraRows, globals: node._doraGlobals });
    }
  );
  wAutoScale.label = "Auto-scale broadcast";

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

  // ---------------- DoRA decompose debug controls (node-adjustable) ----------------
  const wDoraSliceFix = node.addWidget(
    "toggle",
    "dora_slice_fix",
    !!node._doraGlobals.dora_slice_fix,
    (v) => {
      node._doraGlobals.dora_slice_fix = !!v;
      setState(node, { rows: node._doraRows, globals: node._doraGlobals });
    }
  );
  wDoraSliceFix.label = "DoRA slice-fix for offset patches (Flux2)";

  const wDoraAdaLNSwap = node.addWidget(
    "toggle",
    "dora_adaln_swap_fix",
    !!node._doraGlobals.dora_adaln_swap_fix,
    (v) => {
      node._doraGlobals.dora_adaln_swap_fix = !!v;
      setState(node, { rows: node._doraRows, globals: node._doraGlobals });
    }
  );
  wDoraAdaLNSwap.label = "DoRA adaLN swap_scale_shift fix";

  const wDoraDbg = node.addWidget(
    "toggle",
    "dora_decompose_debug",
    !!node._doraGlobals.dora_decompose_debug,
    (v) => {
      node._doraGlobals.dora_decompose_debug = !!v;
      setState(node, { rows: node._doraRows, globals: node._doraGlobals });
    }
  );
  wDoraDbg.label = "DoRA decompose debug logs";

  const wDoraDbgN = node.addWidget(
    "number",
    "dora_decompose_debug_n",
    Number.isFinite(+node._doraGlobals.dora_decompose_debug_n) ? +node._doraGlobals.dora_decompose_debug_n : 30,
    (v) => {
      node._doraGlobals.dora_decompose_debug_n = Math.max(0, Math.floor(+v || 0));
      setState(node, { rows: node._doraRows, globals: node._doraGlobals });
    },
    { min: 0, max: 500, step: 1 }
  );
  wDoraDbgN.label = "DoRA debug lines";

  const wDoraDbgStack = node.addWidget(
    "number",
    "dora_decompose_debug_stack_depth",
    Number.isFinite(+node._doraGlobals.dora_decompose_debug_stack_depth) ? +node._doraGlobals.dora_decompose_debug_stack_depth : 10,
    (v) => {
      node._doraGlobals.dora_decompose_debug_stack_depth = Math.max(2, Math.min(64, Math.floor(+v || 10)));
      setState(node, { rows: node._doraRows, globals: node._doraGlobals });
    },
    { min: 2, max: 64, step: 1 }
  );
  wDoraDbgStack.label = "DoRA debug stack depth";

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
                broadcast_modulations: true,
                broadcast_auto_scale: true,
                broadcast_scale: 1.0,
                broadcast_include_dora_scale: false,
                dora_decompose_debug: false,
                dora_decompose_debug_n: 30,
                dora_decompose_debug_stack_depth: 10,
                dora_slice_fix: true,
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
