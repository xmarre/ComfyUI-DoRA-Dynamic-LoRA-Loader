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
  // Be resilient across ComfyUI/LiteGraph variants.
  if (!node.widgets) {
    node.widgets = [];
    return;
  }
  // Some builds don't expose removeWidget; clearing array works.
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

function parseRowsFromWidgetValues(widgetsValues) {
  // Our serialized widgets are:
  //   per-row: enabled, name, strengthModel, strengthClip  (4)
  //   globals: stack_enabled, verbose, log_unloaded_keys   (3)
  const vals = Array.isArray(widgetsValues) ? widgetsValues : [];
  const GLOBALS = 3;
  const PER_ROW = 4;
  const rows = [];

  if (vals.length < GLOBALS) return { rows, globals: { stack_enabled: true, verbose: false, log_unloaded_keys: false } };
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

function normalizeState(state) {
  const rows = Array.isArray(state?.rows) ? state.rows : [];
  const globals = state?.globals && typeof state.globals === "object" ? state.globals : {};
  return {
    rows: rows.length ? rows.map((r) => ({
      enabled: !!r.enabled,
      name: typeof r.name === "string" ? r.name : "None",
      strengthModel: Number.isFinite(+r.strengthModel) ? +r.strengthModel : 1.0,
      strengthClip: Number.isFinite(+r.strengthClip) ? +r.strengthClip : (Number.isFinite(+r.strengthModel) ? +r.strengthModel : 1.0),
    })) : [makeRowDefaults()],
    globals: {
      stack_enabled: globals.stack_enabled !== undefined ? !!globals.stack_enabled : true,
      verbose: globals.verbose !== undefined ? !!globals.verbose : false,
      log_unloaded_keys: globals.log_unloaded_keys !== undefined ? !!globals.log_unloaded_keys : false,
    },
  };
}

function getStateFromNode(node) {
  const p = node?.properties?.dora_power_lora;
  if (p && typeof p === "object") return normalizeState(p);
  return normalizeState({ rows: [makeRowDefaults()], globals: { stack_enabled: true, verbose: false, log_unloaded_keys: false } });
}

function setStateOnNode(node, state) {
  if (!node.properties) node.properties = {};
  node.properties.dora_power_lora = normalizeState(state);
}

function noSerialize(widget) {
  if (!widget) return widget;
  widget.serialize = false;
  widget.options = widget.options || {};
  widget.options.serialize = false;
  return widget;
}

function buildUI(node, rows, globals, loraValues) {
  removeAllWidgets(node);

  node._doraRows = rows || [];
  node._doraGlobals = globals || { stack_enabled: true, verbose: false, log_unloaded_keys: false };
  setStateOnNode(node, { rows: node._doraRows, globals: node._doraGlobals });

  const loras = Array.isArray(loraValues) ? loraValues : ["None"];

  // Refresh button (helps if new LoRAs are added without restarting)
  const wRefresh = node.addWidget("button", "↻ Refresh LoRAs", null, () => {
    invalidateLorasCache();
    fetchLoras().then((newList) => {
      buildUI(node, node._doraRows || [], node._doraGlobals || globals, newList);
      node.setDirtyCanvas(true, true);
    });
  });
  noSerialize(wRefresh);

  // Rows
  node._doraRows.forEach((row, i) => {
    const n = i + 1;

    const wEnabled = node.addWidget("toggle", `LoRA ${n} Enabled`, !!row.enabled, (v) => {
      row.enabled = !!v;
      setStateOnNode(node, { rows: node._doraRows, globals: node._doraGlobals });
    });
    wEnabled.label = `LoRA ${n} Enabled`;
    noSerialize(wEnabled);

    const wName = node.addWidget("combo", `LoRA ${n}`, row.name ?? "None", (v) => {
      row.name = v;
      setStateOnNode(node, { rows: node._doraRows, globals: node._doraGlobals });
    }, {
      values: loras,
    });
    wName.label = `LoRA ${n}`;
    noSerialize(wName);

    const wSm = node.addWidget(
      "number",
      `Strength (Model) ${n}`,
      Number.isFinite(row.strengthModel) ? row.strengthModel : 1.0,
      (v) => {
        row.strengthModel = Number(v);
        setStateOnNode(node, { rows: node._doraRows, globals: node._doraGlobals });
      },
      { min: -10.0, max: 10.0, step: 0.05 }
    );
    wSm.label = `Strength (Model)`;
    noSerialize(wSm);

    const wSc = node.addWidget(
      "number",
      `Strength (Clip) ${n}`,
      Number.isFinite(row.strengthClip) ? row.strengthClip : (Number.isFinite(row.strengthModel) ? row.strengthModel : 1.0),
      (v) => {
        row.strengthClip = Number(v);
        setStateOnNode(node, { rows: node._doraRows, globals: node._doraGlobals });
      },
      { min: -10.0, max: 10.0, step: 0.05 }
    );
    wSc.label = `Strength (Clip)`;
    noSerialize(wSc);

    const wRemove = node.addWidget("button", `✖ Remove LoRA ${n}`, null, () => {
      node._doraRows.splice(i, 1);
      if (!node._doraRows.length) node._doraRows.push(makeRowDefaults());
      setStateOnNode(node, { rows: node._doraRows, globals: node._doraGlobals });
      // Rebuild with current cached loras (or "None")
      buildUI(node, node._doraRows, node._doraGlobals, _cachedLoras || ["None"]);
      node.setDirtyCanvas(true, true);
    });
    noSerialize(wRemove);
  });

  // Add row button
  const wAdd = node.addWidget("button", "➕ Add LoRA", null, () => {
    node._doraRows.push(makeRowDefaults());
    setStateOnNode(node, { rows: node._doraRows, globals: node._doraGlobals });
    buildUI(node, node._doraRows, node._doraGlobals, _cachedLoras || ["None"]);
    node.setDirtyCanvas(true, true);
  });
  noSerialize(wAdd);

  // Globals (serialized)
  const wStackEnabled = node.addWidget(
    "toggle",
    "stack_enabled",
    !!node._doraGlobals.stack_enabled,
    (v) => {
      node._doraGlobals.stack_enabled = !!v;
      setStateOnNode(node, { rows: node._doraRows, globals: node._doraGlobals });
    }
  );
  wStackEnabled.label = "Stack Enabled";
  noSerialize(wStackEnabled);

  const wVerbose = node.addWidget("toggle", "verbose", !!node._doraGlobals.verbose, (v) => {
    node._doraGlobals.verbose = !!v;
    setStateOnNode(node, { rows: node._doraRows, globals: node._doraGlobals });
  });
  wVerbose.label = "Verbose";
  noSerialize(wVerbose);

  const wLogMissing = node.addWidget(
    "toggle",
    "log_unloaded_keys",
    !!node._doraGlobals.log_unloaded_keys,
    (v) => {
      node._doraGlobals.log_unloaded_keys = !!v;
      setStateOnNode(node, { rows: node._doraRows, globals: node._doraGlobals });
    }
  );
  wLogMissing.label = "Log Unloaded Keys";
  noSerialize(wLogMissing);

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
      // Default persisted state
      const state = getStateFromNode(this);
      buildUI(this, state.rows, state.globals, _cachedLoras || ["None"]);
      // async populate lora dropdown
      fetchLoras().then((loras) => {
        const st = getStateFromNode(this);
        buildUI(this, st.rows, st.globals, loras);
        this.setDirtyCanvas(true, true);
      });
      return r;
    };

    // Make workflow loading safe:
    // - NEVER allow base configure to try to apply widgets_values (it causes "Widget not found" and aborts workflow load)
    // - Load/save state via node.properties.dora_power_lora instead of widgets_values.
    const origConfigure = nodeType.prototype.configure;
    nodeType.prototype.configure = function (info) {
      try {
        // 1) Restore state from properties if present (new format)
        let state = null;
        if (info?.properties?.dora_power_lora) {
          state = normalizeState(info.properties.dora_power_lora);
        } else if (Array.isArray(info?.widgets_values)) {
          // 2) Backward compat: old workflows saved widget values
          const parsed = parseRowsFromWidgetValues(info.widgets_values);
          state = normalizeState({ rows: parsed.rows, globals: parsed.globals });
        } else {
          state = normalizeState(null);
        }
        setStateOnNode(this, state);

        // Call base configure but strip widgets_values to prevent workflow load aborts.
        if (origConfigure) {
          const safeInfo = info ? { ...info } : info;
          if (safeInfo) safeInfo.widgets_values = [];
          origConfigure.call(this, safeInfo);
        }

        // Build UI from restored state
        buildUI(this, state.rows, state.globals, _cachedLoras || ["None"]);

        // async populate lora dropdown
        fetchLoras().then((loras) => {
          const st = getStateFromNode(this);
          buildUI(this, st.rows, st.globals, loras);
          this.setDirtyCanvas(true, true);
        });
      } catch (e) {
        console.warn(`[${EXT_NAME}] configure failed (swallowed to prevent workflow abort)`, e);
        try {
          if (origConfigure) {
            const safeInfo = info ? { ...info } : info;
            if (safeInfo) safeInfo.widgets_values = [];
            origConfigure.call(this, safeInfo);
          }
        } catch (_) {}
      }
      return this;
    };

    // Ensure future saves don't store widgets_values (we store our state in properties instead).
    const origOnSerialize = nodeType.prototype.onSerialize;
    nodeType.prototype.onSerialize = function (o) {
      try {
        if (origOnSerialize) origOnSerialize.call(this, o);
      } catch (_) {}
      o.properties = o.properties || {};
      o.properties.dora_power_lora = getStateFromNode(this);
      // prevent "Widget not found" across versions
      o.widgets_values = [];
    };
  },
});
