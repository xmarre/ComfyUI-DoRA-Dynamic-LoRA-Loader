import { app } from "../../scripts/app.js";

const NODE_CLASS = "DoRA Power LoRA Loader";
const EXT_NAME = "comfyui_dora_dynamic_lora.power_lora_loader";

let _cachedLoras = null;
let _cachedLorasPromise = null;

async function fetchLoras() {
  if (_cachedLoras) return _cachedLoras;
  if (_cachedLorasPromise) return _cachedLorasPromise;

  _cachedLorasPromise = (async () => {
    try {
      // Use ComfyUI's object_info endpoint to read the lora dropdown values from the built-in LoraLoader node.
      const resp = await fetch("/object_info/LoraLoader");
      const json = await resp.json();
      const nodeInfo = json?.LoraLoader ?? json?.[Object.keys(json || {})[0]];
      const spec = nodeInfo?.input?.required?.lora_name;
      const values = Array.isArray(spec) ? spec[0] : null;
      const loras = Array.isArray(values) ? values.slice() : [];
      // Normalize "None"
      const out = ["None", ...loras.filter((x) => x && x !== "None" && x !== "NONE")];
      _cachedLoras = out;
      return out;
    } catch (e) {
      console.warn(`[${EXT_NAME}] Failed to fetch loras via /object_info/LoraLoader`, e);
      _cachedLoras = ["None"];
      return _cachedLoras;
    } finally {
      _cachedLorasPromise = null;
    }
  })();

  return _cachedLorasPromise;
}

function removeAllWidgets(node) {
  if (!node.widgets) return;
  while (node.widgets.length) node.removeWidget(0);
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

function buildUI(node, rows, globals, loraValues) {
  removeAllWidgets(node);

  node._doraRows = rows || [];
  node._doraGlobals = globals || { stack_enabled: true, verbose: false, log_unloaded_keys: false };

  const loras = Array.isArray(loraValues) ? loraValues : ["None"];

  // Rows
  node._doraRows.forEach((row, i) => {
    const n = i + 1;

    const wEnabled = node.addWidget("toggle", `lora_${n}_enabled`, !!row.enabled, (v) => (row.enabled = !!v));
    wEnabled.label = `LoRA ${n} Enabled`;

    const wName = node.addWidget("combo", `lora_${n}_name`, row.name ?? "None", (v) => (row.name = v), {
      values: loras,
    });
    wName.label = `LoRA ${n}`;

    const wSm = node.addWidget(
      "number",
      `lora_${n}_strength_model`,
      Number.isFinite(row.strengthModel) ? row.strengthModel : 1.0,
      (v) => (row.strengthModel = Number(v)),
      { min: -10.0, max: 10.0, step: 0.05 }
    );
    wSm.label = `Strength (Model)`;

    const wSc = node.addWidget(
      "number",
      `lora_${n}_strength_clip`,
      Number.isFinite(row.strengthClip) ? row.strengthClip : (Number.isFinite(row.strengthModel) ? row.strengthModel : 1.0),
      (v) => (row.strengthClip = Number(v)),
      { min: -10.0, max: 10.0, step: 0.05 }
    );
    wSc.label = `Strength (Clip)`;

    const wRemove = node.addWidget("button", `lora_${n}_remove`, "✖ Remove", () => {
      node._doraRows.splice(i, 1);
      // Rebuild with current cached loras (or "None")
      buildUI(node, node._doraRows, node._doraGlobals, _cachedLoras || ["None"]);
      node.setDirtyCanvas(true, true);
    });
    wRemove.serialize = false;
  });

  // Add row button
  const wAdd = node.addWidget("button", "add_lora", "➕ Add LoRA", () => {
    node._doraRows.push(makeRowDefaults());
    buildUI(node, node._doraRows, node._doraGlobals, _cachedLoras || ["None"]);
    node.setDirtyCanvas(true, true);
  });
  wAdd.serialize = false;

  // Globals (serialized)
  const wStackEnabled = node.addWidget(
    "toggle",
    "stack_enabled",
    !!node._doraGlobals.stack_enabled,
    (v) => (node._doraGlobals.stack_enabled = !!v)
  );
  wStackEnabled.label = "Stack Enabled";

  const wVerbose = node.addWidget("toggle", "verbose", !!node._doraGlobals.verbose, (v) => (node._doraGlobals.verbose = !!v));
  wVerbose.label = "Verbose";

  const wLogMissing = node.addWidget(
    "toggle",
    "log_unloaded_keys",
    !!node._doraGlobals.log_unloaded_keys,
    (v) => (node._doraGlobals.log_unloaded_keys = !!v)
  );
  wLogMissing.label = "Log Unloaded Keys";

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
      // Default: one row
      const rows = [makeRowDefaults()];
      const globals = { stack_enabled: true, verbose: false, log_unloaded_keys: false };
      buildUI(this, rows, globals, _cachedLoras || ["None"]);
      // async populate lora dropdown
      fetchLoras().then((loras) => {
        buildUI(this, this._doraRows || rows, this._doraGlobals || globals, loras);
        this.setDirtyCanvas(true, true);
      });
      return r;
    };

    // Rebuild from saved workflow widget values (so rows persist across save/load)
    const origConfigure = nodeType.prototype.configure;
    nodeType.prototype.configure = function (info) {
      // Do NOT call origConfigure first, because it would try to apply widgets_values to widgets we haven't created yet.
      // Instead: reconstruct rows based on widgets_values length, create widgets, then apply values by assignment.
      const widgetsValues = info?.widgets_values;
      const parsed = parseRowsFromWidgetValues(widgetsValues);
      const rows = parsed.rows.length ? parsed.rows : [makeRowDefaults()];
      const globals = parsed.globals || { stack_enabled: true, verbose: false, log_unloaded_keys: false };
      buildUI(this, rows, globals, _cachedLoras || ["None"]);

      // Now let the base configure handle position/size/etc (it won't re-apply widgets_values meaningfully now).
      const r = origConfigure?.call(this, info);

      fetchLoras().then((loras) => {
        buildUI(this, this._doraRows || rows, this._doraGlobals || globals, loras);
        this.setDirtyCanvas(true, true);
      });
      return r;
    };
  },
});
