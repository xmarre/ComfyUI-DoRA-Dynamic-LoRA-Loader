import { app } from "../../scripts/app.js";

const NODE_CLASS = "DoRA Power LoRA Loader";
const EXT_NAME = "comfyui_dora_dynamic_lora.power_lora_loader";
const LORA_API = "/dora_dynamic_lora/loras";
const ROW_HEIGHT = 24;
const HORIZ_MARGIN = 15;
const WEIGHT_STEP = 0.05;

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

      try {
        const r = await fetch(LORA_API, { cache: "no-store" });
        if (r.ok) {
          const j = await r.json();
          if (Array.isArray(j)) loras = j;
        }
      } catch (_) {}

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

      const list = Array.isArray(loras) ? loras : [];
      const out = ["None", ...list.filter((x) => x && x !== "None" && x !== "NONE")];
      _cachedLoras = out;
      return out;
    } catch (e) {
      console.warn(`[${EXT_NAME}] Failed to fetch LoRAs`, e);
      return ["None"];
    } finally {
      _cachedLorasPromise = null;
    }
  })();

  return _cachedLorasPromise;
}

function getThemeColors() {
  const lg = globalThis.LiteGraph || {};
  return {
    bg: lg.WIDGET_BGCOLOR || "#222",
    text: lg.WIDGET_TEXT_COLOR || "#DDD",
    secondary: lg.WIDGET_SECONDARY_TEXT_COLOR || "#999",
    outline: lg.WIDGET_OUTLINE_COLOR || "#666",
  };
}

function drawRoundedRect(ctx, x, y, w, h, r) {
  if (typeof ctx.roundRect === "function") {
    ctx.beginPath();
    ctx.roundRect(x, y, w, h, [r]);
    return;
  }
  const rr = Math.max(0, Math.min(r, Math.min(w, h) / 2));
  ctx.beginPath();
  ctx.moveTo(x + rr, y);
  ctx.arcTo(x + w, y, x + w, y + h, rr);
  ctx.arcTo(x + w, y + h, x, y + h, rr);
  ctx.arcTo(x, y + h, x, y, rr);
  ctx.arcTo(x, y, x + w, y, rr);
  ctx.closePath();
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

function parseRowsFromWidgetValues(widgetsValues) {
  const vals = Array.isArray(widgetsValues) ? widgetsValues : [];

  const parseWithLayout = (perRow, globalsCount) => {
    if (vals.length < globalsCount) return null;
    if ((vals.length - globalsCount) % perRow !== 0) return null;

    const outRows = [];
    let idx = 0;
    const rowsCount = Math.max(0, Math.floor((vals.length - globalsCount) / perRow));
    for (let i = 0; i < rowsCount; i++) {
      const r = makeRowDefaults();
      r.enabled = Boolean(vals[idx++]);
      const nm = vals[idx++];
      r.name = typeof nm === "string" ? nm : (nm ?? "None");
      const sm = Number(vals[idx++]);
      r.strengthModel = sm;
      if (perRow === 4) {
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

    if (globalsCount >= 7) {
      globals.dora_decompose_debug = Boolean(vals[idx++]);
      globals.dora_decompose_debug_n = Number(vals[idx++]);
      globals.dora_decompose_debug_stack_depth = Number(vals[idx++]);
      globals.dora_slice_fix = Boolean(vals[idx++]);
      globals.dora_adaln_swap_fix = true;
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

  for (const [perRow, globalsCount] of layouts) {
    const parsed = parseWithLayout(perRow, globalsCount);
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
        strengthClip: Number.isFinite(+r.strengthClip)
          ? +r.strengthClip
          : (Number.isFinite(+r.strengthModel) ? +r.strengthModel : 1.0),
      }))
    : [makeRowDefaults()];

  const globals = {
    stack_enabled: globalsIn.stack_enabled !== undefined ? !!globalsIn.stack_enabled : true,
    verbose: globalsIn.verbose !== undefined ? !!globalsIn.verbose : false,
    log_unloaded_keys: globalsIn.log_unloaded_keys !== undefined ? !!globalsIn.log_unloaded_keys : false,
    broadcast_modulations: globalsIn.broadcast_modulations !== undefined ? !!globalsIn.broadcast_modulations : true,
    broadcast_auto_scale: globalsIn.broadcast_auto_scale !== undefined ? !!globalsIn.broadcast_auto_scale : true,
    broadcast_scale: Number.isFinite(+globalsIn.broadcast_scale) ? +globalsIn.broadcast_scale : 1.0,
    broadcast_include_dora_scale:
      globalsIn.broadcast_include_dora_scale !== undefined ? !!globalsIn.broadcast_include_dora_scale : false,
    dora_decompose_debug: globalsIn.dora_decompose_debug !== undefined ? !!globalsIn.dora_decompose_debug : false,
    dora_decompose_debug_n: Number.isFinite(+globalsIn.dora_decompose_debug_n)
      ? Math.max(0, Math.floor(+globalsIn.dora_decompose_debug_n))
      : 30,
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

function persistNodeState(node) {
  setState(node, { rows: node._doraRows || [], globals: node._doraGlobals || {} });
}

function setButtonCallback(widget, cb) {
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

class DoraLoraRowWidget {
  constructor(name, parentNode, row) {
    this.name = name;
    this.type = "custom";
    this.parentNode = parentNode;
    this.row = row;
    this.last_y = 0;
    this.hitAreas = {
      toggle: [0, 0],
      name: [0, 0],
      weightDec: [0, 0],
      weightInc: [0, 0],
      weightVal: [0, 0],
    };
  }

  computeSize(width) {
    return [width, ROW_HEIGHT];
  }

  serializeValue() {
    return {
      on: !!this.row.enabled,
      lora: this.row.name ?? "None",
      strength: Number.isFinite(+this.row.strengthModel) ? +this.row.strengthModel : 1.0,
      strengthTwo: Number.isFinite(+this.row.strengthClip)
        ? +this.row.strengthClip
        : (Number.isFinite(+this.row.strengthModel) ? +this.row.strengthModel : 1.0),
    };
  }

  draw(ctx, node, width, posY, height) {
    const colors = getThemeColors();
    const x = HORIZ_MARGIN;
    const widgetWidth = width - HORIZ_MARGIN * 2;
    const midY = posY + height / 2;
    this.last_y = posY;

    const globalOn = node.widgets.find((w) => w.name === "stack_enabled")?.value !== false;
    const effectiveOn = !!this.row.enabled && globalOn;

    ctx.save();

    ctx.fillStyle = colors.bg;
    ctx.strokeStyle = colors.outline;
    drawRoundedRect(ctx, x, posY + 2, widgetWidth, height - 4, 4);
    ctx.fill();
    ctx.stroke();

    const toggleWidth = 20;
    const toggleHeight = 12;
    const toggleX = x + 8;
    const toggleY = midY - toggleHeight / 2;

    ctx.fillStyle = colors.outline;
    drawRoundedRect(ctx, toggleX, toggleY, toggleWidth, toggleHeight, toggleHeight / 2);
    ctx.fill();

    ctx.fillStyle = globalOn ? colors.text : colors.secondary;
    const knobX = this.row.enabled ? toggleX + toggleWidth - 6 : toggleX + 6;
    ctx.beginPath();
    ctx.arc(knobX, midY, 4, 0, Math.PI * 2);
    ctx.fill();
    this.hitAreas.toggle = [toggleX, toggleX + toggleWidth];

    const weightAreaWidth = 74;
    const nameX = toggleX + toggleWidth + 10;
    const nameWidth = widgetWidth - (nameX - x) - weightAreaWidth - 4;

    ctx.fillStyle = effectiveOn ? colors.text : colors.secondary;
    ctx.font = "12px Arial";
    ctx.textAlign = "left";

    let displayLabel = this.row.name || "None";
    const metrics = ctx.measureText(displayLabel);
    if (metrics.width > nameWidth) {
      const avgCharWidth = 7;
      const maxChars = Math.max(4, Math.floor(nameWidth / avgCharWidth));
      if (displayLabel.length > maxChars) {
        displayLabel = `${displayLabel.substring(0, maxChars - 3)}...`;
      }
    }

    ctx.fillText(displayLabel, nameX, midY + 4);
    this.hitAreas.name = [nameX, nameX + nameWidth];

    const rightX = x + widgetWidth - 5;
    const valueCenter = rightX - 35;
    const arrowGap = 22;
    const weightValue = Number.isFinite(+this.row.strengthModel) ? +this.row.strengthModel : 1.0;

    ctx.textAlign = "center";
    ctx.fillText(">", valueCenter + arrowGap, midY + 4);
    this.hitAreas.weightInc = [valueCenter + arrowGap - 10, valueCenter + arrowGap + 10];

    ctx.fillStyle = effectiveOn ? colors.text : colors.secondary;
    ctx.fillText(weightValue.toFixed(2), valueCenter, midY + 4);
    this.hitAreas.weightVal = [valueCenter - 18, valueCenter + 18];

    ctx.fillText("<", valueCenter - arrowGap, midY + 4);
    this.hitAreas.weightDec = [valueCenter - arrowGap - 10, valueCenter - arrowGap + 10];

    ctx.restore();
  }

  async chooseLora(event) {
    const loras = await fetchLoras();
    const ContextMenu = globalThis.LiteGraph?.ContextMenu;
    if (!ContextMenu) return;
    new ContextMenu(loras, {
      event,
      callback: (value) => {
        this.row.name = value || "None";
        persistNodeState(this.parentNode);
        this.parentNode.setDirtyCanvas(true, true);
      },
    });
  }

  updateWeight(next) {
    const value = Number.isFinite(+next) ? +next : 1.0;
    this.row.strengthModel = value;
    this.row.strengthClip = value;
    persistNodeState(this.parentNode);
  }

  openRowMenu(event, node) {
    const ContextMenu = globalThis.LiteGraph?.ContextMenu;
    if (!ContextMenu) return false;

    const rows = node._doraRows || [];
    const rowIndex = rows.indexOf(this.row);
    if (rowIndex < 0) return false;

    const items = [
      {
        content: this.row.enabled ? "Toggle Off" : "Toggle On",
        callback: () => {
          this.row.enabled = !this.row.enabled;
          persistNodeState(this.parentNode);
          this.parentNode.setDirtyCanvas(true, true);
        },
      },
      rowIndex > 0
        ? {
            content: "Move Up",
            callback: () => {
              [rows[rowIndex - 1], rows[rowIndex]] = [rows[rowIndex], rows[rowIndex - 1]];
              persistNodeState(this.parentNode);
              buildUI(this.parentNode, getState(this.parentNode), _cachedLoras || ["None"]);
              this.parentNode.setDirtyCanvas(true, true);
            },
          }
        : null,
      rowIndex < rows.length - 1
        ? {
            content: "Move Down",
            callback: () => {
              [rows[rowIndex], rows[rowIndex + 1]] = [rows[rowIndex + 1], rows[rowIndex]];
              persistNodeState(this.parentNode);
              buildUI(this.parentNode, getState(this.parentNode), _cachedLoras || ["None"]);
              this.parentNode.setDirtyCanvas(true, true);
            },
          }
        : null,
      {
        content: "Remove",
        callback: () => {
          rows.splice(rowIndex, 1);
          if (!rows.length) rows.push(makeRowDefaults());
          persistNodeState(this.parentNode);
          buildUI(this.parentNode, getState(this.parentNode), _cachedLoras || ["None"]);
          this.parentNode.setDirtyCanvas(true, true);
        },
      },
    ].filter(Boolean);

    new ContextMenu(items, { event });
    return true;
  }

  mouse(event, pos, node) {
    if (event.type !== "pointerdown") return false;

    if (event.button === 2) {
      event.preventDefault?.();
      event.stopPropagation?.();
      return this.openRowMenu(event, node);
    }

    const globalOn = node.widgets.find((w) => w.name === "stack_enabled")?.value !== false;
    if (!globalOn) return false;

    const widgetX = pos[0];

    if (widgetX >= this.hitAreas.toggle[0] && widgetX <= this.hitAreas.toggle[1]) {
      this.row.enabled = !this.row.enabled;
      persistNodeState(this.parentNode);
    } else if (widgetX >= this.hitAreas.name[0] && widgetX <= this.hitAreas.name[1]) {
      this.chooseLora(event);
    } else if (widgetX >= this.hitAreas.weightDec[0] && widgetX <= this.hitAreas.weightDec[1]) {
      this.updateWeight(Math.round(((+this.row.strengthModel || 1.0) - WEIGHT_STEP) * 100) / 100);
    } else if (widgetX >= this.hitAreas.weightInc[0] && widgetX <= this.hitAreas.weightInc[1]) {
      this.updateWeight(Math.round(((+this.row.strengthModel || 1.0) + WEIGHT_STEP) * 100) / 100);
    } else if (widgetX >= this.hitAreas.weightVal[0] && widgetX <= this.hitAreas.weightVal[1]) {
      app.canvas.prompt(
        "LoRA Weight",
        Number.isFinite(+this.row.strengthModel) ? +this.row.strengthModel : 1.0,
        (v) => {
          const parsed = parseFloat(v);
          if (!Number.isNaN(parsed)) {
            this.updateWeight(parsed);
            node.setDirtyCanvas(true, true);
          }
        },
        event
      );
    } else {
      return false;
    }

    this.parentNode.setDirtyCanvas(true, true);
    return true;
  }
}

function buildUI(node, state, loraValues) {
  console.log(`[${EXT_NAME}] buildUI`, {
    widgetsBefore: node.widgets?.map((w) => w.name),
    rows: state?.rows,
  });
  const st = sanitizeState(state);
  setState(node, st);
  removeAllWidgets(node);
  node._doraRows = st.rows;
  node._doraGlobals = st.globals;

  const loras = Array.isArray(loraValues) ? loraValues : ["None"];

  const wRefresh = node.addWidget("button", "refresh_loras", "Refresh LoRAs");
  setButtonCallback(wRefresh, () => {
    invalidateLorasCache();
    rebuild(node);
  });
  wRefresh.serialize = false;

  node._doraRows.forEach((row, i) => {
    const widget = new DoraLoraRowWidget(`LORA_${i + 1}`, node, row);
    if (typeof node.addCustomWidget === "function") {
      node.addCustomWidget(widget);
    } else {
      node.widgets.push(widget);
    }
  });

  const wAdd = node.addWidget("button", "add_lora", "Add LoRA");
  setButtonCallback(wAdd, () => {
    node._doraRows.push(makeRowDefaults());
    persistNodeState(node);
    buildUI(node, getState(node), _cachedLoras || ["None"]);
    node.setDirtyCanvas(true, true);
  });
  wAdd.serialize = false;

  const wStackEnabled = node.addWidget(
    "toggle",
    "stack_enabled",
    !!node._doraGlobals.stack_enabled,
    (v) => {
      node._doraGlobals.stack_enabled = !!v;
      persistNodeState(node);
      node.setDirtyCanvas(true, true);
    }
  );
  wStackEnabled.label = "Stack Enabled";

  const wVerbose = node.addWidget("toggle", "verbose", !!node._doraGlobals.verbose, (v) => {
    node._doraGlobals.verbose = !!v;
    persistNodeState(node);
  });
  wVerbose.label = "Verbose";

  const wLogMissing = node.addWidget(
    "toggle",
    "log_unloaded_keys",
    !!node._doraGlobals.log_unloaded_keys,
    (v) => {
      node._doraGlobals.log_unloaded_keys = !!v;
      persistNodeState(node);
    }
  );
  wLogMissing.label = "Log Unloaded Keys";

  const wBroadcastMods = node.addWidget(
    "toggle",
    "broadcast_modulations",
    !!node._doraGlobals.broadcast_modulations,
    (v) => {
      node._doraGlobals.broadcast_modulations = !!v;
      persistNodeState(node);
    }
  );
  wBroadcastMods.label = "Broadcast OneTrainer modulation LoRAs";

  const wIncludeDoraScale = node.addWidget(
    "toggle",
    "broadcast_include_dora_scale",
    !!node._doraGlobals.broadcast_include_dora_scale,
    (v) => {
      node._doraGlobals.broadcast_include_dora_scale = !!v;
      persistNodeState(node);
    }
  );
  wIncludeDoraScale.label = "Include DoRA dora_scale in broadcast";

  const wAutoScale = node.addWidget(
    "toggle",
    "broadcast_auto_scale",
    !!node._doraGlobals.broadcast_auto_scale,
    (v) => {
      node._doraGlobals.broadcast_auto_scale = !!v;
      persistNodeState(node);
    }
  );
  wAutoScale.label = "Auto-scale broadcast";

  const wScale = node.addWidget(
    "number",
    "broadcast_scale",
    Number.isFinite(node._doraGlobals.broadcast_scale) ? node._doraGlobals.broadcast_scale : 1.0,
    (v) => {
      node._doraGlobals.broadcast_scale = Number(v);
      persistNodeState(node);
    },
    { min: 0.0, max: 64.0, step: 0.05 }
  );
  wScale.label = "Broadcast scale";

  const wDoraSliceFix = node.addWidget(
    "toggle",
    "dora_slice_fix",
    !!node._doraGlobals.dora_slice_fix,
    (v) => {
      node._doraGlobals.dora_slice_fix = !!v;
      persistNodeState(node);
    }
  );
  wDoraSliceFix.label = "DoRA slice-fix for offset patches (Flux2)";

  const wDoraAdaLNSwap = node.addWidget(
    "toggle",
    "dora_adaln_swap_fix",
    !!node._doraGlobals.dora_adaln_swap_fix,
    (v) => {
      node._doraGlobals.dora_adaln_swap_fix = !!v;
      persistNodeState(node);
    }
  );
  wDoraAdaLNSwap.label = "DoRA adaLN swap_scale_shift fix";

  const wDoraDbg = node.addWidget(
    "toggle",
    "dora_decompose_debug",
    !!node._doraGlobals.dora_decompose_debug,
    (v) => {
      node._doraGlobals.dora_decompose_debug = !!v;
      persistNodeState(node);
    }
  );
  wDoraDbg.label = "DoRA decompose debug logs";

  const wDoraDbgN = node.addWidget(
    "number",
    "dora_decompose_debug_n",
    Number.isFinite(+node._doraGlobals.dora_decompose_debug_n) ? +node._doraGlobals.dora_decompose_debug_n : 30,
    (v) => {
      node._doraGlobals.dora_decompose_debug_n = Math.max(0, Math.floor(+v || 0));
      persistNodeState(node);
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
      persistNodeState(node);
    },
    { min: 2, max: 64, step: 1 }
  );
  wDoraDbgStack.label = "DoRA debug stack depth";

  const size = node.computeSize();
  node.size[0] = Math.max(node.size[0], size[0]);
  node.size[1] = Math.max(node.size[1], size[1]);
}

app.registerExtension({
  name: EXT_NAME,
  async beforeRegisterNodeDef(nodeType, nodeData) {
    const nodeName = nodeData?.name ?? "";
    const displayName = nodeData?.display_name ?? "";
    const comfyClass = nodeType?.comfyClass ?? "";
    const pythonClass = nodeData?.python_module ?? "";

    const isTarget =
      nodeName === NODE_CLASS ||
      displayName === NODE_CLASS ||
      comfyClass === NODE_CLASS;

    if (!isTarget) return;

    console.log(`[${EXT_NAME}] attaching frontend override`, {
      nodeName,
      displayName,
      comfyClass,
      pythonClass,
    });

    const origOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = origOnNodeCreated?.apply(this, arguments);
      console.log(`[${EXT_NAME}] onNodeCreated`, this);
      rebuild(this);
      return r;
    };

    const origConfigure = nodeType.prototype.configure;
    nodeType.prototype.configure = function (info) {
      try {
        console.log(`[${EXT_NAME}] configure`, info);
        let st = null;
        if (info?.properties?.dora_power_lora) {
          st = sanitizeState(info.properties.dora_power_lora);
        } else if (Array.isArray(info?.widgets_values) && info.widgets_values.length) {
          const parsed = parseRowsFromWidgetValues(info.widgets_values);
          st = sanitizeState({
            rows: parsed.rows,
            globals: {
              ...parsed.globals,
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
          });
        } else {
          st = defaultState();
        }
        setState(this, st);

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
