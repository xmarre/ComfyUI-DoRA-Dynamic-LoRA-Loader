import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_CLASS = "DoRA Power LoRA Loader";
const EXT_NAME = "comfyui_dora_dynamic_lora.power_lora_loader";
const LORA_API = "/dora_dynamic_lora/loras";
const ROW_HEIGHT = 24;
const HORIZ_MARGIN = 15;
const WEIGHT_STEP = 0.05;
const WEIGHT_DRAG_PIXELS_PER_STEP = 6;
const AUTO_STRENGTH_DEVICE_CHOICES = ["auto", "cpu", "gpu"];

function roundToStep(v, step = WEIGHT_STEP) {
  const n = Number.isFinite(+v) ? +v : 0;
  return Number((Math.round(n / step) * step).toFixed(10));
}

function closeActiveLoraPicker() {
  const picker = globalThis.__doraActiveLoraPicker;
  if (picker && typeof picker.close === "function") picker.close();
  globalThis.__doraActiveLoraPicker = null;
}

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

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function formatNumber(value, digits = 3) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "—";
  return n.toFixed(digits);
}

function toFiniteNumberOrNull(value) {
  if (value == null) return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function setCanvasTooltip(text) {
  const canvas = app?.canvas?.canvas;
  if (canvas) {
    canvas.title = text || "";
  }
}

function clearCanvasTooltip() {
  setCanvasTooltip("");
}

function fitText(ctx, text, maxWidth) {
  const s = String(text ?? "");
  if (maxWidth <= 8 || !s) return "";
  if (ctx.measureText(s).width <= maxWidth) return s;
  let out = s;
  while (out.length > 3 && ctx.measureText(`${out}…`).width > maxWidth) {
    out = out.slice(0, -1);
  }
  return out.length ? `${out}…` : "";
}

function syncNodeHeightToContent(node) {
  if (!node || typeof node.computeSize !== "function") return;
  const size = node.computeSize();
  if (!Array.isArray(size) || size.length < 2) return;

  const minSize = Array.isArray(node.min_size) ? node.min_size : [];
  const minWidth = Number(minSize[0]) || 0;
  const minHeight = Number(minSize[1]) || 0;

  node.size = Array.isArray(node.size) ? node.size : [0, 0];
  node.size[0] = Math.max(Number(node.size[0]) || 0, minWidth);
  node.size[1] = Math.max(size[1], minHeight);
}

const DISPLAY_GROUP_LIMIT = 3;
const DISPLAY_RATIO_EPS = 1e-3;
const CLASSIC_DISPLAY_PREFIXES = [
  "lora_unet_",
  "lycoris_",
  "diffusion_model.",
  "diffusion_model_",
  "base_model.model.",
  "base_model_model_",
  "base_model.",
  "base_model_",
  "transformer.",
  "transformer_",
  "model.",
  "model_",
  "unet.",
  "unet_",
];

function normalizeClassicDisplayGroupKind(parts) {
  const tokens = new Set(parts);
  if (tokens.has("attentions") || tokens.has("attention") || tokens.has("attn")) return "attn";
  if (
    tokens.has("resnets") ||
    tokens.has("resnet") ||
    tokens.has("in_layers") ||
    tokens.has("out_layers") ||
    tokens.has("emb_layers") ||
    tokens.has("skip_connection") ||
    tokens.has("nin_shortcut") ||
    tokens.has("conv1") ||
    tokens.has("conv2")
  ) {
    return "resnet";
  }
  if (tokens.has("downsamplers") || tokens.has("downsampler") || tokens.has("downsample")) return "downsampler";
  if (tokens.has("upsamplers") || tokens.has("upsampler") || tokens.has("upsample")) return "upsampler";
  if (tokens.has("proj") || tokens.has("proj_in") || tokens.has("proj_out") || tokens.has("time_emb_proj")) return "proj";
  if (tokens.has("conv") || tokens.has("conv_in") || tokens.has("conv_out")) return "conv";
  if (tokens.has("norm")) return "norm";
  return null;
}

function normalizeClassicDisplayGroupId(base) {
  let text = String(base ?? "").trim();
  if (!text) return null;

  for (const prefix of CLASSIC_DISPLAY_PREFIXES) {
    if (text.startsWith(prefix)) {
      text = text.slice(prefix.length);
      break;
    }
  }

  if (!text) return null;

  const parts = text.replace(/[./]+/g, "_").split("_").filter(Boolean);
  if (!parts.length) return null;

  let root = null;
  let cursor = 0;
  if (parts[0] === "middle" && parts[1] === "block") {
    root = "middle_block";
    cursor = 2;
  } else if ((parts[0] === "input" || parts[0] === "output" || parts[0] === "down" || parts[0] === "up") && parts[1] === "blocks") {
    root = `${parts[0]}_blocks`;
    cursor = 2;
  } else {
    return null;
  }

  let blockIndex = null;
  if (cursor < parts.length && /^\d+$/.test(parts[cursor])) {
    blockIndex = parts[cursor];
    cursor += 1;
  }
  if (cursor < parts.length && /^\d+$/.test(parts[cursor])) {
    cursor += 1;
  }

  const kind = normalizeClassicDisplayGroupKind(parts.slice(cursor));
  if (root === "middle_block") {
    return kind ? `${root}.${kind}` : root;
  }
  if (blockIndex == null) {
    return kind ? `${root}.${kind}` : root;
  }
  return kind ? `${root}.${blockIndex}.${kind}` : `${root}.${blockIndex}`;
}

function buildDisplayGroups(logicalGroups) {
  const buckets = new Map();
  const groups = Array.isArray(logicalGroups) ? logicalGroups : [];

  for (const item of groups) {
    if (!item || typeof item !== "object") continue;

    const rawId = String(item.logical_id ?? "");
    const displayId = normalizeClassicDisplayGroupId(rawId) || rawId || "?";
    let bucket = buckets.get(displayId);
    if (!bucket) {
      bucket = {
        display_id: displayId,
        fanout: 0,
        logical_group_count: 0,
        measured_base_count: 0,
        ratio_weight: 0,
        ratio_weight_count: 0,
      };
      buckets.set(displayId, bucket);
    }

    const fanout = Math.max(1, Number(item.fanout) || 0);
    const ratio = toFiniteNumberOrNull(item.ratio_applied);
    if (ratio !== null) {
      bucket.ratio_weight += ratio * fanout;
      bucket.ratio_weight_count += fanout;
    }
    bucket.fanout += fanout;
    bucket.logical_group_count += 1;
    bucket.measured_base_count += Number(item.measured_base_count) || 0;
  }

  const out = [];
  for (const bucket of buckets.values()) {
    out.push({
      display_id: bucket.display_id,
      fanout: bucket.fanout || bucket.logical_group_count,
      logical_group_count: bucket.logical_group_count,
      measured_base_count: bucket.measured_base_count,
      ratio_applied: bucket.ratio_weight_count > 0 ? bucket.ratio_weight / bucket.ratio_weight_count : null,
    });
  }

  out.sort((a, b) => String(a.display_id).localeCompare(String(b.display_id)));
  return out;
}

function compareDisplayGroupsByRatio(a, b, descending = false) {
  const ar = toFiniteNumberOrNull(a?.ratio_applied) ?? 1.0;
  const br = toFiniteNumberOrNull(b?.ratio_applied) ?? 1.0;
  if (Math.abs(ar - br) > 1e-9) {
    return descending ? br - ar : ar - br;
  }
  return String(a?.display_id ?? "").localeCompare(String(b?.display_id ?? ""));
}

function buildDisplayGroupSections(logicalGroups, expanded) {
  const groups = buildDisplayGroups(logicalGroups);
  const boosts = [];
  const pullbacks = [];
  const neutral = [];

  for (const group of groups) {
    const ratio = toFiniteNumberOrNull(group.ratio_applied);
    if (ratio == null || Math.abs(ratio - 1.0) <= DISPLAY_RATIO_EPS) {
      neutral.push(group);
    } else if (ratio > 1.0) {
      boosts.push(group);
    } else {
      pullbacks.push(group);
    }
  }

  boosts.sort((a, b) => compareDisplayGroupsByRatio(a, b, true));
  pullbacks.sort((a, b) => compareDisplayGroupsByRatio(a, b, false));
  neutral.sort((a, b) => String(a.display_id).localeCompare(String(b.display_id)));

  const sections = [];
  const addSection = (kind, title, items, limit) => {
    if (!items.length) return;
    const visibleItems = expanded || limit == null ? items.slice() : items.slice(0, limit);
    sections.push({
      kind,
      title,
      items: visibleItems,
      totalCount: items.length,
      visibleCount: visibleItems.length,
    });
  };

  addSection("boost", "Strongest boosts", boosts, DISPLAY_GROUP_LIMIT);
  addSection("pullback", "Strongest pullbacks", pullbacks, DISPLAY_GROUP_LIMIT);
  if (expanded) {
    addSection("neutral", "Near global", neutral, null);
  } else if (!boosts.length && !pullbacks.length) {
    addSection("neutral", "Near global", neutral, 6);
  }

  return {
    groups,
    sections,
    hasMore: !expanded && sections.some((section) => section.visibleCount < section.totalCount),
  };
}

function parseExecutionOutputValue(output, key) {
  const value = output?.[key];
  if (Array.isArray(value)) return value[0] ?? null;
  return typeof value === "string" ? value : null;
}

function clearExecutionReport(node) {
  if (!node) return;
  node._doraAutoStrengthReport = null;
  node._doraAutoStrengthReportJson = "";
  node._doraAutoStrengthReportText = "";
  node._doraAutoStrengthExpanded = {};
  syncNodeHeightToContent(node);
  node.setDirtyCanvas?.(true, true);
}

function isTargetNodeInstance(node) {
  return !!node && (node.type === NODE_CLASS || node.comfyClass === NODE_CLASS || node.title === NODE_CLASS);
}

function applyExecutionReportToNode(node, reportJson, reportText) {
  if (!node) return;
  let parsed = null;
  if (typeof reportJson === "string" && reportJson.trim()) {
    try {
      parsed = JSON.parse(reportJson);
    } catch (err) {
      console.warn(`[${EXT_NAME}] Failed to parse auto-strength report JSON`, err);
    }
  }
  node._doraAutoStrengthReport = parsed;
  node._doraAutoStrengthReportJson = typeof reportJson === "string" ? reportJson : "";
  node._doraAutoStrengthReportText = typeof reportText === "string" ? reportText : "";
  node._doraAutoStrengthExpanded = node._doraAutoStrengthExpanded || {};
  syncNodeHeightToContent(node);
  node.setDirtyCanvas?.(true, true);
}

let _executionListenerInstalled = false;
function installExecutionListener() {
  if (_executionListenerInstalled) return;
  _executionListenerInstalled = true;
  if (!api?.addEventListener) return;

  api.addEventListener("executed", (event) => {
    const detail = event?.detail || event || {};
    const output = detail?.output || {};
    const reportJson = parseExecutionOutputValue(output, "auto_strength_report_json");
    const reportText = parseExecutionOutputValue(output, "analysis_report");
    if (typeof reportJson !== "string" && typeof reportText !== "string") return;

    const rawNodeId = detail?.display_node ?? detail?.node;
    const nodeId = Number(rawNodeId);
    const graph = app.graph;
    const node = Number.isFinite(nodeId) && graph?.getNodeById ? graph.getNodeById(nodeId) : null;
    if (!isTargetNodeInstance(node)) return;

    applyExecutionReportToNode(node, reportJson, reportText);
  });
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

function normalizeAutoStrengthDevice(value) {
  if (Number.isInteger(value) && value >= 0 && value < AUTO_STRENGTH_DEVICE_CHOICES.length) {
    return AUTO_STRENGTH_DEVICE_CHOICES[value];
  }
  const v = String(value ?? "auto").trim().toLowerCase();
  return AUTO_STRENGTH_DEVICE_CHOICES.includes(v) ? v : "auto";
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
      auto_strength_enabled: false,
      auto_strength_device: "gpu",
      auto_strength_ratio_floor: 0.30,
      auto_strength_ratio_ceiling: 1.50,
      dora_decompose_debug: false,
      dora_decompose_debug_n: 30,
      dora_decompose_debug_stack_depth: 10,
      dora_slice_fix: true,
      dora_adaln_swap_fix: true,
      zimage_lumina2_compat: true,
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
    auto_strength_enabled: globalsIn.auto_strength_enabled !== undefined ? !!globalsIn.auto_strength_enabled : false,
    auto_strength_device: normalizeAutoStrengthDevice(globalsIn.auto_strength_device ?? "gpu"),
    auto_strength_ratio_floor: Number.isFinite(+globalsIn.auto_strength_ratio_floor)
      ? Math.max(0, +globalsIn.auto_strength_ratio_floor)
      : 0.30,
    auto_strength_ratio_ceiling: Number.isFinite(+globalsIn.auto_strength_ratio_ceiling)
      ? Math.max(0, +globalsIn.auto_strength_ratio_ceiling)
      : 1.50,
    dora_decompose_debug: globalsIn.dora_decompose_debug !== undefined ? !!globalsIn.dora_decompose_debug : false,
    dora_decompose_debug_n: Number.isFinite(+globalsIn.dora_decompose_debug_n)
      ? Math.max(0, Math.floor(+globalsIn.dora_decompose_debug_n))
      : 30,
    dora_decompose_debug_stack_depth: Number.isFinite(+globalsIn.dora_decompose_debug_stack_depth)
      ? Math.max(2, Math.min(64, Math.floor(+globalsIn.dora_decompose_debug_stack_depth)))
      : 10,
    dora_slice_fix: globalsIn.dora_slice_fix !== undefined ? !!globalsIn.dora_slice_fix : true,
    dora_adaln_swap_fix: globalsIn.dora_adaln_swap_fix !== undefined ? !!globalsIn.dora_adaln_swap_fix : true,
    zimage_lumina2_compat: globalsIn.zimage_lumina2_compat !== undefined ? !!globalsIn.zimage_lumina2_compat : true,
  };

  if (globals.auto_strength_ratio_ceiling < globals.auto_strength_ratio_floor) {
    const tmp = globals.auto_strength_ratio_floor;
    globals.auto_strength_ratio_floor = globals.auto_strength_ratio_ceiling;
    globals.auto_strength_ratio_ceiling = tmp;
  }

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
  clearExecutionReport(node);
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
    this._drag = null;
    this.nameTooltip = "";
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

    const fullLabel = String(this.row.name || "None");
    const displayLabel = fitText(ctx, fullLabel, nameWidth);
    this.nameTooltip = displayLabel !== fullLabel ? fullLabel : "";

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

  async openLoraPicker(event) {
    const loras = await fetchLoras();
    closeActiveLoraPicker();

    const picker = document.createElement("div");
    picker.style.position = "fixed";
    picker.style.left = `${Math.max(8, Math.floor(event.clientX || 0) - 12)}px`;
    picker.style.top = `${Math.max(8, Math.floor(event.clientY || 0) - 12)}px`;
    picker.style.width = "520px";
    picker.style.maxWidth = "min(520px, calc(100vw - 16px))";
    picker.style.maxHeight = "min(70vh, 560px)";
    picker.style.background = "#18181b";
    picker.style.border = "1px solid #3f3f46";
    picker.style.borderRadius = "10px";
    picker.style.boxShadow = "0 12px 30px rgba(0,0,0,0.45)";
    picker.style.padding = "10px";
    picker.style.zIndex = "100000";
    picker.style.display = "flex";
    picker.style.flexDirection = "column";
    picker.style.gap = "8px";

    const input = document.createElement("input");
    input.type = "text";
    input.placeholder = "Filter list";
    input.value = "";
    input.style.width = "100%";
    input.style.boxSizing = "border-box";
    input.style.background = "#09090b";
    input.style.color = "#e4e4e7";
    input.style.border = "1px solid #52525b";
    input.style.borderRadius = "8px";
    input.style.padding = "8px 10px";
    input.style.outline = "none";

    const list = document.createElement("div");
    list.style.overflowY = "auto";
    list.style.maxHeight = "min(52vh, 420px)";
    list.style.display = "flex";
    list.style.flexDirection = "column";
    list.style.gap = "4px";
    list.style.paddingRight = "2px";
    list.style.boxSizing = "border-box";

    picker.append(input, list);
    document.body.appendChild(picker);

    let filtered = [];
    let activeIndex = Math.max(0, loras.indexOf(this.row.name ?? "None"));

    const selectValue = (value) => {
      this.row.name = value || "None";
      persistNodeState(this.parentNode);
      this.parentNode.setDirtyCanvas(true, true);
      close();
    };

    const render = () => {
      const q = input.value.trim().toLowerCase();
      filtered = loras.filter((name) => !q || String(name).toLowerCase().includes(q));
      if (!filtered.length) {
        activeIndex = 0;
        list.innerHTML = `<div style="padding:8px 10px;color:#a1a1aa;">No matches</div>`;
        return;
      }
      activeIndex = Math.max(0, Math.min(activeIndex, filtered.length - 1));
      list.innerHTML = "";
      filtered.forEach((name, index) => {
        const item = document.createElement("div");
        item.textContent = name;
        item.style.display = "flex";
        item.style.alignItems = "center";
        item.style.width = "100%";
        item.style.boxSizing = "border-box";
        item.style.minHeight = "32px";
        item.style.height = "32px";
        item.style.padding = "0 10px";
        item.style.background = index === activeIndex ? "#27272a" : "#111215";
        item.style.color = "#e4e4e7";
        item.style.border = "1px solid #3f3f46";
        item.style.borderRadius = "8px";
        item.style.fontSize = "13px";
        item.style.lineHeight = "1.2";
        item.style.fontFamily = "Arial, sans-serif";
        item.style.cursor = "pointer";
        item.style.whiteSpace = "nowrap";
        item.style.overflow = "hidden";
        item.style.textOverflow = "ellipsis";
        item.style.userSelect = "none";
        item.onmouseenter = () => {
          activeIndex = index;
          render();
        };
        item.onpointerdown = (e) => {
          e.preventDefault();
          e.stopPropagation();
          selectValue(name);
        };
        list.appendChild(item);
      });
    };

    const onPointerDownOutside = (e) => {
      if (!picker.contains(e.target)) close();
    };

    const onKeyDown = (e) => {
      if (e.key === "Escape") {
        e.preventDefault();
        close();
      } else if (e.key === "ArrowDown") {
        if (!filtered.length) return;
        e.preventDefault();
        activeIndex = Math.min(filtered.length - 1, activeIndex + 1);
        render();
      } else if (e.key === "ArrowUp") {
        if (!filtered.length) return;
        e.preventDefault();
        activeIndex = Math.max(0, activeIndex - 1);
        render();
      } else if (e.key === "Enter") {
        if (!filtered.length) return;
        e.preventDefault();
        selectValue(filtered[activeIndex]);
      }
    };

    const close = () => {
      document.removeEventListener("pointerdown", onPointerDownOutside, true);
      input.removeEventListener("keydown", onKeyDown);
      picker.remove();
      if (globalThis.__doraActiveLoraPicker?.close === close) {
        globalThis.__doraActiveLoraPicker = null;
      }
    };

    globalThis.__doraActiveLoraPicker = { close };
    document.addEventListener("pointerdown", onPointerDownOutside, true);
    input.addEventListener("keydown", onKeyDown);
    input.addEventListener("input", () => {
      activeIndex = 0;
      render();
    });

    render();
    queueMicrotask(() => {
      input.focus();
      input.select();
    });
  }

  openNumericWeightPrompt(event) {
    app.canvas.prompt(
      "LoRA Weight",
      Number.isFinite(+this.row.strengthModel) ? +this.row.strengthModel : 1.0,
      (v) => {
        const parsed = parseFloat(v);
        if (!Number.isNaN(parsed)) {
          this.updateWeight(parsed);
          this.parentNode.setDirtyCanvas(true, true);
        }
      },
      event
    );
  }

  updateWeight(next) {
    const value = roundToStep(Number.isFinite(+next) ? +next : 1.0);
    this.row.strengthModel = value;
    this.row.strengthClip = value;
    persistNodeState(this.parentNode);
  }

  startWeightDrag(event) {
    if (this._drag) this.finishWeightDrag(event);

    const onPointerMove = (moveEvent) => this.handleWeightDrag(moveEvent);
    const onPointerUp = (upEvent) => this.finishWeightDrag(upEvent);

    this._drag = {
      startClientX: Number.isFinite(event?.clientX) ? event.clientX : 0,
      startValue: Number.isFinite(+this.row.strengthModel) ? +this.row.strengthModel : 1.0,
      moved: false,
      lastSteps: 0,
      onPointerMove,
      onPointerUp,
    };

    window.addEventListener("pointermove", onPointerMove, true);
    window.addEventListener("pointerup", onPointerUp, true);
    window.addEventListener("pointercancel", onPointerUp, true);
  }

  handleWeightDrag(event) {
    if (!this._drag) return false;
    const clientX = Number.isFinite(event?.clientX) ? event.clientX : this._drag.startClientX;
    const dx = clientX - this._drag.startClientX;
    const steps = Math.trunc(dx / WEIGHT_DRAG_PIXELS_PER_STEP);
    const next = this._drag.startValue + steps * WEIGHT_STEP;
    if (Math.abs(dx) >= 2) this._drag.moved = true;
    if (steps !== this._drag.lastSteps) {
      this._drag.lastSteps = steps;
      this.updateWeight(next);
      this.parentNode.setDirtyCanvas(true, true);
    }
    event.preventDefault?.();
    return true;
  }

  finishWeightDrag(event) {
    if (!this._drag) return false;
    const drag = this._drag;
    this._drag = null;
    window.removeEventListener("pointermove", drag.onPointerMove, true);
    window.removeEventListener("pointerup", drag.onPointerUp, true);
    window.removeEventListener("pointercancel", drag.onPointerUp, true);
    if (!drag.moved) this.openNumericWeightPrompt(event);
    this.parentNode.setDirtyCanvas(true, true);
    event.preventDefault?.();
    return true;
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
    if (event.type === "pointermove" || event.type === "mousemove") {
      const dragging = this.handleWeightDrag(event);
      const widgetX = pos[0];
      const hoverText =
        !this._drag &&
        this.nameTooltip &&
        widgetX >= this.hitAreas.name[0] &&
        widgetX <= this.hitAreas.name[1]
          ? this.nameTooltip
          : "";
      setCanvasTooltip(hoverText);
      return dragging;
    }
    if (event.type === "pointerleave" || event.type === "mouseout") {
      clearCanvasTooltip();
      return false;
    }
    if (event.type === "pointerup" || event.type === "pointercancel" || event.type === "mouseup") {
      return this.finishWeightDrag(event);
    }
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
      this.openLoraPicker(event);
    } else if (widgetX >= this.hitAreas.weightDec[0] && widgetX <= this.hitAreas.weightDec[1]) {
      this.updateWeight((+this.row.strengthModel || 1.0) - WEIGHT_STEP);
    } else if (widgetX >= this.hitAreas.weightInc[0] && widgetX <= this.hitAreas.weightInc[1]) {
      this.updateWeight((+this.row.strengthModel || 1.0) + WEIGHT_STEP);
    } else if (widgetX >= this.hitAreas.weightVal[0] && widgetX <= this.hitAreas.weightVal[1]) {
      event.preventDefault?.();
      event.stopPropagation?.();
      this.startWeightDrag(event);
    } else {
      return false;
    }

    this.parentNode.setDirtyCanvas(true, true);
    return true;
  }
}

class DoraAutoStrengthReportWidget {
  constructor(name, parentNode) {
    this.name = name;
    this.type = "custom";
    this.parentNode = parentNode;
    this.last_y = 0;
    this.hitAreas = [];
    this.hoverAreas = [];
  }

  serializeValue() {
    return null;
  }

  _rows() {
    const rows = this.parentNode?._doraAutoStrengthReport?.rows;
    return Array.isArray(rows) ? rows : [];
  }

  _displayRows() {
    return this._rows().filter((row) => row && typeof row === "object");
  }

  _rowLayout(row) {
    const analyzed = row.status === "analyzed" && Array.isArray(row.report?.logical_groups);
    const logicalGroups = analyzed ? row.report.logical_groups : [];
    const expanded = !!this.parentNode?._doraAutoStrengthExpanded?.[row.row_index];
    const grouped = analyzed ? buildDisplayGroupSections(logicalGroups, expanded) : { groups: [], sections: [], hasMore: false };
    return {
      analyzed,
      expanded,
      logicalGroups,
      displayGroups: grouped.groups,
      sections: grouped.sections,
      hasMore: grouped.hasMore,
    };
  }

  _estimateRowHeight(layout) {
    if (!layout.analyzed) return 68;
    let height = 92;
    for (const section of layout.sections) {
      height += 18;
      height += section.items.length * 20;
      if (section.items.length) height += 6;
    }
    if (layout.hasMore) height += 18;
    return height;
  }

  computeSize(width) {
    const rows = this._displayRows();
    let height = 72;
    if (!rows.length) return [width || 320, height];
    for (const row of rows) {
      const layout = this._rowLayout(row);
      height += this._estimateRowHeight(layout);
      height += 10;
    }
    return [width || 320, height];
  }

  draw(ctx, node, width, posY, height) {
    this.last_y = posY;
    this.hitAreas = [];
    this.hoverAreas = [];
    const colors = getThemeColors();
    const x = HORIZ_MARGIN;
    const cardWidth = width - HORIZ_MARGIN * 2;
    const report = this.parentNode?._doraAutoStrengthReport;
    const rows = this._displayRows();
    const analyzedRows = rows.filter((row) => row?.status === "analyzed" && Array.isArray(row.report?.logical_groups));

    ctx.save();
    ctx.font = "12px Arial";
    ctx.textBaseline = "middle";

    drawRoundedRect(ctx, x, posY + 2, cardWidth, 58, 8);
    ctx.fillStyle = "#14171c";
    ctx.fill();
    ctx.strokeStyle = colors.outline;
    ctx.stroke();

    ctx.fillStyle = colors.text;
    ctx.font = "bold 12px Arial";
    ctx.textAlign = "left";
    ctx.fillText("Auto-strength analysis", x + 12, posY + 20);

    ctx.font = "11px Arial";
    ctx.fillStyle = colors.secondary;
    if (!report) {
      const autoEnabled = node._doraGlobals?.auto_strength_enabled === true;
      ctx.fillText(
        autoEnabled ? "Run the node to populate the logical-group auto-strength report." : "Enable auto-strength and run the node to see the report.",
        x + 12,
        posY + 40
      );
      ctx.restore();
      return;
    }

    const summary = [
      `rows ${rows.length}`,
      `analyzed ${analyzedRows.length}`,
      `device ${report.auto_strength_device ?? "auto"}`,
      `window ${formatNumber(report.ratio_floor, 2)}–${formatNumber(report.ratio_ceiling, 2)}`,
    ].join("  ·  ");
    ctx.fillText(summary, x + 12, posY + 40);

    let y = posY + 68;
    for (const row of rows) {
      const layout = this._rowLayout(row);
      const cardHeight = this._estimateRowHeight(layout);

      drawRoundedRect(ctx, x, y, cardWidth, cardHeight, 8);
      ctx.fillStyle = "#101318";
      ctx.fill();
      ctx.strokeStyle = colors.outline;
      ctx.stroke();

      this.hitAreas.push({
        x,
        y,
        w: cardWidth,
        h: 28,
        rowIndex: row.row_index,
        toggleable: layout.analyzed,
      });

      ctx.fillStyle = colors.text;
      ctx.font = "bold 12px Arial";
      const toggleGlyph = layout.analyzed ? (layout.expanded ? "▾" : "▸") : "•";
      const fullHeader = String(row.lora_name || "None");
      const shownHeader = fitText(ctx, fullHeader, cardWidth - 150);
      ctx.fillText(`${toggleGlyph} ${shownHeader}`, x + 12, y + 16);
      if (shownHeader !== fullHeader) {
        this.hoverAreas.push({
          x: x + 12,
          y: y + 4,
          w: Math.max(40, cardWidth - 170),
          h: 18,
          tooltip: fullHeader,
        });
      }

      const badge = layout.analyzed
        ? `${row.report.analyzable_bases ?? row.report.mapped_bases}/${row.report.measured_bases} bases`
        : String(row.status_detail || row.status || "");
      ctx.textAlign = "right";
      ctx.font = "11px Arial";
      ctx.fillStyle = colors.secondary;
      ctx.fillText(fitText(ctx, badge, 180), x + cardWidth - 12, y + 16);
      ctx.textAlign = "left";

      ctx.strokeStyle = "#2a2f39";
      ctx.beginPath();
      ctx.moveTo(x + 12, y + 30);
      ctx.lineTo(x + cardWidth - 12, y + 30);
      ctx.stroke();

      ctx.fillStyle = colors.secondary;
      if (!layout.analyzed) {
        const line = `${row.status || "unknown"} · model ${formatNumber(row.strength_model, 2)} · clip ${formatNumber(row.strength_clip, 2)}`;
        ctx.fillText(fitText(ctx, line, cardWidth - 24), x + 12, y + 48);
        y += cardHeight + 10;
        continue;
      }

      const displayGroups = layout.displayGroups;
      const floor = Number.isFinite(+row.report?.ratio_floor) ? +row.report.ratio_floor : 0;
      const ceiling = Number.isFinite(+row.report?.ratio_ceiling) ? +row.report.ratio_ceiling : 1;
      const denom = Math.max(1e-6, ceiling - floor);
      const barMinWidth = 96;
      const ratioWidth = 54;
      const availableWidth = Math.max(0, cardWidth - 24 - ratioWidth);
      const labelWidth = Math.max(80, Math.min(520, availableWidth - barMinWidth));
      const barX = x + 12 + labelWidth;
      const barW = Math.max(48, availableWidth - labelWidth);
      const center = barX + ((clamp(1, floor, ceiling) - floor) / denom) * barW;

      const topLine = [
        `model ${formatNumber(row.strength_model, 2)}`,
        `clip ${formatNumber(row.strength_clip, 2)}`,
        `${row.report.logical_groups_measured}/${row.report.logical_groups_total} logical`,
        `display ${displayGroups.length}`,
        `load ${row.report.analysis_load_device}`,
      ].join("  ·  ");
      ctx.fillText(fitText(ctx, topLine, cardWidth - 24), x + 12, y + 48);

      const cohortNames = Array.isArray(row.report?.cohorts)
        ? row.report.cohorts.map((cohort) => `${cohort.group}/${cohort.family}`).join("  ·  ")
        : "";
      ctx.fillText(fitText(ctx, cohortNames, cardWidth - 24), x + 12, y + 66);

      let rowY = y + 88;
      ctx.font = "bold 11px Arial";
      for (const section of layout.sections) {
        const sectionTitle = section.visibleCount < section.totalCount
          ? `${section.title} (${section.visibleCount}/${section.totalCount})`
          : `${section.title} (${section.totalCount})`;
        ctx.fillStyle = section.kind === "boost" ? "#8bb2ff" : section.kind === "pullback" ? "#f0b05b" : colors.secondary;
        ctx.textAlign = "left";
        ctx.fillText(fitText(ctx, sectionTitle, cardWidth - 24), x + 12, rowY + 10);
        rowY += 18;

        ctx.font = "11px Arial";
        for (const item of section.items) {
          const ratio = toFiniteNumberOrNull(item?.ratio_applied);
          const validRatio = ratio !== null;
          const ratioText = validRatio ? `${ratio.toFixed(2)}×` : "global";
          const itemLabel = `${item?.display_id ?? item?.logical_id ?? "?"} ×${item?.fanout ?? 1}`;

          const shownItemLabel = fitText(ctx, itemLabel, labelWidth - 8);

          ctx.fillStyle = colors.text;
          ctx.textAlign = "left";
          ctx.fillText(shownItemLabel, x + 12, rowY);
          if (shownItemLabel !== itemLabel) {
            this.hoverAreas.push({
              x: x + 12,
              y: rowY - 8,
              w: labelWidth,
              h: 16,
              tooltip: itemLabel,
            });
          }

          drawRoundedRect(ctx, barX, rowY - 6, barW, 12, 6);
          ctx.fillStyle = "#0b1020";
          ctx.fill();
          ctx.strokeStyle = "#2a2f39";
          ctx.stroke();

          ctx.strokeStyle = "#7c8597";
          ctx.beginPath();
          ctx.moveTo(center, rowY - 7);
          ctx.lineTo(center, rowY + 7);
          ctx.stroke();

          const px = validRatio ? barX + ((clamp(ratio, floor, ceiling) - floor) / denom) * barW : center;
          if (Math.abs(px - center) > 1) {
            const fillX = Math.min(center, px);
            const fillW = Math.max(1, Math.abs(px - center));
            drawRoundedRect(ctx, fillX, rowY - 4, fillW, 8, 4);
            ctx.fillStyle = px >= center ? "#4f8cff" : "#e3a33b";
            ctx.fill();
          }
          ctx.beginPath();
          ctx.fillStyle = validRatio ? (px >= center ? "#bcd4ff" : "#ffd08a") : colors.secondary;
          ctx.arc(px, rowY, 3, 0, Math.PI * 2);
          ctx.fill();

          ctx.textAlign = "right";
          ctx.fillStyle = colors.secondary;
          ctx.fillText(ratioText, x + cardWidth - 12, rowY);
          rowY += 20;
        }

        if (section.items.length) {
          rowY += 6;
        }
      }

      if (layout.hasMore) {
        ctx.textAlign = "left";
        ctx.fillStyle = colors.secondary;
        ctx.fillText(`click header to ${layout.expanded ? "collapse" : `expand all ${displayGroups.length}`} display groups`, x + 12, rowY + 2);
      }

      y += cardHeight + 10;
    }

    ctx.restore();
  }

  mouse(event, pos) {
    if (event.type === "pointermove" || event.type === "mousemove") {
      const mx = pos[0];
      const my = pos[1];
      let tooltip = "";
      for (const hit of this.hoverAreas) {
        if (mx >= hit.x && mx <= hit.x + hit.w && my >= hit.y && my <= hit.y + hit.h) {
          tooltip = hit.tooltip || "";
          break;
        }
      }
      setCanvasTooltip(tooltip);
      return false;
    }
    if (event.type === "pointerleave" || event.type === "mouseout") {
      clearCanvasTooltip();
      return false;
    }
    if (event.type !== "pointerdown") return false;
    const mx = pos[0];
    const my = pos[1];
    for (const hit of this.hitAreas) {
      if (mx >= hit.x && mx <= hit.x + hit.w && my >= hit.y && my <= hit.y + hit.h) {
        if (!hit.toggleable) return true;
        const expanded = this.parentNode._doraAutoStrengthExpanded || (this.parentNode._doraAutoStrengthExpanded = {});
        expanded[hit.rowIndex] = !expanded[hit.rowIndex];
        syncNodeHeightToContent(this.parentNode);
        this.parentNode.setDirtyCanvas?.(true, true);
        return true;
      }
    }
    return false;
  }
}

function buildUI(node, state, loraValues) {
  const st = sanitizeState(state);
  clearCanvasTooltip();
  setState(node, st);
  removeAllWidgets(node);
  node._doraRows = st.rows;
  node._doraGlobals = st.globals;
  node.resizable = true;

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

  const wZImageCompat = node.addWidget(
    "toggle",
    "zimage_lumina2_compat",
    !!node._doraGlobals.zimage_lumina2_compat,
    (v) => {
      node._doraGlobals.zimage_lumina2_compat = !!v;
      persistNodeState(node);
    }
  );
  wZImageCompat.label = "ZiT/Lumina2 auto-fix (QKV fuse + out remap)";

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

  const wAutoStrengthEnabled = node.addWidget(
    "toggle",
    "auto_strength_enabled",
    !!node._doraGlobals.auto_strength_enabled,
    (v) => {
      node._doraGlobals.auto_strength_enabled = !!v;
      persistNodeState(node);
    }
  );
  wAutoStrengthEnabled.label = "Auto-strength (per-base ΔW rebalance)";

  const wAutoStrengthDevice = node.addWidget(
    "combo",
    "auto_strength_device",
    normalizeAutoStrengthDevice(node._doraGlobals.auto_strength_device),
    (v) => {
      node._doraGlobals.auto_strength_device = normalizeAutoStrengthDevice(v);
      persistNodeState(node);
    },
    { values: AUTO_STRENGTH_DEVICE_CHOICES }
  );
  wAutoStrengthDevice.label = "Auto-strength analysis device (auto = CPU-safe)";

  const wAutoStrengthFloor = node.addWidget(
    "number",
    "auto_strength_ratio_floor",
    Number.isFinite(+node._doraGlobals.auto_strength_ratio_floor) ? +node._doraGlobals.auto_strength_ratio_floor : 0.30,
    (v) => {
      node._doraGlobals.auto_strength_ratio_floor = Math.max(0, +v || 0);
      persistNodeState(node);
    },
    { min: 0.0, max: 16.0, step: 0.01 }
  );
  wAutoStrengthFloor.label = "Auto-strength ratio floor";

  const wAutoStrengthCeiling = node.addWidget(
    "number",
    "auto_strength_ratio_ceiling",
    Number.isFinite(+node._doraGlobals.auto_strength_ratio_ceiling) ? +node._doraGlobals.auto_strength_ratio_ceiling : 1.50,
    (v) => {
      node._doraGlobals.auto_strength_ratio_ceiling = Math.max(0, +v || 0);
      persistNodeState(node);
    },
    { min: 0.0, max: 16.0, step: 0.01 }
  );
  wAutoStrengthCeiling.label = "Auto-strength ratio ceiling";

  const wAutoStrengthReport = new DoraAutoStrengthReportWidget("auto_strength_visualization", node);
  if (typeof node.addCustomWidget === "function") {
    node.addCustomWidget(wAutoStrengthReport);
  } else {
    node.widgets.push(wAutoStrengthReport);
  }
  wAutoStrengthReport.serialize = false;

  const size = node.computeSize();
  const minWidth = 360;
  const minSize = Array.isArray(node.min_size) ? node.min_size.slice(0, 2) : [0, 0];
  minSize[0] = Math.max(Number(minSize[0]) || 0, minWidth);
  minSize[1] = Number(minSize[1]) || 0;
  node.min_size = minSize;
  node.size[0] = Math.max(node.size[0], size[0], minWidth);
  syncNodeHeightToContent(node);
}

app.registerExtension({
  name: EXT_NAME,
  async beforeRegisterNodeDef(nodeType, nodeData) {
    installExecutionListener();
    const nodeName = nodeData?.name ?? "";
    const displayName = nodeData?.display_name ?? "";
    const comfyClass = nodeType?.comfyClass ?? "";
    const pythonClass = nodeData?.python_module ?? "";

    const isTarget =
      nodeName === NODE_CLASS ||
      displayName === NODE_CLASS ||
      comfyClass === NODE_CLASS;

    if (!isTarget) return;

    const origOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = origOnNodeCreated?.apply(this, arguments);
      this.resizable = true;
      rebuild(this);
      return r;
    };

    const origConfigure = nodeType.prototype.configure;
    nodeType.prototype.configure = function (info) {
      try {
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
              auto_strength_enabled: false,
              auto_strength_device: "gpu",
              auto_strength_ratio_floor: 0.30,
              auto_strength_ratio_ceiling: 1.50,
              dora_decompose_debug: false,
              dora_decompose_debug_n: 30,
              dora_decompose_debug_stack_depth: 10,
              dora_slice_fix: true,
              dora_adaln_swap_fix: true,
              zimage_lumina2_compat: true,
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
