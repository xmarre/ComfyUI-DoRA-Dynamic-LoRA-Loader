import { app } from "../../scripts/app.js";

const NODE_CLASS = "DoRA Power LoRA Loader";
const EXT_NAME = "comfyui_dora_dynamic_lora.power_lora_loader";
const LORA_API = "/dora_dynamic_lora/loras";
const ROW_HEIGHT = 24;
const HORIZ_MARGIN = 15;
const WEIGHT_STEP = 0.05;
const WEIGHT_DRAG_PIXELS_PER_STEP = 6;

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
      auto_strength_enabled: false,
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
    if (event.type === "pointermove") return this.handleWeightDrag(event);
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

function buildUI(node, state, loraValues) {
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

    const origOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = origOnNodeCreated?.apply(this, arguments);
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
