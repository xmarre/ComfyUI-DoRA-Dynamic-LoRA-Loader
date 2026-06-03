import { app } from "../../scripts/app.js";
import "../../scripts/domWidget.js";

const EXT_NAME = "comfyui_dora_dynamic_lora.state_manager";
const NODE_CLASS = "DoRA State Manager";
const LORA_API = "/dora_dynamic_lora/loras";
const STATE_WIDGET = "state_json";
const SELECTED_CHARACTER_WIDGET = "selected_character_id";
const SELECTED_PROMPT_WIDGET = "selected_prompt_id";
const DOM_WIDGET_NAME = "dora_state_manager_ui";
const DOM_WIDGET_TYPE = "DORA_STATE_MANAGER_UI";
const STYLE_ID = "dora-state-manager-style";
const MIN_NODE_WIDTH = 620;
const MIN_WIDGET_HEIGHT = 700;
const AUTO_STRENGTH_DEVICE_CHOICES = ["auto", "cpu", "gpu"];

function clone(value) {
  if (typeof structuredClone === "function") return structuredClone(value);
  return JSON.parse(JSON.stringify(value));
}

function makeId(prefix) {
  if (globalThis.crypto?.randomUUID) return `${prefix}_${globalThis.crypto.randomUUID()}`;
  return `${prefix}_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 10)}`;
}

function defaultPrompt() {
  return {
    id: "default_prompt",
    name: "Default Prompt",
    positive: "",
    negative: "",
    settings: {},
  };
}

function defaultCharacter() {
  return {
    id: "default_character",
    name: "Default Character",
    loras: [],
    loader_globals: {},
    prompts: [defaultPrompt()],
  };
}

function defaultState() {
  return {
    version: 1,
    characters: [defaultCharacter()],
  };
}

function safeJsonParse(raw, fallback) {
  if (raw && typeof raw === "object") return clone(raw);
  if (typeof raw !== "string" || !raw.trim()) return clone(fallback);
  try {
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === "object" ? parsed : clone(fallback);
  } catch {
    return clone(fallback);
  }
}

function cleanId(value, fallback) {
  const text = String(value ?? "").trim().replace(/[^A-Za-z0-9_.:-]+/g, "_").replace(/^_+|_+$/g, "");
  return text || fallback;
}

function normalizeNumber(value, fallback = 1.0) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function normalizeDevice(value) {
  const v = String(value ?? "gpu").toLowerCase();
  return AUTO_STRENGTH_DEVICE_CHOICES.includes(v) ? v : "gpu";
}

function normalizeBoolean(value, fallback = false) {
  if (typeof value === "boolean") return value;
  if (value == null) return fallback;
  if (typeof value === "string") {
    const normalized = value.trim().toLowerCase();
    if (["1", "true", "yes", "on"].includes(normalized)) return true;
    if (["0", "false", "no", "off", ""].includes(normalized)) return false;
    return fallback;
  }
  return Boolean(value);
}

function normalizeLoaderGlobals(globalsIn) {
  const src = globalsIn && typeof globalsIn === "object" ? globalsIn : {};
  const out = {};
  for (const key of [
    "stack_enabled",
    "verbose",
    "log_unloaded_keys",
    "broadcast_auto_scale",
    "broadcast_modulations",
    "broadcast_include_dora_scale",
    "dora_decompose_debug",
    "dora_slice_fix",
    "dora_adaln_swap_fix",
    "zimage_lumina2_compat",
    "auto_strength_enabled",
  ]) {
    if (Object.prototype.hasOwnProperty.call(src, key)) out[key] = normalizeBoolean(src[key]);
  }
  for (const key of ["broadcast_scale", "auto_strength_ratio_floor", "auto_strength_ratio_ceiling"]) {
    if (Object.prototype.hasOwnProperty.call(src, key)) out[key] = normalizeNumber(src[key], key === "broadcast_scale" ? 1.0 : 0.0);
  }
  for (const key of ["dora_decompose_debug_n", "dora_decompose_debug_stack_depth"]) {
    if (Object.prototype.hasOwnProperty.call(src, key)) out[key] = Math.floor(normalizeNumber(src[key], key === "dora_decompose_debug_n" ? 30 : 10));
  }
  if (Object.prototype.hasOwnProperty.call(src, "auto_strength_device")) out.auto_strength_device = normalizeDevice(src.auto_strength_device);
  return out;
}

function normalizeLoraRow(row) {
  const r = row && typeof row === "object" ? row : {};
  const strengthModel = normalizeNumber(r.strength_model ?? r.strengthModel ?? r.strength, 1.0);
  const strengthClip = normalizeNumber(r.strength_clip ?? r.strengthClip ?? r.strengthTwo, strengthModel);
  return {
    enabled: r.enabled !== undefined ? normalizeBoolean(r.enabled, true) : r.on !== undefined ? normalizeBoolean(r.on, true) : true,
    name: String(r.name ?? r.lora ?? r.lora_name ?? "None").trim() || "None",
    strength_model: strengthModel,
    strength_clip: strengthClip,
  };
}

function normalizeSettings(value) {
  if (value && typeof value === "object" && !Array.isArray(value)) return clone(value);
  const parsed = safeJsonParse(value, {});
  return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed : {};
}

function normalizePrompt(prompt, index) {
  const p = prompt && typeof prompt === "object" ? prompt : {};
  return {
    id: cleanId(p.id, `prompt_${index + 1}`),
    name: String(p.name ?? `Prompt ${index + 1}`).trim() || `Prompt ${index + 1}`,
    positive: String(p.positive ?? p.positive_prompt ?? ""),
    negative: String(p.negative ?? p.negative_prompt ?? ""),
    settings: normalizeSettings(p.settings),
  };
}

function normalizeCharacter(character, index) {
  const c = character && typeof character === "object" ? character : {};
  const promptsIn = Array.isArray(c.prompts) ? c.prompts : [];
  const prompts = promptsIn.map(normalizePrompt).filter(Boolean);
  return {
    id: cleanId(c.id, `character_${index + 1}`),
    name: String(c.name ?? `Character ${index + 1}`).trim() || `Character ${index + 1}`,
    loras: Array.isArray(c.loras) ? c.loras.map(normalizeLoraRow) : [],
    loader_globals: normalizeLoaderGlobals(c.loader_globals ?? c.globals),
    prompts: prompts.length ? prompts : [defaultPrompt()],
  };
}

function normalizeState(raw) {
  const parsed = safeJsonParse(raw, defaultState());
  const charsIn = Array.isArray(parsed.characters) ? parsed.characters : [];
  const chars = charsIn.map(normalizeCharacter).filter(Boolean);
  return {
    version: 1,
    characters: chars.length ? chars : [defaultCharacter()],
  };
}

function serializeState(state) {
  return JSON.stringify(normalizeState(state), null, 0);
}

function getWidgets(node) {
  const map = new Map();
  for (const widget of node.widgets ?? []) map.set(widget.name, widget);
  return {
    stateWidget: map.get(STATE_WIDGET),
    characterWidget: map.get(SELECTED_CHARACTER_WIDGET),
    promptWidget: map.get(SELECTED_PROMPT_WIDGET),
  };
}

function setWidgetValue(widget, value) {
  if (!widget) return;
  widget.value = value;
  widget.callback?.(value);
}

function widgetValue(widget, fallback = "") {
  return widget?.value ?? fallback;
}

function hideWidget(widget) {
  if (!widget || widget.__dsmHidden) return;
  widget.__dsmHidden = true;
  widget.type = "hidden";
  widget.computeSize = () => [0, 0];
  widget.draw = () => {};
}

function selectedCharacter(state, selectedId) {
  return state.characters.find((c) => c.id === selectedId) || state.characters[0];
}

function selectedPrompt(character, selectedId) {
  return character.prompts.find((p) => p.id === selectedId) || character.prompts[0];
}

function ensureSelection(node, state) {
  const { characterWidget, promptWidget } = getWidgets(node);
  let charId = String(widgetValue(characterWidget, "") || "").trim();
  let character = selectedCharacter(state, charId);
  if (!character || character.id !== charId) {
    character = state.characters[0];
    charId = character.id;
    setWidgetValue(characterWidget, charId);
  }
  let promptId = String(widgetValue(promptWidget, "") || "").trim();
  let prompt = selectedPrompt(character, promptId);
  if (!prompt || prompt.id !== promptId) {
    prompt = character.prompts[0];
    setWidgetValue(promptWidget, prompt.id);
  }
  return { character, prompt };
}

function markNodeDirty(node) {
  node.setDirtyCanvas?.(true, true);
  node.graph?.change?.();
}

function saveState(node, state, opts = {}) {
  const widgets = getWidgets(node);
  const normalized = normalizeState(state);
  setWidgetValue(widgets.stateWidget, serializeState(normalized));
  if (opts.characterId) setWidgetValue(widgets.characterWidget, opts.characterId);
  if (opts.promptId) setWidgetValue(widgets.promptWidget, opts.promptId);
  node.properties = node.properties || {};
  node.properties.dora_state_manager = {
    state: normalized,
    selected_character_id: widgetValue(widgets.characterWidget, ""),
    selected_prompt_id: widgetValue(widgets.promptWidget, ""),
  };
  if (opts.dirty !== false) markNodeDirty(node);
}

function readNodeState(node) {
  const { stateWidget } = getWidgets(node);
  return normalizeState(widgetValue(stateWidget, serializeState(defaultState())));
}

let loraCache = null;
async function fetchLoras() {
  if (loraCache) return loraCache;
  try {
    const response = await fetch(LORA_API, { cache: "no-store" });
    if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
    const list = await response.json();
    loraCache = Array.isArray(list) ? ["None", ...list.filter((x) => x && x !== "None" && x !== "NONE")] : ["None"];
  } catch (err) {
    console.warn(`[${EXT_NAME}] failed to fetch LoRA list`, err);
    loraCache = ["None"];
  }
  return loraCache;
}

function ensureStyles() {
  if (document.getElementById(STYLE_ID)) return;
  const style = document.createElement("style");
  style.id = STYLE_ID;
  style.textContent = `
    .dsm-root {
      --dsm-gap: 8px;
      box-sizing: border-box;
      height: 100%;
      min-height: ${MIN_WIDGET_HEIGHT}px;
      padding: 8px;
      overflow: auto;
      display: flex;
      flex-direction: column;
      gap: var(--dsm-gap);
      color: var(--input-text, #ddd);
      background: rgba(20, 20, 20, 0.20);
      font: 12px/1.35 system-ui, sans-serif;
    }
    .dsm-root * { box-sizing: border-box; }
    .dsm-row { display: flex; gap: 6px; align-items: center; }
    .dsm-row.wrap { flex-wrap: wrap; }
    .dsm-section { border: 1px solid rgba(128,128,128,.35); border-radius: 8px; padding: 8px; display: flex; flex-direction: column; gap: 7px; }
    .dsm-section-title { font-weight: 650; opacity: .95; }
    .dsm-root input, .dsm-root select, .dsm-root textarea, .dsm-root button {
      font: inherit;
      color: var(--input-text, #ddd);
      background: var(--comfy-input-bg, #222);
      border: 1px solid rgba(128,128,128,.45);
      border-radius: 5px;
      padding: 4px 6px;
    }
    .dsm-root button { cursor: pointer; white-space: nowrap; }
    .dsm-root textarea { width: 100%; min-height: 82px; resize: vertical; }
    .dsm-root input[type="checkbox"] { width: auto; }
    .dsm-flex { flex: 1 1 auto; min-width: 0; }
    .dsm-small { width: 74px; }
    .dsm-mid { width: 132px; }
    .dsm-muted { opacity: .68; }
    .dsm-lora-row { display: grid; grid-template-columns: auto minmax(160px,1fr) 80px 80px auto; gap: 6px; align-items: center; }
    .dsm-grid2 { display: grid; grid-template-columns: minmax(0,1fr) minmax(0,1fr); gap: 6px; }
    .dsm-details { display: flex; flex-direction: column; gap: 7px; }
    .dsm-details summary { cursor: pointer; user-select: none; }
  `;
  document.head.appendChild(style);
}

function makeButton(label, callback, title = "") {
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = label;
  if (title) button.title = title;
  button.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    callback(event);
  });
  return button;
}

function makeInput(value, onChange, attrs = {}) {
  const input = document.createElement("input");
  input.value = value ?? "";
  Object.assign(input, attrs);
  input.addEventListener("input", () => onChange(input.value));
  input.addEventListener("change", () => onChange(input.value));
  return input;
}

function makeCheckbox(value, onChange) {
  const input = document.createElement("input");
  input.type = "checkbox";
  input.checked = Boolean(value);
  input.addEventListener("change", () => onChange(input.checked));
  return input;
}

function makeSelect(values, selected, onChange) {
  const select = document.createElement("select");
  for (const value of values) {
    const option = document.createElement("option");
    option.value = value.value ?? value;
    option.textContent = value.label ?? value;
    select.appendChild(option);
  }
  select.value = selected;
  select.addEventListener("change", () => onChange(select.value));
  return select;
}

function addLabel(container, text, control) {
  const label = document.createElement("label");
  label.className = "dsm-row dsm-flex";
  const span = document.createElement("span");
  span.textContent = text;
  span.className = "dsm-muted";
  label.append(span, control);
  container.appendChild(label);
  return label;
}

function updateGlobalsFromUi(globals, patch) {
  Object.assign(globals, patch);
  return normalizeLoaderGlobals(globals);
}

function render(node) {
  const root = node.__dsmRoot;
  if (!root) return;
  ensureStyles();

  const state = readNodeState(node);
  const { character, prompt } = ensureSelection(node, state);
  const selectedCharId = character.id;
  const selectedPromptId = prompt.id;

  root.innerHTML = "";
  root.className = "dsm-root";

  const datalistId = `dsm_loras_${node.id ?? "node"}`;
  const datalist = document.createElement("datalist");
  datalist.id = datalistId;
  for (const name of loraCache || ["None"]) {
    const option = document.createElement("option");
    option.value = name;
    datalist.appendChild(option);
  }
  root.appendChild(datalist);

  const header = document.createElement("div");
  header.className = "dsm-section";
  const headerTitle = document.createElement("div");
  headerTitle.className = "dsm-section-title";
  headerTitle.textContent = "Character state";
  const charRow = document.createElement("div");
  charRow.className = "dsm-row wrap";
  const charSelect = makeSelect(
    state.characters.map((c) => ({ value: c.id, label: c.name })),
    selectedCharId,
    (id) => {
      const next = selectedCharacter(state, id);
      saveState(node, state, { characterId: next.id, promptId: next.prompts[0]?.id || "" });
      render(node);
    }
  );
  charSelect.className = "dsm-flex";
  charRow.append(
    charSelect,
    makeButton("New", () => {
      const c = defaultCharacter();
      c.id = makeId("character");
      c.name = `Character ${state.characters.length + 1}`;
      c.prompts[0].id = makeId("prompt");
      state.characters.push(c);
      saveState(node, state, { characterId: c.id, promptId: c.prompts[0].id });
      render(node);
    }),
    makeButton("Duplicate", () => {
      const copy = clone(character);
      copy.id = makeId("character");
      copy.name = `${copy.name} Copy`;
      copy.prompts = copy.prompts.map((p) => ({ ...p, id: makeId("prompt") }));
      state.characters.push(copy);
      saveState(node, state, { characterId: copy.id, promptId: copy.prompts[0]?.id || "" });
      render(node);
    }),
    makeButton("Delete", () => {
      if (state.characters.length <= 1) return;
      const index = state.characters.findIndex((c) => c.id === selectedCharId);
      if (index >= 0) state.characters.splice(index, 1);
      const next = state.characters[Math.max(0, Math.min(index, state.characters.length - 1))];
      saveState(node, state, { characterId: next.id, promptId: next.prompts[0]?.id || "" });
      render(node);
    })
  );
  const nameInput = makeInput(character.name, (value) => {
    character.name = value.trim() || character.name;
    saveState(node, state, { characterId: character.id, promptId: selectedPromptId });
  });
  nameInput.className = "dsm-flex";
  const nameRow = document.createElement("div");
  nameRow.className = "dsm-row";
  nameRow.append(Object.assign(document.createElement("span"), { textContent: "Name", className: "dsm-muted" }), nameInput);
  header.append(headerTitle, charRow, nameRow);
  root.appendChild(header);

  const loraSection = document.createElement("div");
  loraSection.className = "dsm-section";
  const loraTitle = document.createElement("div");
  loraTitle.className = "dsm-row";
  const titleText = document.createElement("div");
  titleText.className = "dsm-section-title dsm-flex";
  titleText.textContent = "LoRA combination";
  loraTitle.append(
    titleText,
    makeButton("Refresh", async () => {
      loraCache = null;
      await fetchLoras();
      render(node);
    }),
    makeButton("Add LoRA", () => {
      character.loras.push({ enabled: true, name: "None", strength_model: 1.0, strength_clip: 1.0 });
      saveState(node, state, { characterId: selectedCharId, promptId: selectedPromptId });
      render(node);
    })
  );
  loraSection.appendChild(loraTitle);
  if (!character.loras.length) {
    const empty = document.createElement("div");
    empty.className = "dsm-muted";
    empty.textContent = "No LoRA rows saved for this character yet.";
    loraSection.appendChild(empty);
  }
  character.loras.forEach((row, index) => {
    const loraRow = document.createElement("div");
    loraRow.className = "dsm-lora-row";
    const enabled = makeCheckbox(row.enabled, (checked) => {
      row.enabled = checked;
      saveState(node, state, { characterId: selectedCharId, promptId: selectedPromptId });
    });
    const name = makeInput(row.name, (value) => {
      row.name = value.trim() || "None";
      saveState(node, state, { characterId: selectedCharId, promptId: selectedPromptId });
    }, { list: datalistId });
    const sm = makeInput(row.strength_model, (value) => {
      row.strength_model = normalizeNumber(value, row.strength_model);
      saveState(node, state, { characterId: selectedCharId, promptId: selectedPromptId });
    }, { type: "number", step: "0.05" });
    const sc = makeInput(row.strength_clip, (value) => {
      row.strength_clip = normalizeNumber(value, row.strength_clip);
      saveState(node, state, { characterId: selectedCharId, promptId: selectedPromptId });
    }, { type: "number", step: "0.05" });
    loraRow.append(enabled, name, sm, sc, makeButton("×", () => {
      character.loras.splice(index, 1);
      saveState(node, state, { characterId: selectedCharId, promptId: selectedPromptId });
      render(node);
    }, "Remove LoRA row"));
    loraSection.appendChild(loraRow);
  });
  root.appendChild(loraSection);

  const promptSection = document.createElement("div");
  promptSection.className = "dsm-section";
  const promptTitle = document.createElement("div");
  promptTitle.className = "dsm-row";
  const promptTitleText = document.createElement("div");
  promptTitleText.className = "dsm-section-title dsm-flex";
  promptTitleText.textContent = "Prompt preset";
  promptTitle.append(
    promptTitleText,
    makeButton("New", () => {
      const p = defaultPrompt();
      p.id = makeId("prompt");
      p.name = `Prompt ${character.prompts.length + 1}`;
      character.prompts.push(p);
      saveState(node, state, { characterId: selectedCharId, promptId: p.id });
      render(node);
    }),
    makeButton("Duplicate", () => {
      const p = { ...clone(prompt), id: makeId("prompt"), name: `${prompt.name} Copy` };
      character.prompts.push(p);
      saveState(node, state, { characterId: selectedCharId, promptId: p.id });
      render(node);
    }),
    makeButton("Delete", () => {
      if (character.prompts.length <= 1) return;
      const index = character.prompts.findIndex((p) => p.id === selectedPromptId);
      if (index >= 0) character.prompts.splice(index, 1);
      const next = character.prompts[Math.max(0, Math.min(index, character.prompts.length - 1))];
      saveState(node, state, { characterId: selectedCharId, promptId: next.id });
      render(node);
    })
  );
  const promptSelect = makeSelect(
    character.prompts.map((p) => ({ value: p.id, label: p.name })),
    selectedPromptId,
    (id) => {
      saveState(node, state, { characterId: selectedCharId, promptId: id });
      render(node);
    }
  );
  promptSelect.className = "dsm-flex";
  const promptSelectRow = document.createElement("div");
  promptSelectRow.className = "dsm-row";
  promptSelectRow.append(promptSelect);
  const promptName = makeInput(prompt.name, (value) => {
    prompt.name = value.trim() || prompt.name;
    saveState(node, state, { characterId: selectedCharId, promptId: prompt.id });
  });
  promptName.className = "dsm-flex";
  const promptNameRow = document.createElement("div");
  promptNameRow.className = "dsm-row";
  promptNameRow.append(Object.assign(document.createElement("span"), { textContent: "Name", className: "dsm-muted" }), promptName);
  const positive = document.createElement("textarea");
  positive.value = prompt.positive;
  positive.placeholder = "Positive prompt";
  positive.addEventListener("input", () => {
    prompt.positive = positive.value;
    saveState(node, state, { characterId: selectedCharId, promptId: prompt.id });
  });
  const negative = document.createElement("textarea");
  negative.value = prompt.negative;
  negative.placeholder = "Negative prompt";
  negative.addEventListener("input", () => {
    prompt.negative = negative.value;
    saveState(node, state, { characterId: selectedCharId, promptId: prompt.id });
  });
  promptSection.append(promptTitle, promptSelectRow, promptNameRow, positive, negative);
  root.appendChild(promptSection);

  const settingsSection = document.createElement("details");
  settingsSection.className = "dsm-section dsm-details";
  const settingsSummary = document.createElement("summary");
  settingsSummary.className = "dsm-section-title";
  settingsSummary.textContent = "Saved loader settings / future node settings";
  settingsSection.appendChild(settingsSummary);

  const globals = character.loader_globals || (character.loader_globals = {});
  const row1 = document.createElement("div");
  row1.className = "dsm-grid2";
  const stackEnabled = makeCheckbox(globals.stack_enabled ?? true, (checked) => {
    character.loader_globals = updateGlobalsFromUi(globals, { stack_enabled: checked });
    saveState(node, state, { characterId: selectedCharId, promptId: selectedPromptId });
  });
  const autoStrength = makeCheckbox(globals.auto_strength_enabled ?? false, (checked) => {
    character.loader_globals = updateGlobalsFromUi(globals, { auto_strength_enabled: checked });
    saveState(node, state, { characterId: selectedCharId, promptId: selectedPromptId });
  });
  addLabel(row1, "Stack", stackEnabled);
  addLabel(row1, "Auto-strength", autoStrength);
  settingsSection.appendChild(row1);

  const row2 = document.createElement("div");
  row2.className = "dsm-grid2";
  const device = makeSelect(AUTO_STRENGTH_DEVICE_CHOICES, normalizeDevice(globals.auto_strength_device ?? "gpu"), (value) => {
    character.loader_globals = updateGlobalsFromUi(globals, { auto_strength_device: value });
    saveState(node, state, { characterId: selectedCharId, promptId: selectedPromptId });
  });
  const broadcastMods = makeCheckbox(globals.broadcast_modulations ?? true, (checked) => {
    character.loader_globals = updateGlobalsFromUi(globals, { broadcast_modulations: checked });
    saveState(node, state, { characterId: selectedCharId, promptId: selectedPromptId });
  });
  addLabel(row2, "Device", device);
  addLabel(row2, "Broadcast modulation", broadcastMods);
  settingsSection.appendChild(row2);

  const row3 = document.createElement("div");
  row3.className = "dsm-grid2";
  const floor = makeInput(globals.auto_strength_ratio_floor ?? 0.30, (value) => {
    character.loader_globals = updateGlobalsFromUi(globals, { auto_strength_ratio_floor: normalizeNumber(value, 0.30) });
    saveState(node, state, { characterId: selectedCharId, promptId: selectedPromptId });
  }, { type: "number", step: "0.01" });
  const ceiling = makeInput(globals.auto_strength_ratio_ceiling ?? 1.50, (value) => {
    character.loader_globals = updateGlobalsFromUi(globals, { auto_strength_ratio_ceiling: normalizeNumber(value, 1.50) });
    saveState(node, state, { characterId: selectedCharId, promptId: selectedPromptId });
  }, { type: "number", step: "0.01" });
  addLabel(row3, "Ratio floor", floor);
  addLabel(row3, "Ratio ceiling", ceiling);
  settingsSection.appendChild(row3);

  const row4 = document.createElement("div");
  row4.className = "dsm-grid2";
  const sliceFix = makeCheckbox(globals.dora_slice_fix ?? true, (checked) => {
    character.loader_globals = updateGlobalsFromUi(globals, { dora_slice_fix: checked });
    saveState(node, state, { characterId: selectedCharId, promptId: selectedPromptId });
  });
  const adalnFix = makeCheckbox(globals.dora_adaln_swap_fix ?? true, (checked) => {
    character.loader_globals = updateGlobalsFromUi(globals, { dora_adaln_swap_fix: checked });
    saveState(node, state, { characterId: selectedCharId, promptId: selectedPromptId });
  });
  addLabel(row4, "Slice fix", sliceFix);
  addLabel(row4, "AdaLN swap fix", adalnFix);
  settingsSection.appendChild(row4);

  const settingsJson = document.createElement("textarea");
  settingsJson.value = JSON.stringify(prompt.settings || {}, null, 2);
  settingsJson.placeholder = "Arbitrary JSON for future nodes, e.g. autoguidance/scale-locked guidance settings";
  settingsJson.addEventListener("change", () => {
    const parsed = safeJsonParse(settingsJson.value, null);
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      prompt.settings = parsed;
      saveState(node, state, { characterId: selectedCharId, promptId: selectedPromptId });
      settingsJson.value = JSON.stringify(prompt.settings, null, 2);
    }
  });
  const settingsLabel = document.createElement("div");
  settingsLabel.className = "dsm-muted";
  settingsLabel.textContent = "Prompt/settings JSON output";
  settingsSection.append(settingsLabel, settingsJson);
  root.appendChild(settingsSection);
}

function syncDomWidgetSize(node, widget) {
  if (!node || !widget) return;
  node.size = Array.isArray(node.size) ? node.size : [MIN_NODE_WIDTH, MIN_WIDGET_HEIGHT];
  node.size[0] = Math.max(Number(node.size[0]) || 0, MIN_NODE_WIDTH);
  node.size[1] = Math.max(Number(node.size[1]) || 0, MIN_WIDGET_HEIGHT + 90);
  node.min_size = [MIN_NODE_WIDTH, MIN_WIDGET_HEIGHT + 90];
}

function installNode(node) {
  if (!node) return;
  node.resizable = true;

  const state = readNodeState(node);
  const { character, prompt } = ensureSelection(node, state);
  saveState(node, state, { characterId: character.id, promptId: prompt.id });

  if (typeof node.addDOMWidget !== "function") {
    console.warn(`[${EXT_NAME}] addDOMWidget is unavailable; falling back to raw JSON widgets.`);
    return;
  }

  const widgets = getWidgets(node);
  hideWidget(widgets.stateWidget);
  hideWidget(widgets.characterWidget);
  hideWidget(widgets.promptWidget);

  if (node.__dsmInstalled) {
    render(node);
    return;
  }
  node.__dsmInstalled = true;

  const root = document.createElement("div");
  node.__dsmRoot = root;
  const widget = node.addDOMWidget(DOM_WIDGET_NAME, DOM_WIDGET_TYPE, root, {
    getMinHeight: () => MIN_WIDGET_HEIGHT,
    getHeight: () => "100%",
    onDraw: (domWidget) => syncDomWidgetSize(node, domWidget),
    afterResize: (domWidgetNode) => syncDomWidgetSize(domWidgetNode, widget),
    serialize: false,
  });
  widget.serialize = false;
  node.__dsmWidget = widget;
  syncDomWidgetSize(node, widget);

  fetchLoras().then(() => {
    render(node);
    markNodeDirty(node);
  });
  render(node);
}

app.registerExtension({
  name: EXT_NAME,

  async beforeRegisterNodeDef(nodeType, nodeData) {
    const nodeName = nodeData?.name ?? "";
    const displayName = nodeData?.display_name ?? "";
    const comfyClass = nodeType?.comfyClass ?? "";
    if (nodeName !== NODE_CLASS && displayName !== NODE_CLASS && comfyClass !== NODE_CLASS) return;

    const origOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = origOnNodeCreated?.apply(this, arguments);
      queueMicrotask(() => installNode(this));
      return result;
    };

    const origConfigure = nodeType.prototype.configure;
    nodeType.prototype.configure = function (info) {
      const result = origConfigure?.apply(this, arguments);
      queueMicrotask(() => installNode(this));
      return result;
    };

    const origOnSerialize = nodeType.prototype.onSerialize;
    nodeType.prototype.onSerialize = function (o) {
      const result = origOnSerialize?.apply(this, arguments);
      try {
        const state = readNodeState(this);
        const widgets = getWidgets(this);
        saveState(this, state, {
          characterId: widgetValue(widgets.characterWidget, "default_character"),
          promptId: widgetValue(widgets.promptWidget, "default_prompt"),
          dirty: false,
        });
        o.properties = o.properties || {};
        o.properties.dora_state_manager = this.properties?.dora_state_manager;
      } catch (err) {
        console.warn(`[${EXT_NAME}] serialize sync failed`, err);
      }
      return result;
    };
  },
});
