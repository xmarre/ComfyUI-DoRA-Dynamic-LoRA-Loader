import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import "../../scripts/domWidget.js";

const EXT_NAME = "comfyui_dora_dynamic_lora.state_manager";
const NODE_CLASS = "DoRA State Manager";
const CUSTOM_WIDGET_INPUT = "state_manager_ui";
const CUSTOM_WIDGET_TYPE = "DORA_STATE_MANAGER_UI";
const STATE_WIDGET = "state_json";
const UI_STATE_WIDGET = "ui_state_json";
const SELECTED_CHARACTER_WIDGET = "selected_character_id";
const SELECTED_PROMPT_WIDGET = "selected_prompt_id";
const USE_RUNTIME_INPUTS_WIDGET = "use_runtime_inputs";
const STYLE_ID = "dora-state-manager-style";
const MIN_WIDGET_HEIGHT = 520;
const MIN_NODE_WIDTH = 620;
const MIN_NODE_HEIGHT = 680;
const THUMBNAIL_SUBFOLDER = "dora_state_manager";
const AUTO_STRENGTH_DEVICE_CHOICES = ["auto", "cpu", "gpu"];

function structuredCloneCompat(value) {
  if (typeof structuredClone === "function") return structuredClone(value);
  return JSON.parse(JSON.stringify(value));
}

function safeJsonParse(raw, fallback) {
  if (raw && typeof raw === "object") return structuredCloneCompat(raw);
  if (typeof raw !== "string" || !raw.trim()) return structuredCloneCompat(fallback);
  try {
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === "object" ? parsed : structuredCloneCompat(fallback);
  } catch {
    return structuredCloneCompat(fallback);
  }
}

function makeId(prefix) {
  if (globalThis.crypto?.randomUUID) return `${prefix}_${globalThis.crypto.randomUUID()}`;
  return `${prefix}_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 10)}`;
}

function cleanId(value, fallback) {
  const text = String(value ?? "")
    .trim()
    .replace(/[^A-Za-z0-9_.:-]+/g, "_")
    .replace(/^_+|_+$/g, "");
  return text || fallback;
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
    thumbnail: {},
    loras: [],
    loader_globals: {},
    prompts: [defaultPrompt()],
  };
}

function defaultState() {
  return { version: 1, characters: [defaultCharacter()] };
}

function defaultUiState() {
  return { version: 1, panel: "prompts", status: "" };
}

function normalizeNumber(value, fallback = 1.0) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
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

function normalizeDevice(value) {
  const text = String(value ?? "gpu").trim().toLowerCase();
  return AUTO_STRENGTH_DEVICE_CHOICES.includes(text) ? text : "gpu";
}

function normalizeSettings(value) {
  if (value && typeof value === "object" && !Array.isArray(value)) return structuredCloneCompat(value);
  const parsed = safeJsonParse(value, {});
  return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed : {};
}

function normalizeThumbnail(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    const url = String(value ?? "").trim();
    return url ? { url } : {};
  }
  const filename = String(value.filename ?? "").trim();
  const subfolder = String(value.subfolder ?? "").trim();
  const type = String(value.type ?? "input").trim() || "input";
  const url = String(value.url ?? "").trim();
  if (filename) return { filename, subfolder, type };
  if (url) return { url };
  return {};
}

function thumbnailUrl(thumbnail) {
  const t = normalizeThumbnail(thumbnail);
  if (t.url) return t.url;
  if (!t.filename) return "";
  const params = new URLSearchParams();
  params.set("filename", t.filename);
  if (t.subfolder) params.set("subfolder", t.subfolder);
  params.set("type", t.type || "input");
  params.set("rand", String(Date.now()));
  return api.apiURL(`/view?${params.toString()}`);
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
  const prompts = Array.isArray(c.prompts) ? c.prompts.map(normalizePrompt).filter(Boolean) : [];
  return {
    id: cleanId(c.id, `character_${index + 1}`),
    name: String(c.name ?? `Character ${index + 1}`).trim() || `Character ${index + 1}`,
    thumbnail: normalizeThumbnail(c.thumbnail),
    loras: Array.isArray(c.loras) ? c.loras.map(normalizeLoraRow) : [],
    loader_globals: normalizeLoaderGlobals(c.loader_globals ?? c.globals),
    prompts: prompts.length ? prompts : [defaultPrompt()],
  };
}

function normalizeState(raw) {
  const parsed = safeJsonParse(raw, defaultState());
  const charsIn = Array.isArray(parsed.characters) ? parsed.characters : [];
  const characters = charsIn.map(normalizeCharacter).filter(Boolean);
  return { version: 1, characters: characters.length ? characters : [defaultCharacter()] };
}

function normalizeUiState(raw) {
  const parsed = safeJsonParse(raw, defaultUiState());
  return {
    version: 1,
    panel: ["prompts", "loras", "settings"].includes(parsed.panel) ? parsed.panel : "prompts",
    status: String(parsed.status ?? ""),
  };
}

function serializeState(state) {
  return JSON.stringify(normalizeState(state), null, 0);
}

function serializeUiState(uiState) {
  return JSON.stringify(normalizeUiState(uiState), null, 0);
}

function getWidgets(node) {
  const map = new Map();
  for (const widget of node.widgets ?? []) map.set(widget.name, widget);
  return {
    stateWidget: map.get(STATE_WIDGET),
    uiStateWidget: map.get(UI_STATE_WIDGET),
    characterWidget: map.get(SELECTED_CHARACTER_WIDGET),
    promptWidget: map.get(SELECTED_PROMPT_WIDGET),
    useRuntimeInputsWidget: map.get(USE_RUNTIME_INPUTS_WIDGET),
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
  widget.hidden = true;
  widget.options = widget.options || {};
  widget.options.hidden = true;
  widget.type = "hidden";
  widget.computeSize = () => [0, 0];
  widget.draw = () => {};
}

function getCurrentState(node) {
  const widgets = getWidgets(node);
  const state = normalizeState(widgetValue(widgets.stateWidget, serializeState(defaultState())));
  const uiState = normalizeUiState(widgetValue(widgets.uiStateWidget, serializeUiState(defaultUiState())));
  return { state, uiState };
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

function updateState(node, state, uiState, opts = {}) {
  const widgets = getWidgets(node);
  const normalizedState = normalizeState(state);
  const normalizedUiState = normalizeUiState(uiState || defaultUiState());
  if (opts.status !== undefined) normalizedUiState.status = String(opts.status || "");
  setWidgetValue(widgets.stateWidget, serializeState(normalizedState));
  setWidgetValue(widgets.uiStateWidget, serializeUiState(normalizedUiState));
  if (opts.characterId) setWidgetValue(widgets.characterWidget, opts.characterId);
  if (opts.promptId) setWidgetValue(widgets.promptWidget, opts.promptId);
  if (opts.useRuntimeInputs !== undefined) setWidgetValue(widgets.useRuntimeInputsWidget, Boolean(opts.useRuntimeInputs));
  node.properties = node.properties || {};
  node.properties.dora_state_manager = {
    state: normalizedState,
    ui_state: normalizedUiState,
    selected_character_id: widgetValue(widgets.characterWidget, ""),
    selected_prompt_id: widgetValue(widgets.promptWidget, ""),
    use_runtime_inputs: Boolean(widgetValue(widgets.useRuntimeInputsWidget, false)),
  };
  if (opts.dirty !== false) markNodeDirty(node);
  cacheRenderableState(node, normalizedState, normalizedUiState);
  if (opts.render !== false) scheduleRender(node);
}

function cacheRenderableState(node, state, uiState) {
  const ctx = node.__dsm;
  if (!ctx) return;
  ctx.state = state;
  ctx.uiState = uiState;
}

function getRenderableState(node) {
  const ctx = node.__dsm;
  if (ctx?.state && ctx?.uiState) return { state: ctx.state, uiState: ctx.uiState };
  const snapshot = getCurrentState(node);
  cacheRenderableState(node, snapshot.state, snapshot.uiState);
  return snapshot;
}

function setStatus(node, text) {
  const { state, uiState } = getCurrentState(node);
  uiState.status = text;
  updateState(node, state, uiState, { dirty: false, render: true });
}

function getFiniteNumber(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function getWidgetOuterHeight(node, widget) {
  const nodeHeight = Math.max(MIN_NODE_HEIGHT, getFiniteNumber(node?.size?.[1], MIN_NODE_HEIGHT));
  const widgetY = Math.max(0, getFiniteNumber(widget?.y, getFiniteNumber(widget?.last_y, 0)));
  return Math.max(MIN_WIDGET_HEIGHT, Math.floor(nodeHeight - widgetY));
}

function syncDomWidgetSize(node, widget) {
  const ctx = node.__dsm;
  if (!ctx || !widget) return;
  const margin = Math.max(0, getFiniteNumber(widget.margin, 10));
  const outerHeight = getWidgetOuterHeight(node, widget);
  const innerHeight = Math.max(0, outerHeight - margin * 2);
  const width = Math.max(MIN_NODE_WIDTH, getFiniteNumber(node?.size?.[0], MIN_NODE_WIDTH));
  widget.computedHeight = outerHeight;
  widget.width = width;
  ctx.root.style.height = `${innerHeight}px`;
  ctx.root.style.minHeight = `${Math.max(0, MIN_WIDGET_HEIGHT - margin * 2)}px`;
}

function scheduleRender(node) {
  const ctx = node.__dsm;
  if (!ctx || ctx.renderFrame) return;
  ctx.renderFrame = requestAnimationFrame(() => {
    ctx.renderFrame = 0;
    renderNode(node);
  });
}

function chainNodeCallback(node, name, fn) {
  const original = node[name];
  node[name] = function (...args) {
    const result = original?.apply(this, args);
    fn.apply(this, args);
    return result;
  };
}

function isTargetNode(nodeData, nodeType) {
  const nodeName = nodeData?.name ?? "";
  const displayName = nodeData?.display_name ?? "";
  const comfyClass = nodeType?.comfyClass ?? "";
  return nodeName === NODE_CLASS || displayName === NODE_CLASS || comfyClass === NODE_CLASS;
}

function isDoraLoaderNode(node) {
  const values = [node?.comfyClass, node?.type, node?.title, node?.constructor?.title].map((v) => String(v ?? ""));
  return values.some((v) => v === "DoRA Power LoRA Loader" || v.includes("DoRA Power LoRA Loader"));
}

function normalizeDoraLoaderState(state) {
  const st = state && typeof state === "object" ? state : {};
  const rows = Array.isArray(st.rows) ? st.rows.map(normalizeLoraRow) : [];
  return {
    loras: rows.filter((row) => row.name && row.name !== "None"),
    loader_globals: normalizeLoaderGlobals(st.globals),
  };
}

function extractLoraStackFromNode(sourceNode) {
  if (!sourceNode) return null;
  if (sourceNode.properties?.dora_power_lora) return normalizeDoraLoaderState(sourceNode.properties.dora_power_lora);
  if (sourceNode._doraRows || sourceNode._doraGlobals) {
    return normalizeDoraLoaderState({ rows: sourceNode._doraRows || [], globals: sourceNode._doraGlobals || {} });
  }
  if (sourceNode.properties?.dora_state_manager?.state) {
    const state = normalizeState(sourceNode.properties.dora_state_manager.state);
    const charId = sourceNode.properties.dora_state_manager.selected_character_id;
    const character = selectedCharacter(state, charId);
    return { loras: character.loras || [], loader_globals: character.loader_globals || {} };
  }
  return null;
}

function getSelectedGraphNodes() {
  const selected = app?.canvas?.selected_nodes;
  if (Array.isArray(selected)) return selected;
  if (selected && typeof selected === "object") return Object.values(selected).filter(Boolean);
  return [];
}

function getInputLink(node, inputName) {
  const input = (node.inputs || []).find((candidate) => candidate.name === inputName);
  if (!input || input.link == null) return null;
  const graph = node.graph || app.graph;
  return graph?.links?.[input.link] || null;
}

function getConnectedOriginNode(node, inputName) {
  const link = getInputLink(node, inputName);
  if (!link) return null;
  const graph = node.graph || app.graph;
  return graph?.getNodeById?.(link.origin_id) || null;
}

function extractTextFromNode(sourceNode) {
  if (!sourceNode) return "";
  const preferredNames = new Set(["text", "positive", "negative", "prompt", "string", "value"]);
  const widgets = sourceNode.widgets || [];
  for (const widget of widgets) {
    if (preferredNames.has(String(widget.name ?? "").toLowerCase()) && typeof widget.value === "string") return widget.value;
  }
  for (const widget of widgets) {
    if (typeof widget.value === "string" && widget.value.trim()) return widget.value;
  }
  return "";
}

function captureConnectedLoraStack(node, character) {
  const stackSource = getConnectedOriginNode(node, "lora_stack");
  const stack = extractLoraStackFromNode(stackSource);
  if (!stack) return false;
  character.loras = stack.loras;
  character.loader_globals = stack.loader_globals;
  return true;
}

function captureSelectedLoaderStack(targetNode, character) {
  const loader = getSelectedGraphNodes().filter((node) => node && node !== targetNode).find(isDoraLoaderNode);
  const stack = extractLoraStackFromNode(loader);
  if (!stack) return false;
  character.loras = stack.loras;
  character.loader_globals = stack.loader_globals;
  return true;
}

function captureConnectedInputs(node, character, prompt) {
  const changes = [];
  const stackSource = getConnectedOriginNode(node, "lora_stack");
  const stack = extractLoraStackFromNode(stackSource);
  if (stack) {
    character.loras = stack.loras;
    character.loader_globals = stack.loader_globals;
    changes.push("LoRA stack");
  }

  const positive = extractTextFromNode(getConnectedOriginNode(node, "positive_prompt"));
  if (positive || getInputLink(node, "positive_prompt")) {
    prompt.positive = positive;
    changes.push("positive prompt");
  }

  const negative = extractTextFromNode(getConnectedOriginNode(node, "negative_prompt"));
  if (negative || getInputLink(node, "negative_prompt")) {
    prompt.negative = negative;
    changes.push("negative prompt");
  }

  const settingsText = extractTextFromNode(getConnectedOriginNode(node, "settings_json_input"));
  if (settingsText || getInputLink(node, "settings_json_input")) {
    const parsed = normalizeSettings(settingsText);
    prompt.settings = parsed;
    changes.push("settings JSON");
  }

  return changes;
}

function captureSelectedNodes(targetNode, character, prompt) {
  const selected = getSelectedGraphNodes().filter((node) => node && node !== targetNode);
  const changes = [];
  const loader = selected.find(isDoraLoaderNode);
  const stack = extractLoraStackFromNode(loader);
  if (stack) {
    character.loras = stack.loras;
    character.loader_globals = stack.loader_globals;
    changes.push("LoRA stack");
  }

  const textNodes = selected.filter((node) => node !== loader).map((node) => ({ node, text: extractTextFromNode(node) })).filter((item) => item.text);
  const negativeCandidate = textNodes.find((item) => /negative|neg/i.test(`${item.node?.title ?? ""} ${item.node?.type ?? ""}`));
  const positiveCandidate = textNodes.find((item) => item !== negativeCandidate);
  if (positiveCandidate) {
    prompt.positive = positiveCandidate.text;
    changes.push("positive prompt");
  }
  if (negativeCandidate) {
    prompt.negative = negativeCandidate.text;
    changes.push("negative prompt");
  } else if (textNodes.length >= 2) {
    prompt.negative = textNodes[1].text;
    changes.push("negative prompt");
  }
  return changes;
}

async function uploadThumbnailFile(file) {
  const body = new FormData();
  body.append("image", file);
  body.append("type", "input");
  body.append("subfolder", THUMBNAIL_SUBFOLDER);
  const response = await api.fetchApi("/upload/image", { method: "POST", body });
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  const payload = await response.json();
  const filename = String(payload?.name ?? "").trim();
  if (!filename) throw new Error("upload returned no filename");
  return {
    filename,
    subfolder: String(payload?.subfolder ?? THUMBNAIL_SUBFOLDER).trim(),
    type: String(payload?.type ?? "input").trim() || "input",
  };
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

function makeTextarea(value, onChange, attrs = {}) {
  const textarea = document.createElement("textarea");
  textarea.value = value ?? "";
  Object.assign(textarea, attrs);
  textarea.addEventListener("input", () => onChange(textarea.value));
  return textarea;
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

function labelledControl(labelText, control) {
  const label = document.createElement("label");
  label.className = "dsm-labelled";
  const span = document.createElement("span");
  span.textContent = labelText;
  label.append(span, control);
  return label;
}

function setPanel(node, panel) {
  const { state, uiState } = getCurrentState(node);
  uiState.panel = panel;
  updateState(node, state, uiState, { dirty: false });
}

function renderCharacterTile(node, state, uiState, character, selectedId) {
  const tile = document.createElement("button");
  tile.type = "button";
  tile.className = `dsm-character-tile${character.id === selectedId ? " selected" : ""}`;
  const thumb = document.createElement("div");
  thumb.className = "dsm-thumb";
  const url = thumbnailUrl(character.thumbnail);
  if (url) {
    const img = document.createElement("img");
    img.src = url;
    img.alt = character.name;
    thumb.appendChild(img);
  } else {
    const initials = character.name.trim().slice(0, 2).toUpperCase() || "?";
    thumb.textContent = initials;
  }
  const name = document.createElement("div");
  name.className = "dsm-character-name";
  name.textContent = character.name;
  const meta = document.createElement("div");
  meta.className = "dsm-muted";
  meta.textContent = `${character.loras.length} LoRA${character.loras.length === 1 ? "" : "s"} · ${character.prompts.length} preset${character.prompts.length === 1 ? "" : "s"}`;
  tile.append(thumb, name, meta);
  tile.addEventListener("click", () => {
    const promptId = character.prompts[0]?.id || "";
    updateState(node, state, uiState, { characterId: character.id, promptId });
  });
  return tile;
}

function renderHeader(node, state, uiState, character, prompt) {
  const widgets = getWidgets(node);
  const section = document.createElement("div");
  section.className = "dsm-section dsm-top";

  const toolbar = document.createElement("div");
  toolbar.className = "dsm-toolbar";
  const title = document.createElement("div");
  title.className = "dsm-title";
  title.textContent = "DoRA State Manager";

  const liveInputs = makeCheckbox(Boolean(widgetValue(widgets.useRuntimeInputsWidget, false)), (checked) => {
    updateState(node, state, uiState, { useRuntimeInputs: checked, status: checked ? "Runtime inputs override saved state during execution." : "Saved state drives execution." });
  });

  toolbar.append(
    title,
    makeButton("Capture connected", () => {
      const changes = captureConnectedInputs(node, character, prompt);
      updateState(node, state, uiState, {
        characterId: character.id,
        promptId: prompt.id,
        status: changes.length ? `Captured ${changes.join(", ")} from connected inputs.` : "No connected prompt/lora inputs found.",
      });
    }, "Capture positive/negative/settings/lora_stack inputs into the selected character/preset"),
    makeButton("Capture selected nodes", () => {
      const changes = captureSelectedNodes(node, character, prompt);
      updateState(node, state, uiState, {
        characterId: character.id,
        promptId: prompt.id,
        status: changes.length ? `Captured ${changes.join(", ")} from selected graph nodes.` : "Select a DoRA loader and/or text nodes first.",
      });
    }, "Read the selected DoRA Power LoRA Loader and selected text nodes from the graph"),
    labelledControl("Runtime override", liveInputs)
  );

  const grid = document.createElement("div");
  grid.className = "dsm-character-grid";
  for (const item of state.characters) grid.appendChild(renderCharacterTile(node, state, uiState, item, character.id));

  const controls = document.createElement("div");
  controls.className = "dsm-toolbar";
  controls.append(
    makeButton("New character", () => {
      const c = defaultCharacter();
      c.id = makeId("character");
      c.name = `Character ${state.characters.length + 1}`;
      c.prompts[0].id = makeId("prompt");
      state.characters.push(c);
      updateState(node, state, uiState, { characterId: c.id, promptId: c.prompts[0].id, status: "Created character." });
    }),
    makeButton("Duplicate", () => {
      const copy = structuredCloneCompat(character);
      copy.id = makeId("character");
      copy.name = `${copy.name} Copy`;
      copy.prompts = copy.prompts.map((p) => ({ ...p, id: makeId("prompt") }));
      state.characters.push(copy);
      updateState(node, state, uiState, { characterId: copy.id, promptId: copy.prompts[0]?.id || "", status: "Duplicated character." });
    }),
    makeButton("Delete", () => {
      if (state.characters.length <= 1) {
        setStatus(node, "At least one character must remain.");
        return;
      }
      const index = state.characters.findIndex((c) => c.id === character.id);
      if (index >= 0) state.characters.splice(index, 1);
      const next = state.characters[Math.max(0, Math.min(index, state.characters.length - 1))];
      updateState(node, state, uiState, { characterId: next.id, promptId: next.prompts[0]?.id || "", status: "Deleted character." });
    })
  );

  section.append(toolbar, grid, controls);
  return section;
}

function renderCharacterPanel(node, state, uiState, character) {
  const section = document.createElement("div");
  section.className = "dsm-section dsm-panel";
  const title = document.createElement("div");
  title.className = "dsm-section-title";
  title.textContent = "Selected character";

  const nameInput = makeInput(character.name, (value) => {
    character.name = value.trim() || character.name;
    updateState(node, state, uiState, { characterId: character.id, render: true });
  });

  const preview = document.createElement("div");
  preview.className = "dsm-large-thumb";
  const url = thumbnailUrl(character.thumbnail);
  if (url) {
    const img = document.createElement("img");
    img.src = url;
    img.alt = character.name;
    preview.appendChild(img);
  } else {
    preview.textContent = "Drop thumbnail here";
  }

  const fileInput = document.createElement("input");
  fileInput.type = "file";
  fileInput.accept = "image/*";
  fileInput.style.display = "none";
  fileInput.addEventListener("change", async () => {
    const file = fileInput.files?.[0];
    if (!file) return;
    try {
      character.thumbnail = await uploadThumbnailFile(file);
      updateState(node, state, uiState, { characterId: character.id, status: "Thumbnail uploaded." });
    } catch (err) {
      setStatus(node, `Thumbnail upload failed: ${err?.message || err}`);
    } finally {
      fileInput.value = "";
    }
  });
  preview.addEventListener("click", () => fileInput.click());
  preview.addEventListener("dragover", (event) => {
    event.preventDefault();
    preview.classList.add("dragging");
  });
  preview.addEventListener("dragleave", () => preview.classList.remove("dragging"));
  preview.addEventListener("drop", async (event) => {
    event.preventDefault();
    preview.classList.remove("dragging");
    const file = [...(event.dataTransfer?.files || [])].find((candidate) => candidate.type.startsWith("image/"));
    if (!file) return;
    try {
      character.thumbnail = await uploadThumbnailFile(file);
      updateState(node, state, uiState, { characterId: character.id, status: "Thumbnail uploaded." });
    } catch (err) {
      setStatus(node, `Thumbnail upload failed: ${err?.message || err}`);
    }
  });

  const loraSummary = document.createElement("div");
  loraSummary.className = "dsm-lora-summary";
  const activeRows = character.loras.filter((row) => row.enabled && row.name && row.name !== "None");
  loraSummary.textContent = activeRows.length
    ? activeRows.map((row) => `${row.name} (${row.strength_model}/${row.strength_clip})`).join("\n")
    : "No saved LoRA stack for this character.";

  const thumbnailButtons = document.createElement("div");
  thumbnailButtons.className = "dsm-toolbar";
  thumbnailButtons.append(
    makeButton("Choose thumbnail", () => fileInput.click()),
    makeButton("Clear", () => {
      character.thumbnail = {};
      updateState(node, state, uiState, { characterId: character.id, status: "Thumbnail cleared." });
    })
  );

  section.append(title, labelledControl("Name", nameInput), preview, fileInput, thumbnailButtons, labelledControl("Saved LoRA stack", loraSummary));
  return section;
}

function renderPromptPanel(node, state, uiState, character, prompt) {
  const section = document.createElement("div");
  section.className = "dsm-section dsm-panel";

  const tabs = document.createElement("div");
  tabs.className = "dsm-tabs";
  for (const [panel, label] of [["prompts", "Prompts"], ["loras", "LoRA stack"], ["settings", "Settings"]]) {
    const btn = makeButton(label, () => setPanel(node, panel));
    if (uiState.panel === panel) btn.classList.add("selected");
    tabs.appendChild(btn);
  }
  section.appendChild(tabs);

  if (uiState.panel === "loras") {
    renderLoraPanelContent(section, node, state, uiState, character, prompt);
  } else if (uiState.panel === "settings") {
    renderSettingsPanelContent(section, node, state, uiState, character, prompt);
  } else {
    renderPromptPanelContent(section, node, state, uiState, character, prompt);
  }

  if (uiState.status) {
    const status = document.createElement("div");
    status.className = "dsm-status";
    status.textContent = uiState.status;
    section.appendChild(status);
  }
  return section;
}

function renderPromptPanelContent(section, node, state, uiState, character, prompt) {
  const header = document.createElement("div");
  header.className = "dsm-toolbar";
  const select = makeSelect(character.prompts.map((p) => ({ value: p.id, label: p.name })), prompt.id, (id) => {
    updateState(node, state, uiState, { characterId: character.id, promptId: id });
  });
  select.className = "dsm-flex";
  header.append(
    select,
    makeButton("New", () => {
      const p = defaultPrompt();
      p.id = makeId("prompt");
      p.name = `Prompt ${character.prompts.length + 1}`;
      character.prompts.push(p);
      updateState(node, state, uiState, { characterId: character.id, promptId: p.id, status: "Created prompt preset." });
    }),
    makeButton("Duplicate", () => {
      const copy = { ...structuredCloneCompat(prompt), id: makeId("prompt"), name: `${prompt.name} Copy` };
      character.prompts.push(copy);
      updateState(node, state, uiState, { characterId: character.id, promptId: copy.id, status: "Duplicated prompt preset." });
    }),
    makeButton("Delete", () => {
      if (character.prompts.length <= 1) {
        setStatus(node, "At least one prompt preset must remain.");
        return;
      }
      const index = character.prompts.findIndex((p) => p.id === prompt.id);
      if (index >= 0) character.prompts.splice(index, 1);
      const next = character.prompts[Math.max(0, Math.min(index, character.prompts.length - 1))];
      updateState(node, state, uiState, { characterId: character.id, promptId: next.id, status: "Deleted prompt preset." });
    })
  );

  const name = makeInput(prompt.name, (value) => {
    prompt.name = value.trim() || prompt.name;
    updateState(node, state, uiState, { characterId: character.id, promptId: prompt.id });
  });

  const positive = makeTextarea(prompt.positive, (value) => {
    prompt.positive = value;
    updateState(node, state, uiState, { characterId: character.id, promptId: prompt.id, render: false });
  }, { placeholder: "Positive prompt" });

  const negative = makeTextarea(prompt.negative, (value) => {
    prompt.negative = value;
    updateState(node, state, uiState, { characterId: character.id, promptId: prompt.id, render: false });
  }, { placeholder: "Negative prompt" });

  section.append(header, labelledControl("Preset name", name), labelledControl("Positive", positive), labelledControl("Negative", negative));
}

function renderLoraPanelContent(section, node, state, uiState, character, prompt) {
  const header = document.createElement("div");
  header.className = "dsm-toolbar";
  header.append(
    makeButton("Capture connected", () => {
      const changed = captureConnectedLoraStack(node, character);
      updateState(node, state, uiState, { characterId: character.id, promptId: prompt.id, status: changed ? "Captured LoRA stack from connected input." : "No connected lora_stack input found." });
    }),
    makeButton("Capture selected loader", () => {
      const changed = captureSelectedLoaderStack(node, character);
      updateState(node, state, uiState, { characterId: character.id, promptId: prompt.id, status: changed ? "Captured LoRA stack from selected DoRA loader." : "Select a DoRA Power LoRA Loader first." });
    }),
    makeButton("Add manual row", () => {
      character.loras.push({ enabled: true, name: "None", strength_model: 1.0, strength_clip: 1.0 });
      updateState(node, state, uiState, { characterId: character.id, promptId: prompt.id, status: "Added manual LoRA row." });
    })
  );
  section.appendChild(header);

  if (!character.loras.length) {
    const empty = document.createElement("div");
    empty.className = "dsm-muted";
    empty.textContent = "No LoRA stack saved. Capture an existing DoRA loader or connect a lora_stack input.";
    section.appendChild(empty);
    return;
  }

  character.loras.forEach((row, index) => {
    const line = document.createElement("div");
    line.className = "dsm-lora-row";
    const enabled = makeCheckbox(row.enabled, (checked) => {
      row.enabled = checked;
      updateState(node, state, uiState, { characterId: character.id, promptId: prompt.id });
    });
    const name = makeInput(row.name, (value) => {
      row.name = value.trim() || "None";
      updateState(node, state, uiState, { characterId: character.id, promptId: prompt.id });
    });
    const sm = makeInput(row.strength_model, (value) => {
      row.strength_model = normalizeNumber(value, row.strength_model);
      updateState(node, state, uiState, { characterId: character.id, promptId: prompt.id });
    }, { type: "number", step: "0.05" });
    const sc = makeInput(row.strength_clip, (value) => {
      row.strength_clip = normalizeNumber(value, row.strength_clip);
      updateState(node, state, uiState, { characterId: character.id, promptId: prompt.id });
    }, { type: "number", step: "0.05" });
    line.append(enabled, name, sm, sc, makeButton("×", () => {
      character.loras.splice(index, 1);
      updateState(node, state, uiState, { characterId: character.id, promptId: prompt.id, status: "Removed LoRA row." });
    }));
    section.appendChild(line);
  });
}

function renderSettingsPanelContent(section, node, state, uiState, character, prompt) {
  const globals = character.loader_globals || (character.loader_globals = {});
  const grid = document.createElement("div");
  grid.className = "dsm-grid2";

  const stackEnabled = makeCheckbox(globals.stack_enabled ?? true, (checked) => {
    character.loader_globals = normalizeLoaderGlobals({ ...globals, stack_enabled: checked });
    updateState(node, state, uiState, { characterId: character.id, promptId: prompt.id });
  });
  const autoStrength = makeCheckbox(globals.auto_strength_enabled ?? false, (checked) => {
    character.loader_globals = normalizeLoaderGlobals({ ...globals, auto_strength_enabled: checked });
    updateState(node, state, uiState, { characterId: character.id, promptId: prompt.id });
  });
  const device = makeSelect(AUTO_STRENGTH_DEVICE_CHOICES, normalizeDevice(globals.auto_strength_device ?? "gpu"), (value) => {
    character.loader_globals = normalizeLoaderGlobals({ ...globals, auto_strength_device: value });
    updateState(node, state, uiState, { characterId: character.id, promptId: prompt.id });
  });
  const broadcastMods = makeCheckbox(globals.broadcast_modulations ?? true, (checked) => {
    character.loader_globals = normalizeLoaderGlobals({ ...globals, broadcast_modulations: checked });
    updateState(node, state, uiState, { characterId: character.id, promptId: prompt.id });
  });
  const floor = makeInput(globals.auto_strength_ratio_floor ?? 0.3, (value) => {
    character.loader_globals = normalizeLoaderGlobals({ ...globals, auto_strength_ratio_floor: normalizeNumber(value, 0.3) });
    updateState(node, state, uiState, { characterId: character.id, promptId: prompt.id });
  }, { type: "number", step: "0.01" });
  const ceiling = makeInput(globals.auto_strength_ratio_ceiling ?? 1.5, (value) => {
    character.loader_globals = normalizeLoaderGlobals({ ...globals, auto_strength_ratio_ceiling: normalizeNumber(value, 1.5) });
    updateState(node, state, uiState, { characterId: character.id, promptId: prompt.id });
  }, { type: "number", step: "0.01" });
  const sliceFix = makeCheckbox(globals.dora_slice_fix ?? true, (checked) => {
    character.loader_globals = normalizeLoaderGlobals({ ...globals, dora_slice_fix: checked });
    updateState(node, state, uiState, { characterId: character.id, promptId: prompt.id });
  });
  const adalnFix = makeCheckbox(globals.dora_adaln_swap_fix ?? true, (checked) => {
    character.loader_globals = normalizeLoaderGlobals({ ...globals, dora_adaln_swap_fix: checked });
    updateState(node, state, uiState, { characterId: character.id, promptId: prompt.id });
  });

  grid.append(
    labelledControl("Stack enabled", stackEnabled),
    labelledControl("Auto-strength", autoStrength),
    labelledControl("Analysis device", device),
    labelledControl("Broadcast modulation", broadcastMods),
    labelledControl("Ratio floor", floor),
    labelledControl("Ratio ceiling", ceiling),
    labelledControl("Slice fix", sliceFix),
    labelledControl("AdaLN swap fix", adalnFix)
  );

  const settingsJson = makeTextarea(JSON.stringify(prompt.settings || {}, null, 2), (value) => {
    const parsed = safeJsonParse(value, null);
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      prompt.settings = parsed;
      updateState(node, state, uiState, { characterId: character.id, promptId: prompt.id, render: false });
    }
  }, { placeholder: "Prompt/settings JSON for future nodes" });
  settingsJson.addEventListener("change", () => {
    settingsJson.value = JSON.stringify(prompt.settings || {}, null, 2);
    updateState(node, state, uiState, { characterId: character.id, promptId: prompt.id });
  });

  section.append(grid, labelledControl("Prompt/settings JSON", settingsJson));
}

function renderNode(node) {
  const ctx = node.__dsm;
  if (!ctx?.root) return;
  const { state, uiState } = getRenderableState(node);
  const { character, prompt } = ensureSelection(node, state);
  ctx.root.innerHTML = "";
  ctx.root.className = "dsm-root";
  ctx.root.append(renderHeader(node, state, uiState, character, prompt));
  const main = document.createElement("div");
  main.className = "dsm-main";
  main.append(renderCharacterPanel(node, state, uiState, character), renderPromptPanel(node, state, uiState, character, prompt));
  ctx.root.appendChild(main);
}

function ensureStyles() {
  if (document.getElementById(STYLE_ID)) return;
  const style = document.createElement("style");
  style.id = STYLE_ID;
  style.textContent = `
    .dsm-root {
      box-sizing: border-box;
      width: 100%;
      height: 100%;
      min-height: 0;
      padding: 8px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      gap: 8px;
      color: var(--input-text, #ddd);
      background: rgba(20, 20, 20, 0.18);
      font: 12px/1.35 system-ui, sans-serif;
    }
    .dsm-root * { box-sizing: border-box; }
    .dsm-section {
      border: 1px solid rgba(128,128,128,.35);
      border-radius: 8px;
      padding: 8px;
      background: rgba(0,0,0,.10);
    }
    .dsm-top { flex: 0 0 auto; min-height: 0; }
    .dsm-main { flex: 1 1 auto; min-height: 0; display: grid; grid-template-columns: minmax(200px, .8fr) minmax(300px, 1.2fr); gap: 8px; overflow: hidden; }
    .dsm-panel { min-height: 0; overflow: auto; display: flex; flex-direction: column; gap: 8px; }
    .dsm-toolbar { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; margin-bottom: 6px; }
    .dsm-title, .dsm-section-title { font-weight: 650; opacity: .95; }
    .dsm-title { flex: 1 1 auto; font-size: 13px; }
    .dsm-muted { opacity: .68; white-space: pre-wrap; }
    .dsm-status { border-top: 1px solid rgba(128,128,128,.25); padding-top: 6px; opacity: .78; }
    .dsm-flex { flex: 1 1 auto; min-width: 0; }
    .dsm-root input, .dsm-root select, .dsm-root textarea, .dsm-root button {
      font: inherit;
      color: var(--input-text, #ddd);
      background: var(--comfy-input-bg, #222);
      border: 1px solid rgba(128,128,128,.45);
      border-radius: 5px;
      padding: 4px 6px;
      min-width: 0;
    }
    .dsm-root button { cursor: pointer; white-space: nowrap; }
    .dsm-root button.selected, .dsm-character-tile.selected { border-color: rgba(180, 210, 255, .8); background: rgba(100, 140, 220, .22); }
    .dsm-root textarea { width: 100%; min-height: 86px; resize: vertical; }
    .dsm-root input[type="checkbox"] { width: auto; }
    .dsm-labelled { display: flex; flex-direction: column; gap: 4px; min-width: 0; }
    .dsm-labelled > span { opacity: .68; }
    .dsm-character-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(112px, 1fr)); gap: 6px; max-height: 190px; overflow: auto; padding-right: 2px; }
    .dsm-character-tile { display: flex; flex-direction: column; align-items: stretch; gap: 4px; text-align: left; min-height: 126px; }
    .dsm-thumb, .dsm-large-thumb { display: flex; align-items: center; justify-content: center; border: 1px solid rgba(128,128,128,.35); border-radius: 7px; overflow: hidden; background: rgba(0,0,0,.22); color: rgba(220,220,220,.72); }
    .dsm-thumb { height: 76px; font-size: 22px; font-weight: 700; }
    .dsm-large-thumb { min-height: 170px; cursor: pointer; }
    .dsm-large-thumb.dragging { border-color: rgba(180, 210, 255, .9); }
    .dsm-thumb img, .dsm-large-thumb img { width: 100%; height: 100%; object-fit: cover; display: block; }
    .dsm-character-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-weight: 600; }
    .dsm-tabs { display: flex; gap: 6px; flex-wrap: wrap; }
    .dsm-grid2 { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 8px; }
    .dsm-lora-row { display: grid; grid-template-columns: auto minmax(110px, 1fr) 76px 76px auto; gap: 6px; align-items: center; }
    .dsm-lora-summary { white-space: pre-wrap; border: 1px solid rgba(128,128,128,.35); border-radius: 6px; padding: 6px; min-height: 58px; max-height: 190px; overflow: auto; background: rgba(0,0,0,.16); }
    @media (max-width: 720px) { .dsm-main { grid-template-columns: 1fr; } }
  `;
  document.head.appendChild(style);
}

function initializeNode(node, widget) {
  if (node.__dsmInitialized) return widget;
  node.__dsmInitialized = true;
  node.__dsmWidget = widget;
  node.resizable = true;

  const widgets = getWidgets(node);
  for (const hidden of [widgets.stateWidget, widgets.uiStateWidget, widgets.characterWidget, widgets.promptWidget, widgets.useRuntimeInputsWidget]) hideWidget(hidden);

  const oldSize = node.size || [MIN_NODE_WIDTH, MIN_NODE_HEIGHT];
  node.min_size = [MIN_NODE_WIDTH, MIN_NODE_HEIGHT];
  node.setSize?.([Math.max(oldSize[0], MIN_NODE_WIDTH), Math.max(oldSize[1], MIN_NODE_HEIGHT)]);

  const snapshot = getCurrentState(node);
  const { character, prompt } = ensureSelection(node, snapshot.state);
  updateState(node, snapshot.state, snapshot.uiState, { characterId: character.id, promptId: prompt.id, dirty: false, render: false });

  chainNodeCallback(node, "onConfigure", function () {
    const next = getCurrentState(node);
    const selection = ensureSelection(node, next.state);
    updateState(node, next.state, next.uiState, { characterId: selection.character.id, promptId: selection.prompt.id, dirty: false, render: false });
    scheduleRender(node);
  });

  chainNodeCallback(node, "onResize", function () {
    syncDomWidgetSize(node, widget);
    scheduleRender(node);
  });

  chainNodeCallback(node, "onRemoved", function () {
    const ctx = node.__dsm;
    if (!ctx?.renderFrame) return;
    cancelAnimationFrame(ctx.renderFrame);
    ctx.renderFrame = 0;
  });

  scheduleRender(node);
  return widget;
}

function maybeInjectWidgetInput(nodeData) {
  if (nodeData?.name !== NODE_CLASS && nodeData?.display_name !== NODE_CLASS) return;
  const required = nodeData?.input?.required;
  if (!required || required[CUSTOM_WIDGET_INPUT]) return;
  nodeData.input.required = { ...required, [CUSTOM_WIDGET_INPUT]: [CUSTOM_WIDGET_TYPE, {}] };
}

app.registerExtension({
  name: EXT_NAME,

  getCustomWidgets() {
    return {
      [CUSTOM_WIDGET_TYPE](node, inputName) {
        ensureStyles();
        if (node.__dsmWidget) return { widget: node.__dsmWidget, minHeight: MIN_WIDGET_HEIGHT, minWidth: MIN_NODE_WIDTH };
        const root = document.createElement("div");
        node.__dsm = { root, renderFrame: 0, state: null, uiState: null };
        const widget = node.addDOMWidget(inputName, CUSTOM_WIDGET_TYPE, root, {
          getMinHeight: () => MIN_WIDGET_HEIGHT,
          getHeight: () => "100%",
          onDraw: (domWidget) => syncDomWidgetSize(node, domWidget),
          afterResize: (domWidgetNode) => syncDomWidgetSize(domWidgetNode, widget),
          serialize: false,
        });
        widget.serialize = false;
        syncDomWidgetSize(node, widget);
        initializeNode(node, widget);
        return { widget, minHeight: MIN_WIDGET_HEIGHT, minWidth: MIN_NODE_WIDTH };
      },
    };
  },

  async beforeRegisterNodeDef(nodeType, nodeData) {
    maybeInjectWidgetInput(nodeData);
    if (!isTargetNode(nodeData, nodeType)) return;
    const originalOnSerialize = nodeType.prototype.onSerialize;
    nodeType.prototype.onSerialize = function (o) {
      const result = originalOnSerialize?.apply(this, arguments);
      try {
        const snapshot = getCurrentState(this);
        const selection = ensureSelection(this, snapshot.state);
        updateState(this, snapshot.state, snapshot.uiState, {
          characterId: selection.character.id,
          promptId: selection.prompt.id,
          dirty: false,
          render: false,
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
