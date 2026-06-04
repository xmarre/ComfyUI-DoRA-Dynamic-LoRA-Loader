import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import "../../scripts/domWidget.js";

const EXT_NAME = "comfyui_dora_dynamic_lora.state_manager";
const NODE_CLASS = "State Manager";
const LEGACY_NODE_CLASS = "DoRA State Manager";
const DORA_LOADER_CLASS = "DoRA Power LoRA Loader";
const STATE_TEXT_CLASS = "State Manager Text Box";
const STATE_TEXT_DISPLAY_CLASS = "State Text Box";
const STATE_SEED_CLASS = "State Manager Seed";
const STATE_SEED_DISPLAY_CLASS = "State Seed";
const CUSTOM_WIDGET_INPUT = "state_manager_ui";
const CUSTOM_WIDGET_TYPE = "DORA_STATE_MANAGER_UI";
const STATE_WIDGET = "state_json";
const UI_STATE_WIDGET = "ui_state_json";
const SELECTED_CHARACTER_WIDGET = "selected_character_id";
const SELECTED_PROMPT_WIDGET = "selected_prompt_id";
const STYLE_ID = "dora-state-manager-style";
const MIN_WIDGET_HEIGHT = 520;
const MIN_NODE_WIDTH = 620;
const MIN_NODE_HEIGHT = 680;
const THUMBNAIL_SUBFOLDER = "dora_state_manager";
const AUTO_STRENGTH_DEVICE_CHOICES = ["auto", "cpu", "gpu"];
const OUTPUT_NAMES = {
  control: ["state_control"],
  // Legacy/runtime outputs remain readable for compatibility, but Save/Load connected
  // should primarily use the control edge so editable prompt/seed widgets are not
  // replaced by linked runtime values.
  lora: ["dora_state", "selected_lora_stack"],
  positive: ["positive_prompt_template", "positive_prompt"],
  negative: ["negative_prompt_template", "negative_prompt"],
  settings: ["settings_json", "state_settings"],
  seed: ["seed"],
};
const TEXT_WIDGET_NAMES = ["text", "prompt", "positive", "negative", "string", "value", "wildcard", "wildcards"];
const POSITIVE_HINT_RE = /positive|pos|prompt/i;
const NEGATIVE_HINT_RE = /negative|neg/i;
const SEED_HINT_RE = /seed|noise_seed|rgthree|control_after_generate|randomize|variation|subseed/i;
const SKIP_SETTING_NODE_RE = /clip text encode|conditioning|preview|reroute/i;

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

function normalizeInteger(value, fallback = 0) {
  if (typeof value === "boolean") return fallback;
  const n = Number(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.max(0, Math.min(Number.MAX_SAFE_INTEGER, Math.floor(n)));
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
  const cacheKey = String(value.cacheKey ?? value.hash ?? value.etag ?? value.updatedAt ?? value.version ?? "").trim();
  if (filename) return cacheKey ? { filename, subfolder, type, cacheKey } : { filename, subfolder, type };
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
  if (t.cacheKey) params.set("cache_key", t.cacheKey);
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
  node?.setDirtyCanvas?.(true, true);
  node?.graph?.change?.();
}

function updateState(node, state, uiState, opts = {}) {
  const widgets = getWidgets(node);
  const normalizedState = normalizeState(state);
  const normalizedUiState = normalizeUiState(uiState || defaultUiState());
  const { character, prompt } = ensureSelection(node, normalizedState);
  setWidgetValue(widgets.stateWidget, serializeState(normalizedState));
  setWidgetValue(widgets.uiStateWidget, serializeUiState({ ...normalizedUiState, status: opts.status ?? normalizedUiState.status }));
  setWidgetValue(widgets.characterWidget, opts.characterId ?? character.id);
  setWidgetValue(widgets.promptWidget, opts.promptId ?? prompt.id);
  node.properties = node.properties || {};
  node.properties.dora_state_manager = {
    state: normalizedState,
    selected_character_id: widgetValue(widgets.characterWidget, ""),
    selected_prompt_id: widgetValue(widgets.promptWidget, ""),
  };
  cacheRenderableState(node, normalizedState, normalizeUiState(widgetValue(widgets.uiStateWidget, "")));
  if (opts.dirty !== false) markNodeDirty(node);
  if (opts.render !== false) scheduleRender(node);
}

function cacheRenderableState(node, state, uiState) {
  const ctx = node.__dsm;
  if (!ctx) return;
  ctx.state = normalizeState(state);
  ctx.uiState = normalizeUiState(uiState);
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
  updateState(node, state, { ...uiState, status: String(text ?? "") });
}

function getFiniteNumber(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function getWidgetOuterHeight(node, widget) {
  const nodeHeight = Math.max(MIN_NODE_HEIGHT, getFiniteNumber(node?.size?.[1], MIN_NODE_HEIGHT));
  const widgetY = Math.max(0, getFiniteNumber(widget?.y, getFiniteNumber(widget?.last_y, 0)));
  return Math.max(MIN_WIDGET_HEIGHT, nodeHeight - widgetY - 16);
}

function syncDomWidgetSize(node, widget) {
  const ctx = node.__dsm;
  if (!ctx?.root) return;
  const margin = Math.max(0, getFiniteNumber(widget.margin, 10));
  const outerHeight = getWidgetOuterHeight(node, widget);
  const innerHeight = Math.max(0, outerHeight - margin * 2);
  const width = Math.max(MIN_NODE_WIDTH, getFiniteNumber(node?.size?.[0], MIN_NODE_WIDTH));
  ctx.root.style.width = `${Math.max(0, width - margin * 2)}px`;
  ctx.root.style.height = `${innerHeight}px`;
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
  return [nodeName, displayName, comfyClass].some((name) => name === NODE_CLASS || name === LEGACY_NODE_CLASS);
}

function nodeNameText(node) {
  return [node?.comfyClass, node?.type, node?.title, node?.constructor?.title].map((v) => String(v ?? "")).join(" ");
}

function isDoraLoaderNode(node) {
  return nodeNameText(node).includes(DORA_LOADER_CLASS);
}

function hasExactNodeClass(node, className) {
  return [node?.comfyClass, node?.type, node?.constructor?.title]
    .map((value) => String(value ?? ""))
    .some((value) => value === className);
}

function hasAnyNodeClassOrTitle(node, classNames) {
  const values = [node?.comfyClass, node?.type, node?.title, node?.constructor?.title].map((value) => String(value ?? ""));
  return classNames.some((className) => values.includes(className));
}

function isStateManagerNode(node) {
  return hasExactNodeClass(node, NODE_CLASS) || hasExactNodeClass(node, LEGACY_NODE_CLASS);
}

function isStateTextNode(node) {
  return hasAnyNodeClassOrTitle(node, [STATE_TEXT_CLASS, STATE_TEXT_DISPLAY_CLASS]);
}

function isStateSeedNode(node) {
  return hasAnyNodeClassOrTitle(node, [STATE_SEED_CLASS, STATE_SEED_DISPLAY_CLASS]);
}

function getRoleWidget(node) {
  const map = getWidgetMap(node);
  return map.get("role") || null;
}

function getStateTextRole(node, fallback = "generic") {
  const role = String(getRoleWidget(node)?.value ?? fallback).toLowerCase();
  if (role.includes("positive")) return "positive";
  if (role.includes("negative")) return "negative";
  return "generic";
}

function getControlledTargets(node) {
  return getOutputTargets(node, OUTPUT_NAMES.control);
}

function getControlledNodes(node) {
  return uniqueNodes(getControlledTargets(node));
}

function normalizeDoraLoaderState(state) {
  const st = state && typeof state === "object" ? state : {};
  const rowsIn = Array.isArray(st.rows) ? st.rows : Array.isArray(st.loras) ? st.loras : [];
  const rows = rowsIn.map(normalizeLoraRow).filter((row) => row.name && row.name !== "None");
  return {
    loras: rows,
    loader_globals: normalizeLoaderGlobals(st.globals ?? st.loader_globals),
  };
}

function extractLoraStackFromNode(sourceNode) {
  if (!sourceNode) return null;
  const loaderApi = globalThis.__doraPowerLoraLoaderApi;
  if (loaderApi?.getState && isDoraLoaderNode(sourceNode)) return normalizeDoraLoaderState(loaderApi.getState(sourceNode));
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

function applyLoraStackToNode(targetNode, character) {
  if (!targetNode || !isDoraLoaderNode(targetNode)) return false;
  const payload = { loras: character.loras || [], loader_globals: character.loader_globals || {} };
  const loaderApi = globalThis.__doraPowerLoraLoaderApi;
  if (loaderApi?.setState) return !!loaderApi.setState(targetNode, payload);

  const rows = (payload.loras || []).map((row) => ({
    enabled: !!row.enabled,
    name: row.name || "None",
    strengthModel: normalizeNumber(row.strength_model, 1.0),
    strengthClip: normalizeNumber(row.strength_clip, normalizeNumber(row.strength_model, 1.0)),
  }));
  targetNode.properties = targetNode.properties || {};
  targetNode.properties.dora_power_lora = { rows, globals: normalizeLoaderGlobals(payload.loader_globals) };
  targetNode._doraRows = rows;
  targetNode._doraGlobals = normalizeLoaderGlobals(payload.loader_globals);
  markNodeDirty(targetNode);
  return true;
}

function getSelectedGraphNodes() {
  const selected = app?.canvas?.selected_nodes;
  if (Array.isArray(selected)) return selected;
  if (selected && typeof selected === "object") return Object.values(selected).filter(Boolean);
  return [];
}

function getGraph(node) {
  return node?.graph || app.graph;
}

function getOutputTargets(node, outputNames) {
  const names = new Set(outputNames);
  const graph = getGraph(node);
  const out = [];
  for (const output of node.outputs || []) {
    if (!names.has(output.name)) continue;
    for (const linkId of output.links || []) {
      const link = graph?.links?.[linkId];
      if (!link) continue;
      const targetNode = graph?.getNodeById?.(link.target_id);
      if (!targetNode) continue;
      const input = (targetNode.inputs || [])[link.target_slot] || null;
      out.push({ node: targetNode, link, outputName: output.name, inputName: input?.name || "" });
    }
  }
  return out;
}

function uniqueNodes(targets) {
  const seen = new Set();
  const out = [];
  for (const target of targets) {
    const node = target?.node || target;
    if (!node || seen.has(node.id)) continue;
    seen.add(node.id);
    out.push(node);
  }
  return out;
}

function getWidgetMap(node) {
  const map = new Map();
  for (const widget of node?.widgets || []) {
    if (!widget?.name) continue;
    map.set(String(widget.name).toLowerCase(), widget);
  }
  return map;
}

function setNodeWidget(widgetNode, widget, value) {
  if (!widget) return false;
  widget.value = value;
  widget.callback?.(value);
  widgetNode?.onWidgetChanged?.(widget.name, value, widget.value, widget);
  markNodeDirty(widgetNode);
  return true;
}

function isTextWidget(widget) {
  if (!widget) return false;
  if (typeof widget.value === "string") return true;
  const t = String(widget.type ?? "").toLowerCase();
  return t === "string" || t === "text" || t === "customtext";
}

function findTextWidget(node, role = "", inputName = "") {
  const widgets = node?.widgets || [];
  const hints = [];
  const roleText = String(role || "").toLowerCase();
  const inputText = String(inputName || "").toLowerCase();
  if (inputText) hints.push(inputText);
  if (roleText === "positive") hints.push("positive", "pos", "prompt");
  if (roleText === "negative") hints.push("negative", "neg");
  hints.push(...TEXT_WIDGET_NAMES);

  for (const hint of hints) {
    const exact = widgets.find((w) => isTextWidget(w) && String(w.name ?? "").toLowerCase() === hint);
    if (exact) return exact;
  }
  for (const hint of hints) {
    const partial = widgets.find((w) => isTextWidget(w) && `${w.name ?? ""} ${w.label ?? ""}`.toLowerCase().includes(hint));
    if (partial) return partial;
  }
  return widgets.find(isTextWidget) || null;
}

function extractTextFromNode(sourceNode, role = "", inputName = "") {
  const widget = findTextWidget(sourceNode, role, inputName);
  return typeof widget?.value === "string" ? widget.value : "";
}

function applyTextToNode(targetNode, text, role = "", inputName = "") {
  const widget = findTextWidget(targetNode, role, inputName);
  return setNodeWidget(targetNode, widget, String(text ?? ""));
}

function isJsonSafeWidgetValue(value) {
  if (value == null) return true;
  if (["string", "number", "boolean"].includes(typeof value)) return true;
  if (Array.isArray(value)) return value.every(isJsonSafeWidgetValue);
  if (typeof value === "object") {
    try {
      const encoded = JSON.stringify(value);
      return encoded.length < 20000;
    } catch {
      return false;
    }
  }
  return false;
}

function nodeIdentity(node) {
  return {
    id: node?.id ?? null,
    type: String(node?.type ?? ""),
    comfyClass: String(node?.comfyClass ?? ""),
    title: String(node?.title ?? ""),
  };
}

function nodeIdentityKey(identity) {
  return `${identity.comfyClass || identity.type || "?"}::${identity.title || ""}`;
}

function isSeedNode(node) {
  const text = nodeNameText(node);
  if (/rgthree/i.test(text) && /seed/i.test(text)) return true;
  if (/seed/i.test(text)) return true;
  return (node?.widgets || []).some((widget) => SEED_HINT_RE.test(`${widget?.name ?? ""} ${widget?.label ?? ""}`));
}

function isPromptLikeNode(node) {
  if (isStateTextNode(node)) return true;
  const text = nodeNameText(node);
  if (SKIP_SETTING_NODE_RE.test(text)) return true;
  return (node?.widgets || []).some((widget) => isTextWidget(widget) && /prompt|text|positive|negative/i.test(`${widget.name ?? ""} ${widget.label ?? ""}`));
}

function captureNodeSnapshot(node) {
  if (!node || isStateManagerNode(node) || isDoraLoaderNode(node)) return null;
  const widgets = {};
  const seedWidgets = {};
  for (const widget of node.widgets || []) {
    if (!widget?.name) continue;
    if (widget.serialize === false) continue;
    if (String(widget.type ?? "").toLowerCase() === "button") continue;
    if (!isJsonSafeWidgetValue(widget.value)) continue;
    widgets[widget.name] = structuredCloneCompat(widget.value);
    if (SEED_HINT_RE.test(`${widget.name ?? ""} ${widget.label ?? ""}`)) {
      seedWidgets[widget.name] = structuredCloneCompat(widget.value);
    }
  }
  if (!Object.keys(widgets).length) return null;
  const identity = nodeIdentity(node);
  const seed = normalizeSeedFromWidgets(widgets, null);
  return {
    version: 1,
    identity,
    key: nodeIdentityKey(identity),
    is_seed_node: isSeedNode(node),
    widgets,
    seed_widgets: seedWidgets,
    seed,
  };
}

function normalizeSeedFromWidgets(widgets, fallback = null) {
  if (!widgets || typeof widgets !== "object") return fallback;
  for (const key of ["seed", "noise_seed", "value"]) {
    if (Object.prototype.hasOwnProperty.call(widgets, key)) return normalizeInteger(widgets[key], fallback ?? 0);
  }
  for (const [key, value] of Object.entries(widgets)) {
    if (/seed/i.test(key)) return normalizeInteger(value, fallback ?? 0);
  }
  return fallback;
}

function normalizeNodeSnapshots(raw) {
  if (!Array.isArray(raw)) return [];
  return raw.filter((item) => item && typeof item === "object" && item.widgets && typeof item.widgets === "object");
}

function mergeCapturedSettings(prompt, nodes, { replaceNodes = true } = {}) {
  const settings = normalizeSettings(prompt.settings);
  const snapshots = nodes.map(captureNodeSnapshot).filter(Boolean);
  if (snapshots.length) {
    if (replaceNodes) {
      settings.nodes = snapshots;
    } else {
      const byKey = new Map(normalizeNodeSnapshots(settings.nodes).map((snap) => [snap.key || nodeIdentityKey(snap.identity || {}), snap]));
      for (const snap of snapshots) byKey.set(snap.key || nodeIdentityKey(snap.identity || {}), snap);
      settings.nodes = [...byKey.values()];
    }
  }

  const seedSnapshot = snapshots.find((snap) => snap.is_seed_node || snap.seed != null || Object.keys(snap.seed_widgets || {}).length);
  if (seedSnapshot) {
    settings.seed = normalizeInteger(seedSnapshot.seed ?? normalizeSeedFromWidgets(seedSnapshot.widgets, 0), 0);
    settings.rgthree_seed = seedSnapshot;
  }
  prompt.settings = settings;
  return snapshots;
}

function nodeMatchesSnapshot(node, snapshot) {
  if (!node || !snapshot) return false;
  const identity = snapshot.identity || {};
  if (identity.id != null && node.id != null && String(identity.id) === String(node.id)) return true;
  const nodeClass = String(node.comfyClass || node.type || "");
  const snapClass = String(identity.comfyClass || identity.type || "");
  const nodeTitle = String(node.title || "");
  const snapTitle = String(identity.title || "");
  if (nodeClass && snapClass && nodeClass === snapClass && (!snapTitle || nodeTitle === snapTitle)) return true;
  if (snapshot.key && snapshot.key === nodeIdentityKey(nodeIdentity(node))) return true;
  return false;
}

function findSnapshotForNode(settings, node) {
  const normalized = normalizeSettings(settings);
  if (isSeedNode(node) && normalized.rgthree_seed) return normalized.rgthree_seed;
  const snapshots = normalizeNodeSnapshots(normalized.nodes);
  return snapshots.find((snap) => nodeMatchesSnapshot(node, snap)) || null;
}

function applySnapshotToNode(node, snapshot) {
  if (!node || !snapshot?.widgets) return 0;
  let changed = 0;
  const widgetMap = getWidgetMap(node);
  for (const [name, value] of Object.entries(snapshot.widgets)) {
    const widget = widgetMap.get(String(name).toLowerCase());
    if (!widget) continue;
    if (setNodeWidget(node, widget, structuredCloneCompat(value))) changed += 1;
  }
  return changed;
}

function extractSeedFromSettings(settings) {
  const normalized = normalizeSettings(settings);
  if (normalized.seed != null) return normalizeInteger(normalized.seed, 0);
  if (normalized.rgthree_seed?.seed != null) return normalizeInteger(normalized.rgthree_seed.seed, 0);
  const widgets = normalized.rgthree_seed?.widgets;
  const fromWidgets = normalizeSeedFromWidgets(widgets, null);
  if (fromWidgets != null) return fromWidgets;
  for (const snap of normalizeNodeSnapshots(normalized.nodes)) {
    const seed = normalizeSeedFromWidgets(snap.widgets, null);
    if (seed != null) return seed;
  }
  return null;
}

function applySeedToNode(node, settings) {
  const seed = extractSeedFromSettings(settings);
  if (seed == null) return 0;
  const widgetMap = getWidgetMap(node);
  let changed = 0;
  for (const name of ["seed", "noise_seed", "value"]) {
    const widget = widgetMap.get(name);
    if (widget && typeof widget.value !== "boolean") {
      if (setNodeWidget(node, widget, seed)) changed += 1;
      return changed;
    }
  }
  for (const widget of node.widgets || []) {
    if (SEED_HINT_RE.test(`${widget?.name ?? ""} ${widget?.label ?? ""}`) && typeof widget.value !== "boolean") {
      if (setNodeWidget(node, widget, seed)) changed += 1;
      break;
    }
  }
  return changed;
}

function applySettingsToNodes(nodes, settings) {
  let changedNodes = 0;
  for (const node of nodes) {
    const snapshot = findSnapshotForNode(settings, node);
    let changed = snapshot ? applySnapshotToNode(node, snapshot) : 0;
    if (isSeedNode(node)) changed += applySeedToNode(node, settings);
    if (changed > 0) changedNodes += 1;
  }
  return changedNodes;
}

function saveConnectedState(targetNode, character, prompt) {
  const changes = [];
  const controlledNodes = getControlledNodes(targetNode).filter((node) => node && node !== targetNode);

  // Preferred save/load-only path: manager.state_control -> target.state_control.
  const controlledLoaderTargets = controlledNodes.filter(isDoraLoaderNode);
  const legacyLoaderTargets = controlledLoaderTargets.length ? [] : uniqueNodes([...getOutputTargets(targetNode, OUTPUT_NAMES.lora)]).filter(isDoraLoaderNode);
  const loaderTargets = controlledLoaderTargets.length ? controlledLoaderTargets : legacyLoaderTargets;
  const loaderStack = loaderTargets.map(extractLoraStackFromNode).find(Boolean);
  if (loaderStack) {
    character.loras = loaderStack.loras || [];
    character.loader_globals = loaderStack.loader_globals || {};
    changes.push("LoRA stack");
  }

  const controlledTextNodes = controlledNodes.filter(isStateTextNode);
  let savedPositive = false;
  let savedNegative = false;
  for (const textNode of controlledTextNodes) {
    const role = getStateTextRole(textNode);
    const value = extractTextFromNode(textNode, role);
    if (role === "positive") {
      prompt.positive = value;
      savedPositive = true;
    } else if (role === "negative") {
      prompt.negative = value;
      savedNegative = true;
    }
  }
  if (savedPositive) changes.push("positive prompt template");
  if (savedNegative) changes.push("negative prompt template");

  // Compatibility fallback for old graphs. Avoid depending on this for normal use;
  // it can make editable widgets link-controlled if users connect STRING outputs.
  if (!savedPositive) {
    for (const target of getOutputTargets(targetNode, OUTPUT_NAMES.positive)) {
      const widget = findTextWidget(target.node, "positive", target.inputName);
      if (widget) {
        prompt.positive = typeof widget.value === "string" ? widget.value : "";
        changes.push("positive prompt template");
        break;
      }
    }
  }
  if (!savedNegative) {
    for (const target of getOutputTargets(targetNode, OUTPUT_NAMES.negative)) {
      const widget = findTextWidget(target.node, "negative", target.inputName);
      if (widget) {
        prompt.negative = typeof widget.value === "string" ? widget.value : "";
        changes.push("negative prompt template");
        break;
      }
    }
  }

  const controlledSettingTargets = controlledNodes.filter((node) =>
    node !== targetNode && !isDoraLoaderNode(node) && !isStateTextNode(node)
  );
  const legacySettingTargets = controlledSettingTargets.length ? [] : uniqueNodes([
    ...getOutputTargets(targetNode, OUTPUT_NAMES.settings),
    ...getOutputTargets(targetNode, OUTPUT_NAMES.seed),
  ]).filter((node) => node !== targetNode && !isDoraLoaderNode(node) && !isStateTextNode(node));
  const settingTargets = controlledSettingTargets.length ? controlledSettingTargets : legacySettingTargets;
  const snapshots = mergeCapturedSettings(prompt, settingTargets, { replaceNodes: true });
  if (snapshots.length) changes.push(`settings from ${snapshots.length} node${snapshots.length === 1 ? "" : "s"}`);
  if (prompt.settings?.seed != null) changes.push("seed");
  return [...new Set(changes)];
}

function applyConnectedState(targetNode, character, prompt) {
  const changes = [];
  const controlledNodes = getControlledNodes(targetNode).filter((node) => node && node !== targetNode);

  const controlledLoaderTargets = controlledNodes.filter(isDoraLoaderNode);
  const legacyLoaderTargets = controlledLoaderTargets.length ? [] : uniqueNodes([...getOutputTargets(targetNode, OUTPUT_NAMES.lora)]).filter(isDoraLoaderNode);
  const loaderTargets = controlledLoaderTargets.length ? controlledLoaderTargets : legacyLoaderTargets;
  let loaderChanged = 0;
  for (const loader of loaderTargets) if (applyLoraStackToNode(loader, character)) loaderChanged += 1;
  if (loaderChanged) changes.push(`LoRA stack to ${loaderChanged} loader${loaderChanged === 1 ? "" : "s"}`);

  const controlledTextNodes = controlledNodes.filter(isStateTextNode);
  let posChanged = 0;
  let negChanged = 0;
  for (const textNode of controlledTextNodes) {
    const role = getStateTextRole(textNode);
    if (role === "positive") {
      if (applyTextToNode(textNode, prompt.positive, "positive")) posChanged += 1;
    } else if (role === "negative") {
      if (applyTextToNode(textNode, prompt.negative, "negative")) negChanged += 1;
    }
  }
  if (posChanged) changes.push(`positive template to ${posChanged} node${posChanged === 1 ? "" : "s"}`);
  if (negChanged) changes.push(`negative template to ${negChanged} node${negChanged === 1 ? "" : "s"}`);

  const controlledSettingTargets = controlledNodes.filter((node) =>
    node !== targetNode && !isDoraLoaderNode(node) && !isStateTextNode(node)
  );
  const legacySettingTargets = controlledSettingTargets.length ? [] : uniqueNodes([
    ...getOutputTargets(targetNode, OUTPUT_NAMES.settings),
    ...getOutputTargets(targetNode, OUTPUT_NAMES.seed),
  ]).filter((node) => node !== targetNode && !isDoraLoaderNode(node) && !isStateTextNode(node));
  const settingTargets = controlledSettingTargets.length ? controlledSettingTargets : legacySettingTargets;
  const settingsChanged = applySettingsToNodes(settingTargets, prompt.settings);
  if (settingsChanged) changes.push(`settings/seed to ${settingsChanged} node${settingsChanged === 1 ? "" : "s"}`);
  return changes;
}

function classifySelectedTextNodes(nodes) {
  const textNodes = nodes
    .map((node) => ({
      node,
      text: extractTextFromNode(node, isStateTextNode(node) ? getStateTextRole(node) : ""),
      name: nodeNameText(node),
      role: isStateTextNode(node) ? getStateTextRole(node) : "",
    }))
    .filter((item) => item.text || findTextWidget(item.node));
  const negative = textNodes.find((item) => item.role === "negative") || textNodes.find((item) => NEGATIVE_HINT_RE.test(item.name));
  const positive = textNodes.find((item) => item.role === "positive") || textNodes.find((item) => item !== negative && POSITIVE_HINT_RE.test(item.name)) || textNodes.find((item) => item !== negative);
  const fallbackNegative = negative || (textNodes.length >= 2 ? textNodes.find((item) => item !== positive) : null);
  return { positive, negative: fallbackNegative, used: [positive?.node, fallbackNegative?.node].filter(Boolean) };
}

function saveSelectedState(targetNode, character, prompt) {
  const selected = getSelectedGraphNodes().filter((node) => node && node !== targetNode);
  const changes = [];
  const loader = selected.find(isDoraLoaderNode);
  const stack = extractLoraStackFromNode(loader);
  if (stack) {
    character.loras = stack.loras || [];
    character.loader_globals = stack.loader_globals || {};
    changes.push("LoRA stack");
  }

  const classified = classifySelectedTextNodes(selected.filter((node) => node !== loader && !isSeedNode(node)));
  if (classified.positive) {
    prompt.positive = classified.positive.text;
    changes.push("positive prompt template");
  }
  if (classified.negative) {
    prompt.negative = classified.negative.text;
    changes.push("negative prompt template");
  }

  const used = new Set([loader, ...classified.used].filter(Boolean));
  const settingNodes = selected.filter((node) => !used.has(node));
  const snapshots = mergeCapturedSettings(prompt, settingNodes, { replaceNodes: true });
  if (snapshots.length) changes.push(`settings from ${snapshots.length} node${snapshots.length === 1 ? "" : "s"}`);
  if (prompt.settings?.seed != null) changes.push("seed");
  return [...new Set(changes)];
}

function applySelectedState(targetNode, character, prompt) {
  const selected = getSelectedGraphNodes().filter((node) => node && node !== targetNode);
  const changes = [];
  const loaderTargets = selected.filter(isDoraLoaderNode);
  let loaderChanged = 0;
  for (const loader of loaderTargets) if (applyLoraStackToNode(loader, character)) loaderChanged += 1;
  if (loaderChanged) changes.push(`LoRA stack to ${loaderChanged} loader${loaderChanged === 1 ? "" : "s"}`);

  const classified = classifySelectedTextNodes(selected.filter((node) => !isDoraLoaderNode(node) && !isSeedNode(node)));
  if (classified.positive && applyTextToNode(classified.positive.node, prompt.positive, "positive")) changes.push("positive template");
  if (classified.negative && applyTextToNode(classified.negative.node, prompt.negative, "negative")) changes.push("negative template");

  const used = new Set([...loaderTargets, ...classified.used].filter(Boolean));
  const settingNodes = selected.filter((node) => !used.has(node));
  const settingsChanged = applySettingsToNodes(settingNodes, prompt.settings);
  if (settingsChanged) changes.push(`settings/seed to ${settingsChanged} node${settingsChanged === 1 ? "" : "s"}`);
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
    cacheKey: String(Date.now()),
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
    option.value = typeof value === "object" ? value.value : value;
    option.textContent = typeof value === "object" ? value.label : value;
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
  updateState(node, state, { ...uiState, panel });
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
    thumb.textContent = character.name.trim().slice(0, 2).toUpperCase() || "?";
  }
  const name = document.createElement("div");
  name.className = "dsm-character-name";
  name.textContent = character.name;
  const meta = document.createElement("div");
  meta.className = "dsm-muted";
  meta.textContent = `${character.loras.filter((row) => row.enabled && row.name !== "None").length} LoRA · ${character.prompts.length} preset${character.prompts.length === 1 ? "" : "s"}`;
  tile.append(thumb, name, meta);
  tile.addEventListener("click", () => {
    const promptId = character.prompts[0]?.id || "";
    updateState(node, state, uiState, { characterId: character.id, promptId, status: `Selected ${character.name}. Use Load/Apply to push it into connected nodes.` });
  });
  return tile;
}

function renderHeader(node, state, uiState, character, prompt) {
  const section = document.createElement("div");
  section.className = "dsm-section dsm-top";

  const toolbar = document.createElement("div");
  toolbar.className = "dsm-toolbar";
  const title = document.createElement("div");
  title.className = "dsm-title";
  title.textContent = "State Manager";

  toolbar.append(
    title,
    makeButton("Save connected", () => {
      const changes = saveConnectedState(node, character, prompt);
      updateState(node, state, uiState, {
        characterId: character.id,
        promptId: prompt.id,
        status: changes.length ? `Saved ${changes.join(", ")} from connected nodes.` : "No connected downstream nodes had capturable state.",
      });
    }, "Capture current values from nodes connected to this manager's outputs"),
    makeButton("Load connected", () => {
      const changes = applyConnectedState(node, character, prompt);
      updateState(node, state, uiState, {
        characterId: character.id,
        promptId: prompt.id,
        status: changes.length ? `Loaded ${changes.join(", ")} into connected nodes.` : "No connected downstream nodes accepted this state.",
      });
    }, "Push the selected character/preset into nodes connected to this manager's outputs"),
    makeButton("Save selected", () => {
      const changes = saveSelectedState(node, character, prompt);
      updateState(node, state, uiState, {
        characterId: character.id,
        promptId: prompt.id,
        status: changes.length ? `Saved ${changes.join(", ")} from selected nodes.` : "Select a loader, text, seed, or settings node first.",
      });
    }, "Capture selected graph nodes into the current character/preset"),
    makeButton("Apply selected", () => {
      const changes = applySelectedState(node, character, prompt);
      updateState(node, state, uiState, {
        characterId: character.id,
        promptId: prompt.id,
        status: changes.length ? `Applied ${changes.join(", ")} to selected nodes.` : "Selected nodes did not match this state.",
      });
    }, "Apply the current character/preset to selected graph nodes")
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
  for (const [panel, label] of [["prompts", "Prompts"], ["loras", "LoRA stack"], ["settings", "Settings/seed"]]) {
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
    updateState(node, state, uiState, { characterId: character.id, promptId: id, status: "Selected prompt preset. Use Load/Apply to push it into graph nodes." });
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
  }, { placeholder: "Positive prompt template. Wildcards stay here and expand downstream." });

  const negative = makeTextarea(prompt.negative, (value) => {
    prompt.negative = value;
    updateState(node, state, uiState, { characterId: character.id, promptId: prompt.id, render: false });
  }, { placeholder: "Negative prompt template. Wildcards stay here and expand downstream." });

  section.append(header, labelledControl("Preset name", name), labelledControl("Positive template", positive), labelledControl("Negative template", negative));
}

function renderLoraPanelContent(section, node, state, uiState, character, prompt) {
  const header = document.createElement("div");
  header.className = "dsm-toolbar";
  header.append(
    makeButton("Save connected loader", () => {
      const controlledLoaders = getControlledNodes(node).filter(isDoraLoaderNode);
      const loaders = controlledLoaders.length ? controlledLoaders : uniqueNodes(getOutputTargets(node, OUTPUT_NAMES.lora)).filter(isDoraLoaderNode);
      const stack = loaders.map(extractLoraStackFromNode).find(Boolean);
      if (stack) {
        character.loras = stack.loras;
        character.loader_globals = stack.loader_globals;
      }
      updateState(node, state, uiState, { characterId: character.id, promptId: prompt.id, status: stack ? "Saved LoRA stack from connected loader." : "No connected DoRA loader found." });
    }),
    makeButton("Apply connected loader", () => {
      const controlledLoaders = getControlledNodes(node).filter(isDoraLoaderNode);
      const loaders = controlledLoaders.length ? controlledLoaders : uniqueNodes(getOutputTargets(node, OUTPUT_NAMES.lora)).filter(isDoraLoaderNode);
      let count = 0;
      for (const loader of loaders) if (applyLoraStackToNode(loader, character)) count += 1;
      updateState(node, state, uiState, { characterId: character.id, promptId: prompt.id, status: count ? `Applied LoRA stack to ${count} connected loader${count === 1 ? "" : "s"}.` : "No connected DoRA loader accepted the stack." });
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
    empty.textContent = "No LoRA stack saved. Save a connected/selected DoRA loader or add manual rows.";
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

  const seedValue = extractSeedFromSettings(prompt.settings) ?? 0;
  const seedInput = makeInput(seedValue, (value) => {
    const settings = normalizeSettings(prompt.settings);
    settings.seed = normalizeInteger(value, 0);
    if (settings.rgthree_seed?.widgets) settings.rgthree_seed.widgets.seed = settings.seed;
    if (settings.rgthree_seed) settings.rgthree_seed.seed = settings.seed;
    prompt.settings = settings;
    updateState(node, state, uiState, { characterId: character.id, promptId: prompt.id });
  }, { type: "number", step: "1", min: "0" });

  const settingsJson = makeTextarea(JSON.stringify(prompt.settings || {}, null, 2), (value) => {
    const parsed = safeJsonParse(value, null);
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      prompt.settings = parsed;
      updateState(node, state, uiState, { characterId: character.id, promptId: prompt.id, render: false });
    }
  }, { placeholder: "Saved node settings JSON. Save connected/selected nodes to populate this." });
  settingsJson.addEventListener("change", () => {
    settingsJson.value = JSON.stringify(prompt.settings || {}, null, 2);
    updateState(node, state, uiState, { characterId: character.id, promptId: prompt.id });
  });

  const snapshots = normalizeNodeSnapshots(prompt.settings?.nodes);
  const summary = document.createElement("div");
  summary.className = "dsm-lora-summary";
  const seed = extractSeedFromSettings(prompt.settings);
  const lines = [];
  if (seed != null) lines.push(`Seed: ${seed}`);
  if (prompt.settings?.rgthree_seed) lines.push("rgthree seed snapshot: yes");
  if (snapshots.length) {
    lines.push(`Captured settings nodes: ${snapshots.length}`);
    lines.push(...snapshots.slice(0, 12).map((snap) => `- ${snap.identity?.title || snap.identity?.comfyClass || snap.identity?.type || snap.key || "node"}`));
  }
  summary.textContent = lines.length ? lines.join("\n") : "No saved settings/seed snapshots. Use Save connected or Save selected.";

  const settingsToolbar = document.createElement("div");
  settingsToolbar.className = "dsm-toolbar";
  settingsToolbar.append(
    makeButton("Clear settings", () => {
      prompt.settings = {};
      updateState(node, state, uiState, { characterId: character.id, promptId: prompt.id, status: "Cleared prompt settings and seed." });
    }),
    makeButton("Save selected settings", () => {
      const selected = getSelectedGraphNodes().filter((target) => target && target !== node && !isDoraLoaderNode(target) && !isPromptLikeNode(target));
      const snapshots = mergeCapturedSettings(prompt, selected, { replaceNodes: true });
      updateState(node, state, uiState, { characterId: character.id, promptId: prompt.id, status: snapshots.length ? `Saved ${snapshots.length} selected settings node${snapshots.length === 1 ? "" : "s"}.` : "No selected settings/seed nodes found." });
    }),
    makeButton("Apply selected settings", () => {
      const selected = getSelectedGraphNodes().filter((target) => target && target !== node && !isDoraLoaderNode(target));
      const count = applySettingsToNodes(selected, prompt.settings);
      updateState(node, state, uiState, { characterId: character.id, promptId: prompt.id, status: count ? `Applied settings/seed to ${count} selected node${count === 1 ? "" : "s"}.` : "No selected node matched the saved settings." });
    })
  );

  section.append(grid, labelledControl("Seed output / saved seed", seedInput), settingsToolbar, labelledControl("Settings summary", summary), labelledControl("Settings JSON", settingsJson));
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
  for (const hidden of [widgets.stateWidget, widgets.uiStateWidget, widgets.characterWidget, widgets.promptWidget]) hideWidget(hidden);

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
  const name = nodeData?.name ?? "";
  const displayName = nodeData?.display_name ?? "";
  if (![name, displayName].some((value) => value === NODE_CLASS || value === LEGACY_NODE_CLASS)) return;
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
