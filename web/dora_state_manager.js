import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import "../../scripts/domWidget.js";

const EXT_NAME = "comfyui_dora_dynamic_lora.state_manager";
const NODE_CLASS = "State Manager";
const LEGACY_NODE_CLASS = "DoRA State Manager";
const DORA_LOADER_CLASS = "DoRA Power LoRA Loader";
const DORA_STATE_KIND = "dora_state_manager_state";
const STATE_SCHEMA_VERSION = 3;
const STATE_MANAGER_CONTROL_KIND = "state_manager_control";
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
const MIN_WIDGET_HEIGHT = 560;
const MIN_NODE_WIDTH = 820;
const MIN_NODE_HEIGHT = 720;
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
  image: ["character_image"],
  fileimagePrefix: ["fileimage_prefix"],
};
const TEXT_WIDGET_NAMES = ["text", "prompt", "positive", "negative", "string", "value", "wildcard", "wildcards", "wildcard_text", "populated_text"];
const STATE_TEXT_OUTPUT_NAMES = ["text"];
const STATE_TEXT_SLOT_WIDGET = "state_slot";
const TEXT_BOX_ROLE_CHOICES = ["positive", "negative", "generic"];
const POSITIVE_HINT_RE = /positive|pos|prompt/i;
const NEGATIVE_HINT_RE = /negative|neg/i;
const SEED_HINT_RE = /seed|noise_seed|rgthree|control_after_generate|randomize|variation|subseed/i;
const SKIP_SETTING_NODE_RE = /clip text encode|conditioning|preview|reroute/i;
const STATE_SEED_MIN = -1125899906842624;
const STATE_SEED_MAX = 1125899906842624;
const STATE_SEED_RANDOM = -1;
const STATE_SEED_INCREMENT = -2;
const STATE_SEED_DECREMENT = -3;
const STATE_SEED_SPECIALS = [STATE_SEED_RANDOM, STATE_SEED_INCREMENT, STATE_SEED_DECREMENT];
const LAST_SEED_BUTTON_LABEL = "♻️ (Use Last Queued Seed)";
const LIBRARY_API = "/dora_dynamic_lora/state-library";
const BINDING_KIND = "dora_state_manager_binding";
const BINDING_VERSION = 1;
const QUEUE_SESSION_MAX_AGE_MS = 30000;

const dsmQueueSession = {
  active: false,
  total: 1,
  nextIndex: 0,
  startedAt: 0,
  promptPools: new Map(),
};

const stateLibraryClient = {
  revision: 0,
  state: { version: STATE_SCHEMA_VERSION, characters: [] },
  canonical: "[]",
  pending: [],
  writing: false,
  blocked: false,
  lastAppliedNode: null,
  nodes: new Set(),
};


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

function makeId(_prefix) {
  if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID();
  const bytes = new Uint8Array(16);
  globalThis.crypto?.getRandomValues?.(bytes);
  if (!bytes.some(Boolean)) {
    for (let index = 0; index < bytes.length; index++) bytes[index] = Math.floor(Math.random() * 256);
  }
  bytes[6] = (bytes[6] & 0x0f) | 0x40;
  bytes[8] = (bytes[8] & 0x3f) | 0x80;
  const hex = [...bytes].map((value) => value.toString(16).padStart(2, "0")).join("");
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

function cleanId(value, fallback) {
  const text = String(value ?? "")
    .trim()
    .replace(/[^A-Za-z0-9_.:-]+/g, "_")
    .replace(/^_+|_+$/g, "");
  return text || fallback;
}

function defaultPrompt(name = "Default Character") {
  return {
    id: "default_prompt",
    name: String(name || "Default Character"),
    positive: "",
    negative: "",
    text_boxes: [
      { role: "positive", slot: "default", label: "Default positive", text: "" },
      { role: "negative", slot: "default", label: "Default negative", text: "" },
    ],
    settings: {},
    reference_image: {},
    fileimage_prefix: "",
  };
}

function normalizeTextRole(value, fallback = "generic") {
  const text = String(value ?? fallback).trim().toLowerCase();
  if (text.includes("positive") || text === "pos") return "positive";
  if (text.includes("negative") || text === "neg") return "negative";
  return TEXT_BOX_ROLE_CHOICES.includes(text) ? text : fallback;
}

function isTextRoleKey(value) {
  return TEXT_BOX_ROLE_CHOICES.includes(normalizeTextRole(value, ""));
}

function normalizeTextSlot(value, fallback = "default") {
  return cleanId(value, fallback);
}

function defaultTextBox(role = "positive", slot = "default", text = "") {
  const normalizedRole = normalizeTextRole(role, "generic");
  const normalizedSlot = normalizeTextSlot(slot, "default");
  const labelRole = normalizedRole.charAt(0).toUpperCase() + normalizedRole.slice(1);
  return {
    role: normalizedRole,
    slot: normalizedSlot,
    label: `${labelRole} ${normalizedSlot}`,
    text: String(text ?? ""),
  };
}

function textBoxKey(role, slot) {
  return `${normalizeTextRole(role, "generic")}::${normalizeTextSlot(slot, "default")}`;
}

function normalizeTextBox(raw, index = 0) {
  if (raw == null) return null;
  const src = raw && typeof raw === "object" && !Array.isArray(raw) ? raw : { text: raw };
  const role = normalizeTextRole(src.role ?? src.kind ?? src.type, "generic");
  const slot = normalizeTextSlot(src.slot ?? src.id ?? src.name, role === "generic" ? `text_${index + 1}` : "default");
  return {
    role,
    slot,
    label: String(src.label ?? src.name ?? `${role} ${slot}`).trim() || `${role} ${slot}`,
    text: String(src.text ?? src.value ?? src.prompt ?? ""),
  };
}

function rawTextBoxesFromPrompt(prompt) {
  const p = prompt && typeof prompt === "object" ? prompt : {};
  const raw = p.text_boxes ?? p.textBoxes ?? p.prompt_boxes ?? p.promptBoxes;
  const out = [];
  if (Array.isArray(raw)) {
    out.push(...raw);
  } else if (raw && typeof raw === "object") {
    for (const [key, value] of Object.entries(raw)) {
      const roleKey = isTextRoleKey(key);
      if (value && typeof value === "object" && !Array.isArray(value)) {
        const merged = { ...value };
        if (roleKey && merged.role == null) merged.role = key;
        else if (!roleKey && merged.slot == null) merged.slot = key;
        out.push(merged);
      } else {
        out.push(roleKey ? { role: key, text: value } : { slot: key, text: value });
      }
    }
  }
  return out;
}

function normalizePromptTextBoxes(prompt) {
  const p = prompt && typeof prompt === "object" ? prompt : {};
  const legacyPositive = String(p.positive ?? p.positive_prompt ?? "");
  const legacyNegative = String(p.negative ?? p.negative_prompt ?? "");
  const raw = rawTextBoxesFromPrompt(p);
  const boxes = [];
  const used = new Set();
  raw.forEach((box, index) => {
    const normalized = normalizeTextBox(box, index);
    if (!normalized) return;
    const base = normalized.slot;
    let slot = base;
    let suffix = 2;
    while (used.has(textBoxKey(normalized.role, slot))) slot = `${base}_${suffix++}`;
    normalized.slot = slot;
    used.add(textBoxKey(normalized.role, normalized.slot));
    boxes.push(normalized);
  });

  const hadRawBoxes = raw.length > 0;
  const upsertLegacy = (role, text) => {
    const key = textBoxKey(role, "default");
    const existing = boxes.find((box) => textBoxKey(box.role, box.slot) === key);
    if (existing) {
      if (text && !existing.text) existing.text = text;
      return;
    }
    if (text || !hadRawBoxes) boxes.push(defaultTextBox(role, "default", text));
  };

  upsertLegacy("positive", legacyPositive);
  upsertLegacy("negative", legacyNegative);
  return boxes;
}

function findPromptTextBox(prompt, role, slot, { allowRoleFallback = true } = {}) {
  const boxes = normalizePromptTextBoxes(prompt);
  const normalizedRole = normalizeTextRole(role, "generic");
  const normalizedSlot = normalizeTextSlot(slot, "default");
  const exact = boxes.find((box) => box.role === normalizedRole && normalizeTextSlot(box.slot, "default") === normalizedSlot);
  if (exact || !allowRoleFallback) return exact || null;
  return boxes.find((box) => box.role === normalizedRole && normalizeTextSlot(box.slot, "default") === "default") || boxes.find((box) => box.role === normalizedRole) || null;
}

function syncPromptTextMirror(prompt) {
  if (!prompt || typeof prompt !== "object") return prompt;
  const boxes = normalizePromptTextBoxes(prompt);
  const positive = boxes.find((box) => box.role === "positive" && box.slot === "default") || boxes.find((box) => box.role === "positive");
  const negative = boxes.find((box) => box.role === "negative" && box.slot === "default") || boxes.find((box) => box.role === "negative");
  prompt.text_boxes = boxes;
  prompt.positive = String(positive?.text ?? prompt.positive ?? "");
  prompt.negative = String(negative?.text ?? prompt.negative ?? "");
  return prompt;
}

function setPromptTextBox(prompt, role, slot, text, label = "") {
  const boxes = normalizePromptTextBoxes(prompt);
  const normalizedRole = normalizeTextRole(role, "generic");
  const normalizedSlot = normalizeTextSlot(slot, "default");
  let box = boxes.find((item) => item.role === normalizedRole && normalizeTextSlot(item.slot, "default") === normalizedSlot);
  if (!box) {
    box = defaultTextBox(normalizedRole, normalizedSlot, text);
    boxes.push(box);
  }
  box.text = String(text ?? "");
  if (label) box.label = String(label).trim() || box.label;
  prompt.text_boxes = boxes;
  syncPromptTextMirror(prompt);
  return box;
}

function getPromptText(prompt, role, slot = "default") {
  const box = findPromptTextBox(prompt, role, slot);
  if (box) return box.text;
  if (normalizeTextRole(role, "generic") === "positive") return String(prompt?.positive ?? "");
  if (normalizeTextRole(role, "generic") === "negative") return String(prompt?.negative ?? "");
  return "";
}

function defaultLoaderStack(slot = "default", label = "Default loader") {
  return {
    slot: normalizeLoaderSlot(slot, "default"),
    label: String(label || slot || "Default loader"),
    loras: [],
    loader_globals: {},
  };
}

function defaultCharacter() {
  const stack = defaultLoaderStack();
  return {
    id: "default_character",
    name: "Default Character",
    thumbnail: {},
    loader_stacks: [stack],
    // Legacy/default stack mirror. Kept for old workflows and old consumers.
    loras: stack.loras,
    loader_globals: stack.loader_globals,
    prompts: [defaultPrompt("Default Character")],
  };
}

function defaultState() {
  return { version: STATE_SCHEMA_VERSION, characters: [defaultCharacter()] };
}

function defaultUiState() {
  return { version: 1, panel: "prompts", status: "", queue_randomize_saved_seed: false };
}

function stripBackupRestoreStatus(uiState) {
  return normalizeUiState(uiState || defaultUiState());
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

function normalizeSeedInteger(value, fallback = 0) {
  if (typeof value === "boolean") return fallback;
  const n = Number(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.max(STATE_SEED_MIN, Math.min(STATE_SEED_MAX, Math.floor(n)));
}

function isStateSeedSpecial(value) {
  return STATE_SEED_SPECIALS.includes(normalizeSeedInteger(value, 0));
}

function generateStateManagerRuntimeSeed() {
  return Math.floor(Math.random() * STATE_SEED_MAX) + 1;
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

function normalizeLoaderSlot(value, fallback = "default") {
  const text = String(value ?? "")
    .trim()
    .replace(/[^A-Za-z0-9_.:-]+/g, "_")
    .replace(/^_+|_+$/g, "");
  return text || fallback;
}

function normalizeLoaderStack(stack, index = 0) {
  const src = stack && typeof stack === "object" ? stack : {};
  const slot = normalizeLoaderSlot(src.slot ?? src.id ?? src.name, `loader_${index + 1}`);
  const label = String(src.label ?? src.name ?? slot).trim() || slot;
  const rowsIn = Array.isArray(src.loras) ? src.loras : Array.isArray(src.rows) ? src.rows : [];
  return {
    slot,
    label,
    loras: rowsIn.map(normalizeLoraRow).filter(Boolean),
    loader_globals: normalizeLoaderGlobals(src.loader_globals ?? src.globals),
  };
}

function normalizeLoaderStacks(character) {
  const c = character && typeof character === "object" ? character : {};
  const raw = [];
  if (Array.isArray(c.loader_stacks)) {
    raw.push(...c.loader_stacks);
  } else if (c.loader_stacks && typeof c.loader_stacks === "object") {
    for (const [slot, stack] of Object.entries(c.loader_stacks)) {
      if (stack && typeof stack === "object") raw.push({ ...stack, slot: stack.slot ?? slot });
    }
  }

  const stacks = [];
  const used = new Set();
  raw.forEach((stack, index) => {
    const normalized = normalizeLoaderStack(stack, index);
    let slot = normalized.slot;
    const base = slot;
    let suffix = 2;
    while (used.has(slot)) slot = `${base}_${suffix++}`;
    used.add(slot);
    normalized.slot = slot;
    stacks.push(normalized);
  });

  const legacyGlobals = normalizeLoaderGlobals({
    ...(c.globals && typeof c.globals === "object" ? c.globals : {}),
    ...(c.loader_globals && typeof c.loader_globals === "object" ? c.loader_globals : {}),
  });
  if (!stacks.length) {
    stacks.push({
      slot: "default",
      label: "Default loader",
      loras: Array.isArray(c.loras) ? c.loras.map(normalizeLoraRow).filter(Boolean) : [],
      loader_globals: legacyGlobals,
    });
  } else if (Object.keys(legacyGlobals).length) {
    const primary = stacks.find((stack) => normalizeLoaderSlot(stack.slot, "default") === "default") || stacks[0];
    if (primary && !Object.keys(primary.loader_globals || {}).length) primary.loader_globals = legacyGlobals;
  }
  return stacks;
}

function syncLegacyLoaderMirror(character) {
  const stacks = normalizeLoaderStacks(character);
  character.loader_stacks = stacks;
  const primary = stacks.find((stack) => stack.slot === "default") || stacks[0] || defaultLoaderStack();
  character.loras = primary.loras || [];
  character.loader_globals = primary.loader_globals || {};
  return character;
}

function getCharacterLoaderStacks(character) {
  syncLegacyLoaderMirror(character);
  return character.loader_stacks;
}

function findCharacterLoaderStack(character, slot, { allowFallback = true } = {}) {
  const stacks = getCharacterLoaderStacks(character);
  const wanted = normalizeLoaderSlot(slot, "default");
  const exact = stacks.find((stack) => normalizeLoaderSlot(stack.slot, "default") === wanted);
  if (exact) return exact;
  if (!allowFallback) return null;
  return stacks.find((stack) => stack.slot === "default") || stacks[0] || null;
}

function updateCharacterLoaderGlobals(character, stackOrSlot, patch) {
  const slot = normalizeLoaderSlot(
    typeof stackOrSlot === "string" ? stackOrSlot : stackOrSlot?.slot,
    "default"
  );
  const stack = findCharacterLoaderStack(character, slot, { allowFallback: false });
  if (!stack) return {};
  const current = normalizeLoaderGlobals(stack.loader_globals || {});
  stack.loader_globals = normalizeLoaderGlobals({ ...current, ...(patch || {}) });
  syncLegacyLoaderMirror(character);
  return stack.loader_globals;
}

function setCharacterLoaderStack(character, stack) {
  const normalized = normalizeLoaderStack(stack, getCharacterLoaderStacks(character).length);
  const stacks = getCharacterLoaderStacks(character);
  const index = stacks.findIndex((existing) => normalizeLoaderSlot(existing.slot, "default") === normalized.slot);
  if (index >= 0) stacks[index] = normalized;
  else stacks.push(normalized);
  syncLegacyLoaderMirror(character);
  return normalized;
}

function loaderStackActiveRowCount(stack) {
  return (stack?.loras || []).filter((row) => row.enabled && row.name && row.name !== "None").length;
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

function normalizePrompt(prompt, index, nameOverride = "") {
  const p = prompt && typeof prompt === "object" ? prompt : {};
  const fallbackName = `Character ${index + 1}`;
  return syncPromptTextMirror({
    id: cleanId(p.id, `prompt_${index + 1}`),
    name: String(nameOverride || p.name || fallbackName).trim() || fallbackName,
    positive: String(p.positive ?? p.positive_prompt ?? ""),
    negative: String(p.negative ?? p.negative_prompt ?? ""),
    text_boxes: normalizePromptTextBoxes(p),
    settings: normalizeSettings(p.settings),
    reference_image: normalizeThumbnail(p.reference_image ?? p.referenceImage ?? p.prompt_image ?? p.image),
    fileimage_prefix: String(p.fileimage_prefix ?? p.filename_prefix ?? p.file_image_prefix ?? "").trim(),
  });
}

function normalizeCharacter(character, index, { migrateLegacyPresetNames = false } = {}) {
  const c = character && typeof character === "object" ? character : {};
  const name = String(c.name ?? `Character ${index + 1}`).trim() || `Character ${index + 1}`;
  const prompts = Array.isArray(c.prompts)
    ? c.prompts.map((prompt, promptIndex) => normalizePrompt(prompt, promptIndex, migrateLegacyPresetNames ? name : "")).filter(Boolean)
    : [];
  const normalized = {
    id: cleanId(c.id, `character_${index + 1}`),
    name,
    thumbnail: normalizeThumbnail(c.thumbnail),
    loader_stacks: normalizeLoaderStacks(c),
    prompts: prompts.length ? prompts : [defaultPrompt(name)],
  };
  if (c.__dsm_ephemeral) normalized.__dsm_ephemeral = true;
  return syncLegacyLoaderMirror(normalized);
}


function normalizeState(raw) {
  const parsed = safeJsonParse(raw, defaultState());
  const charsIn = Array.isArray(parsed.characters) ? parsed.characters : [];
  const migrateLegacyPresetNames = Number(parsed.version ?? 0) < STATE_SCHEMA_VERSION;
  const characters = charsIn
    .map((character, index) => normalizeCharacter(character, index, { migrateLegacyPresetNames }))
    .filter(Boolean);
  return { version: STATE_SCHEMA_VERSION, characters: characters.length ? characters : [defaultCharacter()] };
}

function updatePromptName(state, characterId, promptId, value) {
  const nextState = normalizeState(state);
  const character = nextState.characters.find((item) => item.id === characterId);
  const promptIndex = character?.prompts.findIndex((item) => item.id === promptId) ?? -1;
  if (!character || promptIndex < 0) return nextState;

  const prompt = { ...character.prompts[promptIndex] };
  prompt.name = String(value ?? "").trim() || prompt.name;
  character.prompts[promptIndex] = prompt;
  return nextState;
}

function normalizeIdList(value) {
  if (Array.isArray(value)) return value.map((item) => cleanId(item, "")).filter(Boolean);
  if (typeof value === "string") {
    return value.split(/[\n,]+/g).map((item) => cleanId(item, "")).filter(Boolean);
  }
  return [];
}

function normalizeUiState(raw) {
  const parsed = safeJsonParse(raw, defaultUiState());
  return {
    version: 2,
    panel: ["prompts", "character", "loras", "settings", "queue"].includes(parsed.panel) ? parsed.panel : "prompts",
    status: String(parsed.status ?? ""),
    queue_prompt_wildcard: normalizeBoolean(parsed.queue_prompt_wildcard ?? parsed.prompt_wildcard_enabled, false),
    queue_character_wildcard: normalizeBoolean(parsed.queue_character_wildcard ?? parsed.character_wildcard_enabled, false),
    queue_randomize_saved_seed: normalizeBoolean(parsed.queue_randomize_saved_seed ?? parsed.randomize_saved_seed, false),
    queue_character_ids: normalizeIdList(parsed.queue_character_ids ?? parsed.selected_character_ids),
  };
}

function serializeState(state) {
  return JSON.stringify(normalizeState(state), null, 0);
}

function serializeUiState(uiState) {
  return JSON.stringify(stripBackupRestoreStatus(uiState), null, 0);
}

function serializeWorkflowUiState(uiState) {
  const normalized = normalizeUiState(uiState);
  return JSON.stringify({
    version: 2,
    queue_prompt_wildcard: normalized.queue_prompt_wildcard,
    queue_character_wildcard: normalized.queue_character_wildcard,
    queue_randomize_saved_seed: normalized.queue_randomize_saved_seed,
    queue_character_ids: normalized.queue_character_ids,
  });
}

function canonicalJson(value) {
  const canonicalize = (item) => {
    if (Array.isArray(item)) return item.map(canonicalize);
    if (!item || typeof item !== "object") return item;
    return Object.fromEntries(
      Object.keys(item).sort().map((key) => [key, canonicalize(item[key])])
    );
  };
  return JSON.stringify(canonicalize(value));
}

function defaultBinding() {
  return { version: BINDING_VERSION, kind: BINDING_KIND };
}

function serializeBinding() {
  return JSON.stringify(defaultBinding());
}

function parseLegacyEmbeddedState(raw) {
  const parsed = safeJsonParse(raw, null);
  return parsed && Array.isArray(parsed.characters) ? parsed : null;
}

function isDefaultStateValue(value) {
  try {
    return canonicalJson(normalizeState(value)) === canonicalJson(normalizeState(defaultState()));
  } catch {
    return true;
  }
}

function persistentCharacters(state) {
  return normalizeState(state).characters.filter(
    (character) => character.id !== "default_character" && !character.__dsm_ephemeral
  );
}

function materializeEditedDefault(state, characterId, promptId) {
  const normalized = normalizeState(state);
  const defaultIndex = normalized.characters.findIndex((character) => character.id === "default_character");
  if (defaultIndex < 0) return { state: normalized, characterId, promptId };
  const character = normalized.characters[defaultIndex];
  if (canonicalJson(character) === canonicalJson(defaultCharacter())) return { state: normalized, characterId, promptId };
  const oldPromptId = String(promptId || "");
  character.id = makeId("character");
  const promptIds = new Map();
  character.prompts = character.prompts.map((prompt) => {
    const id = makeId("prompt");
    promptIds.set(String(prompt.id || ""), id);
    return { ...prompt, id };
  });
  return {
    state: normalized,
    characterId: String(characterId || "") === "default_character" ? character.id : characterId,
    promptId: promptIds.get(oldPromptId) || character.prompts[0]?.id || "",
  };
}

async function stateLibraryRequest(path = "", options = {}) {
  const response = await api.fetchApi(`${LIBRARY_API}${path}`, options);
  let payload = null;
  try {
    payload = await response.json();
  } catch {
    payload = null;
  }
  if (!response.ok) {
    const error = new Error(payload?.error || `State Manager library request failed (${response.status}).`);
    error.status = response.status;
    error.payload = payload;
    throw error;
  }
  return payload;
}

function installLibrarySnapshot(snapshot, { force = false } = {}) {
  const characters = Array.isArray(snapshot?.characters) ? snapshot.characters : [];
  const revision = Math.max(0, Number(snapshot?.revision) || 0);
  if (!force && revision < stateLibraryClient.revision) return false;
  stateLibraryClient.revision = revision;
  stateLibraryClient.state = { version: STATE_SCHEMA_VERSION, characters: structuredCloneCompat(characters) };
  stateLibraryClient.canonical = canonicalJson(characters);
  stateLibraryClient.blocked = false;
  return true;
}

function selectionIsDefault(characterId, promptId) {
  return ["", "default_character"].includes(String(characterId || ""))
    && ["", "default_prompt"].includes(String(promptId || ""));
}

function stateViewForSelection(characterId, promptId) {
  const characters = Array.isArray(stateLibraryClient.state?.characters)
    ? stateLibraryClient.state.characters.map((character, index) => normalizeCharacter(character, index)).filter(Boolean)
    : [];
  const state = { version: STATE_SCHEMA_VERSION, characters };
  if (selectionIsDefault(characterId, promptId)) {
    return normalizeState({ version: STATE_SCHEMA_VERSION, characters: [defaultCharacter(), ...state.characters] });
  }
  const character = state.characters.find((item) => item.id === characterId);
  const prompt = character?.prompts?.find((item) => item.id === promptId);
  if (character && prompt) return state;
  const missing = defaultCharacter();
  missing.__dsm_ephemeral = true;
  missing.id = String(characterId || "missing_character");
  missing.name = "Selected character preset is not available locally";
  missing.prompts[0].id = String(promptId || "missing_prompt");
  missing.prompts[0].name = "Select or create a local character";
  return normalizeState({ version: STATE_SCHEMA_VERSION, characters: [missing, ...state.characters] });
}

function refreshNodeFromLibrary(node, { status = "" } = {}) {
  if (!node?.__dsm) return;
  const widgets = getWidgets(node);
  const characterId = String(widgetValue(widgets.characterWidget, "") || "");
  const promptId = String(widgetValue(widgets.promptWidget, "") || "");
  const state = stateViewForSelection(characterId, promptId);
  const uiState = normalizeUiState(
    node.__dsm?.uiState || widgetValue(widgets.uiStateWidget, serializeWorkflowUiState(defaultUiState()))
  );
  cacheRenderableState(node, state, status ? { ...uiState, status } : uiState);
  scheduleRender(node);
}

function refreshAllNodesFromLibrary() {
  for (const node of stateLibraryClient.nodes) refreshNodeFromLibrary(node);
}

function mergeScheduledLibraryUpdate(baseCharacters, desiredCharacters, currentCharacters, { allowOverwrite = false } = {}) {
  const base = new Map((baseCharacters || []).map((character) => [character.id, character]));
  const desired = new Map((desiredCharacters || []).map((character) => [character.id, character]));
  const current = new Map((currentCharacters || []).map((character) => [character.id, character]));
  const changedIds = new Set();
  for (const id of new Set([...base.keys(), ...desired.keys()])) {
    if (canonicalJson(base.get(id)) !== canonicalJson(desired.get(id))) changedIds.add(id);
  }
  for (const id of changedIds) {
    const baseValue = base.get(id);
    const desiredValue = desired.get(id);
    const currentValue = current.get(id);
    const currentMatchesBase = canonicalJson(currentValue) === canonicalJson(baseValue);
    const currentMatchesDesired = canonicalJson(currentValue) === canonicalJson(desiredValue);
    if (!currentMatchesBase && !currentMatchesDesired && !allowOverwrite) {
      return { conflict: id, characters: currentCharacters };
    }
    if (desiredValue === undefined) current.delete(id);
    else current.set(id, structuredCloneCompat(desiredValue));
  }
  const order = [];
  for (const character of currentCharacters || []) {
    if (current.has(character.id)) order.push(current.get(character.id));
  }
  for (const character of desiredCharacters || []) {
    if (!order.some((entry) => entry.id === character.id) && current.has(character.id)) {
      order.push(current.get(character.id));
    }
  }
  return { conflict: null, characters: order };
}

async function writePendingLibrary() {
  if (stateLibraryClient.writing || stateLibraryClient.blocked) return;
  stateLibraryClient.writing = true;
  try {
    while (stateLibraryClient.pending.length && !stateLibraryClient.blocked) {
      const pending = stateLibraryClient.pending.shift();
      const merged = mergeScheduledLibraryUpdate(
        pending.baseCharacters,
        pending.desiredCharacters,
        stateLibraryClient.state.characters,
        { allowOverwrite: pending.node === stateLibraryClient.lastAppliedNode },
      );
      if (merged.conflict) {
        stateLibraryClient.blocked = true;
        for (const node of stateLibraryClient.nodes) {
          const current = getRenderableState(node);
          cacheRenderableState(node, current.state, {
            ...current.uiState,
            status: "Two local State Manager instances edited the same character concurrently. Reload the library; no conflicting write was sent.",
          });
          scheduleRender(node);
        }
        break;
      }
      if (canonicalJson(merged.characters) === stateLibraryClient.canonical) {
        stateLibraryClient.lastAppliedNode = pending.node;
        continue;
      }
      try {
        const snapshot = await stateLibraryRequest("", {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            expected_revision: stateLibraryClient.revision,
            characters: merged.characters,
          }),
        });
        installLibrarySnapshot(snapshot);
        stateLibraryClient.lastAppliedNode = pending.node;
        refreshAllNodesFromLibrary();
      } catch (error) {
        if (error.status === 409) {
          stateLibraryClient.blocked = true;
          for (const node of stateLibraryClient.nodes) {
            const current = getRenderableState(node);
            cacheRenderableState(node, current.state, {
              ...current.uiState,
              status: "Library changed in another tab or State Manager. Reload the library before editing again; the stale write was rejected.",
            });
            scheduleRender(node);
          }
          break;
        }
        stateLibraryClient.blocked = true;
        for (const node of stateLibraryClient.nodes) {
          const current = getRenderableState(node);
          cacheRenderableState(node, current.state, {
            ...current.uiState,
            status: `Library save failed: ${error?.message || error}`,
          });
          scheduleRender(node);
        }
        break;
      }
    }
  } finally {
    stateLibraryClient.writing = false;
  }
}

function scheduleLibraryPersist(node, state) {
  const desiredCharacters = persistentCharacters(state);
  const canonical = canonicalJson(desiredCharacters);
  if (canonical === stateLibraryClient.canonical || stateLibraryClient.blocked) return;
  const last = stateLibraryClient.pending[stateLibraryClient.pending.length - 1];
  const scheduled = {
    node,
    baseCharacters: structuredCloneCompat(last?.node === node ? last.baseCharacters : stateLibraryClient.state.characters),
    desiredCharacters: structuredCloneCompat(desiredCharacters),
  };
  if (last?.node === node) stateLibraryClient.pending[stateLibraryClient.pending.length - 1] = scheduled;
  else stateLibraryClient.pending.push(scheduled);
  void writePendingLibrary();
}

async function reloadStateLibrary(node, { status = "Reloaded State Manager library." } = {}) {
  stateLibraryClient.blocked = false;
  stateLibraryClient.pending = [];
  stateLibraryClient.lastAppliedNode = null;
  const snapshot = await stateLibraryRequest();
  installLibrarySnapshot(snapshot, { force: true });
  refreshAllNodesFromLibrary();
  refreshNodeFromLibrary(node, { status });
}

async function initializeStateLibrary(node, legacyState, selectedCharacterId, selectedPromptId, token = null) {
  let snapshot;
  let characterId = String(selectedCharacterId || "");
  let promptId = String(selectedPromptId || "");
  let status = "";
  if (legacyState && !isDefaultStateValue(legacyState)) {
    const migration = await stateLibraryRequest("/migrate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        state: legacyState,
        selected_character_id: characterId,
        selected_prompt_id: promptId,
      }),
    });
    snapshot = migration.snapshot;
    characterId = migration.selected_character_id || "";
    promptId = migration.selected_prompt_id || "";
    status = migration.already_migrated
      ? "Legacy embedded library was already migrated; restored its local UUID binding."
      : "Migrated the legacy embedded State Manager library into persistent user storage.";
  } else {
    snapshot = await stateLibraryRequest();
  }
  if (token != null && node.__dsmLibraryLoadToken !== token) return;
  installLibrarySnapshot(snapshot);
  const widgets = getWidgets(node);
  setWidgetValue(widgets.stateWidget, serializeBinding());
  setWidgetValue(widgets.characterWidget, characterId || "default_character");
  setWidgetValue(widgets.promptWidget, promptId || "default_prompt");
  const state = stateViewForSelection(
    widgetValue(widgets.characterWidget, ""),
    widgetValue(widgets.promptWidget, ""),
  );
  const uiState = normalizeUiState(widgetValue(widgets.uiStateWidget, serializeUiState(defaultUiState())));
  setWidgetValue(widgets.uiStateWidget, serializeWorkflowUiState(uiState));
  refreshAllNodesFromLibrary();
  cacheRenderableState(node, state, status ? { ...uiState, status } : uiState);
  scheduleRender(node);
}

function selectionIdsForState(state, characterId = "", promptId = "") {
  const normalizedState = normalizeState(state);
  const character = normalizedState.characters.find((item) => item.id === characterId) || normalizedState.characters[0];
  const prompt = character?.prompts?.find((item) => item.id === promptId) || character?.prompts?.[0];
  return { characterId: character?.id || "", promptId: prompt?.id || "" };
}

function downloadJsonFile(filename, payload) {
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  try {
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    anchor.rel = "noopener";
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
  } finally {
    setTimeout(() => URL.revokeObjectURL(url), 0);
  }
}

async function exportCharacterJson(characterId) {
  const payload = await stateLibraryRequest(`/characters/${encodeURIComponent(characterId)}/export`);
  downloadJsonFile(`dora-state-manager-character-${characterId}.json`, payload);
}

async function exportLibraryJson() {
  const payload = await stateLibraryRequest("/export");
  downloadJsonFile("dora-state-manager-library.json", payload);
}

async function importStateJsonFile(node, file) {
  if (!file) return;
  const text = await file.text();
  const parsed = JSON.parse(String(text || ""));
  const result = await stateLibraryRequest("/import", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(parsed),
  });
  installLibrarySnapshot(result.snapshot);
  refreshAllNodesFromLibrary();
  const characterId = result.character?.id || result.character_ids?.[0] || "";
  if (characterId) {
    const character = stateLibraryClient.state.characters.find((item) => item.id === characterId);
    const widgets = getWidgets(node);
    setWidgetValue(widgets.characterWidget, characterId);
    setWidgetValue(widgets.promptWidget, character?.prompts?.[0]?.id || "");
  }
  refreshNodeFromLibrary(node, { status: `Imported ${file.name || "State Manager JSON"}.` });
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
  if (node?.__dsm?.state && node?.__dsm?.uiState) {
    return { state: node.__dsm.state, uiState: node.__dsm.uiState };
  }
  const widgets = getWidgets(node);
  const state = stateViewForSelection(
    String(widgetValue(widgets.characterWidget, "") || ""),
    String(widgetValue(widgets.promptWidget, "") || ""),
  );
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

function markDownstreamDirty(node) {
  const graph = getGraph(node);
  if (!graph || !node) return 0;
  const queue = [node];
  const seen = new Set([node.id]);
  let changed = 0;
  while (queue.length) {
    const current = queue.shift();
    for (const output of current?.outputs || []) {
      for (const linkId of output?.links || []) {
        const link = graph.links?.[linkId];
        if (!link) continue;
        const target = graph.getNodeById?.(link.target_id);
        if (!target || seen.has(target.id)) continue;
        seen.add(target.id);
        markNodeDirty(target);
        changed += 1;
        queue.push(target);
      }
    }
  }
  return changed;
}

function updateState(node, state, uiState, opts = {}) {
  const widgets = getWidgets(node);
  const currentCharacterId = opts.characterId ?? String(widgetValue(widgets.characterWidget, "") || "");
  const currentPromptId = opts.promptId ?? String(widgetValue(widgets.promptWidget, "") || "");
  const materialized = materializeEditedDefault(state, currentCharacterId, currentPromptId);
  const normalizedState = materialized.state;
  const normalizedUiState = stripBackupRestoreStatus(uiState || defaultUiState());
  const nextUiState = normalizeUiState({
    ...normalizedUiState,
    status: opts.status ?? normalizedUiState.status,
  });
  setWidgetValue(widgets.characterWidget, materialized.characterId || "default_character");
  setWidgetValue(widgets.promptWidget, materialized.promptId || "default_prompt");
  const { character, prompt } = ensureSelection(node, normalizedState);
  setWidgetValue(widgets.stateWidget, serializeBinding());
  setWidgetValue(widgets.uiStateWidget, serializeWorkflowUiState(nextUiState));
  setWidgetValue(widgets.characterWidget, materialized.characterId || character.id);
  setWidgetValue(widgets.promptWidget, materialized.promptId || prompt.id);
  node.properties = node.properties || {};
  delete node.properties.dora_state_manager;
  delete node.properties.dora_state_manager_backup_node_uid;
  cacheRenderableState(node, normalizedState, nextUiState);
  if (opts.persist !== false) scheduleLibraryPersist(node, normalizedState);
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

function isNodeDefForClass(nodeData, nodeType, classNames) {
  const values = [nodeData?.name, nodeData?.display_name, nodeType?.comfyClass, nodeType?.title]
    .map((value) => String(value ?? ""));
  return classNames.some((className) => values.includes(className));
}

function isTargetNode(nodeData, nodeType) {
  return isNodeDefForClass(nodeData, nodeType, [NODE_CLASS, LEGACY_NODE_CLASS]);
}

function isStateTextNodeDef(nodeData, nodeType) {
  return isNodeDefForClass(nodeData, nodeType, [STATE_TEXT_CLASS, STATE_TEXT_DISPLAY_CLASS]);
}

function isStateSeedNodeDef(nodeData, nodeType) {
  return isNodeDefForClass(nodeData, nodeType, [STATE_SEED_CLASS, STATE_SEED_DISPLAY_CLASS]);
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

function getStateTextSlotWidget(node) {
  const map = getWidgetMap(node);
  return map.get(STATE_TEXT_SLOT_WIDGET) || map.get("slot") || null;
}

function getStateTextRole(node, fallback = "generic") {
  return normalizeTextRole(getRoleWidget(node)?.value, fallback);
}

function getStateTextSlot(node, role = "generic", fallback = "default") {
  const widgetValue = getStateTextSlotWidget(node)?.value;
  const propValue = node?.properties?.dora_state_text_slot;
  const normalizedRole = normalizeTextRole(role, "generic");
  return normalizeTextSlot(widgetValue ?? propValue, normalizedRole === "generic" ? fallback : "default");
}

function setStateTextSlot(node, slot) {
  const normalized = normalizeTextSlot(slot, "default");
  const widget = getStateTextSlotWidget(node);
  if (widget) setNodeWidget(node, widget, normalized);
  node.properties = node.properties || {};
  node.properties.dora_state_text_slot = normalized;
  return normalized;
}

function stateTextLabel(node, role, slot) {
  const title = String(node?.title || "").trim();
  if (title && title !== STATE_TEXT_CLASS && title !== STATE_TEXT_DISPLAY_CLASS) return title;
  return `${normalizeTextRole(role, "generic")} ${normalizeTextSlot(slot, "default")}`;
}

function ensureUniqueStateTextSlot(node, role, usedKeys, fallbackIndex = 0) {
  let slot = getStateTextSlot(node, role, `${normalizeTextRole(role, "generic")}_${node?.id ?? fallbackIndex + 1}`);
  const baseFallback = `${normalizeTextRole(role, "generic")}_${node?.id ?? fallbackIndex + 1}`;
  const original = slot;
  let key = textBoxKey(role, slot);
  if (!slot || usedKeys.has(key)) {
    slot = normalizeTextSlot(baseFallback, `text_${fallbackIndex + 1}`);
    key = textBoxKey(role, slot);
  }
  let suffix = 2;
  const base = slot;
  while (usedKeys.has(key)) {
    slot = `${base}_${suffix++}`;
    key = textBoxKey(role, slot);
  }
  usedKeys.add(key);
  if (slot !== original) setStateTextSlot(node, slot);
  return slot;
}

function getControlledTargets(node) {
  return getOutputTargets(node, OUTPUT_NAMES.control);
}

function getControlledNodes(node) {
  return uniqueNodes(getControlledTargets(node));
}

function getDoraLoaderSlot(sourceNode, fallbackIndex = 0) {
  const loaderApi = globalThis.__doraPowerLoraLoaderApi;
  if (loaderApi?.getSlot && isDoraLoaderNode(sourceNode)) {
    return normalizeLoaderSlot(loaderApi.getSlot(sourceNode), `loader_${sourceNode?.id ?? fallbackIndex + 1}`);
  }
  const widgetMap = getWidgetMap(sourceNode);
  const widgetValue = widgetMap.get("state_slot")?.value;
  const propValue = sourceNode?.properties?.dora_state_slot;
  return normalizeLoaderSlot(widgetValue ?? propValue, `loader_${sourceNode?.id ?? fallbackIndex + 1}`);
}

function getDoraLoaderLabel(sourceNode, slot) {
  const title = String(sourceNode?.title || "").trim();
  if (title && title !== DORA_LOADER_CLASS) return title;
  return slot || "loader";
}

function normalizeDoraLoaderState(state, sourceNode = null, fallbackIndex = 0) {
  const st = state && typeof state === "object" ? state : {};
  const rowsIn = Array.isArray(st.rows) ? st.rows : Array.isArray(st.loras) ? st.loras : [];
  const rows = rowsIn.map(normalizeLoraRow).filter((row) => row.name && row.name !== "None");
  const slot = normalizeLoaderSlot(st.slot ?? (sourceNode ? getDoraLoaderSlot(sourceNode, fallbackIndex) : "default"), `loader_${sourceNode?.id ?? fallbackIndex + 1}`);
  return {
    slot,
    label: String(st.label ?? getDoraLoaderLabel(sourceNode, slot)).trim() || slot,
    loras: rows,
    loader_globals: normalizeLoaderGlobals(st.globals ?? st.loader_globals),
  };
}

function extractLoraStackFromNode(sourceNode, fallbackIndex = 0) {
  if (!sourceNode) return null;
  const loaderApi = globalThis.__doraPowerLoraLoaderApi;
  if (loaderApi?.getState && isDoraLoaderNode(sourceNode)) return normalizeDoraLoaderState(loaderApi.getState(sourceNode), sourceNode, fallbackIndex);
  if (sourceNode.properties?.dora_power_lora) return normalizeDoraLoaderState(sourceNode.properties.dora_power_lora, sourceNode, fallbackIndex);
  if (sourceNode._doraRows || sourceNode._doraGlobals) {
    return normalizeDoraLoaderState({ rows: sourceNode._doraRows || [], globals: sourceNode._doraGlobals || {} }, sourceNode, fallbackIndex);
  }
  return null;
}

function applyLoraStackToNode(targetNode, character) {
  if (!targetNode || !isDoraLoaderNode(targetNode)) return false;
  const slot = getDoraLoaderSlot(targetNode);
  let stack = findCharacterLoaderStack(character, slot, { allowFallback: false });
  if (!stack) {
    const stacks = getCharacterLoaderStacks(character);
    const legacyDefault = stacks.length === 1 && normalizeLoaderSlot(stacks[0]?.slot, "default") === "default";
    if (!legacyDefault) return false;
    stack = stacks[0];
  }
  if (!stack) return false;
  const payload = {
    slot,
    label: stack.label,
    loras: stack.loras || [],
    loader_globals: stack.loader_globals || {},
  };
  const loaderApi = globalThis.__doraPowerLoraLoaderApi;
  if (loaderApi?.setState) {
    loaderApi.setSlot?.(targetNode, slot);
    return !!loaderApi.setState(targetNode, payload);
  }

  const rows = (payload.loras || []).map((row) => ({
    enabled: !!row.enabled,
    name: row.name || "None",
    strengthModel: normalizeNumber(row.strength_model, 1.0),
    strengthClip: normalizeNumber(row.strength_clip, normalizeNumber(row.strength_model, 1.0)),
  }));
  targetNode.properties = targetNode.properties || {};
  targetNode.properties.dora_state_slot = slot;
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

function syncNodeWidgetsValueCache(widgetNode, widget) {
  const widgets = widgetNode?.widgets || [];
  const index = widgets.indexOf(widget);
  if (index < 0) return;
  if (!Array.isArray(widgetNode.widgets_values)) widgetNode.widgets_values = [];
  widgetNode.widgets_values[index] = widget.value;
}

function setNodeWidget(widgetNode, widget, value) {
  if (!widget) return false;
  widget.value = value;
  widget.callback?.(value);
  widgetNode?.onWidgetChanged?.(widget.name, value, widget.value, widget);
  markNodeDirty(widgetNode);
  return true;
}

function setMirroredNodeWidget(widgetNode, widget, value) {
  if (!widget) return false;
  const oldValue = widget.value;
  widget.value = value;
  syncNodeWidgetsValueCache(widgetNode, widget);
  if (Object.is(oldValue, value)) return false;
  widget.callback?.call?.(widget, value, app?.canvas, widgetNode, null, null);
  widgetNode?.onWidgetChanged?.(widget.name, value, oldValue, widget);
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


function textInputLooksMirrorable(inputName) {
  return /text|prompt|wildcard|positive|negative|populated|string/i.test(String(inputName ?? ""));
}

function widgetByExactName(node, name) {
  const wanted = String(name ?? "").toLowerCase();
  return (node?.widgets || []).find((widget) => String(widget?.name ?? "").toLowerCase() === wanted) || null;
}

function isImpactWildcardNode(node) {
  const text = nodeNameText(node);
  if (/ImpactWildcard(?:Processor|Encode)?/i.test(text)) return true;
  return !!(widgetByExactName(node, "wildcard_text") && widgetByExactName(node, "populated_text"));
}

function mirrorTextToImpactWildcardNode(targetNode, text, inputName = "") {
  if (!isImpactWildcardNode(targetNode)) return false;
  const input = String(inputName ?? "").toLowerCase();
  const next = String(text ?? "");
  let changed = 0;

  // ImpactWildcardProcessor and ImpactWildcardEncode execute from populated_text.
  // wildcard_text is mostly a UI/template field; if only that widget changes,
  // the backend can keep processing the previous populated_text value.
  const populated = widgetByExactName(targetNode, "populated_text");
  if (populated && setMirroredNodeWidget(targetNode, populated, next)) changed += 1;

  // Keep wildcard_text visually synchronized when the State Manager is connected
  // to that input, or when no specific input name is available.
  if (!input || input === "wildcard_text") {
    const wildcard = widgetByExactName(targetNode, "wildcard_text");
    if (wildcard && wildcard !== populated && setMirroredNodeWidget(targetNode, wildcard, next)) changed += 1;
  }

  return changed > 0;
}

function mirrorTextToLinkedWidgetTarget(targetNode, text, role = "", inputName = "") {
  if (!targetNode || isStateManagerNode(targetNode)) return false;
  if (isImpactWildcardNode(targetNode) && mirrorTextToImpactWildcardNode(targetNode, text, inputName)) return true;
  if (inputName && !textInputLooksMirrorable(inputName) && !isPromptLikeNode(targetNode)) return false;
  const widget = findTextWidget(targetNode, role, inputName);
  if (!widget) return false;
  return setMirroredNodeWidget(targetNode, widget, String(text ?? ""));
}

function mirrorStateTextToDownstreamWidgets(sourceNode, text, role = "") {
  let changed = 0;
  const changedTargets = [];
  for (const target of getOutputTargets(sourceNode, STATE_TEXT_OUTPUT_NAMES)) {
    if (target.node === sourceNode) continue;
    if (mirrorTextToLinkedWidgetTarget(target.node, text, role, target.inputName)) {
      changed += 1;
      changedTargets.push(target.node);
    }
  }
  for (const targetNode of changedTargets) markDownstreamDirty(targetNode);
  return changed;
}

function syncStateTextNodeDownstream(sourceNode) {
  if (!sourceNode || !isStateTextNode(sourceNode)) return 0;
  const role = getStateTextRole(sourceNode);
  const text = extractTextFromNode(sourceNode, role);
  return mirrorStateTextToDownstreamWidgets(sourceNode, text, role);
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
    if (Object.prototype.hasOwnProperty.call(widgets, key)) return normalizeSeedInteger(widgets[key], fallback ?? 0);
  }
  for (const [key, value] of Object.entries(widgets)) {
    if (/seed/i.test(key)) return normalizeSeedInteger(value, fallback ?? 0);
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
    const seed = normalizeSeedInteger(seedSnapshot.seed ?? normalizeSeedFromWidgets(seedSnapshot.widgets, 0), 0);
    settings.seed = seed;
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
  if (normalized.seed != null) {
    return normalizeSeedInteger(normalized.seed, 0);
  }
  if (normalized.rgthree_seed?.seed != null) {
    return normalizeSeedInteger(normalized.rgthree_seed.seed, 0);
  }
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
  let savedLoaders = 0;
  loaderTargets.forEach((loader, index) => {
    const stack = extractLoraStackFromNode(loader, index);
    if (!stack) return;
    setCharacterLoaderStack(character, stack);
    savedLoaders += 1;
  });
  if (savedLoaders) changes.push(`${savedLoaders} LoRA loader${savedLoaders === 1 ? "" : "s"}`);

  const controlledTextNodes = controlledNodes.filter(isStateTextNode);
  const usedTextKeys = new Set();
  const savedTextCounts = { positive: 0, negative: 0, generic: 0 };
  for (const [index, textNode] of controlledTextNodes.entries()) {
    const role = getStateTextRole(textNode);
    const slot = ensureUniqueStateTextSlot(textNode, role, usedTextKeys, index);
    const value = extractTextFromNode(textNode, role);
    setPromptTextBox(prompt, role, slot, value, stateTextLabel(textNode, role, slot));
    savedTextCounts[role] = (savedTextCounts[role] || 0) + 1;
  }
  if (savedTextCounts.positive) changes.push(`${savedTextCounts.positive} positive text box${savedTextCounts.positive === 1 ? "" : "es"}`);
  if (savedTextCounts.negative) changes.push(`${savedTextCounts.negative} negative text box${savedTextCounts.negative === 1 ? "" : "es"}`);
  if (savedTextCounts.generic) changes.push(`${savedTextCounts.generic} generic text box${savedTextCounts.generic === 1 ? "" : "es"}`);

  // Compatibility fallback for old graphs. Avoid depending on this for normal use;
  // it can make editable widgets link-controlled if users connect STRING outputs.
  if (!savedTextCounts.positive) {
    for (const target of getOutputTargets(targetNode, OUTPUT_NAMES.positive)) {
      const widget = findTextWidget(target.node, "positive", target.inputName);
      if (widget) {
        setPromptTextBox(prompt, "positive", "default", typeof widget.value === "string" ? widget.value : "", "Default positive");
        changes.push("positive prompt template");
        break;
      }
    }
  }
  if (!savedTextCounts.negative) {
    for (const target of getOutputTargets(targetNode, OUTPUT_NAMES.negative)) {
      const widget = findTextWidget(target.node, "negative", target.inputName);
      if (widget) {
        setPromptTextBox(prompt, "negative", "default", typeof widget.value === "string" ? widget.value : "", "Default negative");
        changes.push("negative prompt template");
        break;
      }
    }
  }
  syncPromptTextMirror(prompt);

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
  if (loaderChanged) changes.push(`LoRA stacks to ${loaderChanged} loader${loaderChanged === 1 ? "" : "s"}`);

  const controlledTextNodes = controlledNodes.filter(isStateTextNode);
  let posChanged = 0;
  let negChanged = 0;
  let genericChanged = 0;
  const usedTextKeys = new Set();
  for (const [index, textNode] of controlledTextNodes.entries()) {
    const role = getStateTextRole(textNode);
    const slot = ensureUniqueStateTextSlot(textNode, role, usedTextKeys, index);
    const saved = findPromptTextBox(prompt, role, slot, { allowRoleFallback: false });
    if (!saved) continue;
    if (applyTextToNode(textNode, saved.text, role)) {
      mirrorStateTextToDownstreamWidgets(textNode, saved.text, role);
      if (role === "positive") posChanged += 1;
      else if (role === "negative") negChanged += 1;
      else genericChanged += 1;
    }
  }

  if (!controlledTextNodes.length) {
    for (const target of getOutputTargets(targetNode, OUTPUT_NAMES.positive)) {
      if (applyTextToNode(target.node, getPromptText(prompt, "positive", "default"), "positive", target.inputName)) {
        posChanged += 1;
        break;
      }
    }
    for (const target of getOutputTargets(targetNode, OUTPUT_NAMES.negative)) {
      if (applyTextToNode(target.node, getPromptText(prompt, "negative", "default"), "negative", target.inputName)) {
        negChanged += 1;
        break;
      }
    }
  }

  if (posChanged) changes.push(`positive template to ${posChanged} node${posChanged === 1 ? "" : "s"}`);
  if (negChanged) changes.push(`negative template to ${negChanged} node${negChanged === 1 ? "" : "s"}`);
  if (genericChanged) changes.push(`generic text to ${genericChanged} node${genericChanged === 1 ? "" : "s"}`);

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
  const loaders = selected.filter(isDoraLoaderNode);
  let savedLoaders = 0;
  loaders.forEach((loader, index) => {
    const stack = extractLoraStackFromNode(loader, index);
    if (!stack) return;
    setCharacterLoaderStack(character, stack);
    savedLoaders += 1;
  });
  if (savedLoaders) changes.push(`${savedLoaders} LoRA loader${savedLoaders === 1 ? "" : "s"}`);

  const selectedStateTextNodes = selected.filter(isStateTextNode);
  const usedTextKeys = new Set();
  const savedTextCounts = { positive: 0, negative: 0, generic: 0 };
  for (const [index, textNode] of selectedStateTextNodes.entries()) {
    const role = getStateTextRole(textNode);
    const slot = ensureUniqueStateTextSlot(textNode, role, usedTextKeys, index);
    setPromptTextBox(prompt, role, slot, extractTextFromNode(textNode, role), stateTextLabel(textNode, role, slot));
    savedTextCounts[role] = (savedTextCounts[role] || 0) + 1;
  }
  if (savedTextCounts.positive) changes.push(`${savedTextCounts.positive} positive text box${savedTextCounts.positive === 1 ? "" : "es"}`);
  if (savedTextCounts.negative) changes.push(`${savedTextCounts.negative} negative text box${savedTextCounts.negative === 1 ? "" : "es"}`);
  if (savedTextCounts.generic) changes.push(`${savedTextCounts.generic} generic text box${savedTextCounts.generic === 1 ? "" : "es"}`);

  const classified = selectedStateTextNodes.length ? { used: [] } : classifySelectedTextNodes(selected.filter((node) => !isDoraLoaderNode(node) && !isSeedNode(node)));
  if (classified.positive) {
    setPromptTextBox(prompt, "positive", "default", classified.positive.text, "Default positive");
    changes.push("positive prompt template");
  }
  if (classified.negative) {
    setPromptTextBox(prompt, "negative", "default", classified.negative.text, "Default negative");
    changes.push("negative prompt template");
  }
  syncPromptTextMirror(prompt);

  const used = new Set([...loaders, ...selectedStateTextNodes, ...(classified.used || [])].filter(Boolean));
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
  if (loaderChanged) changes.push(`LoRA stacks to ${loaderChanged} loader${loaderChanged === 1 ? "" : "s"}`);

  const selectedStateTextNodes = selected.filter(isStateTextNode);
  const usedTextKeys = new Set();
  const changedTextCounts = { positive: 0, negative: 0, generic: 0 };
  for (const [index, textNode] of selectedStateTextNodes.entries()) {
    const role = getStateTextRole(textNode);
    const slot = ensureUniqueStateTextSlot(textNode, role, usedTextKeys, index);
    const saved = findPromptTextBox(prompt, role, slot, { allowRoleFallback: false });
    if (saved && applyTextToNode(textNode, saved.text, role)) {
      mirrorStateTextToDownstreamWidgets(textNode, saved.text, role);
      changedTextCounts[role] = (changedTextCounts[role] || 0) + 1;
    }
  }
  if (changedTextCounts.positive) changes.push(`positive template to ${changedTextCounts.positive} node${changedTextCounts.positive === 1 ? "" : "s"}`);
  if (changedTextCounts.negative) changes.push(`negative template to ${changedTextCounts.negative} node${changedTextCounts.negative === 1 ? "" : "s"}`);
  if (changedTextCounts.generic) changes.push(`generic text to ${changedTextCounts.generic} node${changedTextCounts.generic === 1 ? "" : "s"}`);

  const classified = selectedStateTextNodes.length ? { used: [] } : classifySelectedTextNodes(selected.filter((node) => !isDoraLoaderNode(node) && !isSeedNode(node)));
  if (classified.positive && applyTextToNode(classified.positive.node, getPromptText(prompt, "positive", "default"), "positive")) changes.push("positive template");
  if (classified.negative && applyTextToNode(classified.negative.node, getPromptText(prompt, "negative", "default"), "negative")) changes.push("negative template");

  const used = new Set([...loaderTargets, ...selectedStateTextNodes, ...(classified.used || [])].filter(Boolean));
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

function stopComfyFileDropEvent(event) {
  event.preventDefault();
  event.stopPropagation();
  event.stopImmediatePropagation?.();
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

function makeInlineCheckbox(labelText, value, onChange, description = "") {
  const label = document.createElement("label");
  label.className = "dsm-checkline";
  const input = document.createElement("input");
  input.type = "checkbox";
  input.checked = Boolean(value);
  const text = document.createElement("span");
  text.className = "dsm-checkline-text";
  const main = document.createElement("strong");
  main.textContent = labelText;
  text.appendChild(main);
  if (description) {
    const detail = document.createElement("small");
    detail.textContent = description;
    text.appendChild(detail);
  }
  input.addEventListener("change", () => onChange(input.checked));
  label.append(input, text);
  return label;
}

function setPanel(node, panel) {
  const { state, uiState } = getCurrentState(node);
  updateState(node, state, { ...uiState, panel });
}

function queueCharacterIdsExplicit(state, uiState) {
  const available = new Set((state?.characters || []).map((character) => character.id));
  return normalizeIdList(uiState?.queue_character_ids).filter((id) => available.has(id));
}

function validQueueCharacterIds(state, uiState, fallbackId = "") {
  const available = new Set((state?.characters || []).map((character) => character.id));
  const ids = queueCharacterIdsExplicit(state, uiState);
  if (ids.length) return ids;
  return fallbackId && available.has(fallbackId) ? [fallbackId] : [];
}

function toggleQueueCharacterId(state, uiState, characterId, enabled) {
  const ids = queueCharacterIdsExplicit(state, uiState);
  const next = ids.filter((id) => id !== characterId);
  if (enabled) next.push(characterId);
  return [...new Set(next)];
}

function characterNamesForIds(state, ids) {
  const byId = new Map((state?.characters || []).map((character) => [character.id, character.name]));
  return ids.map((id) => byId.get(id) || id);
}

function queueSettingsSummary(state, uiState, currentCharacterId) {
  const explicitIds = queueCharacterIdsExplicit(state, uiState);
  const runtimeIds = validQueueCharacterIds(state, uiState, currentCharacterId);
  if (!uiState.queue_prompt_wildcard && !uiState.queue_character_wildcard && !uiState.queue_randomize_saved_seed) {
    return "Queue wildcarding is off. Queued jobs use the selected character and selected prompt preset.";
  }

  const lines = [];
  if (uiState.queue_character_wildcard) {
    if (explicitIds.length) {
      lines.push(`Character chunks: ${characterNamesForIds(state, runtimeIds).join(" -> ")}. The ComfyUI queue uses up to ${runtimeIds.length} contiguous block${runtimeIds.length === 1 ? "" : "s"} depending on queued job count.`);
    } else {
      lines.push("Character chunks are on, but no chunk characters are checked. Runtime falls back to the currently selected character.");
    }
  } else {
    lines.push("Characters: selected character only.");
  }

  lines.push(uiState.queue_prompt_wildcard
    ? "Prompts: each queued job randomly selects one saved prompt preset from the active character."
    : "Prompts: selected prompt preset only.");
  if (uiState.queue_randomize_saved_seed) {
    lines.push("Saved prompt preset seeds randomize only when no connected State Seed provides a fixed live seed.");
  }
  return lines.join(" ");
}

function renderCharacterTile(node, state, uiState, character, selectedId, selectedPromptId = "") {
  const queueIds = queueCharacterIdsExplicit(state, uiState);
  const queueIndex = queueIds.indexOf(character.id);
  const isSelected = character.id === selectedId;
  const isQueued = queueIndex >= 0;
  const isActiveQueueChunk = Boolean(uiState.queue_character_wildcard && isQueued);

  const tile = document.createElement("div");
  tile.tabIndex = 0;
  tile.role = "button";
  tile.dataset.libraryItem = "character";
  tile.dataset.characterId = character.id;
  tile.dataset.search = librarySearchText(character);
  tile.setAttribute("aria-label", character.name);
  if (isSelected) tile.setAttribute("aria-current", "true");
  tile.className = `dsm-character-tile${isSelected ? " selected" : ""}${isActiveQueueChunk ? " queued" : ""}${isQueued && !isActiveQueueChunk ? " prepared" : ""}`;

  const thumb = document.createElement("div");
  thumb.className = "dsm-thumb";
  const url = thumbnailUrl(character.thumbnail);
  if (url) {
    const img = document.createElement("img");
    img.src = url;
    img.alt = character.name;
    img.loading = "lazy";
    img.decoding = "async";
    thumb.appendChild(img);
  } else {
    thumb.textContent = character.name.trim().slice(0, 2).toUpperCase() || "?";
  }

  const name = document.createElement("div");
  name.className = "dsm-character-name";
  name.textContent = character.name;
  name.title = character.name;

  const meta = document.createElement("div");
  meta.className = "dsm-muted";
  const stacks = getCharacterLoaderStacks(character);
  const rowCount = stacks.reduce((sum, stack) => sum + loaderStackActiveRowCount(stack), 0);
  meta.textContent = `${stacks.length} loader${stacks.length === 1 ? "" : "s"} · ${rowCount} LoRA · ${character.prompts.length} preset${character.prompts.length === 1 ? "" : "s"}`;

  const queueRow = document.createElement("label");
  queueRow.className = "dsm-character-queue";
  const queueCheckbox = makeCheckbox(isQueued, (checked) => {
    const current = getRenderableState(node);
    const ids = toggleQueueCharacterId(current.state, current.uiState, character.id, checked);
    const nextUiState = {
      ...current.uiState,
      queue_character_ids: ids,
      queue_character_wildcard: checked ? true : current.uiState.queue_character_wildcard,
    };
    const selected = ensureSelection(node, current.state);
    updateState(node, current.state, nextUiState, {
      characterId: selected.character.id,
      promptId: selected.prompt.id,
      status: queueSettingsSummary(current.state, nextUiState, selected.character.id),
    });
  });
  queueRow.addEventListener("click", (event) => event.stopPropagation());
  queueRow.addEventListener("mousedown", (event) => event.stopPropagation());
  queueRow.addEventListener("keydown", (event) => event.stopPropagation());
  const queueLabel = document.createElement("span");
  queueLabel.textContent = isQueued ? (uiState.queue_character_wildcard ? `Chunk ${queueIndex + 1}` : "Prepared") : "Use in chunks";
  queueRow.append(queueCheckbox, queueLabel);

  tile.append(thumb, name, meta, queueRow);

  const selectCharacter = () => {
    const current = getRenderableState(node);
    const currentSelection = ensureSelection(node, current.state);
    const nextCharacter = current.state.characters.find((item) => item.id === character.id) || currentSelection.character;
    const promptId = nextCharacter.id === currentSelection.character.id && nextCharacter.prompts.some((item) => item.id === currentSelection.prompt.id)
      ? currentSelection.prompt.id
      : nextCharacter.prompts[0]?.id || "";
    updateState(node, current.state, current.uiState, { characterId: nextCharacter.id, promptId, status: `Editing ${nextCharacter.name}. Use Load/Apply to push it into connected nodes.` });
  };
  tile.addEventListener("click", selectCharacter);
  tile.addEventListener("keydown", (event) => {
    if (event.target !== tile) return;
    if (event.key !== "Enter" && event.key !== " ") return;
    event.preventDefault();
    selectCharacter();
  });
  return tile;
}

function characterLibrarySearchBase(character) {
  const stacks = normalizeLoaderStacks(character);
  const loraNames = stacks.flatMap((stack) => (stack.loras || []).map((row) => row.name));
  return [character.name, ...loraNames]
    .filter(Boolean)
    .join(" ")
    .normalize("NFKD")
    .toLocaleLowerCase();
}

function librarySearchText(character, prompt = null) {
  const base = characterLibrarySearchBase(character);
  const promptNames = prompt
    ? [prompt.name, prompt.fileimage_prefix]
    : (character.prompts || []).flatMap((item) => [item.name, item.fileimage_prefix]);
  return [base, ...promptNames]
    .filter(Boolean)
    .join(" ")
    .normalize("NFKD")
    .toLocaleLowerCase();
}

function renderPresetTile(node, character, prompt, selectedCharacterId, selectedPromptId, searchText = "") {
  const isSelected = character.id === selectedCharacterId && prompt.id === selectedPromptId;
  const tile = document.createElement("div");
  tile.tabIndex = 0;
  tile.role = "button";
  tile.dataset.libraryItem = "preset";
  tile.dataset.characterId = character.id;
  tile.dataset.promptId = prompt.id;
  tile.dataset.search = searchText || librarySearchText(character, prompt);
  tile.className = `dsm-preset-tile${isSelected ? " selected" : ""}`;
  tile.setAttribute("aria-label", prompt.name);
  if (isSelected) tile.setAttribute("aria-current", "true");

  const thumb = document.createElement("div");
  thumb.className = "dsm-preset-thumb";
  const url = thumbnailUrl(prompt.reference_image) || thumbnailUrl(character.thumbnail);
  if (url) {
    const img = document.createElement("img");
    img.src = url;
    img.alt = prompt.name;
    img.loading = "lazy";
    img.decoding = "async";
    thumb.appendChild(img);
  } else {
    thumb.textContent = prompt.name.trim().slice(0, 2).toUpperCase() || "?";
  }

  const body = document.createElement("div");
  body.className = "dsm-preset-body";
  const name = document.createElement("div");
  name.className = "dsm-preset-name";
  name.textContent = prompt.name;
  name.title = prompt.name;
  const meta = document.createElement("div");
  meta.className = "dsm-muted dsm-preset-meta";
  const seed = extractSeedFromSettings(prompt.settings);
  const textBoxCount = normalizePromptTextBoxes(prompt).length;
  meta.textContent = `${textBoxCount} text box${textBoxCount === 1 ? "" : "es"}${seed == null ? "" : ` · seed ${seed}`}`;
  body.append(name, meta);
  tile.append(thumb, body);

  const selectPreset = () => {
    const current = getRenderableState(node);
    const nextCharacter = current.state.characters.find((item) => item.id === character.id);
    const nextPrompt = nextCharacter?.prompts.find((item) => item.id === prompt.id);
    if (!nextCharacter || !nextPrompt) return;
    updateState(node, current.state, current.uiState, {
      characterId: character.id,
      promptId: prompt.id,
      status: `Selected ${nextPrompt.name}. Use Load/Apply to push it into graph nodes.`,
    });
  };
  tile.addEventListener("click", selectPreset);
  tile.addEventListener("keydown", (event) => {
    if (event.target !== tile || (event.key !== "Enter" && event.key !== " ")) return;
    event.preventDefault();
    selectPreset();
  });
  return tile;
}

function normalizeLibraryQuery(value) {
  return String(value ?? "").normalize("NFKD").trim().toLocaleLowerCase();
}

function applyLibraryFilter(library, query) {
  if (!library) return;
  const terms = normalizeLibraryQuery(query).split(/\s+/g).filter(Boolean);
  const items = [...library.querySelectorAll("[data-library-item]")];
  let visible = 0;
  for (const item of items) {
    const haystack = item.dataset.search || "";
    const matches = terms.every((term) => haystack.includes(term));
    item.hidden = !matches;
    if (matches) visible += 1;
  }
  const count = library.querySelector("[data-library-count]");
  if (count) {
    const noun = library.dataset.libraryMode === "characters" ? "character" : "preset";
    count.textContent = `${visible} of ${items.length} ${noun}${items.length === 1 ? "" : "s"}`;
  }
  const empty = library.querySelector("[data-library-empty]");
  if (empty) empty.hidden = visible !== 0;
}

function rememberLibraryScroll(node, element, key) {
  const ctx = node.__dsm;
  if (!ctx || !element) return;
  ctx.scrollPositions = ctx.scrollPositions || {};
  element.addEventListener("scroll", () => {
    if (!element.isConnected) return;
    ctx.scrollPositions[key] = element.scrollTop;
  }, { passive: true });
  requestAnimationFrame(() => {
    if (!element.isConnected) return;
    element.scrollTop = Number(ctx.scrollPositions[key] || 0);
    if (ctx.focusLibrarySelection) {
      ctx.focusLibrarySelection = false;
      element.querySelector('[aria-current="true"]:not([hidden])')?.scrollIntoView?.({ block: "nearest" });
    }
  });
}

function renderHeader(node, state, uiState, character, prompt) {
  const section = document.createElement("div");
  section.className = "dsm-section dsm-global-header";

  const toolbar = document.createElement("div");
  toolbar.className = "dsm-toolbar dsm-global-toolbar";
  const heading = document.createElement("div");
  heading.className = "dsm-heading";
  const title = document.createElement("div");
  title.className = "dsm-title";
  title.textContent = "State Manager";
  const selection = document.createElement("div");
  selection.className = "dsm-selection-path";
  selection.textContent = prompt.name;
  selection.title = selection.textContent;
  heading.append(title, selection);

  const importInput = document.createElement("input");
  importInput.type = "file";
  importInput.accept = "application/json,.json";
  importInput.style.display = "none";
  importInput.addEventListener("change", async () => {
    const file = importInput.files?.[0];
    if (!file) return;
    try {
      await importStateJsonFile(node, file);
    } catch (err) {
      setStatus(node, `Import failed: ${err?.message || err}`);
    } finally {
      importInput.value = "";
    }
  });

  const saveConnected = makeButton("Save connected", () => {
    const changes = saveConnectedState(node, character, prompt);
    updateState(node, state, uiState, {
      characterId: character.id,
      promptId: prompt.id,
      status: changes.length ? `Saved ${changes.join(", ")} from connected nodes.` : "No connected downstream nodes had capturable state.",
    });
  }, "Capture current values from nodes connected to this manager's outputs");
  saveConnected.classList.add("dsm-primary");

  const loadConnected = makeButton("Load connected", () => {
    const changes = applyConnectedState(node, character, prompt);
    updateState(node, state, uiState, {
      characterId: character.id,
      promptId: prompt.id,
      status: changes.length ? `Loaded ${changes.join(", ")} into connected nodes.` : "No connected downstream nodes accepted this state.",
    });
  }, "Push the selected character/preset into nodes connected to this manager's outputs");
  loadConnected.classList.add("dsm-primary");

  toolbar.append(
    heading,
    saveConnected,
    loadConnected,
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
    }, "Apply the current character/preset to selected graph nodes"),
    makeButton("Export character", async () => {
      try {
        if (character.id === "default_character" || character.__dsm_ephemeral) {
          throw new Error("The built-in or missing placeholder is not a persistent character.");
        }
        await exportCharacterJson(character.id);
        setStatus(node, "Exported the selected character.");
      } catch (err) {
        setStatus(node, `Character export failed: ${err?.message || err}`);
      }
    }, "Download only the selected character and its prompts"),
    makeButton("Export library", async () => {
      try {
        await exportLibraryJson();
        setStatus(node, "Exported the complete State Manager library.");
      } catch (err) {
        setStatus(node, `Library export failed: ${err?.message || err}`);
      }
    }, "Download the complete persistent State Manager library"),
    makeButton("Import", () => importInput.click(), "Import a character or State Manager library JSON"),
    makeButton("Reload library", async () => {
      try {
        await reloadStateLibrary(node);
      } catch (err) {
        setStatus(node, `Library reload failed: ${err?.message || err}`);
      }
    }, "Discard unsaved stale edits and reload persistent storage")
  );

  section.append(toolbar, importInput);
  return section;
}

function renderLibrary(node, state, uiState, character, prompt) {
  const ctx = node.__dsm;
  const mode = ctx.libraryMode === "characters" ? "characters" : "presets";
  const section = document.createElement("section");
  section.className = "dsm-section dsm-library";
  section.dataset.libraryMode = mode;

  const header = document.createElement("div");
  header.className = "dsm-library-header";
  const titleGroup = document.createElement("div");
  titleGroup.className = "dsm-library-title-group";
  const title = document.createElement("div");
  title.className = "dsm-section-title";
  title.textContent = "State library";
  const count = document.createElement("div");
  count.className = "dsm-library-count";
  count.dataset.libraryCount = "";
  titleGroup.append(title, count);

  const modes = document.createElement("div");
  modes.className = "dsm-tabs dsm-library-tabs";
  for (const [value, label] of [["presets", "All presets"], ["characters", "Characters"]]) {
    const button = makeButton(label, () => {
      ctx.libraryMode = value;
      scheduleRender(node);
    });
    if (mode === value) button.classList.add("selected");
    modes.appendChild(button);
  }
  header.append(titleGroup, modes);

  const search = document.createElement("input");
  search.type = "search";
  search.className = "dsm-library-search";
  search.placeholder = mode === "presets"
    ? "Search presets, characters, filename prefixes, or LoRAs…"
    : "Search characters, presets, or LoRAs…";
  search.value = ctx.searchQuery || "";
  search.autocomplete = "off";
  search.spellcheck = false;
  search.addEventListener("input", () => {
    ctx.searchQuery = search.value;
    applyLibraryFilter(section, search.value);
  });
  search.addEventListener("keydown", (event) => event.stopPropagation());

  const grid = document.createElement("div");
  grid.className = mode === "presets" ? "dsm-library-grid dsm-preset-grid" : "dsm-library-grid dsm-character-grid";
  grid.dataset.scrollKey = mode;
  const fragment = document.createDocumentFragment();
  if (mode === "presets") {
    for (const item of state.characters) {
      const characterSearch = characterLibrarySearchBase(item);
      for (const itemPrompt of item.prompts) {
        const presetSearch = `${characterSearch} ${String(itemPrompt.name || "")} ${String(itemPrompt.fileimage_prefix || "")}`
          .normalize("NFKD")
          .toLocaleLowerCase();
        fragment.appendChild(renderPresetTile(node, item, itemPrompt, character.id, prompt.id, presetSearch));
      }
    }
  } else {
    for (const item of state.characters) {
      fragment.appendChild(renderCharacterTile(node, state, uiState, item, character.id, prompt.id));
    }
  }
  grid.appendChild(fragment);

  const empty = document.createElement("div");
  empty.className = "dsm-library-empty";
  empty.dataset.libraryEmpty = "";
  empty.textContent = "No saved states match this search.";
  grid.appendChild(empty);

  const controls = document.createElement("div");
  controls.className = "dsm-toolbar dsm-library-actions";
  const currentLibrarySelection = () => {
    const current = getRenderableState(node);
    const selected = ensureSelection(node, current.state);
    return { ...current, ...selected };
  };
  if (mode === "presets") {
    controls.append(
      makeButton("New preset", () => {
        const current = currentLibrarySelection();
        const next = defaultPrompt(current.prompt.name);
        next.id = makeId("prompt");
        current.character.prompts.push(next);
        ctx.focusLibrarySelection = true;
        updateState(node, current.state, current.uiState, { characterId: current.character.id, promptId: next.id, status: `Created preset for ${current.character.name}.` });
      }),
      makeButton("Duplicate preset", () => {
        const current = currentLibrarySelection();
        const copy = { ...structuredCloneCompat(current.prompt), id: makeId("prompt"), name: `${current.prompt.name} Copy` };
        current.character.prompts.push(copy);
        ctx.focusLibrarySelection = true;
        updateState(node, current.state, current.uiState, { characterId: current.character.id, promptId: copy.id, status: "Duplicated prompt preset." });
      }),
      makeButton("Delete preset", () => {
        const current = currentLibrarySelection();
        if (current.character.prompts.length <= 1) {
          setStatus(node, "At least one prompt preset must remain for this character.");
          return;
        }
        const index = current.character.prompts.findIndex((item) => item.id === current.prompt.id);
        if (index >= 0) current.character.prompts.splice(index, 1);
        const next = current.character.prompts[Math.max(0, Math.min(index, current.character.prompts.length - 1))];
        updateState(node, current.state, current.uiState, { characterId: current.character.id, promptId: next.id, status: "Deleted prompt preset." });
      })
    );
  } else {
    controls.append(
      makeButton("New character", () => {
        const current = currentLibrarySelection();
        const next = defaultCharacter();
        next.id = makeId("character");
        next.name = `Character ${current.state.characters.length + 1}`;
        next.prompts[0].id = makeId("prompt");
        next.prompts[0].name = next.name;
        current.state.characters.push(next);
        ctx.focusLibrarySelection = true;
        updateState(node, current.state, current.uiState, { characterId: next.id, promptId: next.prompts[0].id, status: "Created character." });
      }),
      makeButton("Duplicate character", () => {
        const current = currentLibrarySelection();
        const copy = structuredCloneCompat(current.character);
        copy.id = makeId("character");
        copy.name = `${copy.name} Copy`;
        copy.prompts = copy.prompts.map((item) => ({ ...item, id: makeId("prompt"), name: `${item.name} Copy` }));
        current.state.characters.push(copy);
        ctx.focusLibrarySelection = true;
        updateState(node, current.state, current.uiState, { characterId: copy.id, promptId: copy.prompts[0]?.id || "", status: "Duplicated character." });
      }),
      makeButton("Delete character", () => {
        const current = currentLibrarySelection();
        if (current.state.characters.length <= 1) {
          setStatus(node, "At least one character must remain.");
          return;
        }
        const index = current.state.characters.findIndex((item) => item.id === current.character.id);
        if (index >= 0) current.state.characters.splice(index, 1);
        const next = current.state.characters[Math.max(0, Math.min(index, current.state.characters.length - 1))];
        updateState(node, current.state, current.uiState, { characterId: next.id, promptId: next.prompts[0]?.id || "", status: "Deleted character." });
      })
    );
  }

  section.append(header, search, grid, controls);
  applyLibraryFilter(section, ctx.searchQuery || "");
  rememberLibraryScroll(node, grid, mode);
  return section;
}

function renderQueuePanelContent(section, node, state, uiState, character, prompt) {
  const queueRows = document.createElement("div");
  queueRows.className = "dsm-queue-options";
  queueRows.append(
    makeInlineCheckbox(
      "Random prompt preset per queued image",
      uiState.queue_prompt_wildcard,
      (checked) => {
        const nextUiState = { ...uiState, queue_prompt_wildcard: checked };
        updateState(node, state, nextUiState, { characterId: character.id, promptId: prompt.id, status: queueSettingsSummary(state, nextUiState, character.id) });
      },
      "Each queued job samples one saved prompt preset from the active character."
    ),
    makeInlineCheckbox(
      "Randomize saved prompt seed when live seed is random",
      uiState.queue_randomize_saved_seed,
      (checked) => {
        const nextUiState = { ...uiState, queue_randomize_saved_seed: checked };
        updateState(node, state, nextUiState, { characterId: character.id, promptId: prompt.id, status: queueSettingsSummary(state, nextUiState, character.id) });
      },
      "A fixed connected State Seed keeps precedence."
    ),
    makeInlineCheckbox(
      "Split queued images into contiguous character chunks",
      uiState.queue_character_wildcard,
      (checked) => {
        const ids = queueCharacterIdsExplicit(state, uiState);
        const nextUiState = {
          ...uiState,
          queue_character_wildcard: checked,
          queue_character_ids: checked && !ids.length ? [character.id] : ids,
        };
        updateState(node, state, nextUiState, { characterId: character.id, promptId: prompt.id, status: queueSettingsSummary(state, nextUiState, character.id) });
      },
      "Choose the participating characters below."
    )
  );

  const picker = document.createElement("div");
  picker.className = "dsm-queue-character-picker";
  const queueIds = queueCharacterIdsExplicit(state, uiState);
  for (const item of state.characters) {
    const row = document.createElement("label");
    row.className = "dsm-queue-character-row";
    const checked = queueIds.includes(item.id);
    const checkbox = makeCheckbox(checked, (enabled) => {
      const ids = toggleQueueCharacterId(state, uiState, item.id, enabled);
      const nextUiState = {
        ...uiState,
        queue_character_ids: ids,
        queue_character_wildcard: enabled ? true : uiState.queue_character_wildcard,
      };
      updateState(node, state, nextUiState, { characterId: character.id, promptId: prompt.id, status: queueSettingsSummary(state, nextUiState, character.id) });
    });
    const name = document.createElement("span");
    name.textContent = item.name;
    const order = document.createElement("small");
    const index = queueIds.indexOf(item.id);
    order.textContent = index >= 0 ? `Chunk ${index + 1}` : "";
    row.append(checkbox, name, order);
    picker.appendChild(row);
  }

  const actions = document.createElement("div");
  actions.className = "dsm-toolbar";
  actions.append(
    makeButton("Use current character", () => {
      const nextUiState = { ...uiState, queue_character_wildcard: true, queue_character_ids: [character.id] };
      updateState(node, state, nextUiState, { characterId: character.id, promptId: prompt.id, status: queueSettingsSummary(state, nextUiState, character.id) });
    }),
    makeButton("Use all characters", () => {
      const nextUiState = { ...uiState, queue_character_wildcard: true, queue_character_ids: state.characters.map((item) => item.id) };
      updateState(node, state, nextUiState, { characterId: character.id, promptId: prompt.id, status: queueSettingsSummary(state, nextUiState, character.id) });
    }),
    makeButton("Clear", () => {
      const nextUiState = { ...uiState, queue_character_wildcard: false, queue_character_ids: [] };
      updateState(node, state, nextUiState, { characterId: character.id, promptId: prompt.id, status: queueSettingsSummary(state, nextUiState, character.id) });
    })
  );

  const summary = document.createElement("div");
  summary.className = "dsm-queue-summary";
  summary.textContent = queueSettingsSummary(state, uiState, character.id);
  section.append(queueRows, labelledControl("Character chunks", picker), actions, summary);
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
  preview.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    fileInput.click();
  });
  preview.addEventListener("dragenter", (event) => {
    stopComfyFileDropEvent(event);
    preview.classList.add("dragging");
  }, true);
  preview.addEventListener("dragover", (event) => {
    stopComfyFileDropEvent(event);
    preview.classList.add("dragging");
  }, true);
  preview.addEventListener("dragleave", (event) => {
    event.stopPropagation();
    preview.classList.remove("dragging");
  }, true);
  preview.addEventListener("drop", async (event) => {
    stopComfyFileDropEvent(event);
    preview.classList.remove("dragging");
    const file = [...(event.dataTransfer?.files || [])].find((candidate) => candidate.type.startsWith("image/"));
    if (!file) return;
    try {
      character.thumbnail = await uploadThumbnailFile(file);
      updateState(node, state, uiState, { characterId: character.id, status: "Original character image uploaded. UI preview is CSS-scaled only; backend image output loads the original file." });
    } catch (err) {
      setStatus(node, `Thumbnail upload failed: ${err?.message || err}`);
    }
  }, true);

  const loraSummary = document.createElement("div");
  loraSummary.className = "dsm-lora-summary";
  const stackLines = getCharacterLoaderStacks(character).flatMap((stack) => {
    const activeRows = (stack.loras || []).filter((row) => row.enabled && row.name && row.name !== "None");
    if (!activeRows.length) return [`[${stack.slot}] ${stack.label}: empty`];
    return [`[${stack.slot}] ${stack.label}`, ...activeRows.map((row) => `  ${row.name} (${row.strength_model}/${row.strength_clip})`)];
  });
  loraSummary.textContent = stackLines.length ? stackLines.join("\n") : "No saved LoRA stack for this character.";

  const thumbnailButtons = document.createElement("div");
  thumbnailButtons.className = "dsm-toolbar";
  thumbnailButtons.append(
    makeButton("Choose thumbnail", () => fileInput.click()),
    makeButton("Clear", () => {
      character.thumbnail = {};
      updateState(node, state, uiState, { characterId: character.id, status: "Thumbnail cleared." });
    })
  );

  const imageNote = document.createElement("div");
  imageNote.className = "dsm-muted";
  imageNote.textContent = "The State Manager image output loads the original uploaded file, not the CSS-scaled preview.";

  section.append(title, labelledControl("Shared group name (all states)", nameInput), preview, fileInput, thumbnailButtons, imageNote, labelledControl("Saved LoRA stacks", loraSummary));
  return section;
}

function renderPromptPanel(node, state, uiState, character, prompt) {
  const section = document.createElement("div");
  section.className = "dsm-section dsm-panel dsm-editor";

  const tabs = document.createElement("div");
  tabs.className = "dsm-tabs";
  for (const [panel, label] of [
    ["prompts", "Preset"],
    ["character", "Character"],
    ["loras", "LoRA stacks"],
    ["settings", "Settings / seed"],
    ["queue", "Queue"],
  ]) {
    const btn = makeButton(label, () => setPanel(node, panel));
    if (uiState.panel === panel) btn.classList.add("selected");
    tabs.appendChild(btn);
  }
  section.appendChild(tabs);

  if (uiState.panel === "character") {
    const characterPanel = renderCharacterPanel(node, state, uiState, character);
    characterPanel.className = "dsm-editor-content";
    section.appendChild(characterPanel);
  } else if (uiState.panel === "loras") {
    renderLoraPanelContent(section, node, state, uiState, character, prompt);
  } else if (uiState.panel === "settings") {
    renderSettingsPanelContent(section, node, state, uiState, character, prompt);
  } else if (uiState.panel === "queue") {
    renderQueuePanelContent(section, node, state, uiState, character, prompt);
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
      const p = defaultPrompt(prompt.name);
      p.id = makeId("prompt");
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
    const current = getRenderableState(node);
    const nextState = updatePromptName(current.state, character.id, prompt.id, value);
    updateState(node, nextState, current.uiState, { characterId: character.id, promptId: prompt.id });
  });

  const positive = makeTextarea(getPromptText(prompt, "positive", "default"), (value) => {
    setPromptTextBox(prompt, "positive", "default", value, "Default positive");
    updateState(node, state, uiState, { characterId: character.id, promptId: prompt.id, render: false });
  }, { placeholder: "Positive prompt template. Wildcards stay here and expand downstream." });

  const negative = makeTextarea(getPromptText(prompt, "negative", "default"), (value) => {
    setPromptTextBox(prompt, "negative", "default", value, "Default negative");
    updateState(node, state, uiState, { characterId: character.id, promptId: prompt.id, render: false });
  }, { placeholder: "Negative prompt template. Wildcards stay here and expand downstream." });

  const fileimagePrefix = makeInput(prompt.fileimage_prefix ?? "", (value) => {
    prompt.fileimage_prefix = String(value ?? "").trim();
    updateState(node, state, uiState, { characterId: character.id, promptId: prompt.id });
  }, { placeholder: "e.g. jl_dryfs2/AIM" });

  const referencePreview = document.createElement("div");
  referencePreview.className = "dsm-large-thumb";
  const referenceUrl = thumbnailUrl(prompt.reference_image);
  if (referenceUrl) {
    const img = document.createElement("img");
    img.src = referenceUrl;
    img.alt = `${prompt.name || "Prompt"} reference`;
    referencePreview.appendChild(img);
  } else {
    referencePreview.textContent = "Drop prompt reference image here";
  }

  const referenceInput = document.createElement("input");
  referenceInput.type = "file";
  referenceInput.accept = "image/*";
  referenceInput.style.display = "none";
  referenceInput.addEventListener("change", async () => {
    const file = referenceInput.files?.[0];
    if (!file) return;
    try {
      prompt.reference_image = await uploadThumbnailFile(file);
      updateState(node, state, uiState, { characterId: character.id, promptId: prompt.id, status: "Prompt reference image uploaded." });
    } catch (err) {
      setStatus(node, `Prompt reference upload failed: ${err?.message || err}`);
    } finally {
      referenceInput.value = "";
    }
  });
  referencePreview.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    referenceInput.click();
  });
  referencePreview.addEventListener("dragenter", (event) => {
    stopComfyFileDropEvent(event);
    referencePreview.classList.add("dragging");
  }, true);
  referencePreview.addEventListener("dragover", (event) => {
    stopComfyFileDropEvent(event);
    referencePreview.classList.add("dragging");
  }, true);
  referencePreview.addEventListener("dragleave", (event) => {
    event.stopPropagation();
    referencePreview.classList.remove("dragging");
  }, true);
  referencePreview.addEventListener("drop", async (event) => {
    stopComfyFileDropEvent(event);
    referencePreview.classList.remove("dragging");
    const file = [...(event.dataTransfer?.files || [])].find((candidate) => candidate.type.startsWith("image/"));
    if (!file) return;
    try {
      prompt.reference_image = await uploadThumbnailFile(file);
      updateState(node, state, uiState, { characterId: character.id, promptId: prompt.id, status: "Prompt reference image uploaded." });
    } catch (err) {
      setStatus(node, `Prompt reference upload failed: ${err?.message || err}`);
    }
  }, true);

  const referenceButtons = document.createElement("div");
  referenceButtons.className = "dsm-toolbar";
  referenceButtons.append(
    makeButton("Choose prompt image", () => referenceInput.click()),
    makeButton("Clear", () => {
      prompt.reference_image = {};
      updateState(node, state, uiState, { characterId: character.id, promptId: prompt.id, status: "Prompt reference image cleared. Character image fallback will be used." });
    })
  );

  const referenceNote = document.createElement("div");
  referenceNote.className = "dsm-muted";
  referenceNote.textContent = "The image output uses this prompt reference first, then falls back to the character thumbnail.";

  const textBoxToolbar = document.createElement("div");
  textBoxToolbar.className = "dsm-toolbar";
  textBoxToolbar.append(
    makeButton("Add positive box", () => {
      const slot = normalizeTextSlot(`positive_${normalizePromptTextBoxes(prompt).filter((box) => box.role === "positive").length + 1}`);
      setPromptTextBox(prompt, "positive", slot, "", `Positive ${slot}`);
      updateState(node, state, uiState, { characterId: character.id, promptId: prompt.id, status: `Added positive text box ${slot}. Match this slot on a State Manager Text Box node.` });
    }),
    makeButton("Add negative box", () => {
      const slot = normalizeTextSlot(`negative_${normalizePromptTextBoxes(prompt).filter((box) => box.role === "negative").length + 1}`);
      setPromptTextBox(prompt, "negative", slot, "", `Negative ${slot}`);
      updateState(node, state, uiState, { characterId: character.id, promptId: prompt.id, status: `Added negative text box ${slot}. Match this slot on a State Manager Text Box node.` });
    })
  );

  const textBoxNote = document.createElement("div");
  textBoxNote.className = "dsm-muted";
  textBoxNote.textContent = "Additional prompt boxes are matched by Role + State slot. Use unique slots, e.g. main, detailer, refiner, upscale.";

  const boxes = normalizePromptTextBoxes(prompt);
  const savedTextBoxes = document.createElement("div");
  savedTextBoxes.className = "dsm-stack-box";
  for (const [index, box] of boxes.entries()) {
    const row = document.createElement("div");
    row.className = "dsm-prompt-box";
    const role = makeSelect(TEXT_BOX_ROLE_CHOICES, box.role, (value) => {
      box.role = normalizeTextRole(value, box.role);
      prompt.text_boxes = boxes;
      syncPromptTextMirror(prompt);
      updateState(node, state, uiState, { characterId: character.id, promptId: prompt.id });
    });
    const slot = makeInput(box.slot, (value) => {
      const oldSlot = box.slot;
      box.slot = normalizeTextSlot(value, oldSlot || `text_${index + 1}`);
      prompt.text_boxes = boxes;
      syncPromptTextMirror(prompt);
      updateState(node, state, uiState, { characterId: character.id, promptId: prompt.id, status: `Renamed text slot ${oldSlot} -> ${box.slot}. Update the matching State Manager Text Box node too.` });
    });
    const label = makeInput(box.label, (value) => {
      box.label = String(value || `${box.role} ${box.slot}`).trim() || `${box.role} ${box.slot}`;
      prompt.text_boxes = boxes;
      syncPromptTextMirror(prompt);
      updateState(node, state, uiState, { characterId: character.id, promptId: prompt.id });
    });
    const text = makeTextarea(box.text, (value) => {
      box.text = value;
      prompt.text_boxes = boxes;
      syncPromptTextMirror(prompt);
      updateState(node, state, uiState, { characterId: character.id, promptId: prompt.id, render: false });
    }, { placeholder: `${box.role} text for slot ${box.slot}` });
    row.append(
      labelledControl("Role", role),
      labelledControl("State slot", slot),
      labelledControl("Label", label),
      makeButton("Delete", () => {
        if (boxes.length <= 1) {
          setStatus(node, "At least one prompt text box must remain.");
          return;
        }
        boxes.splice(index, 1);
        prompt.text_boxes = boxes;
        syncPromptTextMirror(prompt);
        updateState(node, state, uiState, { characterId: character.id, promptId: prompt.id, status: `Deleted text box ${box.role}/${box.slot}.` });
      }),
      labelledControl("Text", text)
    );
    savedTextBoxes.appendChild(row);
  }

  section.append(
    header,
    labelledControl("Character name", name),
    labelledControl("fileimage_prefix", fileimagePrefix),
    referencePreview,
    referenceInput,
    referenceButtons,
    referenceNote,
    labelledControl("Default positive template", positive),
    labelledControl("Default negative template", negative),
    textBoxToolbar,
    textBoxNote,
    labelledControl("Saved prompt text boxes", savedTextBoxes)
  );
}

function renderLoraStackEditor(section, node, state, uiState, character, prompt, stack, stackIndex) {
  const box = document.createElement("div");
  box.className = "dsm-stack-box";

  const title = document.createElement("div");
  title.className = "dsm-toolbar";
  const slot = makeInput(stack.slot, (value) => {
    const oldSlot = stack.slot;
    stack.slot = normalizeLoaderSlot(value, oldSlot || `loader_${stackIndex + 1}`);
    if (stack.slot !== oldSlot) syncLegacyLoaderMirror(character);
    updateState(node, state, uiState, { characterId: character.id, promptId: prompt.id, status: `Renamed loader slot ${oldSlot} -> ${stack.slot}. Update the matching DoRA loader's State slot too.` });
  });
  const label = makeInput(stack.label, (value) => {
    stack.label = String(value || stack.slot).trim() || stack.slot;
    syncLegacyLoaderMirror(character);
    updateState(node, state, uiState, { characterId: character.id, promptId: prompt.id });
  });
  title.append(
    labelledControl("Slot", slot),
    labelledControl("Label", label),
    makeButton("Add row", () => {
      stack.loras.push({ enabled: true, name: "None", strength_model: 1.0, strength_clip: 1.0 });
      syncLegacyLoaderMirror(character);
      updateState(node, state, uiState, { characterId: character.id, promptId: prompt.id, status: `Added LoRA row to ${stack.slot}.` });
    }),
    makeButton("Delete stack", () => {
      const stacks = getCharacterLoaderStacks(character);
      if (stacks.length <= 1) {
        setStatus(node, "At least one loader stack must remain.");
        return;
      }
      const idx = stacks.findIndex((candidate) => normalizeLoaderSlot(candidate.slot, "default") === normalizeLoaderSlot(stack.slot, "default"));
      if (idx >= 0) stacks.splice(idx, 1);
      syncLegacyLoaderMirror(character);
      updateState(node, state, uiState, { characterId: character.id, promptId: prompt.id, status: `Deleted loader stack ${stack.slot}.` });
    })
  );
  box.appendChild(title);

  const globals = stack.loader_globals || (stack.loader_globals = {});
  const globalsLine = document.createElement("div");
  globalsLine.className = "dsm-grid2";
  globalsLine.append(
    labelledControl("Stack enabled", makeCheckbox(globals.stack_enabled ?? true, (checked) => {
      updateCharacterLoaderGlobals(character, stack, { stack_enabled: checked });
      updateState(node, state, uiState, { characterId: character.id, promptId: prompt.id });
    })),
    labelledControl("Auto-strength", makeCheckbox(globals.auto_strength_enabled ?? false, (checked) => {
      updateCharacterLoaderGlobals(character, stack, { auto_strength_enabled: checked });
      updateState(node, state, uiState, { characterId: character.id, promptId: prompt.id });
    })),
    labelledControl("Analysis device", makeSelect(AUTO_STRENGTH_DEVICE_CHOICES, normalizeDevice(globals.auto_strength_device ?? "gpu"), (value) => {
      updateCharacterLoaderGlobals(character, stack, { auto_strength_device: value });
      updateState(node, state, uiState, { characterId: character.id, promptId: prompt.id });
    })),
    labelledControl("Ratio ceiling", makeInput(globals.auto_strength_ratio_ceiling ?? 1.5, (value) => {
      updateCharacterLoaderGlobals(character, stack, { auto_strength_ratio_ceiling: normalizeNumber(value, 1.5) });
      updateState(node, state, uiState, { characterId: character.id, promptId: prompt.id });
    }, { type: "number", step: "0.01" }))
  );
  box.appendChild(globalsLine);

  if (!stack.loras.length) {
    const empty = document.createElement("div");
    empty.className = "dsm-muted";
    empty.textContent = `No LoRA rows saved for slot ${stack.slot}.`;
    box.appendChild(empty);
  }

  stack.loras.forEach((row, index) => {
    const line = document.createElement("div");
    line.className = "dsm-lora-row";
    const enabled = makeCheckbox(row.enabled, (checked) => {
      row.enabled = checked;
      syncLegacyLoaderMirror(character);
      updateState(node, state, uiState, { characterId: character.id, promptId: prompt.id });
    });
    const name = makeInput(row.name, (value) => {
      row.name = value.trim() || "None";
      syncLegacyLoaderMirror(character);
      updateState(node, state, uiState, { characterId: character.id, promptId: prompt.id });
    });
    const sm = makeInput(row.strength_model, (value) => {
      row.strength_model = normalizeNumber(value, row.strength_model);
      syncLegacyLoaderMirror(character);
      updateState(node, state, uiState, { characterId: character.id, promptId: prompt.id });
    }, { type: "number", step: "0.05" });
    const sc = makeInput(row.strength_clip, (value) => {
      row.strength_clip = normalizeNumber(value, row.strength_clip);
      syncLegacyLoaderMirror(character);
      updateState(node, state, uiState, { characterId: character.id, promptId: prompt.id });
    }, { type: "number", step: "0.05" });
    line.append(enabled, name, sm, sc, makeButton("×", () => {
      stack.loras.splice(index, 1);
      syncLegacyLoaderMirror(character);
      updateState(node, state, uiState, { characterId: character.id, promptId: prompt.id, status: `Removed LoRA row from ${stack.slot}.` });
    }));
    box.appendChild(line);
  });

  section.appendChild(box);
}

function renderLoraPanelContent(section, node, state, uiState, character, prompt) {
  const header = document.createElement("div");
  header.className = "dsm-toolbar";
  header.append(
    makeButton("Save connected loaders", () => {
      const controlledLoaders = getControlledNodes(node).filter(isDoraLoaderNode);
      const loaders = controlledLoaders.length ? controlledLoaders : uniqueNodes(getOutputTargets(node, OUTPUT_NAMES.lora)).filter(isDoraLoaderNode);
      let count = 0;
      loaders.forEach((loader, index) => {
        const stack = extractLoraStackFromNode(loader, index);
        if (!stack) return;
        setCharacterLoaderStack(character, stack);
        count += 1;
      });
      updateState(node, state, uiState, { characterId: character.id, promptId: prompt.id, status: count ? `Saved ${count} connected DoRA loader${count === 1 ? "" : "s"}.` : "No connected DoRA loader found." });
    }),
    makeButton("Apply connected loaders", () => {
      const controlledLoaders = getControlledNodes(node).filter(isDoraLoaderNode);
      const loaders = controlledLoaders.length ? controlledLoaders : uniqueNodes(getOutputTargets(node, OUTPUT_NAMES.lora)).filter(isDoraLoaderNode);
      let count = 0;
      for (const loader of loaders) if (applyLoraStackToNode(loader, character)) count += 1;
      updateState(node, state, uiState, { characterId: character.id, promptId: prompt.id, status: count ? `Applied saved LoRA stacks to ${count} connected loader${count === 1 ? "" : "s"}.` : "No connected DoRA loader accepted a matching stack." });
    }),
    makeButton("Add loader stack", () => {
      const stack = defaultLoaderStack(`loader_${getCharacterLoaderStacks(character).length + 1}`, `Loader ${getCharacterLoaderStacks(character).length + 1}`);
      character.loader_stacks.push(stack);
      syncLegacyLoaderMirror(character);
      updateState(node, state, uiState, { characterId: character.id, promptId: prompt.id, status: `Added loader stack ${stack.slot}.` });
    })
  );
  section.appendChild(header);

  const note = document.createElement("div");
  note.className = "dsm-muted";
  note.textContent = "Each DoRA loader is matched by its State slot widget. Use unique slots, e.g. face, outfit, style, refiner.";
  section.appendChild(note);

  for (const [index, stack] of getCharacterLoaderStacks(character).entries()) {
    renderLoraStackEditor(section, node, state, uiState, character, prompt, stack, index);
  }
}


function renderSettingsPanelContent(section, node, state, uiState, character, prompt) {
  const primaryStack = findCharacterLoaderStack(character, "default") || getCharacterLoaderStacks(character)[0];
  const globals = primaryStack.loader_globals || (primaryStack.loader_globals = {});
  const grid = document.createElement("div");
  grid.className = "dsm-grid2";

  const stackEnabled = makeCheckbox(globals.stack_enabled ?? true, (checked) => {
    updateCharacterLoaderGlobals(character, primaryStack, { stack_enabled: checked });
    updateState(node, state, uiState, { characterId: character.id, promptId: prompt.id });
  });
  const autoStrength = makeCheckbox(globals.auto_strength_enabled ?? false, (checked) => {
    updateCharacterLoaderGlobals(character, primaryStack, { auto_strength_enabled: checked });
    updateState(node, state, uiState, { characterId: character.id, promptId: prompt.id });
  });
  const device = makeSelect(AUTO_STRENGTH_DEVICE_CHOICES, normalizeDevice(globals.auto_strength_device ?? "gpu"), (value) => {
    updateCharacterLoaderGlobals(character, primaryStack, { auto_strength_device: value });
    updateState(node, state, uiState, { characterId: character.id, promptId: prompt.id });
  });
  const broadcastMods = makeCheckbox(globals.broadcast_modulations ?? true, (checked) => {
    updateCharacterLoaderGlobals(character, primaryStack, { broadcast_modulations: checked });
    updateState(node, state, uiState, { characterId: character.id, promptId: prompt.id });
  });
  const floor = makeInput(globals.auto_strength_ratio_floor ?? 0.3, (value) => {
    updateCharacterLoaderGlobals(character, primaryStack, { auto_strength_ratio_floor: normalizeNumber(value, 0.3) });
    updateState(node, state, uiState, { characterId: character.id, promptId: prompt.id });
  }, { type: "number", step: "0.01" });
  const ceiling = makeInput(globals.auto_strength_ratio_ceiling ?? 1.5, (value) => {
    updateCharacterLoaderGlobals(character, primaryStack, { auto_strength_ratio_ceiling: normalizeNumber(value, 1.5) });
    updateState(node, state, uiState, { characterId: character.id, promptId: prompt.id });
  }, { type: "number", step: "0.01" });
  const sliceFix = makeCheckbox(globals.dora_slice_fix ?? true, (checked) => {
    updateCharacterLoaderGlobals(character, primaryStack, { dora_slice_fix: checked });
    updateState(node, state, uiState, { characterId: character.id, promptId: prompt.id });
  });
  const adalnFix = makeCheckbox(globals.dora_adaln_swap_fix ?? true, (checked) => {
    updateCharacterLoaderGlobals(character, primaryStack, { dora_adaln_swap_fix: checked });
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
    settings.seed = normalizeSeedInteger(value, -1);
    if (settings.rgthree_seed?.widgets) settings.rgthree_seed.widgets.seed = settings.seed;
    if (settings.rgthree_seed) settings.rgthree_seed.seed = settings.seed;
    prompt.settings = settings;
    syncLegacyLoaderMirror(character);
    updateState(node, state, uiState, { characterId: character.id, promptId: prompt.id });
  }, { type: "number", step: "1", min: String(STATE_SEED_MIN), max: String(STATE_SEED_MAX) });

  const settingsJson = makeTextarea(JSON.stringify(prompt.settings || {}, null, 2), (value) => {
    const parsed = safeJsonParse(value, null);
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      prompt.settings = parsed;
      updateState(node, state, uiState, { characterId: character.id, promptId: prompt.id, render: false });
    }
  }, { placeholder: "Saved node settings JSON. Save connected/selected nodes to populate this." });
  settingsJson.addEventListener("change", () => {
    settingsJson.value = JSON.stringify(prompt.settings || {}, null, 2);
    syncLegacyLoaderMirror(character);
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

function libraryRenderKey(node, state, uiState, mode) {
  const stateKey = canonicalJson(state);
  return [
    mode,
    stateKey,
    uiState.queue_character_wildcard ? "1" : "0",
    queueCharacterIdsExplicit(state, uiState).join(","),
  ].join("\u0000");
}

function updateLibrarySelection(library, characterId, promptId) {
  if (!library) return;
  for (const item of library.querySelectorAll("[data-library-item]")) {
    const selected = item.dataset.libraryItem === "preset"
      ? item.dataset.characterId === characterId && item.dataset.promptId === promptId
      : item.dataset.characterId === characterId;
    item.classList.toggle("selected", selected);
    if (selected) item.setAttribute("aria-current", "true");
    else item.removeAttribute("aria-current");
  }
}

function renderNode(node) {
  const ctx = node.__dsm;
  if (!ctx?.root) return;
  const { state, uiState } = getRenderableState(node);
  const { character, prompt } = ensureSelection(node, state);
  const mode = ctx.libraryMode === "characters" ? "characters" : "presets";
  const key = libraryRenderKey(node, state, uiState, mode);
  let library = ctx.libraryElement;
  let preservedScroll = null;
  if (!library || ctx.libraryKey !== key) {
    library = renderLibrary(node, state, uiState, character, prompt);
    ctx.libraryElement = library;
    ctx.libraryKey = key;
  } else {
    const grid = library.querySelector(`[data-scroll-key="${mode}"]`);
    if (grid) {
      preservedScroll = { element: grid, top: grid.scrollTop };
      ctx.scrollPositions[mode] = preservedScroll.top;
    }
    library.remove();
    updateLibrarySelection(library, character.id, prompt.id);
  }
  ctx.root.replaceChildren();
  ctx.root.className = "dsm-root";
  ctx.root.append(renderHeader(node, state, uiState, character, prompt));
  const main = document.createElement("div");
  main.className = "dsm-main";
  main.append(library, renderPromptPanel(node, state, uiState, character, prompt));
  ctx.root.appendChild(main);
  if (preservedScroll) {
    const restoreScroll = () => {
      if (!preservedScroll.element.isConnected) return;
      preservedScroll.element.scrollTop = preservedScroll.top;
      ctx.scrollPositions[mode] = preservedScroll.element.scrollTop;
    };
    restoreScroll();
    requestAnimationFrame(restoreScroll);
  }
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
      background: rgba(20, 20, 20, .18);
      font: 12px/1.35 system-ui, sans-serif;
    }
    .dsm-root *, .dsm-root *::before, .dsm-root *::after { box-sizing: border-box; }
    .dsm-section {
      border: 1px solid rgba(128,128,128,.35);
      border-radius: 8px;
      padding: 8px;
      background: rgba(0,0,0,.10);
    }
    .dsm-global-header { flex: 0 0 auto; min-width: 0; padding: 7px 8px; }
    .dsm-global-toolbar { margin: 0; }
    .dsm-heading { flex: 1 1 180px; min-width: 130px; overflow: hidden; }
    .dsm-title, .dsm-section-title { font-weight: 700; opacity: .96; }
    .dsm-title { font-size: 13px; }
    .dsm-selection-path { margin-top: 1px; opacity: .66; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .dsm-main {
      flex: 1 1 auto;
      min-height: 0;
      display: grid;
      grid-template-columns: minmax(390px, 1.35fr) minmax(340px, .9fr);
      gap: 8px;
      overflow: hidden;
    }
    .dsm-library {
      min-height: 0;
      min-width: 0;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      gap: 7px;
      border-color: rgba(150,175,220,.46);
      background: rgba(20,30,48,.16);
    }
    .dsm-library-header { display: flex; align-items: center; justify-content: space-between; gap: 10px; min-width: 0; }
    .dsm-library-title-group { display: flex; align-items: baseline; gap: 7px; min-width: 0; }
    .dsm-library-count { opacity: .62; white-space: nowrap; }
    .dsm-library-tabs { flex: 0 0 auto; }
    .dsm-library-search { width: 100%; flex: 0 0 auto; }
    .dsm-library-grid {
      flex: 1 1 auto;
      min-height: 0;
      overflow: auto;
      overscroll-behavior: contain;
      display: grid;
      align-content: start;
      gap: 6px;
      padding: 1px 3px 2px 1px;
      scrollbar-gutter: stable;
    }
    .dsm-preset-grid { grid-template-columns: repeat(auto-fill, minmax(184px, 1fr)); }
    .dsm-character-grid { grid-template-columns: repeat(auto-fill, minmax(148px, 1fr)); }
    .dsm-library-actions { flex: 0 0 auto; margin: 0; padding-top: 1px; }
    .dsm-library-empty { grid-column: 1 / -1; padding: 24px 10px; text-align: center; opacity: .6; }
    .dsm-library [hidden] { display: none !important; }
    .dsm-preset-tile, .dsm-character-tile {
      color: var(--input-text, #ddd);
      background: var(--comfy-input-bg, #222);
      border: 1px solid rgba(128,128,128,.42);
      border-radius: 7px;
      cursor: pointer;
      min-width: 0;
      content-visibility: auto;
      contain-intrinsic-size: 80px;
    }
    .dsm-preset-tile {
      display: grid;
      grid-template-columns: 58px minmax(0, 1fr);
      gap: 7px;
      min-height: 72px;
      padding: 6px;
      align-items: stretch;
    }
    .dsm-preset-body { min-width: 0; display: flex; flex-direction: column; justify-content: center; gap: 2px; }
    .dsm-preset-name, .dsm-character-name {
      font-size: 13px;
      font-weight: 700;
      line-height: 1.25;
      overflow-wrap: anywhere;
      word-break: break-word;
      white-space: normal;
    }
    .dsm-preset-meta { font-size: 11px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .dsm-preset-thumb, .dsm-thumb, .dsm-large-thumb {
      display: flex;
      align-items: center;
      justify-content: center;
      border: 1px solid rgba(128,128,128,.32);
      border-radius: 6px;
      overflow: hidden;
      background: rgba(0,0,0,.22);
      color: rgba(220,220,220,.72);
    }
    .dsm-preset-thumb { width: 58px; min-height: 58px; font-size: 16px; font-weight: 700; }
    .dsm-preset-thumb img, .dsm-thumb img, .dsm-large-thumb img { width: 100%; height: 100%; object-fit: cover; display: block; }
    .dsm-character-tile {
      display: flex;
      flex-direction: column;
      align-items: stretch;
      gap: 4px;
      text-align: left;
      min-height: 132px;
      padding: 6px;
      contain-intrinsic-size: 132px;
    }
    .dsm-thumb { height: 62px; font-size: 20px; font-weight: 700; }
    .dsm-character-tile:focus, .dsm-preset-tile:focus { outline: 1px solid rgba(180,210,255,.75); outline-offset: 1px; }
    .dsm-root button.selected, .dsm-character-tile.selected, .dsm-preset-tile.selected {
      border-color: rgba(180,210,255,.86);
      background: rgba(100,140,220,.22);
    }
    .dsm-character-tile.queued:not(.selected) { border-color: rgba(180,210,255,.55); background: rgba(100,140,220,.12); }
    .dsm-character-tile.prepared:not(.selected) { border-color: rgba(128,128,128,.62); background: rgba(255,255,255,.035); }
    .dsm-character-queue { display: flex; align-items: center; gap: 5px; margin-top: auto; opacity: .86; cursor: pointer; }
    .dsm-character-queue input { margin: 0; }
    .dsm-editor {
      min-height: 0;
      min-width: 0;
      overflow: auto;
      display: flex;
      flex-direction: column;
      gap: 8px;
      scrollbar-gutter: stable;
    }
    .dsm-editor > .dsm-tabs { position: sticky; top: -8px; z-index: 2; padding: 1px 0 7px; background: var(--comfy-menu-bg, #1b1b1b); background: color-mix(in srgb, var(--comfy-menu-bg, #1b1b1b) 92%, transparent); }
    .dsm-editor-content { display: flex; flex-direction: column; gap: 8px; min-width: 0; }
    .dsm-toolbar { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; margin-bottom: 6px; }
    .dsm-tabs { display: flex; gap: 5px; flex-wrap: wrap; }
    .dsm-muted { opacity: .68; white-space: pre-wrap; }
    .dsm-status { border-top: 1px solid rgba(128,128,128,.25); padding-top: 6px; opacity: .78; }
    .dsm-warning { margin-top: 6px; border: 1px solid rgba(255,190,90,.7); border-radius: 6px; padding: 6px; background: rgba(255,170,0,.12); color: var(--input-text, #f2e6cc); display: flex; gap: 8px; align-items: flex-start; justify-content: space-between; }
    .dsm-warning span { min-width: 0; }
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
    .dsm-root button.dsm-primary { border-color: rgba(145,180,240,.66); background: rgba(70,110,185,.24); }
    .dsm-root textarea { width: 100%; min-height: 86px; resize: vertical; }
    .dsm-root input[type="checkbox"] { width: auto; }
    .dsm-labelled { display: flex; flex-direction: column; gap: 4px; min-width: 0; }
    .dsm-labelled > span { opacity: .68; }
    .dsm-large-thumb { min-height: 170px; max-height: 320px; cursor: pointer; }
    .dsm-large-thumb.dragging { border-color: rgba(180,210,255,.9); }
    .dsm-grid2 { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 8px; }
    .dsm-stack-box { border: 1px solid rgba(128,128,128,.28); border-radius: 7px; padding: 7px; display: flex; flex-direction: column; gap: 7px; background: rgba(0,0,0,.10); }
    .dsm-lora-row { display: grid; grid-template-columns: auto minmax(110px, 1fr) 76px 76px auto; gap: 6px; align-items: center; }
    .dsm-prompt-box { display: grid; grid-template-columns: 94px minmax(90px, .8fr) minmax(110px, 1fr) auto; gap: 6px; align-items: end; }
    .dsm-prompt-box .dsm-labelled:last-child { grid-column: 1 / -1; }
    .dsm-lora-summary { white-space: pre-wrap; border: 1px solid rgba(128,128,128,.35); border-radius: 6px; padding: 6px; min-height: 58px; max-height: 190px; overflow: auto; background: rgba(0,0,0,.16); }
    .dsm-queue-options { display: grid; grid-template-columns: minmax(0, 1fr); gap: 6px; }
    .dsm-checkline { display: grid; grid-template-columns: auto minmax(0, 1fr); gap: 7px; align-items: start; border: 1px solid rgba(128,128,128,.24); border-radius: 6px; padding: 6px; background: rgba(255,255,255,.025); }
    .dsm-checkline input { margin-top: 2px; }
    .dsm-checkline-text { display: flex; flex-direction: column; gap: 2px; min-width: 0; }
    .dsm-checkline-text strong { font-weight: 650; }
    .dsm-checkline-text small { opacity: .68; line-height: 1.3; }
    .dsm-queue-character-picker { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 5px; max-height: 260px; overflow: auto; padding: 2px; }
    .dsm-queue-character-row { display: grid; grid-template-columns: auto minmax(0, 1fr) auto; align-items: center; gap: 6px; padding: 5px 6px; border: 1px solid rgba(128,128,128,.24); border-radius: 6px; }
    .dsm-queue-character-row span { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .dsm-queue-character-row small { opacity: .62; }
    .dsm-queue-summary { opacity: .78; white-space: pre-wrap; border-top: 1px solid rgba(128,128,128,.22); padding-top: 6px; }
    @media (max-width: 760px) {
      .dsm-root { overflow: auto; }
      .dsm-main { grid-template-columns: 1fr; grid-template-rows: minmax(360px, 1fr) minmax(320px, auto); overflow: visible; }
      .dsm-editor { overflow: visible; }
      .dsm-library { min-height: 360px; }
      .dsm-grid2 { grid-template-columns: 1fr; }
    }
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

  stateLibraryClient.nodes.add(node);
  const load = async () => {
    const token = (node.__dsmLibraryLoadToken || 0) + 1;
    node.__dsmLibraryLoadToken = token;
    const currentWidgets = getWidgets(node);
    const legacy = parseLegacyEmbeddedState(widgetValue(currentWidgets.stateWidget, ""));
    setWidgetValue(currentWidgets.stateWidget, serializeBinding());
    node.properties = node.properties || {};
    delete node.properties.dora_state_manager;
    delete node.properties.dora_state_manager_backup_node_uid;
    const characterId = String(widgetValue(currentWidgets.characterWidget, "") || "");
    const promptId = String(widgetValue(currentWidgets.promptWidget, "") || "");
    try {
      await initializeStateLibrary(node, legacy, characterId, promptId, token);
    } catch (err) {
      if (node.__dsmLibraryLoadToken !== token) return;
      const fallback = stateViewForSelection(characterId, promptId);
      const uiState = normalizeUiState(widgetValue(currentWidgets.uiStateWidget, serializeUiState(defaultUiState())));
      cacheRenderableState(node, fallback, {
        ...uiState,
        status: `State Manager library load failed: ${err?.message || err}`,
      });
      scheduleRender(node);
    }
  };
  void load();

  chainNodeCallback(node, "onConfigure", function () {
    void Promise.resolve().then(load);
  });

  chainNodeCallback(node, "onResize", function () {
    syncDomWidgetSize(node, widget);
    scheduleRender(node);
  });

  chainNodeCallback(node, "onRemoved", function () {
    stateLibraryClient.nodes.delete(node);
    const ctx = node.__dsm;
    if (!ctx?.renderFrame) return;
    cancelAnimationFrame(ctx.renderFrame);
    ctx.renderFrame = 0;
  });

  scheduleRender(node);
  return widget;
}


function queueSessionTotalFromArguments(number, batchCount) {
  const batch = Number(batchCount);
  if (Number.isFinite(batch) && batch > 0) return Math.max(1, Math.floor(batch));
  const count = Number(number);
  if (Number.isFinite(count) && count > 1) return Math.max(1, Math.floor(count));
  return 1;
}

function startDsmQueueSession(number, batchCount) {
  dsmQueueSession.active = true;
  dsmQueueSession.total = queueSessionTotalFromArguments(number, batchCount);
  dsmQueueSession.nextIndex = 0;
  dsmQueueSession.startedAt = Date.now();
  dsmQueueSession.promptPools = new Map();
}

function finishDsmQueueSession(startedAt) {
  setTimeout(() => {
    if (dsmQueueSession.startedAt === startedAt) dsmQueueSession.active = false;
  }, 0);
}

function nextDsmQueueIndex(index) {
  const now = Date.now();
  if (!dsmQueueSession.active || now - dsmQueueSession.startedAt > QUEUE_SESSION_MAX_AGE_MS) {
    dsmQueueSession.active = true;
    dsmQueueSession.total = 1;
    dsmQueueSession.nextIndex = 0;
    dsmQueueSession.startedAt = now;
    dsmQueueSession.promptPools = new Map();
  }
  const queueIndex = dsmQueueSession.nextIndex;
  dsmQueueSession.nextIndex += 1;
  return queueIndex;
}

function queueOutputKeysForNode(promptPayload, node) {
  const output = promptPayload?.output || {};
  const id = String(node?.id ?? "");
  const keys = [];
  for (const key of Object.keys(output)) {
    if (key === id || key.endsWith(`:${id}`)) keys.push(key);
  }
  if (id && !keys.includes(id)) keys.push(id);
  return [...new Set(keys)];
}

function findWorkflowNodeForPromptNode(promptPayload, node, promptId = null) {
  const workflowNodes = promptPayload?.workflow?.nodes;
  if (!Array.isArray(workflowNodes)) return null;
  const idText = String(promptId ?? node?.id ?? "");
  return workflowNodes.find((item) => String(item?.id) === idText)
    || workflowNodes.find((item) => String(item?.id) === String(node?.id ?? ""))
    || null;
}

function syncQueuedWorkflowWidget(promptPayload, node, widget, value, promptId = null) {
  if (!widget) return false;
  const workflowNode = findWorkflowNodeForPromptNode(promptPayload, node, promptId);
  const index = (node?.widgets || []).indexOf(widget);
  if (!workflowNode || !Array.isArray(workflowNode.widgets_values) || index < 0) return false;
  workflowNode.widgets_values[index] = value;
  return true;
}

function setQueuedInput(promptPayload, node, inputName, value, { addIfMissing = false, syncWidget = true } = {}) {
  if (!promptPayload?.output || !node || !inputName) return 0;
  let changed = 0;
  const widget = syncWidget ? widgetByExactName(node, inputName) : null;
  for (const promptId of queueOutputKeysForNode(promptPayload, node)) {
    const outputNode = promptPayload.output?.[promptId];
    const inputs = outputNode?.inputs;
    if (!inputs) continue;
    if (!addIfMissing && !Object.prototype.hasOwnProperty.call(inputs, inputName)) continue;
    inputs[inputName] = structuredCloneCompat(value);
    syncQueuedWorkflowWidget(promptPayload, node, widget, value, promptId);
    changed += 1;
  }
  return changed;
}

function setQueuedWidgetInput(promptPayload, node, widgetName, value) {
  return setQueuedInput(promptPayload, node, widgetName, value, { addIfMissing: false, syncWidget: false });
}

function readQueuedInput(promptPayload, node, inputName) {
  if (!promptPayload?.output || !node || !inputName) return undefined;
  for (const promptId of queueOutputKeysForNode(promptPayload, node)) {
    const inputs = promptPayload.output?.[promptId]?.inputs;
    if (inputs && Object.prototype.hasOwnProperty.call(inputs, inputName)) return inputs[inputName];
  }
  return undefined;
}

function readQueuedStateSeed(promptPayload, node) {
  const raw = readQueuedInput(promptPayload, node, "seed");
  if (raw === undefined) return null;
  const seed = normalizeSeedInteger(raw, STATE_SEED_RANDOM);
  return STATE_SEED_SPECIALS.includes(seed) ? null : seed;
}

function firstControlledStateSeed(managerNode, promptPayload) {
  const controlled = getControlledNodes(managerNode).filter((node) => node && node !== managerNode && isStateSeedNode(node));
  for (const node of controlled) {
    const seed = readQueuedStateSeed(promptPayload, node);
    if (seed != null) return { node, seed };
  }
  return null;
}

function serializeQueuedUiStateOverride(uiState, characterId, promptId, runtimeSeed, queueIndex, total) {
  return JSON.stringify({
    ...safeJsonParse(serializeWorkflowUiState(uiState), {}),
    ...(runtimeSeed == null ? {} : { __dsm_runtime_seed: runtimeSeed }),
    __dsm_queued_runtime_character_id: String(characterId ?? ""),
    __dsm_queued_runtime_prompt_id: String(promptId ?? ""),
    __dsm_queued_runtime_queue_index: Math.max(0, Number(queueIndex) || 0),
    __dsm_queued_runtime_queue_total: Math.max(1, Number(total) || 1),
    // The runtime override must differ per queued prompt even if the same prompt
    // is selected again after a prompt-pool reshuffle. This prevents executor
    // cache reuse from keeping a stale STATE_MANAGER_CONTROL payload alive.
    __dsm_queued_runtime_nonce: `${Date.now()}:${Math.random()}:${queueIndex}`,
  }, null, 0);
}

function setQueuedStateManagerRuntimeState(promptPayload, node, uiState, characterId, promptId, runtimeSeed, queueIndex, total) {
  return setQueuedInput(
    promptPayload,
    node,
    UI_STATE_WIDGET,
    serializeQueuedUiStateOverride(uiState, characterId, promptId, runtimeSeed, queueIndex, total),
    { addIfMissing: true, syncWidget: false }
  );
}

function setQueuedStateManagerSelection(promptPayload, node, characterId, promptId) {
  let changed = 0;
  changed += setQueuedInput(promptPayload, node, SELECTED_CHARACTER_WIDGET, characterId, { addIfMissing: false, syncWidget: false });
  changed += setQueuedInput(promptPayload, node, SELECTED_PROMPT_WIDGET, promptId, { addIfMissing: false, syncWidget: false });
  return changed;
}

function shuffledPromptPool(prompts, avoidPromptId = "") {
  const pool = [...prompts];
  for (let i = pool.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [pool[i], pool[j]] = [pool[j], pool[i]];
  }
  if (pool.length > 1 && avoidPromptId && pool[0]?.id === avoidPromptId) {
    const swapIndex = pool.findIndex((prompt, index) => index > 0 && prompt?.id !== avoidPromptId);
    if (swapIndex > 0) [pool[0], pool[swapIndex]] = [pool[swapIndex], pool[0]];
  }
  return pool;
}

function nextQueuedPromptFromPool(managerNode, character, prompts) {
  if (!Array.isArray(prompts) || !prompts.length) return null;
  if (prompts.length === 1) return prompts[0];
  if (!(dsmQueueSession.promptPools instanceof Map)) dsmQueueSession.promptPools = new Map();
  const key = `${managerNode?.id ?? "manager"}:${character?.id ?? "character"}`;
  let entry = dsmQueueSession.promptPools.get(key);
  const promptIds = prompts.map((prompt) => String(prompt?.id ?? "")).join("\n");
  if (!entry || entry.promptIds !== promptIds || !Array.isArray(entry.pool) || !entry.pool.length) {
    entry = { promptIds, lastId: entry?.lastId || "", pool: shuffledPromptPool(prompts, entry?.lastId || "") };
  }
  const prompt = entry.pool.shift() || prompts[0];
  entry.lastId = String(prompt?.id ?? "");
  if (!entry.pool.length) entry.pool = shuffledPromptPool(prompts, entry.lastId);
  dsmQueueSession.promptPools.set(key, entry);
  return prompt;
}

function selectQueuedCharacterAndPrompt(managerNode, state, uiState, currentCharacterId, currentPromptId, queueIndex, total) {
  const current = selectedCharacter(state, currentCharacterId) || state.characters[0];
  let character = current;
  if (uiState.queue_character_wildcard) {
    const ids = validQueueCharacterIds(state, uiState, current?.id || currentCharacterId);
    if (ids.length) {
      const safeTotal = Math.max(1, Number(total) || 1);
      const chunkIndex = Math.min(ids.length - 1, Math.floor((Math.max(0, queueIndex) * ids.length) / safeTotal));
      character = selectedCharacter(state, ids[chunkIndex]) || current;
    }
  }

  let prompt = selectedPrompt(character, character?.id === current?.id ? currentPromptId : "") || character?.prompts?.[0];
  const prompts = Array.isArray(character?.prompts) ? character.prompts : [];
  if (uiState.queue_prompt_wildcard && prompts.length) {
    prompt = nextQueuedPromptFromPool(managerNode, character, prompts) || prompt;
  }
  return { character, prompt };
}

function buildQueuedDoraStatePayload(character, prompt) {
  syncLegacyLoaderMirror(character);
  syncPromptTextMirror(prompt);
  const loaderStacks = getCharacterLoaderStacks(character).map((stack) => structuredCloneCompat(stack));
  const defaultStack = findCharacterLoaderStack(character, "default") || loaderStacks[0] || defaultLoaderStack();
  const textBoxes = normalizePromptTextBoxes(prompt);
  const positiveBox = findPromptTextBox(prompt, "positive", "default") || textBoxes.find((box) => box.role === "positive");
  const negativeBox = findPromptTextBox(prompt, "negative", "default") || textBoxes.find((box) => box.role === "negative");
  const referenceImage = normalizeThumbnail(prompt?.reference_image);
  const fileimagePrefix = String(prompt?.fileimage_prefix ?? "").trim();
  return {
    version: 2,
    kind: "dora_state_manager_state",
    character: {
      id: String(character?.id ?? ""),
      name: String(prompt?.name ?? character?.name ?? ""),
      thumbnail: normalizeThumbnail(character?.thumbnail),
    },
    prompt: {
      id: String(prompt?.id ?? ""),
      name: String(prompt?.name ?? ""),
      reference_image: referenceImage,
      fileimage_prefix: fileimagePrefix,
    },
    loader_stacks: loaderStacks,
    loras: structuredCloneCompat(defaultStack?.loras || []),
    loader_globals: normalizeLoaderGlobals(defaultStack?.loader_globals || character?.loader_globals || {}),
    settings: normalizeSettings(prompt?.settings || {}),
    text_boxes: textBoxes.map((box) => structuredCloneCompat(box)),
    positive_prompt: String(positiveBox?.text ?? prompt?.positive ?? ""),
    negative_prompt: String(negativeBox?.text ?? prompt?.negative ?? ""),
    reference_image: referenceImage,
    fileimage_prefix: fileimagePrefix,
  };
}

function buildQueuedDoraLoaderStatePayload(character, loader = null, fallbackIndex = 0) {
  syncLegacyLoaderMirror(character);
  const slot = loader ? getDoraLoaderSlot(loader, fallbackIndex) : "default";
  const stack = findCharacterLoaderStack(character, slot, { allowFallback: false })
    || findCharacterLoaderStack(character, "default", { allowFallback: true })
    || defaultLoaderStack();
  const normalizedStack = normalizeLoaderStack({
    ...structuredCloneCompat(stack),
    slot,
    label: stack?.label || slot,
  });
  const globals = normalizeLoaderGlobals(normalizedStack.loader_globals || {});

  // Loader-only runtime state. Keep prompts, settings, seeds, text boxes,
  // thumbnails, reference images, and unrelated loader stacks out of the DoRA
  // loader's state_control payload so prompt-only queue wildcarding cannot
  // invalidate/reload LoRA patches.
  return {
    version: 2,
    kind: DORA_STATE_KIND,
    loader_stacks: [{
      slot: normalizedStack.slot,
      label: normalizedStack.label,
      loras: structuredCloneCompat(normalizedStack.loras || []),
      loader_globals: globals,
    }],
    loras: structuredCloneCompat(normalizedStack.loras || []),
    loader_globals: globals,
  };
}

function buildQueuedDoraLoaderControlPayload(character, loader = null, fallbackIndex = 0) {
  const state = buildQueuedDoraLoaderStatePayload(character, loader, fallbackIndex);
  return {
    version: 2,
    kind: STATE_MANAGER_CONTROL_KIND,
    state,
    loader_stacks: structuredCloneCompat(state.loader_stacks || []),
    loras: structuredCloneCompat(state.loras || []),
    loader_globals: structuredCloneCompat(state.loader_globals || {}),
  };
}

function applyRuntimeSeedToSettings(settings, seed) {
  const normalized = normalizeSettings(settings);
  normalized.seed = seed;
  if (normalized.rgthree_seed) {
    normalized.rgthree_seed = structuredCloneCompat(normalized.rgthree_seed);
    normalized.rgthree_seed.seed = seed;
    if (normalized.rgthree_seed.widgets && typeof normalized.rgthree_seed.widgets === "object") {
      for (const name of ["seed", "noise_seed", "value"]) {
        if (Object.prototype.hasOwnProperty.call(normalized.rgthree_seed.widgets, name)) {
          normalized.rgthree_seed.widgets[name] = seed;
        }
      }
    }
  }
  if (Array.isArray(normalized.nodes)) {
    normalized.nodes = normalizeNodeSnapshots(normalized.nodes).map((snapshot) => {
      const next = structuredCloneCompat(snapshot);
      if (!next.widgets || typeof next.widgets !== "object") return next;
      for (const name of ["seed", "noise_seed", "value"]) {
        if (Object.prototype.hasOwnProperty.call(next.widgets, name)) next.widgets[name] = seed;
      }
      if (next.is_seed_node || next.seed != null || Object.keys(next.seed_widgets || {}).length) {
        next.seed = seed;
        if (next.seed_widgets && typeof next.seed_widgets === "object") {
          for (const name of Object.keys(next.seed_widgets)) next.seed_widgets[name] = seed;
        }
      }
      return next;
    });
  }
  return normalized;
}

function withRuntimeSeed(payload, seed) {
  return {
    ...structuredCloneCompat(payload),
    settings: applyRuntimeSeedToSettings(payload?.settings, seed),
  };
}

function mutateQueuedDoraLoaders(promptPayload, managerNode, character) {
  const controlled = getControlledNodes(managerNode).filter((node) => node && node !== managerNode);
  const controlledLoaders = controlled.filter(isDoraLoaderNode);
  const legacyLoaders = controlledLoaders.length ? [] : uniqueNodes(getOutputTargets(managerNode, OUTPUT_NAMES.lora)).filter(isDoraLoaderNode);
  const loaders = controlledLoaders.length ? controlledLoaders : legacyLoaders;
  if (!loaders.length) return 0;
  let changed = 0;
  for (const [index, loader] of loaders.entries()) {
    const loaderControlPayload = buildQueuedDoraLoaderControlPayload(character, loader, index);

    // state_control is the intended State Manager edge. During queued wildcarding,
    // keep using state_control, but replace the queued API input with a loader-only
    // control payload. The editor graph remains connected through state_control.
    changed += setQueuedInput(promptPayload, loader, "state_control", loaderControlPayload, { addIfMissing: true, syncWidget: false });
  }
  return changed;
}

function mutateQueuedStateTextBoxes(promptPayload, managerNode, prompt) {
  const controlled = getControlledNodes(managerNode).filter((node) => node && node !== managerNode);
  const textNodes = controlled.filter(isStateTextNode);
  let changed = 0;
  for (const [index, textNode] of textNodes.entries()) {
    const role = getStateTextRole(textNode);
    const slot = getStateTextSlot(textNode, role, `${role}_${textNode?.id ?? index + 1}`);
    const saved = findPromptTextBox(prompt, role, slot, { allowRoleFallback: false }) || findPromptTextBox(prompt, role, slot, { allowRoleFallback: true });
    if (!saved) continue;
    changed += setQueuedWidgetInput(promptPayload, textNode, "text", String(saved.text ?? ""));
  }
  return changed;
}

function mutateQueuedLegacyTextTargets(promptPayload, managerNode, prompt) {
  let changed = 0;
  const legacyTargets = [
    { targets: getOutputTargets(managerNode, OUTPUT_NAMES.positive), role: "positive", text: getPromptText(prompt, "positive", "default") },
    { targets: getOutputTargets(managerNode, OUTPUT_NAMES.negative), role: "negative", text: getPromptText(prompt, "negative", "default") },
  ];
  for (const group of legacyTargets) {
    for (const target of group.targets) {
      const inputName = target.inputName || (group.role === "negative" ? "negative" : "text");
      changed += setQueuedInput(promptPayload, target.node, inputName, String(group.text ?? ""), { addIfMissing: true, syncWidget: false });
      if (isImpactWildcardNode(target.node)) {
        changed += setQueuedInput(promptPayload, target.node, "wildcard_text", String(group.text ?? ""), { addIfMissing: true, syncWidget: false });
        changed += setQueuedInput(promptPayload, target.node, "populated_text", String(group.text ?? ""), { addIfMissing: true, syncWidget: false });
      }
    }
  }
  return changed;
}


function mutateQueuedSettingsNodes(promptPayload, managerNode, settingsSource) {
  const controlled = getControlledNodes(managerNode).filter((node) => node && node !== managerNode && !isDoraLoaderNode(node) && !isStateTextNode(node));
  const legacy = controlled.length ? [] : uniqueNodes([
    ...getOutputTargets(managerNode, OUTPUT_NAMES.settings),
    ...getOutputTargets(managerNode, OUTPUT_NAMES.seed),
  ]).filter((node) => node && node !== managerNode && !isDoraLoaderNode(node) && !isStateTextNode(node));
  const targets = controlled.length ? controlled : legacy;
  const settings = normalizeSettings(settingsSource || {});
  let changed = 0;
  for (const node of targets) {
    const preserveLiveStateSeed = isStateSeedNode(node) && readQueuedStateSeed(promptPayload, node) != null;
    const snapshot = findSnapshotForNode(settings, node);
    if (snapshot?.widgets) {
      for (const [name, value] of Object.entries(snapshot.widgets)) {
        if (preserveLiveStateSeed && String(name) === "seed") continue;
        changed += setQueuedInput(promptPayload, node, String(name), value, { addIfMissing: false, syncWidget: false });
      }
    }
    if (isSeedNode(node) && !preserveLiveStateSeed) {
      const seed = extractSeedFromSettings(settings);
      if (seed != null) {
        for (const name of ["seed", "noise_seed", "value"]) {
          const before = changed;
          changed += setQueuedInput(promptPayload, node, name, seed, { addIfMissing: false, syncWidget: false });
          if (changed !== before) break;
        }
      }
    }
  }
  return changed;
}

function mutatePromptForStateManagers(promptPayload, queueIndex, total) {
  if (!promptPayload?.output) return 0;
  let changed = 0;
  const graphNodes = app?.graph?._nodes || [];
  for (const node of graphNodes) {
    if (!isStateManagerNode(node)) continue;
    const widgets = getWidgets(node);
    const { state, uiState } = getCurrentState(node);
    if (!uiState.queue_prompt_wildcard && !uiState.queue_character_wildcard && !uiState.queue_randomize_saved_seed) continue;
    const currentCharacterId = String(widgetValue(widgets.characterWidget, "") || "");
    const currentPromptId = String(widgetValue(widgets.promptWidget, "") || "");
    const { character, prompt } = selectQueuedCharacterAndPrompt(node, state, uiState, currentCharacterId, currentPromptId, queueIndex, total);
    if (!character || !prompt) continue;
    let payload = buildQueuedDoraStatePayload(character, prompt);
    let runtimeSeed = null;
    const liveStateSeed = firstControlledStateSeed(node, promptPayload);
    if (liveStateSeed) {
      runtimeSeed = liveStateSeed.seed;
      payload = withRuntimeSeed(payload, runtimeSeed);
    } else if (uiState.queue_randomize_saved_seed) {
      runtimeSeed = generateStateManagerRuntimeSeed();
      payload = withRuntimeSeed(payload, runtimeSeed);
    }
    changed += setQueuedStateManagerSelection(promptPayload, node, character.id, prompt.id);
    changed += setQueuedStateManagerRuntimeState(
      promptPayload,
      node,
      uiState,
      character.id,
      prompt.id,
      runtimeSeed,
      queueIndex,
      total,
    );
    changed += mutateQueuedDoraLoaders(promptPayload, node, character);
    changed += mutateQueuedStateTextBoxes(promptPayload, node, prompt);
    changed += mutateQueuedLegacyTextTargets(promptPayload, node, prompt);
    changed += mutateQueuedSettingsNodes(promptPayload, node, payload.settings);
  }
  return changed;
}


function getStateSeedWidget(node) {
  return widgetByExactName(node, "seed") || (node?.widgets || []).find((widget) => /seed/i.test(`${widget?.name ?? ""} ${widget?.label ?? ""}`)) || null;
}

function setStateSeedWidgetValue(node, value) {
  const widget = getStateSeedWidget(node);
  if (!widget) return false;
  return setNodeWidget(node, widget, normalizeSeedInteger(value, STATE_SEED_RANDOM));
}

function ensureStateSeedRandomRange(node) {
  node.properties = node.properties || {};
  const max = normalizeSeedInteger(node.properties.dora_state_seed_random_max ?? STATE_SEED_MAX, STATE_SEED_MAX);
  const min = normalizeSeedInteger(node.properties.dora_state_seed_random_min ?? 0, 0);
  node.properties.dora_state_seed_random_max = Math.max(min + 1, Math.min(STATE_SEED_MAX, max));
  node.properties.dora_state_seed_random_min = Math.max(0, Math.min(node.properties.dora_state_seed_random_max - 1, min));
}

function generateStateSeedRandom(node) {
  const seedWidget = getStateSeedWidget(node);
  ensureStateSeedRandomRange(node);
  const step = Math.max(1, Number(seedWidget?.options?.step || 1));
  const randomMin = Number(node.properties.dora_state_seed_random_min ?? 0);
  const randomMax = Number(node.properties.dora_state_seed_random_max ?? STATE_SEED_MAX);
  const randomRange = Math.max(1, (randomMax - randomMin) / (step / 10));
  for (let attempt = 0; attempt < 10; attempt++) {
    const seed = normalizeSeedInteger(Math.floor(Math.random() * randomRange) * (step / 10) + randomMin, 0);
    if (!STATE_SEED_SPECIALS.includes(seed)) return seed;
  }
  return 0;
}

function getStateSeedToUse(node) {
  const seedWidget = getStateSeedWidget(node);
  const inputSeed = normalizeSeedInteger(seedWidget?.value, STATE_SEED_RANDOM);
  let seedToUse = null;
  if (STATE_SEED_SPECIALS.includes(inputSeed)) {
    if (typeof node.__dsmLastSeed === "number" && !STATE_SEED_SPECIALS.includes(node.__dsmLastSeed)) {
      if (inputSeed === STATE_SEED_INCREMENT) seedToUse = node.__dsmLastSeed + 1;
      else if (inputSeed === STATE_SEED_DECREMENT) seedToUse = node.__dsmLastSeed - 1;
    }
    if (seedToUse == null || STATE_SEED_SPECIALS.includes(seedToUse)) seedToUse = generateStateSeedRandom(node);
  }
  return normalizeSeedInteger(seedToUse ?? inputSeed, STATE_SEED_RANDOM);
}

function updateStateSeedLastButton(node, seedToUse) {
  const seedWidget = getStateSeedWidget(node);
  const button = node.__dsmLastSeedButton;
  if (!button) return;
  if (seedToUse !== normalizeSeedInteger(seedWidget?.value, STATE_SEED_RANDOM)) {
    button.name = `♻️ ${seedToUse}`;
    button.label = button.name;
    button.disabled = false;
  } else {
    button.name = LAST_SEED_BUTTON_LABEL;
    button.label = LAST_SEED_BUTTON_LABEL;
    button.disabled = true;
  }
}

function findWorkflowNodeForStateSeed(promptPayload, node, promptId) {
  const workflowNodes = promptPayload?.workflow?.nodes;
  if (!Array.isArray(workflowNodes)) return null;
  const idText = String(promptId ?? node?.id ?? "");
  return workflowNodes.find((item) => String(item?.id) === idText)
    || workflowNodes.find((item) => String(item?.id) === String(node?.id ?? "") && hasAnyNodeClassOrTitle(node, [STATE_SEED_CLASS, STATE_SEED_DISPLAY_CLASS]))
    || null;
}

function mutatePromptForStateSeedNode(promptPayload, node) {
  if (!promptPayload || !node || !isStateSeedNode(node)) return false;
  const idCandidates = [String(node.id ?? "")];
  for (const key of Object.keys(promptPayload.output || {})) {
    if (key === String(node.id ?? "") || key.endsWith(`:${node.id}`)) idCandidates.push(key);
  }
  let changed = false;
  const seedToUse = getStateSeedToUse(node);
  for (const promptId of [...new Set(idCandidates)]) {
    const outputNode = promptPayload.output?.[promptId];
    const outputInputs = outputNode?.inputs;
    if (!outputInputs || !Object.prototype.hasOwnProperty.call(outputInputs, "seed")) continue;
    outputInputs.seed = seedToUse;

    const workflowNode = findWorkflowNodeForStateSeed(promptPayload, node, promptId);
    const seedWidget = getStateSeedWidget(node);
    const seedWidgetIndex = seedWidget ? (node.widgets || []).indexOf(seedWidget) : 0;
    if (workflowNode && Array.isArray(workflowNode.widgets_values) && seedWidgetIndex >= 0) {
      workflowNode.widgets_values[seedWidgetIndex] = seedToUse;
    }

    changed = true;
  }
  if (changed) {
    node.__dsmLastSeed = seedToUse;
    updateStateSeedLastButton(node, seedToUse);
  }
  return changed;
}

function mutatePromptForStateSeeds(promptPayload) {
  if (!promptPayload?.output) return 0;
  let changed = 0;
  const graphNodes = app?.graph?._nodes || [];
  for (const node of graphNodes) {
    if (mutatePromptForStateSeedNode(promptPayload, node)) changed += 1;
  }
  return changed;
}

function initializeStateSeedNode(node) {
  if (node.__dsmStateSeedInitialized) return;
  node.__dsmStateSeedInitialized = true;
  node.serialize_widgets = true;
  ensureStateSeedRandomRange(node);

  const seedWidget = getStateSeedWidget(node);
  if (seedWidget) {
    seedWidget.options = seedWidget.options || {};
    seedWidget.options.min = STATE_SEED_MIN;
    seedWidget.options.max = STATE_SEED_MAX;
    seedWidget.options.step = seedWidget.options.step || 1;
    if (seedWidget.value == null) seedWidget.value = STATE_SEED_RANDOM;
  }

  for (let i = (node.widgets || []).length - 1; i >= 0; i--) {
    if (String(node.widgets[i]?.name ?? "") === "control_after_generate") node.widgets.splice(i, 1);
  }

  if (!node.__dsmStateSeedButtonsAdded) {
    node.__dsmStateSeedButtonsAdded = true;
    node.addWidget?.("button", "🎲 Randomize Each Time", "", () => {
      setStateSeedWidgetValue(node, STATE_SEED_RANDOM);
    }, { serialize: false });
    node.addWidget?.("button", "🎲 New Fixed Random", "", () => {
      setStateSeedWidgetValue(node, generateStateSeedRandom(node));
    }, { serialize: false });
    node.__dsmLastSeedButton = node.addWidget?.("button", LAST_SEED_BUTTON_LABEL, "", () => {
      if (typeof node.__dsmLastSeed === "number") setStateSeedWidgetValue(node, node.__dsmLastSeed);
      if (node.__dsmLastSeedButton) {
        node.__dsmLastSeedButton.name = LAST_SEED_BUTTON_LABEL;
        node.__dsmLastSeedButton.label = LAST_SEED_BUTTON_LABEL;
        node.__dsmLastSeedButton.disabled = true;
      }
    }, { width: 50, serialize: false });
    if (node.__dsmLastSeedButton) node.__dsmLastSeedButton.disabled = true;
  }

  chainNodeCallback(node, "onPropertyChanged", function (property, value) {
    if (property === "dora_state_seed_random_min" || property === "dora_state_seed_random_max") ensureStateSeedRandomRange(node);
    return true;
  });
}

function patchStateSeedNodeDef(nodeType) {
  if (nodeType.prototype.__dsmStateSeedPatched) return;
  nodeType.prototype.__dsmStateSeedPatched = true;
  nodeType["@dora_state_seed_random_min"] = { type: "number" };
  nodeType["@dora_state_seed_random_max"] = { type: "number" };

  const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
  nodeType.prototype.onNodeCreated = function () {
    const result = originalOnNodeCreated?.apply(this, arguments);
    initializeStateSeedNode(this);
    return result;
  };

  const originalOnConfigure = nodeType.prototype.onConfigure;
  nodeType.prototype.onConfigure = function () {
    const result = originalOnConfigure?.apply(this, arguments);
    initializeStateSeedNode(this);
    return result;
  };
}

function installStateSeedQueuePatch() {
  if (api.__dsmStateSeedQueuePatchInstalled || typeof api.queuePrompt !== "function") return;
  api.__dsmStateSeedQueuePatchInstalled = true;

  if (!app.__dsmQueueSessionPatchInstalled && typeof app.queuePrompt === "function") {
    app.__dsmQueueSessionPatchInstalled = true;
    const originalAppQueuePrompt = app.queuePrompt;
    app.queuePrompt = async function (number, batchCount, ...args) {
      startDsmQueueSession(number, batchCount);
      const startedAt = dsmQueueSession.startedAt;
      try {
        return await originalAppQueuePrompt.apply(this, [number, batchCount, ...args]);
      } finally {
        finishDsmQueueSession(startedAt);
      }
    };
  }

  const originalQueuePrompt = api.queuePrompt;
  api.queuePrompt = async function (index, promptPayload, ...args) {
    const queueIndex = nextDsmQueueIndex(index);
    const total = Math.max(1, dsmQueueSession.total || 1);
    try {
      mutatePromptForStateSeeds(promptPayload);
      mutatePromptForStateManagers(promptPayload, queueIndex, total);
    } catch (err) {
      console.warn(`[${EXT_NAME}] failed to resolve State Manager queue values before queue`, err);
    }
    return originalQueuePrompt.apply(this, [index, promptPayload, ...args]);
  };
}

function initializeStateTextNode(node) {
  if (node.__dsmStateTextInitialized) return;
  node.__dsmStateTextInitialized = true;

  const sync = () => {
    try {
      syncStateTextNodeDownstream(node);
    } catch (err) {
      console.warn(`[${EXT_NAME}] failed to sync State Manager Text Box downstream widgets`, err);
    }
  };

  chainNodeCallback(node, "onWidgetChanged", function (name, value, oldValue, widget) {
    const widgetName = String(name ?? widget?.name ?? "").toLowerCase();
    if (widgetName === "text" || widgetName === "role" || widgetName === STATE_TEXT_SLOT_WIDGET) sync();
  });

  chainNodeCallback(node, "onConnectionsChange", function () {
    sync();
  });

  const textWidget = widgetByExactName(node, "text");
  if (textWidget && !textWidget.__dsmStateTextBeforeQueued) {
    textWidget.__dsmStateTextBeforeQueued = true;
    const originalBeforeQueued = textWidget.beforeQueued;
    textWidget.beforeQueued = function () {
      const result = originalBeforeQueued?.apply(this, arguments);
      sync();
      return result;
    };
  }

  requestAnimationFrame(sync);
}

function patchStateTextNodeDef(nodeType) {
  if (nodeType.prototype.__dsmStateTextPatched) return;
  nodeType.prototype.__dsmStateTextPatched = true;

  const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
  nodeType.prototype.onNodeCreated = function () {
    const result = originalOnNodeCreated?.apply(this, arguments);
    initializeStateTextNode(this);
    return result;
  };

  const originalOnConfigure = nodeType.prototype.onConfigure;
  nodeType.prototype.onConfigure = function () {
    const result = originalOnConfigure?.apply(this, arguments);
    initializeStateTextNode(this);
    requestAnimationFrame(() => syncStateTextNodeDownstream(this));
    return result;
  };
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
        node.__dsm = {
          root,
          renderFrame: 0,
          state: null,
          uiState: null,
          libraryMode: "presets",
          searchQuery: "",
          scrollPositions: {},
          focusLibrarySelection: false,
          libraryElement: null,
          libraryKey: "",
        };
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
    installStateSeedQueuePatch();
    maybeInjectWidgetInput(nodeData);
    if (isStateTextNodeDef(nodeData, nodeType)) patchStateTextNodeDef(nodeType);
    if (isStateSeedNodeDef(nodeData, nodeType)) patchStateSeedNodeDef(nodeType);
    if (!isTargetNode(nodeData, nodeType)) return;
    const originalOnSerialize = nodeType.prototype.onSerialize;
    nodeType.prototype.onSerialize = function (o) {
      const result = originalOnSerialize?.apply(this, arguments);
      try {
        const widgets = getWidgets(this);
        const snapshot = getCurrentState(this);
        const binding = serializeBinding();
        const workflowUiState = serializeWorkflowUiState(snapshot.uiState);
        setWidgetValue(widgets.stateWidget, binding);
        o.properties = o.properties || {};
        delete o.properties.dora_state_manager;
        delete o.properties.dora_state_manager_backup_node_uid;
        if (this.properties) {
          delete this.properties.dora_state_manager;
          delete this.properties.dora_state_manager_backup_node_uid;
        }
        if (app?.graph?.extra) delete app.graph.extra.dora_state_manager_backup_workflow_id;
        const values = Array.isArray(o.widgets_values) ? o.widgets_values : null;
        const named = o.widgets_values_named && typeof o.widgets_values_named === "object"
          ? o.widgets_values_named
          : null;
        for (const [widget, value] of [
          [widgets.stateWidget, binding],
          [widgets.uiStateWidget, workflowUiState],
        ]) {
          const index = (this.widgets || []).indexOf(widget);
          if (values && index >= 0) values[index] = value;
          if (named && widget?.name) named[widget.name] = value;
        }
      } catch (err) {
        console.warn(`[${EXT_NAME}] serialize sync failed`, err);
      }
      return result;
    };
  },
});
