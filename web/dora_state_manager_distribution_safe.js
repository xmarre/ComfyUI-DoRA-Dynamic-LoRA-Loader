import { app } from "../../scripts/app.js";

const EXT_NAME = "comfyui_dora_dynamic_lora.state_manager_distribution_safe";
const NODE_CLASSES = new Set(["State Manager", "DoRA State Manager"]);
const PROPERTY = "dora_state_manager_distribution_safe_serialization";
const UI_STATE_WIDGET = "ui_state_json";
const SELECTED_CHARACTER_WIDGET = "selected_character_id";
const SELECTED_PROMPT_WIDGET = "selected_prompt_id";
const DEFAULT_CHARACTER_ID = "default_character";
const DEFAULT_PROMPT_ID = "default_prompt";
const SELECTION_MIRROR_PROPERTY = "dora_state_manager_selection_v1";
const CONTROL_ATTR = "data-dsm-distribution-safe-control";

function isStateManagerDef(nodeData, nodeType) {
  return [nodeData?.name, nodeData?.display_name, nodeType?.comfyClass, nodeType?.title]
    .map((value) => String(value ?? ""))
    .some((value) => NODE_CLASSES.has(value));
}

function isStateManagerNode(node) {
  return [node?.comfyClass, node?.type, node?.constructor?.title]
    .map((value) => String(value ?? ""))
    .some((value) => NODE_CLASSES.has(value));
}

function isDistributionSafe(node) {
  return node?.properties?.[PROPERTY] === true;
}

function setDistributionSafe(node, enabled) {
  if (!node) return;
  node.properties = node.properties || {};
  node.properties[PROPERTY] = Boolean(enabled);
  node.graph?.change?.();
  node.setDirtyCanvas?.(true, true);
  renderDistributionSafeControl(node);
}

function setSerializedWidget(output, node, widgetName, value) {
  if (!output || !node) return false;
  const widgets = Array.isArray(node.widgets) ? node.widgets : [];
  const index = widgets.findIndex((widget) => widget?.name === widgetName);
  let changed = false;

  if (Array.isArray(output.widgets_values) && index >= 0) {
    output.widgets_values[index] = value;
    changed = true;
  }
  if (output.widgets_values_named && typeof output.widgets_values_named === "object") {
    output.widgets_values_named[widgetName] = value;
    changed = true;
  }
  return changed;
}

function scrubReservedQueueSnapshot(value) {
  if (typeof value !== "string" || !value.trim()) return value;
  let parsed;
  try {
    parsed = JSON.parse(value);
  } catch {
    return value;
  }
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return value;
  if (!Object.prototype.hasOwnProperty.call(parsed, "__dsm_queue_snapshot_v1")) return value;
  delete parsed.__dsm_queue_snapshot_v1;
  return JSON.stringify(parsed);
}

function applyDistributionSafeSerialization(output, node) {
  if (!isDistributionSafe(node)) return false;
  const uiStateWidget = (Array.isArray(node.widgets) ? node.widgets : [])
    .find((widget) => widget?.name === UI_STATE_WIDGET);
  const uiStateChanged = uiStateWidget
    ? setSerializedWidget(
        output,
        node,
        UI_STATE_WIDGET,
        scrubReservedQueueSnapshot(uiStateWidget.value),
      )
    : false;
  const characterChanged = setSerializedWidget(
    output,
    node,
    SELECTED_CHARACTER_WIDGET,
    DEFAULT_CHARACTER_ID,
  );
  const promptChanged = setSerializedWidget(
    output,
    node,
    SELECTED_PROMPT_WIDGET,
    DEFAULT_PROMPT_ID,
  );
  // The workflow-local mirror used by normal startup recovery contains the same
  // private UUIDs that distribution-safe mode is intended to scrub. The core
  // State Manager keeps a browser-local binding for restart persistence instead.
  if (output.properties && typeof output.properties === "object") {
    delete output.properties[SELECTION_MIRROR_PROPERTY];
  }
  return uiStateChanged || characterChanged || promptChanged;
}

function findSettingsTabs(root) {
  for (const tabs of root?.querySelectorAll?.(".dsm-tabs") || []) {
    const buttons = [...tabs.querySelectorAll("button")];
    const settingsButton = buttons.find(
      (button) => String(button.textContent || "").trim() === "Settings / seed",
    );
    if (settingsButton) return { tabs, settingsButton };
  }
  return null;
}

function renderDistributionSafeControl(node) {
  const root = node?.__dsm?.root;
  if (!root?.querySelectorAll) return false;

  const existing = root.querySelector(`[${CONTROL_ATTR}]`);
  const settings = findSettingsTabs(root);
  const visible = Boolean(settings?.settingsButton?.classList?.contains("selected"));

  if (!visible) {
    existing?.remove?.();
    return false;
  }

  if (existing) {
    const checkbox = existing.querySelector('input[type="checkbox"]');
    if (checkbox) checkbox.checked = isDistributionSafe(node);
    return true;
  }

  const wrapper = document.createElement("div");
  wrapper.setAttribute(CONTROL_ATTR, "");
  wrapper.className = "dsm-section";

  const label = document.createElement("label");
  label.className = "dsm-checkline";

  const checkbox = document.createElement("input");
  checkbox.type = "checkbox";
  checkbox.checked = isDistributionSafe(node);

  const text = document.createElement("span");
  text.className = "dsm-checkline-text";

  const title = document.createElement("strong");
  title.textContent = "Distribution-safe workflow serialization";

  const detail = document.createElement("small");
  detail.textContent =
    "When enabled, workflow saves/exports omit your selected local UUIDs and serialize default_character/default_prompt instead. This browser keeps the live selection in local storage for restart recovery; another machine receives the safe defaults.";

  text.append(title, detail);
  label.append(checkbox, text);
  wrapper.appendChild(label);

  checkbox.addEventListener("change", () => {
    setDistributionSafe(node, checkbox.checked);
  });

  settings.tabs.insertAdjacentElement("afterend", wrapper);
  return true;
}

function attachDistributionSafeUi(node) {
  if (!isStateManagerNode(node) || node.__dsmDistributionSafeAttached) return;
  node.__dsmDistributionSafeAttached = true;

  let attempts = 0;
  const attach = () => {
    const root = node?.__dsm?.root;
    if (!root) {
      if (attempts++ < 120) requestAnimationFrame(attach);
      return;
    }

    renderDistributionSafeControl(node);
    const observer = new MutationObserver(() => renderDistributionSafeControl(node));
    observer.observe(root, { childList: true, subtree: true });
    node.__dsmDistributionSafeObserver = observer;

    const originalRemoved = node.onRemoved;
    node.onRemoved = function (...args) {
      node.__dsmDistributionSafeObserver?.disconnect?.();
      node.__dsmDistributionSafeObserver = null;
      return originalRemoved?.apply(this, args);
    };
  };
  requestAnimationFrame(attach);
}

app.registerExtension({
  name: EXT_NAME,

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!isStateManagerDef(nodeData, nodeType)) return;
    const originalOnSerialize = nodeType.prototype.onSerialize;
    nodeType.prototype.onSerialize = function (output) {
      const result = originalOnSerialize?.apply(this, arguments);
      try {
        applyDistributionSafeSerialization(output, this);
      } catch (error) {
        console.warn(`[${EXT_NAME}] distribution-safe serialization failed`, error);
      }
      return result;
    };
  },

  nodeCreated(node) {
    attachDistributionSafeUi(node);
  },
});
