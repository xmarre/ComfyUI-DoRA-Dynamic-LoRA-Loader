import { app } from "../../scripts/app.js";

const NODE_CLASS = "DoRA Power LoRA Loader";
const EXT_NAME = "comfyui_dora_dynamic_lora.runtime_bypass";
const WIDGET_NAME = "runtime_bypass_lora";
const PROPERTY_NAME = "dora_runtime_bypass_lora";

function isTarget(nodeData, nodeType) {
  const name = nodeData?.name ?? "";
  const displayName = nodeData?.display_name ?? "";
  const comfyClass = nodeType?.comfyClass ?? "";
  return name === NODE_CLASS || displayName === NODE_CLASS || comfyClass === NODE_CLASS;
}

function runtimeValue(node) {
  node.properties = node.properties || {};
  if (node.properties[PROPERTY_NAME] === undefined) {
    node.properties[PROPERTY_NAME] = false;
  }
  return !!node.properties[PROPERTY_NAME];
}

function ensureRuntimeWidget(node) {
  if (!node || !Array.isArray(node.widgets)) return null;

  const existing = node.widgets.find((widget) => widget?.name === WIDGET_NAME);
  if (existing) return existing;

  const widget = node.addWidget(
    "toggle",
    WIDGET_NAME,
    runtimeValue(node),
    (value) => {
      node.properties = node.properties || {};
      node.properties[PROPERTY_NAME] = !!value;
      widget.value = !!value;
      node.setDirtyCanvas?.(true, true);
      node.graph?.change?.();
    }
  );

  widget.label = "Runtime bypass LoRA (low VRAM; standard LoRA only)";
  widget.tooltip =
    "Keeps base model weights untouched and evaluates standard LoRA adapters during forward passes. " +
    "Avoids the full materialized patched-weight copy on very large models. DoRA and unsupported " +
    "adapter/offset forms fail explicitly instead of being approximated.";
  widget.options = widget.options || {};
  widget.options.tooltip = widget.tooltip;

  // The main extension rebuilds the entire custom UI and adds the analysis widget last.
  // Put this global mode directly below Stack Enabled rather than burying it at the bottom.
  const currentIndex = node.widgets.indexOf(widget);
  const stackIndex = node.widgets.findIndex((item) => item?.name === "stack_enabled");
  if (currentIndex >= 0 && stackIndex >= 0 && currentIndex !== stackIndex + 1) {
    node.widgets.splice(currentIndex, 1);
    node.widgets.splice(stackIndex + 1, 0, widget);
  }

  return widget;
}

app.registerExtension({
  name: EXT_NAME,
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!isTarget(nodeData, nodeType)) return;

    // dora_power_lora_loader.js removes and recreates every widget whenever its UI
    // rebuilds. Its auto-strength custom widget is the final build step, so use that
    // as a stable point to reinsert this standard serializable boolean widget.
    const inheritedAddCustomWidget = nodeType.prototype.addCustomWidget;
    if (typeof inheritedAddCustomWidget === "function" && !nodeType.prototype.__doraRuntimeBypassWrapped) {
      nodeType.prototype.addCustomWidget = function (widget) {
        const result = inheritedAddCustomWidget.apply(this, arguments);
        if (widget?.name === "auto_strength_visualization") {
          ensureRuntimeWidget(this);
        }
        return result;
      };
      nodeType.prototype.__doraRuntimeBypassWrapped = true;
    }

    const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      // Fallback for frontend variants where the report widget is not installed.
      ensureRuntimeWidget(this);
      return result;
    };

    const originalConfigure = nodeType.prototype.configure;
    nodeType.prototype.configure = function (info) {
      if (info?.properties && Object.prototype.hasOwnProperty.call(info.properties, PROPERTY_NAME)) {
        this.properties = this.properties || {};
        this.properties[PROPERTY_NAME] = !!info.properties[PROPERTY_NAME];
      }
      const result = originalConfigure?.apply(this, arguments);
      ensureRuntimeWidget(this);
      const widget = this.widgets?.find((item) => item?.name === WIDGET_NAME);
      if (widget) widget.value = runtimeValue(this);
      return result;
    };
  },
});
