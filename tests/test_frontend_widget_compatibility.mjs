import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const NODE_CLASS = "DoRA Power LoRA Loader";

function makeCanvasContext() {
  const labels = [];
  const noop = () => {};
  return {
    labels,
    arc: noop,
    beginPath: noop,
    fill: noop,
    fillText(text) {
      labels.push(String(text));
    },
    measureText(text) {
      return { width: String(text).length * 7 };
    },
    restore: noop,
    roundRect: noop,
    save: noop,
    stroke: noop,
  };
}

async function loadLoaderExtension() {
  const sourceUrl = new URL("../web/dora_power_lora_loader.js", import.meta.url);
  let source = await readFile(sourceUrl, "utf8");
  let extension = null;
  const previousFetch = globalThis.fetch;
  const previousLiteGraph = globalThis.LiteGraph;

  globalThis.fetch = async () => ({
    ok: true,
    async json() {
      return ["loaded.safetensors"];
    },
  });
  globalThis.LiteGraph = {};
  globalThis.__doraWidgetCompatibilityTestApp = {
    canvas: { canvas: {} },
    graph: null,
    registerExtension(value) {
      extension = value;
    },
  };
  globalThis.__doraWidgetCompatibilityTestApi = {
    addEventListener() {},
  };

  source = source
    .replace(
      'import { app } from "../../scripts/app.js";',
      "const app = globalThis.__doraWidgetCompatibilityTestApp;"
    )
    .replace(
      'import { api } from "../../scripts/api.js";',
      "const api = globalThis.__doraWidgetCompatibilityTestApi;"
    );

  const cleanup = () => {
    delete globalThis.__doraWidgetCompatibilityTestApp;
    delete globalThis.__doraWidgetCompatibilityTestApi;
    globalThis.fetch = previousFetch;
    globalThis.LiteGraph = previousLiteGraph;
  };

  try {
    const encoded = Buffer.from(source, "utf8").toString("base64");
    await import(`data:text/javascript;base64,${encoded}#${Date.now()}-${Math.random()}`);
  } catch (error) {
    cleanup();
    throw error;
  }

  assert.ok(extension, "loader extension should register");
  return { cleanup, extension };
}

function makeNodeType() {
  class FakeNode {
    constructor() {
      this.id = 42;
      this.graph = { change() {}, rootGraph: { id: "test-graph" } };
      this.min_size = [0, 0];
      this.properties = {
        dora_power_lora: {
          rows: [
            {
              enabled: true,
              name: "loaded.safetensors",
              strengthModel: 0.85,
              strengthClip: 0.85,
            },
          ],
          globals: {},
        },
      };
      this.size = [360, 200];
      this.widgets = [];
    }

    addWidget(type, name, value, callback, options = {}) {
      const widget = { callback, name, options, type, value };
      this.widgets.push(widget);
      return widget;
    }

    addCustomWidget(widget) {
      Object.setPrototypeOf(widget, Object.prototype);
      this.widgets.push(widget);
      return widget;
    }

    computeSize() {
      const height = this.widgets.reduce((total, widget) => {
        const widgetHeight = widget.computeSize?.(this.size[0])?.[1] ?? 20;
        return total + widgetHeight;
      }, 40);
      return [this.size[0], height];
    }

    onNodeCreated() {
      return this;
    }

    setDirtyCanvas() {}
  }

  FakeNode.comfyClass = NODE_CLASS;
  return FakeNode;
}

test("frontend normalization and rebuilds preserve live DoRA custom widget behavior", async () => {
  const { cleanup, extension } = await loadLoaderExtension();
  try {
    const NodeType = makeNodeType();
    await extension.beforeRegisterNodeDef(NodeType, {
      display_name: NODE_CLASS,
      name: NODE_CLASS,
    });

    const node = new NodeType();
    node.onNodeCreated();

    // Current Vue legacy-widget rendering binds the live widget instance on
    // mount and keeps that object while the WidgetId/type render key is stable.
    // Capture the first-build instances before fetchLoras() completes so the
    // test detects a same-name replacement during the async refresh.
    const initiallyBoundRowWidget = node.widgets.find((widget) => widget.name === "LORA_1");
    const initiallyBoundReportWidget = node.widgets.find(
      (widget) => widget.name === "auto_strength_visualization"
    );
    assert.ok(initiallyBoundRowWidget, "initial LoRA row widget should be registered");
    assert.ok(initiallyBoundReportWidget, "initial auto-strength report should be registered");
    let reboundDraws = 0;
    initiallyBoundRowWidget.triggerDraw = () => {
      reboundDraws += 1;
    };

    await new Promise((resolve) => setImmediate(resolve));

    const rowWidget = node.widgets.find((widget) => widget.name === "LORA_1");
    const reportWidget = node.widgets.find(
      (widget) => widget.name === "auto_strength_visualization"
    );

    assert.ok(rowWidget, "LoRA row widget should be registered");
    assert.strictEqual(
      rowWidget,
      initiallyBoundRowWidget,
      "async LoRA refresh must preserve the live row widget object"
    );
    assert.strictEqual(
      reportWidget,
      initiallyBoundReportWidget,
      "async LoRA refresh must preserve the live report widget object"
    );
    assert.ok(reboundDraws >= 1, "async rebuild should redraw the reused Vue-bound row");
    assert.equal(Object.getPrototypeOf(rowWidget), Object.prototype);
    assert.equal(rowWidget.computeSize(360)[1], 24);
    assert.equal(typeof rowWidget.draw, "function");
    assert.equal(typeof rowWidget.mouse, "function");
    assert.equal(typeof rowWidget.openLoraPicker, "function");
    assert.equal(typeof rowWidget.serializeValue, "function");
    assert.deepEqual(rowWidget.serializeValue(), {
      lora: "loaded.safetensors",
      on: true,
      strength: 0.85,
      strengthTwo: 0.85,
    });

    const context = makeCanvasContext();
    rowWidget.draw(context, node, 360, 0, 24);
    assert.ok(context.labels.includes("loaded.safetensors"));
    assert.ok(context.labels.includes("0.85"));

    const pointerDown = {
      button: 0,
      preventDefault() {},
      stopPropagation() {},
      type: "pointerdown",
    };
    assert.equal(rowWidget.mouse(pointerDown, [30, 12], node), true);
    assert.equal(
      node.properties.dora_power_lora.rows[0].enabled,
      false,
      "a click delivered to the originally bound row must update canonical live state"
    );

    const rowBeforeAdd = rowWidget;
    const addWidget = node.widgets.find((widget) => widget.name === "add_lora");
    assert.equal(typeof addWidget?.callback, "function");
    addWidget.callback();
    assert.equal(node.properties.dora_power_lora.rows.length, 2);
    assert.ok(node.widgets.some((widget) => widget.name === "LORA_2"));
    assert.strictEqual(
      node.widgets.find((widget) => widget.name === "LORA_1"),
      rowBeforeAdd,
      "Add LoRA rebuild must keep the existing row widget object live"
    );

    assert.ok(reportWidget, "auto-strength report widget should be registered");
    assert.equal(Object.getPrototypeOf(reportWidget), Object.prototype);
    assert.equal(reportWidget.computeSize(360)[1], 72);
    assert.equal(typeof reportWidget.draw, "function");
    assert.equal(typeof reportWidget.mouse, "function");
    assert.equal(typeof reportWidget._displayRows, "function");
  } finally {
    cleanup();
  }
});
