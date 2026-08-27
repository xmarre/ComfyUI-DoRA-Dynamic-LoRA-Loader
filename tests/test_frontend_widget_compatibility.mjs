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

test("current frontend prototype normalization preserves DoRA custom widget behavior", async () => {
  const { cleanup, extension } = await loadLoaderExtension();
  try {
    const NodeType = makeNodeType();
    await extension.beforeRegisterNodeDef(NodeType, {
      display_name: NODE_CLASS,
      name: NODE_CLASS,
    });

    const node = new NodeType();
    node.onNodeCreated();
    await new Promise((resolve) => setImmediate(resolve));

    const rowWidget = node.widgets.find((widget) => widget.name === "LORA_1");
    const reportWidget = node.widgets.find(
      (widget) => widget.name === "auto_strength_visualization"
    );

    assert.ok(rowWidget, "LoRA row widget should be registered");
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
