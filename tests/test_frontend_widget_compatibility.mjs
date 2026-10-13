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


function makeConfigurableNodeType() {
  const NodeType = makeNodeType();
  NodeType.prototype.configure = function (info) {
    for (const [key, value] of Object.entries(info?.properties || {})) this.properties[key] = value;
  };
  return NodeType;
}

function legacyRowsOf(node) {
  return node.properties.dora_power_lora.rows.map((row) => ({ ...row }));
}

const LEGACY_ROWS = [
  { enabled: true, name: "face.safetensors", strengthModel: 0.7, strengthClip: 0.6 },
  { enabled: false, name: "style.safetensors", strengthModel: 1.25, strengthClip: 0.5 },
];
const LEGACY_GLOBALS = { stack_enabled: false, verbose: true, log_unloaded_keys: false };
const rowValues = (row) => [row.enabled, row.name, row.strengthModel, row.strengthClip];
const legacyGlobalValues = [false, true, false];

// Widget-only workflows came from the first two loader builds, saved by
// frontends that either skipped non-serialized buttons or kept null holes.
const LEGACY_WIDGET_FIXTURES = {
  "compact rows": [...LEGACY_ROWS.flatMap(rowValues), ...legacyGlobalValues],
  "first build with button holes": [...LEGACY_ROWS.flatMap((row) => [...rowValues(row), null]), null, ...legacyGlobalValues],
  "second build with button holes": [null, ...LEGACY_ROWS.flatMap((row) => [...rowValues(row), null]), null, ...legacyGlobalValues],
};

test("legacy widget-only workflows survive configure, serialize, and reopen", async () => {
  const { cleanup, extension } = await loadLoaderExtension();
  try {
    const NodeType = makeConfigurableNodeType();
    await extension.beforeRegisterNodeDef(NodeType, { display_name: NODE_CLASS, name: NODE_CLASS });

    for (const [label, widgetsValues] of Object.entries(LEGACY_WIDGET_FIXTURES)) {
      const node = new NodeType();
      node.properties = {};
      node.configure({ properties: {}, widgets_values: structuredClone(widgetsValues) });
      assert.deepEqual(legacyRowsOf(node), LEGACY_ROWS, label);
      for (const [key, value] of Object.entries(LEGACY_GLOBALS)) {
        assert.equal(node.properties.dora_power_lora.globals[key], value, `${label}: ${key}`);
      }

      const serialized = { properties: structuredClone(node.properties) };
      node.onSerialize(serialized);
      assert.deepEqual(serialized.widgets_values, [], label);

      const reopened = new NodeType();
      reopened.properties = {};
      reopened.configure(JSON.parse(JSON.stringify(serialized)));
      assert.deepEqual(legacyRowsOf(reopened), LEGACY_ROWS, `${label} after reopen`);
      assert.deepEqual(reopened.properties.dora_power_lora.globals, node.properties.dora_power_lora.globals, `${label} after reopen`);
      assert.equal(reopened.properties.dora_power_lora_unrecognized_widgets_values, undefined, label);
    }
  } finally {
    cleanup();
  }
});

test("unrecognized legacy widget values are kept instead of being reinterpreted", async () => {
  const { cleanup, extension } = await loadLoaderExtension();
  const previousWarn = console.warn;
  console.warn = () => {};
  try {
    const NodeType = makeConfigurableNodeType();
    await extension.beforeRegisterNodeDef(NodeType, { display_name: NODE_CLASS, name: NODE_CLASS });

    // One row followed by seven settings: no loader build wrote this, and the
    // old decoder misread it as two rows plus three toggles.
    const unknown = [true, "face.safetensors", 0.7, 0.7, false, false, false, true, 30, 10, true];
    const node = new NodeType();
    node.properties = {};
    node.configure({ properties: {}, widgets_values: structuredClone(unknown) });
    assert.deepEqual(node.properties.dora_power_lora.rows.map((row) => row.name), ["None"]);
    assert.deepEqual(node.properties.dora_power_lora_unrecognized_widgets_values, unknown);

    const serialized = { properties: structuredClone(node.properties) };
    node.onSerialize(serialized);
    const reopened = new NodeType();
    reopened.properties = {};
    reopened.configure(JSON.parse(JSON.stringify(serialized)));
    assert.deepEqual(reopened.properties.dora_power_lora_unrecognized_widgets_values, unknown);
  } finally {
    console.warn = previousWarn;
    cleanup();
  }
});
