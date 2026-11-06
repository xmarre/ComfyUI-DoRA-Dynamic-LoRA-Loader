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
  const previousStateManagerSyncApi = globalThis.__doraStateManagerSyncApi;
  const syncCalls = [];

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
  globalThis.__doraStateManagerSyncApi = {
    loaderStateChanged(node, details = {}) {
      syncCalls.push({ node, details });
      return 1;
    },
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
    if (previousStateManagerSyncApi === undefined) delete globalThis.__doraStateManagerSyncApi;
    else globalThis.__doraStateManagerSyncApi = previousStateManagerSyncApi;
  };

  try {
    const encoded = Buffer.from(source, "utf8").toString("base64");
    await import(`data:text/javascript;base64,${encoded}#${Date.now()}-${Math.random()}`);
  } catch (error) {
    cleanup();
    throw error;
  }

  assert.ok(extension, "loader extension should register");
  return { cleanup, extension, syncCalls };
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


test("loader-owned settings debounce continuous edits while discrete slot changes stay immediate", async () => {
  const { cleanup, extension, syncCalls } = await loadLoaderExtension();
  try {
    const NodeType = makeNodeType();
    await extension.beforeRegisterNodeDef(NodeType, {
      display_name: NODE_CLASS,
      name: NODE_CLASS,
    });

    const node = new NodeType();
    node.onNodeCreated();
    await new Promise((resolve) => setImmediate(resolve));
    await new Promise((resolve) => setTimeout(resolve, 180));
    syncCalls.length = 0;

    const autoStrength = node.widgets.find((widget) => widget.name === "auto_strength_enabled");
    assert.equal(typeof autoStrength?.callback, "function");
    autoStrength.callback(true);
    autoStrength.callback(false);
    autoStrength.callback(true);
    assert.equal(node.properties.dora_power_lora.globals.auto_strength_enabled, true);
    assert.equal(syncCalls.length, 0, "continuous loader edits should be coalesced before notifying State Manager");

    await new Promise((resolve) => setTimeout(resolve, 180));
    assert.equal(syncCalls.length, 1);
    assert.strictEqual(syncCalls[0]?.node, node);

    const stateSlot = node.widgets.find((widget) => widget.name === "state_slot");
    assert.equal(typeof stateSlot?.callback, "function");
    stateSlot.callback("character_style");
    assert.equal(node.properties.dora_state_slot, "character_style");
    assert.equal(syncCalls.length, 2, "State slot identity changes should synchronize immediately");
    assert.strictEqual(syncCalls.at(-1)?.node, node);
    assert.equal(syncCalls.at(-1)?.details?.previousSlot, "loader_42");

    autoStrength.callback(false);
    assert.equal(syncCalls.length, 2);
    node.onRemoved?.();
    await new Promise((resolve) => setTimeout(resolve, 180));
    assert.equal(syncCalls.length, 2, "removing the loader must cancel a pending debounced notification");
  } finally {
    cleanup();
  }
});

function makeReportFixture(loraCount = 4, groupCount = 60) {
  return {
    auto_strength_device: "gpu",
    ratio_floor: 0.5,
    ratio_ceiling: 1.5,
    rows: Array.from({ length: loraCount }, (_, rowIndex) => ({
      row_index: rowIndex,
      status: "analyzed",
      lora_name: `test-lora-${rowIndex}.safetensors`,
      strength_model: 1,
      strength_clip: 1,
      report: {
        logical_groups: Array.from({ length: groupCount }, (_, groupIndex) => ({
          logical_id: `model.layers.${groupIndex}.attention`,
          ratio_applied: groupIndex % 3 === 0 ? 1.2 : groupIndex % 3 === 1 ? 0.8 : 1,
          fanout: 1,
        })),
        logical_groups_measured: groupCount,
        logical_groups_total: groupCount,
        measured_bases: groupCount,
        mapped_bases: groupCount,
        ratio_floor: 0.5,
        ratio_ceiling: 1.5,
        analysis_load_device: "cpu",
        cohorts: [],
      },
    })),
  };
}

function makeReportCanvas(height = 10000) {
  const labels = [];
  const noop = () => {};
  return {
    labels,
    canvas: { height },
    getTransform() { return { b: 0, c: 0, d: 1, f: 0 }; },
    beginPath: noop, closePath: noop, roundRect: noop,
    arc: noop, moveTo: noop, lineTo: noop, fill: noop, stroke: noop,
    save: noop, restore: noop,
    fillText(text) { labels.push(String(text)); },
    measureText(text) { return { width: String(text).length * 7 }; },
  };
}

test("auto-strength report groups are reused across layout and redraw, invalidated on toggle and replacement", async () => {
  const { cleanup, extension } = await loadLoaderExtension();
  const originalSort = Array.prototype.sort;
  try {
    const NodeType = makeNodeType();
    await extension.beforeRegisterNodeDef(NodeType, { name: NODE_CLASS });
    const node = new NodeType();
    node.onNodeCreated();
    await new Promise((resolve) => setImmediate(resolve));
    const reportWidget = node.widgets.find((widget) => widget.name === "auto_strength_visualization");
    const report = makeReportFixture();
    node._doraAutoStrengthReport = report;
    node._doraAutoStrengthExpanded = {};
    let sortCount = 0;
    Array.prototype.sort = function (...args) {
      sortCount += 1;
      return originalSort.apply(this, args);
    };

    const initialHeight = reportWidget.computeSize(360)[1];
    const firstSorts = sortCount;
    assert.ok(firstSorts >= report.rows.length, "first layout must group the logical projections");
    const context = makeReportCanvas();
    for (let i = 0; i < 8; i++) {
      assert.equal(reportWidget.computeSize(360)[1], initialHeight);
      reportWidget.draw(context, node, 360, 0, initialHeight);
    }
    assert.equal(sortCount, firstSorts, "redrawing unchanged analysis must not regroup logical projections");

    // Row header click expands the report; only that row should be recomputed.
    assert.equal(reportWidget.mouse({ type: "pointerdown" }, [25, 90]), true);
    const expandedHeight = reportWidget.computeSize(360)[1];
    assert.ok(expandedHeight > initialHeight);
    assert.equal(sortCount, firstSorts + 4, "expanding one row regroups only that row");
    reportWidget.draw(context, node, 360, 0, expandedHeight);
    assert.equal(sortCount, firstSorts + 4);

    node._doraAutoStrengthReport = makeReportFixture(2, 20);
    const replacedHeight = reportWidget.computeSize(360)[1];
    assert.ok(replacedHeight > 0 && replacedHeight < expandedHeight);
    assert.equal(sortCount, firstSorts + 4 + 4, "replacement report must rebuild only its first visible row");
  } finally {
    Array.prototype.sort = originalSort;
    cleanup();
  }
});

test("auto-strength report skips offscreen card painting without changing layout height", async () => {
  const { cleanup, extension } = await loadLoaderExtension();
  try {
    const NodeType = makeNodeType();
    await extension.beforeRegisterNodeDef(NodeType, { name: NODE_CLASS });
    const node = new NodeType();
    node.onNodeCreated();
    const reportWidget = node.widgets.find((widget) => widget.name === "auto_strength_visualization");
    node._doraAutoStrengthReport = makeReportFixture(12, 60);
    node._doraAutoStrengthExpanded = {};
    const height = reportWidget.computeSize(360)[1];

    const context = makeReportCanvas(180);
    reportWidget.draw(context, node, 360, 0, height);
    assert.ok(context.labels.some((v) => v.includes("test-lora-0.safetensors")));
    assert.ok(!context.labels.some((v) => v.includes("test-lora-1.safetensors")));
    assert.equal(reportWidget.hitAreas.length, 1, "only visible row headers should be interactive");

    node._doraAutoStrengthExpanded[0] = true;
    const expandedHeight = reportWidget.computeSize(360)[1];
    const expandedContext = makeReportCanvas(180);
    reportWidget.draw(expandedContext, node, 360, 0, expandedHeight);
    assert.ok(expandedContext.labels.length < 30, "expanded cards must not paint offscreen logical-group bars");
    assert.equal(reportWidget.hitAreas.length, 1);

    const allRowsContext = makeReportCanvas();
    delete allRowsContext.getTransform;
    reportWidget.draw(allRowsContext, node, 360, 0, height);
    assert.equal(reportWidget.hitAreas.length, 1, "even without a usable transform, only one LoRA card is drawn");
    assert.ok(!allRowsContext.labels.some((v) => v.includes("test-lora-11.safetensors")));

    // The remaining rows must remain accessible, not dropped from the report.
    for (let index = 1; index < 12; index++) {
      assert.equal(reportWidget.mouse({ type: "pointerdown" }, [327, 20]), true);
      reportWidget.draw(allRowsContext, node, 360, 0, reportWidget.computeSize(360)[1]);
    }
    assert.equal(reportWidget._reportPage, 11);
    assert.ok(allRowsContext.labels.some((v) => v.includes("test-lora-11.safetensors")));
  } finally {
    cleanup();
  }
});


test("auto-strength report bounds the Vue legacy canvas backing bitmap regardless of report size", async () => {
  const { cleanup, extension } = await loadLoaderExtension();
  try {
    const NodeType = makeNodeType();
    await extension.beforeRegisterNodeDef(NodeType, { name: NODE_CLASS });
    const node = new NodeType();
    node.onNodeCreated();
    const widget = node.widgets.find((item) => item.name === "auto_strength_visualization");
    const report = makeReportFixture(48, 480);
    report.rows[0].report.cohorts = Array.from({ length: 800 }, (_, i) => ({
      group: `very-long-cohort-name-${i}`, family: "linear",
    }));
    node._doraAutoStrengthReport = report;
    node._doraAutoStrengthExpanded = {};

    const width = 360;
    const collapsedHeight = widget.computeSize(width)[1];
    assert.ok(collapsedHeight < 400, "48-row collapsed report must not increase the canvas backing height");

    const context = makeReportCanvas();
    let measures = 0;
    const originalMeasureText = context.measureText;
    context.measureText = (text) => {
      measures += 1;
      return originalMeasureText(text);
    };
    widget.draw(context, node, width, 0, collapsedHeight);
    assert.ok(measures < 200, "long cohort labels must be truncated with logarithmic text measurement");

    assert.equal(widget.mouse({ type: "pointerdown" }, [25, 84]), true);
    const expandedHeight = widget.computeSize(width)[1];
    assert.ok(expandedHeight < 550, "expansion must not make the report canvas grow with logical-group count");
    // Current WidgetLegacy.vue allocates width*2 × (height+2)*2 backing pixels on draw.
    const maxBackingPixels = (width * 2) * ((expandedHeight + 2) * 2);
    assert.ok(maxBackingPixels < 800_000, "legacy backing-store allocation must remain bounded");

    const detailContext = makeReportCanvas();
    widget.draw(detailContext, node, width, 0, expandedHeight);
    assert.ok(detailContext.labels.some((v) => v.includes("Groups 1/")));
    assert.equal(widget.groupNavAreas.length, 2);
    const firstPage = widget._rowLayout(report.rows[0]);
    assert.ok(firstPage.sections.reduce((n, section) => n + section.items.length, 0) <= 12);

    const next = widget.groupNavAreas.find((area) => area.direction === 1);
    assert.ok(next);
    assert.equal(widget.mouse({ type: "pointerdown" }, [next.x + 10, next.y + 10]), true);
    widget.draw(detailContext, node, width, 0, widget.computeSize(width)[1]);
    assert.ok(detailContext.labels.some((v) => v.includes("Groups 2/")));
    assert.equal(widget._rowLayout(report.rows[0]).groupPage, 1);

    // A report replacement resets selection, including stale per-row page indexes.
    node._doraAutoStrengthReport = makeReportFixture(2, 6);
    node._doraAutoStrengthExpanded = {};
    widget.computeSize(width);
    assert.equal(widget._reportPage, 0);
    assert.equal(widget._groupPages.size, 0);
  } finally {
    cleanup();
  }
});
