import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

// Keep seed-specific test imports independent of the general State Manager
// fixture so adjacent PR overlays can extend their own exports without conflict.
async function loadStateSeedHelpers() {
  const sourceUrl = new URL("../web/dora_state_manager.js", import.meta.url);
  let source = await readFile(sourceUrl, "utf8");
  source = source
    .replace('import { app } from "../../scripts/app.js";', "const app = { registerExtension() {}, graph: { extra: {} } };")
    .replace('import { api } from "../../scripts/api.js";', "const api = { fetchApi() { throw new Error('not used'); }, apiURL(value) { return value; } };")
    .replace('import "../../scripts/domWidget.js";', "");
  source += "\nexport { mutatePromptForStateSeedNode, patchStateSeedNodeDef };\n";
  const encoded = Buffer.from(source, "utf8").toString("base64");
  return import(`data:text/javascript;base64,${encoded}#${Date.now()}-${Math.random()}`);
}

test("queued State Manager Seed snapshots preserve the effective seed in both serialization formats", async () => {
  const helpers = await loadStateSeedHelpers();
  for (const originalSeed of [-1, -2, -3, 67890]) {
    const node = {
      id: 54,
      type: "State Manager Seed",
      widgets: [{ name: "seed", value: originalSeed, options: { step: 1 } }],
      properties: {},
      __dsmLastSeed: 4000,
    };
    const snapshot = {
      id: 54,
      widgets_values: [originalSeed, ""],
      widgets_values_named: { seed: originalSeed, "🎲 Randomize Each Time": "" },
    };
    const payload = {
      output: { "54": { class_type: "StateManagerSeed", inputs: { seed: originalSeed } } },
      workflow: { nodes: [snapshot] },
    };
    assert.equal(helpers.mutatePromptForStateSeedNode(payload, node), true);
    const effectiveSeed = payload.output["54"].inputs.seed;
    assert.equal(snapshot.widgets_values[0], effectiveSeed);
    assert.equal(snapshot.widgets_values_named.seed, effectiveSeed);
    assert.equal(snapshot.widgets_values_named["🎲 Randomize Each Time"], "");
    assert.equal(node.widgets[0].value, originalSeed, "live -1/-2/-3 must remain active for subsequent queues");
    assert.equal(node.__dsmLastSeed, effectiveSeed);
    if (originalSeed === -2) assert.equal(effectiveSeed, 4001);
    else if (originalSeed === -3) assert.equal(effectiveSeed, 3999);
    else if (originalSeed >= 0) assert.equal(effectiveSeed, originalSeed);
    else assert.ok(Number.isSafeInteger(effectiveSeed) && effectiveSeed >= 0 && effectiveSeed <= 1125899906842624);
  }

  const positionalOnlyNode = {
    id: 54, type: "State Manager Seed",
    widgets: [{ name: "seed", value: 9876 }], properties: {},
  };
  const positionalOnly = {
    output: { "54": { inputs: { seed: 9876 } } },
    workflow: { nodes: [{ id: 54, widgets_values: [0] }] },
  };
  assert.equal(helpers.mutatePromptForStateSeedNode(positionalOnly, positionalOnlyNode), true);
  assert.equal(positionalOnly.workflow.nodes[0].widgets_values[0], 9876);
});

test("State Manager Seed restores the effective seed from older inconsistent workflow snapshots", async () => {
  const helpers = await loadStateSeedHelpers();
  class SeedNode {
    constructor() {
      this.widgets = [{ name: "seed", value: -1, options: {} }];
      this.properties = {};
    }
    addWidget(type, name, value, callback, options) {
      const widget = { type, name, value, callback, options };
      this.widgets.push(widget);
      return widget;
    }
    onConfigure(info) {
      // Mimic the ComfyUI frontend's named-value restore before node.onConfigure.
      this.widgets[0].value = info.widgets_values_named?.seed ?? info.widgets_values?.[0] ?? -1;
    }
  }
  helpers.patchStateSeedNodeDef(SeedNode);
  for (const [named, positional, expected] of [
    [-1, 591911601269354, 591911601269354],
    [-2, 12345, 12345],
    [-3, 12344, 12344],
    [-1, -1, -1],
    [333, 444, 333],
    [-1, null, -1],
  ]) {
    const node = new SeedNode();
    node.onConfigure({ widgets_values: [positional], widgets_values_named: { seed: named } });
    assert.equal(node.widgets[0].value, expected);
    assert.equal(node.widgets.filter((w) => w.name === "seed").length, 1);
  }
  const positionalOnly = new SeedNode();
  positionalOnly.onConfigure({ widgets_values: [98765] });
  assert.equal(positionalOnly.widgets[0].value, 98765);
});
