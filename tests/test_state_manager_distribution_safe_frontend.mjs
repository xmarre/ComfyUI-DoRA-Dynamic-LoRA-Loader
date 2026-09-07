import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";


async function loadDistributionSafeExtension() {
  const sourceUrl = new URL("../web/dora_state_manager_distribution_safe.js", import.meta.url);
  let source = await readFile(sourceUrl, "utf8");
  source = source.replace(
    'import { app } from "../../scripts/app.js";',
    'let capturedExtension = null; const app = { registerExtension(value) { capturedExtension = value; } };',
  );
  source += `\nexport { capturedExtension };\n`;
  const encoded = Buffer.from(source, "utf8").toString("base64");
  return import(`data:text/javascript;base64,${encoded}#${Date.now()}-${Math.random()}`);
}


function makeNodeType(className = "State Manager") {
  class StateManagerNode {
    onSerialize(output) {
      output.widgets_values = this.widgets.map((widget) => widget.value);
      output.widgets_values_named = Object.fromEntries(
        this.widgets.map((widget) => [widget.name, widget.value]),
      );
      output.properties = { ...this.properties };
    }
  }
  StateManagerNode.comfyClass = className;
  return StateManagerNode;
}


function makeNode(StateManagerNode, enabled) {
  const node = new StateManagerNode();
  node.properties = {
    dora_state_manager_distribution_safe_serialization: enabled,
  };
  node.widgets = [
    { name: "state_json", value: '{"version":1,"kind":"dora_state_manager_binding"}' },
    { name: "ui_state_json", value: '{"version":2}' },
    { name: "selected_character_id", value: "private-character-uuid" },
    { name: "selected_prompt_id", value: "private-prompt-uuid" },
  ];
  return node;
}


test("distribution-safe mode scrubs only serialized selection bindings", async () => {
  const helpers = await loadDistributionSafeExtension();
  const StateManagerNode = makeNodeType();
  await helpers.capturedExtension.beforeRegisterNodeDef(StateManagerNode, {
    name: "State Manager",
    input: { required: {} },
  });

  const node = makeNode(StateManagerNode, true);
  const output = {};
  node.onSerialize(output);

  assert.equal(output.widgets_values[2], "default_character");
  assert.equal(output.widgets_values[3], "default_prompt");
  assert.equal(output.widgets_values_named.selected_character_id, "default_character");
  assert.equal(output.widgets_values_named.selected_prompt_id, "default_prompt");
  assert.equal(output.properties.dora_state_manager_distribution_safe_serialization, true);

  // The live widgets remain on the real local selection, so queue/generation state is unchanged.
  assert.equal(node.widgets[2].value, "private-character-uuid");
  assert.equal(node.widgets[3].value, "private-prompt-uuid");
});


test("normal mode preserves selected local UUID bindings", async () => {
  const helpers = await loadDistributionSafeExtension();
  const StateManagerNode = makeNodeType();
  await helpers.capturedExtension.beforeRegisterNodeDef(StateManagerNode, {
    name: "State Manager",
    input: { required: {} },
  });

  const node = makeNode(StateManagerNode, false);
  const output = {};
  node.onSerialize(output);

  assert.equal(output.widgets_values[2], "private-character-uuid");
  assert.equal(output.widgets_values[3], "private-prompt-uuid");
  assert.equal(output.widgets_values_named.selected_character_id, "private-character-uuid");
  assert.equal(output.widgets_values_named.selected_prompt_id, "private-prompt-uuid");
});


test("legacy State Manager class receives the same protection", async () => {
  const helpers = await loadDistributionSafeExtension();
  const LegacyNode = makeNodeType("DoRA State Manager");
  await helpers.capturedExtension.beforeRegisterNodeDef(LegacyNode, {
    name: "DoRA State Manager",
    input: { required: {} },
  });

  const node = makeNode(LegacyNode, true);
  const output = {};
  node.onSerialize(output);

  assert.equal(output.widgets_values_named.selected_character_id, "default_character");
  assert.equal(output.widgets_values_named.selected_prompt_id, "default_prompt");
});
