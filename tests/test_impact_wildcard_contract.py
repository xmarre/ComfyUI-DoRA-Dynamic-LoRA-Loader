from __future__ import annotations

import ast
import importlib.util
import os
from pathlib import Path
import sys
import types

import pytest


IMPACT_SOURCE = os.environ.get("IMPACT_PACK_SOURCE")


def _load_real_wildcards(source_root: Path):
    # Import the reviewed Impact wildcard engine itself while stubbing only the
    # unrelated ComfyUI integration modules required by its module imports.
    stubs = {}
    for name in ("folder_paths", "nodes"):
        module = types.ModuleType(name)
        stubs[name] = module
        sys.modules[name] = module

    impact = types.ModuleType("impact")
    impact.__path__ = []
    config = types.ModuleType("impact.config")
    utils = types.ModuleType("impact.utils")
    impact.config = config
    impact.utils = utils
    sys.modules["impact"] = impact
    sys.modules["impact.config"] = config
    sys.modules["impact.utils"] = utils

    path = source_root / "modules" / "impact" / "wildcards.py"
    spec = importlib.util.spec_from_file_location("_reviewed_impact_wildcards", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    impact.wildcards = module
    return impact, module


def _compile_reviewed_defs(path: Path, names: set[str], namespace: dict):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and node.name in names
    ]
    missing = names - {node.name for node in selected}
    if missing:
        raise AssertionError(f"reviewed Impact source is missing {sorted(missing)}")
    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


class _PromptServerInstance:
    def __init__(self):
        self.feedback = []

    def send_sync(self, event, payload):
        self.feedback.append((event, payload))


@pytest.fixture()
def impact_contract():
    if not IMPACT_SOURCE:
        pytest.skip("IMPACT_PACK_SOURCE is required for reviewed Impact integration")
    source_root = Path(IMPACT_SOURCE).resolve()
    impact, wildcards = _load_real_wildcards(source_root)

    # Controlled dictionary avoids depending on user wildcard assets while still
    # executing Impact's real RNG/traversal implementation.
    wildcards.wildcard_dict.clear()
    wildcards.wildcard_dict.update(
        {
            "managed/scene": [
                "ONE_RED_CUBE_SENTINEL",
                "TWO_GREEN_SPHERE_SENTINEL",
                "THREE_BLUE_PYRAMID_SENTINEL",
            ]
        }
    )
    wildcards.available_wildcards.clear()
    wildcards.loaded_wildcards.clear()
    wildcards._on_demand_mode = False

    prompt_server = types.SimpleNamespace(instance=_PromptServerInstance())
    server_ns = {
        "impact": impact,
        "logging": __import__("logging"),
        "PromptServer": prompt_server,
    }
    _compile_reviewed_defs(
        source_root / "modules" / "impact" / "impact_server.py",
        {"find_input_value", "onprompt_populate_wildcards"},
        server_ns,
    )

    processor_ns = {"impact": impact}
    _compile_reviewed_defs(
        source_root / "modules" / "impact" / "impact_pack.py",
        {"ImpactWildcardProcessor"},
        processor_ns,
    )
    return (
        wildcards,
        server_ns["onprompt_populate_wildcards"],
        processor_ns["ImpactWildcardProcessor"],
        prompt_server.instance,
    )


def _payload(*, mode: str, wildcard_text: str, populated_text: str, seed=123):
    return {
        "prompt": {
            "12": {
                "class_type": "ImpactWildcardProcessor",
                "inputs": {
                    "wildcard_text": wildcard_text,
                    "populated_text": populated_text,
                    "mode": mode,
                    "seed": seed,
                },
            }
        }
    }


def test_real_impact_populate_handler_and_processor_match_reviewed_wildcard_engine(impact_contract):
    wildcards, handler, processor_cls, _server = impact_contract
    template = (
        "SHARED_ENV_SENTINEL __managed/scene__\n\n"
        "[0-5s]\nONE_BODY\n\n"
        "[5-10s]\nTWO_BODY"
    )
    expected = wildcards.process(template, 123)
    payload = _payload(
        mode="populate",
        wildcard_text=template,
        populated_text="STALE_POPULATED_TEXT",
        seed=123,
    )

    handler(payload)
    inputs = payload["prompt"]["12"]["inputs"]
    assert inputs["mode"] == "reproduce"
    assert inputs["wildcard_text"] == template
    assert inputs["populated_text"] == expected
    assert "[0-5s]" in expected and "[5-10s]" in expected

    # Impact execution processes populated_text again. Since queue expansion has
    # consumed the controlled wildcard, this must preserve the exact expanded
    # document passed downstream.
    assert processor_cls().doit(**inputs) == (expected,)


def test_real_impact_linked_seed_supported_queue_lookup_matches_literal_seed(impact_contract):
    wildcards, handler, processor_cls, _server = impact_contract
    template = "Linked __managed/scene__"
    expected = wildcards.process(template, 123)
    payload = _payload(
        mode="populate",
        wildcard_text=template,
        populated_text="STALE_POPULATED_TEXT",
        seed=["13", 0],
    )
    payload["prompt"]["13"] = {
        "class_type": "PrimitiveNode",
        "inputs": {"value": 123},
    }

    handler(payload)
    inputs = payload["prompt"]["12"]["inputs"]
    assert inputs["mode"] == "reproduce"
    assert inputs["populated_text"] == expected

    # ComfyUI resolves the linked INT before execution; the Processor then sees
    # the same seed value as the queue handler.
    execution_inputs = {**inputs, "seed": 123}
    assert processor_cls().doit(**execution_inputs) == (expected,)


def test_real_impact_unresolved_linked_seed_preserves_native_queue_skip_and_later_resolution(impact_contract):
    wildcards, handler, processor_cls, _server = impact_contract
    template = "Deferred __managed/scene__"
    payload = _payload(
        mode="populate",
        wildcard_text=template,
        populated_text=template,
        seed=["13", 0],
    )
    payload["prompt"]["13"] = {
        "class_type": "UnsupportedSeedSource",
        "inputs": {"text": "not an integer"},
    }

    handler(payload)
    inputs = payload["prompt"]["12"]["inputs"]
    assert inputs["mode"] == "populate"
    assert inputs["populated_text"] == template

    # Simulate normal executor resolution after Impact intentionally skipped
    # queue-time expansion. No derived/replacement seed is invented.
    execution_inputs = {**inputs, "seed": 123}
    assert processor_cls().doit(**execution_inputs) == (wildcards.process(template, 123),)


def test_real_impact_fixed_skips_queue_expansion_but_processes_populated_text_at_execution(impact_contract):
    wildcards, handler, processor_cls, _server = impact_contract
    populated = "Fixed __managed/scene__"
    payload = _payload(
        mode="fixed",
        wildcard_text="ignored __managed/scene__",
        populated_text=populated,
        seed=77,
    )

    handler(payload)
    inputs = payload["prompt"]["12"]["inputs"]
    assert inputs["mode"] == "fixed"
    assert inputs["populated_text"] == populated
    assert processor_cls().doit(**inputs) == (wildcards.process(populated, 77),)


def test_real_impact_reproduce_skips_queue_expansion_and_keeps_execution_semantics(impact_contract):
    wildcards, handler, processor_cls, server = impact_contract
    populated = "Reproduce __managed/scene__"
    payload = _payload(
        mode="reproduce",
        wildcard_text="ignored __managed/scene__",
        populated_text=populated,
        seed=91,
    )

    handler(payload)
    inputs = payload["prompt"]["12"]["inputs"]
    assert inputs["mode"] == "reproduce"
    assert inputs["populated_text"] == populated
    assert processor_cls().doit(**inputs) == (wildcards.process(populated, 91),)
    assert any(
        event == "impact-node-feedback"
        and item.get("widget_name") == "mode"
        and item.get("value") == "populate"
        for event, item in server.feedback
    )
