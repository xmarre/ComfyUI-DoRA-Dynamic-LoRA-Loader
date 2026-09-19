from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import types
from types import SimpleNamespace
import uuid

import pytest


IMPACT_SOURCE = os.environ.get("IMPACT_PACK_SOURCE")
CONTINUUM_SOURCE = os.environ.get("H3_CONTINUUM_SOURCE")


def _load_continuum_contract(source_root: Path):
    package_name = "_reviewed_h3_continuum_transport"
    package = types.ModuleType(package_name)
    package.__path__ = [str(source_root)]
    sys.modules[package_name] = package

    v2_name = f"{package_name}.v2"
    v2_package = types.ModuleType(v2_name)
    v2_package.__path__ = [str(source_root / "v2")]
    sys.modules[v2_name] = v2_package

    modules = {}
    for short_name, path in (
        ("constants", source_root / "constants.py"),
        ("version", source_root / "version.py"),
        ("v2.prompts", source_root / "v2" / "prompts.py"),
        ("v2.physical_prompts", source_root / "v2" / "physical_prompts.py"),
        ("v2.prompt_transport", source_root / "v2" / "prompt_transport.py"),
    ):
        full_name = f"{package_name}.{short_name}"
        spec = importlib.util.spec_from_file_location(full_name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load reviewed Continuum module {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = module
        spec.loader.exec_module(module)
        modules[short_name] = module
    physical_runtime = importlib.import_module(f"{package_name}.v2.physical_runtime")
    masked_continuation = importlib.import_module(f"{package_name}.masked_continuation")
    return (
        modules["v2.prompts"],
        modules["v2.prompt_transport"],
        modules["v2.physical_prompts"],
        physical_runtime,
        masked_continuation,
    )


def _compile_reviewed_defs(path: Path, names: set[str], namespace: dict):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    selected = [
        node for node in tree.body
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


def _load_real_impact(source_root: Path):
    saved = {name: sys.modules.get(name) for name in ("nodes", "folder_paths", "impact", "impact.config", "impact.utils")}

    folder_stub = types.ModuleType("folder_paths")
    nodes_stub = types.ModuleType("nodes")
    sys.modules["folder_paths"] = folder_stub
    sys.modules["nodes"] = nodes_stub

    impact = types.ModuleType("impact")
    impact.__path__ = []
    config = types.ModuleType("impact.config")
    utils = types.ModuleType("impact.utils")
    impact.config = config
    impact.utils = utils
    sys.modules["impact"] = impact
    sys.modules["impact.config"] = config
    sys.modules["impact.utils"] = utils

    try:
        wildcard_path = source_root / "modules" / "impact" / "wildcards.py"
        spec = importlib.util.spec_from_file_location("_reviewed_impact_wildcards_e2e", wildcard_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load {wildcard_path}")
        wildcards = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = wildcards
        spec.loader.exec_module(wildcards)
        impact.wildcards = wildcards

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
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _character(text: str):
    character_id = str(uuid.uuid4())
    prompt_id = str(uuid.uuid4())
    return {
        "id": character_id,
        "name": "Managed transport integration",
        "thumbnail": {},
        "loader_stacks": [],
        "loras": [],
        "loader_globals": {},
        "prompts": [{
            "id": prompt_id,
            "name": "Sequence",
            "positive": text,
            "negative": "",
            "text_boxes": [
                {"role": "positive", "slot": "default", "label": "Sequence", "text": text},
                {"role": "negative", "slot": "default", "label": "Negative", "text": ""},
            ],
            "settings": {"seed": 123},
            "reference_image": {},
            "fileimage_prefix": "",
        }],
    }


def _queue(nodes, character, *, mode="populate"):
    return {
        "prompt": {
            "249": {
                "class_type": "State Manager",
                "inputs": {
                    "state_json": json.dumps(nodes._state_manager_default_binding()),
                    "ui_state_json": "",
                    "selected_character_id": character["id"],
                    "selected_prompt_id": character["prompts"][0]["id"],
                },
            },
            "250": {
                "class_type": "State Manager Text Box",
                "inputs": {
                    "role": "positive",
                    "text": "stale",
                    "state_slot": "default",
                    "state_control": ["249", 7],
                },
            },
            "251": {
                "class_type": "ImpactWildcardProcessor",
                "inputs": {
                    "wildcard_text": ["250", 0],
                    "populated_text": "stale populated",
                    "mode": mode,
                    "seed": 123,
                },
            },
            "260": {
                "class_type": "H3 Continuum Production",
                "inputs": {
                    "sequence_prompt": ["251", 0],
                    "managed_prompt_source_json": "",
                },
            },
        }
    }


def _install_document(nodes, character, text, descriptor):
    store = nodes._get_state_manager_store()
    initial = store.replace([character], 0)
    result = store.update_prompt_document(
        character["id"],
        character["prompts"][0]["id"],
        "positive",
        "default",
        text,
        descriptor,
        initial["revision"],
        "Sequence",
    )
    return result


@pytest.fixture()
def reviewed_stack(dora_modules, tmp_path, monkeypatch):
    if not IMPACT_SOURCE or not CONTINUUM_SOURCE:
        pytest.skip("IMPACT_PACK_SOURCE and H3_CONTINUUM_SOURCE are required")

    nodes, _runtime = dora_modules
    store_module = importlib.import_module("dora_loader_testpkg.state_manager_store")
    bridge = importlib.import_module("dora_loader_testpkg.state_manager_prompt_bridge")
    store_module.reset_state_manager_store_for_tests()
    monkeypatch.setattr(nodes.folder_paths, "get_user_directory", lambda: str(tmp_path))

    prompts, transport, physical_prompts, physical_runtime, masked_continuation = (
        _load_continuum_contract(Path(CONTINUUM_SOURCE).resolve())
    )
    wildcards, impact_handler, processor_cls, feedback = _load_real_impact(Path(IMPACT_SOURCE).resolve())

    class ContinuumConsumer:
        H3_CONTINUUM_PROMPT_TRANSPORT_PROVIDER_V1 = transport.PROMPT_TRANSPORT_PROVIDER_V1

        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {"sequence_prompt": ("STRING",)},
                "optional": {"managed_prompt_source_json": ("STRING", {"default": ""})},
            }

    comfy_nodes = importlib.import_module("nodes")
    monkeypatch.setitem(comfy_nodes.NODE_CLASS_MAPPINGS, "H3 Continuum Production", ContinuumConsumer)

    wildcards.wildcard_dict.clear()
    wildcards.wildcard_dict.update({
        "managed/shared": ["SHARED_ENV_SENTINEL"],
        "managed/one": ["ONE_RED_CUBE_SENTINEL"],
        "managed/two": ["TWO_GREEN_SPHERE_SENTINEL"],
        "managed/three": ["THREE_BLUE_PYRAMID_SENTINEL"],
        "managed/header": ["[7-8s]\nINJECTED_HEADER_SENTINEL"],
    })
    wildcards.available_wildcards.clear()
    wildcards.loaded_wildcards.clear()
    wildcards._on_demand_mode = False

    yield (
        nodes,
        bridge,
        prompts,
        impact_handler,
        processor_cls,
        feedback,
        physical_prompts,
        physical_runtime,
        masked_continuation,
    )
    store_module.reset_state_manager_store_for_tests()


def _bridge_and_expand(stack, character, descriptor, *, mode="populate"):
    nodes, bridge, _prompts, impact_handler, processor_cls, _feedback, *_continuum = stack
    raw = character["prompts"][0]["positive"]
    persisted = _install_document(nodes, character, raw, descriptor)
    payload = _queue(nodes, character, mode=mode)

    bridge.materialize_state_manager_impact_prompts(
        payload,
        resolve_payload=nodes._resolve_dora_state_payload,
        resolve_snapshot=nodes._resolve_dora_state_payload_snapshot,
        library_user_from_ui_state=nodes._queued_library_user_from_ui_state,
        text_for_box=nodes._state_payload_text_for_box,
        ordering_verified=True,
    )
    sidecar = json.loads(payload["prompt"]["260"]["inputs"]["managed_prompt_source_json"])
    assert sidecar["library_revision"] == persisted["library_revision"]
    assert sidecar["raw_text_sha256"] == hashlib.sha256(raw.encode("utf-8")).hexdigest()

    impact_handler(payload)
    inputs = payload["prompt"]["251"]["inputs"]
    expanded = processor_cls().doit(**inputs)[0]
    return payload, sidecar, expanded


def test_direct_state_manager_real_impact_to_continuum_sequence_contract(reviewed_stack):
    _nodes, _bridge, prompts, _impact_handler, _processor, _feedback, *_continuum = reviewed_stack
    raw = (
        "__managed/shared__\n\n"
        "[0-5s]\n__managed/one__\n\n"
        "[5-10s]\n__managed/two__\n\n"
        "[10-15s]\n__managed/three__"
    )
    character = _character(raw)
    descriptor = {
        "schema_version": 1,
        "format": "timeline",
        "routing": "logical_chunks",
        "geometry": {"chunks": 3, "chunk_seconds": "5"},
    }
    payload, sidecar, expanded = _bridge_and_expand(reviewed_stack, character, descriptor)

    impact_inputs = payload["prompt"]["251"]["inputs"]
    assert impact_inputs["mode"] == "reproduce"
    assert impact_inputs["wildcard_text"] == raw
    assert "__managed/" not in expanded
    assert sidecar["text"] == raw

    plan = prompts.build_sampler_prompt_plan(
        prompt_mode="Auto",
        prompt_script="legacy",
        sequence_prompt=expanded,
        prompt_plan=None,
        chunks=3,
        chunk_seconds=5.0,
        managed_prompt_source_json=json.dumps(sidecar),
    )
    assert plan["managed_prompt_transport"]["status"] == "verified_sequence"
    assert plan["managed_prompt_transport"]["geometry_match"] is True
    assert plan["managed_prompt_transport"]["skeleton_match"] is True
    assert plan["prompts"] == [
        "SHARED_ENV_SENTINEL\n\nONE_RED_CUBE_SENTINEL",
        "SHARED_ENV_SENTINEL\n\nTWO_GREEN_SPHERE_SENTINEL",
        "SHARED_ENV_SENTINEL\n\nTHREE_BLUE_PYRAMID_SENTINEL",
    ]

    # Carry the exact plan produced by real Store -> queue bridge -> Impact
    # through Continuum's real physical compiler to the Qwen/CLIP input boundary.
    physical_prompts, physical_runtime, masked_continuation = reviewed_stack[6:]
    descriptor = physical_prompts.make_physical_sample_descriptor(
        group_id="managed-e2e-chunk-2",
        logical_indices=(1,),
        retained_before=120,
        context_frames=24,
        total_frames=144,
        target_duration_frames=360,
        continuation_method=masked_continuation.CONTINUATION_GUIDE,
        initial_state_origin="sequence",
        include_first=False,
        include_last=False,
        presentation_contract={"include_first": False, "include_last": False},
        guided_overlap=True,
    )
    original_flag = physical_runtime.physical_prompt_compiler_enabled
    physical_runtime.physical_prompt_compiler_enabled = lambda: True

    class CaptureClip:
        def __init__(self):
            self.prompt = None

        def tokenize(self, prompt, **_kwargs):
            self.prompt = prompt
            return prompt

        def encode_from_tokens_scheduled(self, tokens):
            return [["conditioning", {"captured": tokens}]]

    clip = CaptureClip()
    assets = SimpleNamespace(first_image=None, last_image=None)
    try:
        _conditioning, compiled, _metadata, _cache_key = (
            physical_runtime.encode_physical_prompt_conditioning(
                clip=clip,
                plan=plan,
                descriptor=descriptor,
                legacy_text=plan["prompts"][1],
                assets=assets,
                include_first=False,
                include_last=False,
            )
        )
    finally:
        physical_runtime.physical_prompt_compiler_enabled = original_flag

    assert clip.prompt == compiled.text
    assert "SHARED_ENV_SENTINEL" in clip.prompt
    assert "TWO_GREEN_SPHERE_SENTINEL" in clip.prompt
    assert "ONE_RED_CUBE_SENTINEL" not in clip.prompt
    assert "THREE_BLUE_PYRAMID_SENTINEL" not in clip.prompt


def test_real_impact_header_injection_cannot_become_verified_schedule(reviewed_stack):
    _nodes, _bridge, prompts, _impact_handler, _processor, _feedback = reviewed_stack
    raw = (
        "SHARED_ENV_SENTINEL\n\n"
        "[0-5s]\nONE_RED_CUBE_SENTINEL\n\n"
        "[5-10s]\n__managed/header__\n\n"
        "[10-15s]\nTHREE_BLUE_PYRAMID_SENTINEL"
    )
    character = _character(raw)
    descriptor = {
        "schema_version": 1,
        "format": "timeline",
        "routing": "logical_chunks",
        "geometry": {"chunks": 3, "chunk_seconds": "5"},
    }
    _payload, sidecar, expanded = _bridge_and_expand(reviewed_stack, character, descriptor)

    plan = prompts.build_sampler_prompt_plan(
        prompt_mode="Auto",
        prompt_script="legacy",
        sequence_prompt=expanded,
        prompt_plan=None,
        chunks=3,
        chunk_seconds=5.0,
        managed_prompt_source_json=json.dumps(sidecar),
    )
    receipt = plan["managed_prompt_transport"]
    assert receipt["status"] == "fallback_fixed"
    assert receipt["sequence_verified"] is False
    assert receipt["skeleton_match"] is False
    assert plan["prompts"] == [expanded, expanded, expanded]


def test_explicit_fixed_document_stays_fixed_through_real_impact(reviewed_stack):
    _nodes, _bridge, prompts, _impact_handler, _processor, _feedback = reviewed_stack
    raw = "[0-5s]\nLiteral header-looking prose __managed/one__"
    character = _character(raw)
    descriptor = {"schema_version": 1, "format": "fixed"}
    _payload, sidecar, expanded = _bridge_and_expand(
        reviewed_stack,
        character,
        descriptor,
        mode="fixed",
    )

    plan = prompts.build_sampler_prompt_plan(
        prompt_mode="Auto",
        prompt_script="legacy",
        sequence_prompt=expanded,
        prompt_plan=None,
        chunks=3,
        chunk_seconds=5.0,
        managed_prompt_source_json=json.dumps(sidecar),
    )
    assert plan["mode"] == prompts.PROMPT_MODE_FIXED
    assert plan["prompts"] == [expanded, expanded, expanded]
    assert plan["managed_prompt_transport"]["status"] == "document_applied"
    assert plan["managed_prompt_transport"]["sequence_verified"] is False
