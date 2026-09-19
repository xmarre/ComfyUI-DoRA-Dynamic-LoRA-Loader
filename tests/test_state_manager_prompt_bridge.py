import importlib
import json
import uuid

import pytest


def _persistent_character(text):
    return {
        "id": str(uuid.uuid4()),
        "name": "Managed",
        "thumbnail": {},
        "loader_stacks": [],
        "loras": [],
        "loader_globals": {},
        "prompts": [{
            "id": str(uuid.uuid4()),
            "name": "Prompt",
            "positive": text,
            "negative": "",
            "text_boxes": [
                {"role": "positive", "slot": "default", "label": "Positive", "text": text},
                {"role": "negative", "slot": "default", "label": "Negative", "text": ""},
            ],
            "settings": {"seed": 42},
            "reference_image": {},
            "fileimage_prefix": "",
        }],
    }


@pytest.fixture
def bridge(dora_modules):
    return importlib.import_module("dora_loader_testpkg.state_manager_prompt_bridge")


@pytest.fixture
def configured_nodes(dora_modules, tmp_path, monkeypatch):
    nodes, _ = dora_modules
    store_module = importlib.import_module("dora_loader_testpkg.state_manager_store")
    store_module.reset_state_manager_store_for_tests()
    monkeypatch.setattr(nodes.folder_paths, "get_user_directory", lambda: str(tmp_path))
    yield nodes
    store_module.reset_state_manager_store_for_tests()


def _prompt(nodes, character, *, mode="populate", impact_class="ImpactWildcardProcessor"):
    prompt_id = character["prompts"][0]["id"]
    return {
        "prompt": {
            "249": {
                "class_type": "State Manager",
                "inputs": {
                    "state_json": json.dumps(nodes._state_manager_default_binding()),
                    "ui_state_json": "",
                    "selected_character_id": character["id"],
                    "selected_prompt_id": prompt_id,
                },
            },
            "250": {
                "class_type": "State Manager Text Box",
                "inputs": {
                    "role": "positive",
                    "text": "stale text-box value",
                    "state_slot": "default",
                    "state_control": ["249", 7],
                },
            },
            "251": {
                "class_type": impact_class,
                "inputs": {
                    "wildcard_text": ["250", 0],
                    "populated_text": "stale populated value",
                    "mode": mode,
                    "seed": 123,
                },
            },
        }
    }


@pytest.mark.parametrize("mode", ["populate", "fixed", "reproduce"])
@pytest.mark.parametrize("impact_class", ["ImpactWildcardProcessor", "ImpactWildcardEncode"])
def test_backend_bridge_materializes_authoritative_timeline_for_impact_modes(
    bridge,
    configured_nodes,
    mode,
    impact_class,
):
    nodes = configured_nodes
    timeline = "Global.\n\n[0-7s]\nOne.\n\n[7-14s]\nTwo."
    character = _persistent_character(timeline)
    nodes._get_state_manager_store().replace([character], 0)
    payload = _prompt(nodes, character, mode=mode, impact_class=impact_class)

    changed = bridge.materialize_state_manager_impact_prompts(
        payload,
        resolve_payload=nodes._resolve_dora_state_payload,
        library_user_from_ui_state=nodes._queued_library_user_from_ui_state,
        text_for_box=nodes._state_payload_text_for_box,
    )

    assert changed == 3
    assert payload["prompt"]["250"]["inputs"]["text"] == timeline
    assert payload["prompt"]["251"]["inputs"]["wildcard_text"] == timeline
    assert payload["prompt"]["251"]["inputs"]["populated_text"] == timeline
    assert payload["prompt"]["251"]["inputs"]["mode"] == mode
    assert payload["prompt"]["251"]["inputs"]["seed"] == 123


def test_backend_bridge_recovers_source_after_frontend_literalized_impact_input(configured_nodes, bridge):
    nodes = configured_nodes
    timeline = "[0-7s]\nRecovered from persistent State Manager text."
    character = _persistent_character(timeline)
    nodes._get_state_manager_store().replace([character], 0)
    payload = _prompt(nodes, character)

    # Simulate the previous frontend bridge: it replaced the API-prompt link
    # with a concrete (and potentially stale) string after prompt serialization.
    payload["prompt"]["251"]["inputs"]["wildcard_text"] = "stale frontend literal"
    payload["prompt"]["251"]["inputs"]["populated_text"] = "stale frontend literal"
    payload["extra_data"] = {
        "extra_pnginfo": {
            "workflow": {
                "nodes": [
                    {
                        "id": 250,
                        "type": "State Manager Text Box",
                        "inputs": [
                            {"name": "state_control", "type": "STATE_MANAGER_CONTROL", "link": 12},
                        ],
                    },
                    {
                        "id": 251,
                        "type": "ImpactWildcardProcessor",
                        "inputs": [
                            {"name": "wildcard_text", "type": "STRING", "link": 13},
                        ],
                    },
                ],
                "links": [
                    [12, 249, 7, 250, 0, "STATE_MANAGER_CONTROL"],
                    [13, 250, 0, 251, 0, "STRING"],
                ],
            }
        }
    }

    changed = bridge.materialize_state_manager_impact_prompts(
        payload,
        resolve_payload=nodes._resolve_dora_state_payload,
        library_user_from_ui_state=nodes._queued_library_user_from_ui_state,
        text_for_box=nodes._state_payload_text_for_box,
    )

    assert changed == 3
    assert payload["prompt"]["250"]["inputs"]["text"] == timeline
    assert payload["prompt"]["251"]["inputs"]["wildcard_text"] == timeline
    assert payload["prompt"]["251"]["inputs"]["populated_text"] == timeline


def test_backend_bridge_uses_queued_library_user(configured_nodes, bridge):
    nodes = configured_nodes
    first = _persistent_character("wrong")
    second_text = "[0-7s]\nCorrect managed timeline."
    second = _persistent_character(second_text)
    nodes._get_state_manager_store("first-user").replace([first], 0)
    nodes._get_state_manager_store("second-user").replace([second], 0)

    payload = _prompt(nodes, second)
    payload["prompt"]["249"]["inputs"]["ui_state_json"] = json.dumps({
        "__dsm_library_user_id": "second-user",
    })

    bridge.materialize_state_manager_impact_prompts(
        payload,
        resolve_payload=nodes._resolve_dora_state_payload,
        library_user_from_ui_state=nodes._queued_library_user_from_ui_state,
        text_for_box=nodes._state_payload_text_for_box,
    )

    assert payload["prompt"]["251"]["inputs"]["populated_text"] == second_text


def test_backend_bridge_reads_frontend_contract_provenance(bridge):
    assert bridge._queued_frontend_contract(json.dumps({
        "__dsm_frontend_prompt_contract_version": 4,
        "__dsm_frontend_prompt_contract_revision": "backend-write-v1",
    })) == (4, "backend-write-v1")
    assert bridge._queued_frontend_contract("{}") == (0, "")


def test_backend_bridge_uses_request_local_selection_metadata(configured_nodes, bridge):
    nodes = configured_nodes
    stale = _persistent_character("stale prompt")
    timeline_text = "[0-7s]\nExact queued managed timeline."
    selected = _persistent_character(timeline_text)
    nodes._get_state_manager_store().replace([stale, selected], 0)

    payload = _prompt(nodes, stale)
    payload["prompt"]["249"]["inputs"]["ui_state_json"] = json.dumps({
        "__dsm_library_user_id": "default",
        "__dsm_queued_runtime_character_id": selected["id"],
        "__dsm_queued_runtime_prompt_id": selected["prompts"][0]["id"],
    })

    bridge.materialize_state_manager_impact_prompts(
        payload,
        resolve_payload=nodes._resolve_dora_state_payload,
        library_user_from_ui_state=nodes._queued_library_user_from_ui_state,
        text_for_box=nodes._state_payload_text_for_box,
    )

    assert payload["prompt"]["250"]["inputs"]["text"] == timeline_text
    assert payload["prompt"]["251"]["inputs"]["wildcard_text"] == timeline_text
    assert payload["prompt"]["251"]["inputs"]["populated_text"] == timeline_text


def test_backend_bridge_leaves_unrelated_impact_node_untouched(configured_nodes, bridge):
    nodes = configured_nodes
    timeline = "[0-7s]\nManaged."
    character = _persistent_character(timeline)
    nodes._get_state_manager_store().replace([character], 0)
    payload = _prompt(nodes, character)
    payload["prompt"]["252"] = {
        "class_type": "ImpactWildcardProcessor",
        "inputs": {
            "wildcard_text": "independent",
            "populated_text": "independent populated",
            "mode": "fixed",
            "seed": 9,
        },
    }

    bridge.materialize_state_manager_impact_prompts(
        payload,
        resolve_payload=nodes._resolve_dora_state_payload,
        library_user_from_ui_state=nodes._queued_library_user_from_ui_state,
        text_for_box=nodes._state_payload_text_for_box,
    )

    assert payload["prompt"]["252"]["inputs"]["wildcard_text"] == "independent"
    assert payload["prompt"]["252"]["inputs"]["populated_text"] == "independent populated"


def test_backend_bridge_fails_open_when_selected_preset_is_missing(configured_nodes, bridge):
    nodes = configured_nodes
    character = _persistent_character("[0-7s]\nManaged.")
    payload = _prompt(nodes, character)
    before = json.loads(json.dumps(payload))

    changed = bridge.materialize_state_manager_impact_prompts(
        payload,
        resolve_payload=nodes._resolve_dora_state_payload,
        library_user_from_ui_state=nodes._queued_library_user_from_ui_state,
        text_for_box=nodes._state_payload_text_for_box,
    )

    assert changed == 0
    assert payload == before


def test_prompt_bridge_registration_is_idempotent(bridge):
    class Server:
        def __init__(self):
            self.handlers = []

        def add_on_prompt_handler(self, handler):
            self.handlers.append(handler)

    class PromptServer:
        instance = Server()

    bridge.register_prompt_bridge(
        PromptServer,
        resolve_payload=lambda *_args: {},
        library_user_from_ui_state=lambda _value: "default",
        text_for_box=lambda *_args: None,
    )
    bridge.register_prompt_bridge(
        PromptServer,
        resolve_payload=lambda *_args: {},
        library_user_from_ui_state=lambda _value: "default",
        text_for_box=lambda *_args: None,
    )

    assert len(PromptServer.instance.handlers) == 1


def _install_fake_continuum_provider(monkeypatch):
    import sys
    import types

    provider = {
        "provider_version": 1,
        "managed_source_schema_versions": [1],
        "prompt_document_schema_versions": [1],
        "formats": ["inherit", "fixed", "list", "timeline"],
        "timeline_routings": ["logical_chunks", "physical_timeline"],
        "chunks": {"min": 1, "max": 16},
        "chunk_seconds": {"min": 4.0, "max": 15.0},
        "classify": lambda text: "timeline" if "[0-" in text else "fixed",
        "inspect": lambda text: {"text": text},
    }

    class FakeContinuum:
        H3_CONTINUUM_PROMPT_TRANSPORT_PROVIDER_V1 = provider

        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {"sequence_prompt": ("STRING",)},
                "optional": {"managed_prompt_source_json": ("STRING", {"default": ""})},
            }

    fake_nodes = types.SimpleNamespace(
        NODE_CLASS_MAPPINGS={"H3 Continuum Production": FakeContinuum}
    )
    monkeypatch.setitem(sys.modules, "nodes", fake_nodes)
    return provider


def _add_descriptor(nodes, character, text, descriptor):
    first = nodes._get_state_manager_store().replace([character], 0)
    result = nodes._get_state_manager_store().update_prompt_document(
        character["id"],
        character["prompts"][0]["id"],
        "positive",
        "default",
        text,
        descriptor,
        first["revision"],
        "Positive",
    )
    return result


def test_queue_snapshot_and_direct_continuum_sidecar_share_one_revision(
    configured_nodes, bridge, monkeypatch
):
    nodes = configured_nodes
    _install_fake_continuum_provider(monkeypatch)
    text = "Shared.\n\n[0-5s]\nONE\n\n[5-10s]\nTWO\n\n[10-15s]\nTHREE"
    character = _persistent_character(text)
    character["prompts"][0]["settings"] = {"seed": 77, "sampler": "keep"}
    character["prompts"][0]["reference_image"] = {
        "filename": "ref.png", "subfolder": "dora_state_manager", "type": "input"
    }
    descriptor = {
        "schema_version": 1,
        "format": "timeline",
        "routing": "logical_chunks",
        "geometry": {"chunks": 3, "chunk_seconds": "5"},
    }
    persisted = _add_descriptor(nodes, character, text, descriptor)

    payload = _prompt(nodes, character)
    payload["prompt"].pop("251")
    payload["prompt"]["249"]["inputs"]["ui_state_json"] = json.dumps({
        "__dsm_queue_snapshot_v1": {
            "version": 1,
            "library_revision": 9999,
            "character_id": "client",
            "prompt_id": "client",
            "payload": {"private": "untrusted"},
        }
    })
    payload["prompt"]["260"] = {
        "class_type": "H3 Continuum Production",
        "inputs": {
            "sequence_prompt": ["250", 0],
            "managed_prompt_source_json": "",
        },
    }

    changed = bridge.materialize_state_manager_impact_prompts(
        payload,
        resolve_payload=nodes._resolve_dora_state_payload,
        resolve_snapshot=nodes._resolve_dora_state_payload_snapshot,
        library_user_from_ui_state=nodes._queued_library_user_from_ui_state,
        text_for_box=nodes._state_payload_text_for_box,
        ordering_verified=True,
    )

    assert changed >= 3
    manager_ui = json.loads(payload["prompt"]["249"]["inputs"]["ui_state_json"])
    frozen = manager_ui["__dsm_queue_snapshot_v1"]
    assert frozen["library_revision"] == persisted["library_revision"]
    assert frozen["character_id"] == character["id"]
    assert frozen["prompt_id"] == character["prompts"][0]["id"]
    assert frozen["payload"]["settings"]["seed"] == 77
    assert frozen["payload"]["reference_image"]["filename"] == "ref.png"
    assert "private" not in frozen["payload"]

    sidecar = json.loads(payload["prompt"]["260"]["inputs"]["managed_prompt_source_json"])
    assert sidecar["magic"] == "DSM_H3_PROMPT_SOURCE"
    assert sidecar["schema_version"] == 1
    assert sidecar["text"] == text
    assert sidecar["prompt_document"] == descriptor
    assert sidecar["library_revision"] == persisted["library_revision"]
    assert sidecar["binding"] == {
        "manager_node": "249",
        "text_node": "250",
        "impact_node": None,
        "role": "positive",
        "slot": "default",
    }
    assert sidecar["queue_contract"] == "ordered-impact-v1"


def test_impact_to_continuum_sidecar_proves_exact_output_zero_path(
    configured_nodes, bridge, monkeypatch
):
    nodes = configured_nodes
    _install_fake_continuum_provider(monkeypatch)
    text = "Shared.\n\n[0-5s]\nONE\n\n[5-10s]\nTWO\n\n[10-15s]\nTHREE"
    character = _persistent_character(text)
    descriptor = {
        "schema_version": 1,
        "format": "timeline",
        "routing": "logical_chunks",
        "geometry": {"chunks": 3, "chunk_seconds": "5"},
    }
    persisted = _add_descriptor(nodes, character, text, descriptor)
    payload = _prompt(nodes, character, mode="populate")
    payload["prompt"]["260"] = {
        "class_type": "H3 Continuum Production",
        "inputs": {
            "sequence_prompt": ["251", 0],
            "managed_prompt_source_json": "",
        },
    }

    bridge.materialize_state_manager_impact_prompts(
        payload,
        resolve_payload=nodes._resolve_dora_state_payload,
        resolve_snapshot=nodes._resolve_dora_state_payload_snapshot,
        library_user_from_ui_state=nodes._queued_library_user_from_ui_state,
        text_for_box=nodes._state_payload_text_for_box,
        ordering_verified=True,
    )

    assert payload["prompt"]["251"]["inputs"]["wildcard_text"] == text
    assert payload["prompt"]["251"]["inputs"]["populated_text"] == text
    assert payload["prompt"]["251"]["inputs"]["mode"] == "populate"
    assert payload["prompt"]["251"]["inputs"]["seed"] == 123
    sidecar = json.loads(payload["prompt"]["260"]["inputs"]["managed_prompt_source_json"])
    assert sidecar["binding"]["impact_node"] == "251"
    assert sidecar["library_revision"] == persisted["library_revision"]


def test_nonzero_impact_output_never_receives_managed_provenance(
    configured_nodes, bridge, monkeypatch
):
    nodes = configured_nodes
    _install_fake_continuum_provider(monkeypatch)
    text = "[0-5s]\nONE"
    character = _persistent_character(text)
    _add_descriptor(
        nodes,
        character,
        text,
        {
            "schema_version": 1,
            "format": "timeline",
            "routing": "logical_chunks",
            "geometry": {"chunks": 1, "chunk_seconds": "5"},
        },
    )
    payload = _prompt(nodes, character)
    payload["prompt"]["260"] = {
        "class_type": "H3 Continuum Production",
        "inputs": {
            "sequence_prompt": ["251", 1],
            "managed_prompt_source_json": "",
        },
    }

    bridge.materialize_state_manager_impact_prompts(
        payload,
        resolve_payload=nodes._resolve_dora_state_payload,
        resolve_snapshot=nodes._resolve_dora_state_payload_snapshot,
        library_user_from_ui_state=nodes._queued_library_user_from_ui_state,
        text_for_box=nodes._state_payload_text_for_box,
        ordering_verified=True,
    )

    assert payload["prompt"]["251"]["inputs"]["wildcard_text"] == text
    assert payload["prompt"]["260"]["inputs"]["managed_prompt_source_json"] == ""


def test_transport_receipts_are_bounded_and_do_not_log_prompt_text(
    configured_nodes, bridge, monkeypatch, caplog
):
    nodes = configured_nodes
    _install_fake_continuum_provider(monkeypatch)
    text = (
        "PRIVATE_SHARED_SENTINEL\n\n"
        "[0-5s]\nPRIVATE_ONE_SENTINEL"
    )
    character = _persistent_character(text)
    descriptor = {
        "schema_version": 1,
        "format": "timeline",
        "routing": "logical_chunks",
        "geometry": {"chunks": 1, "chunk_seconds": "5"},
    }
    persisted = _add_descriptor(nodes, character, text, descriptor)
    payload = _prompt(nodes, character, mode="populate")
    submitted_populated = "PRIVATE_STALE_POPULATED_SENTINEL"
    payload["prompt"]["251"]["inputs"]["populated_text"] = submitted_populated
    payload["prompt"]["260"] = {
        "class_type": "H3 Continuum Production",
        "inputs": {
            "sequence_prompt": ["251", 0],
            "managed_prompt_source_json": "",
        },
    }

    with caplog.at_level("INFO", logger=bridge.__name__):
        bridge.materialize_state_manager_impact_prompts(
            payload,
            resolve_payload=nodes._resolve_dora_state_payload,
            resolve_snapshot=nodes._resolve_dora_state_payload_snapshot,
            library_user_from_ui_state=nodes._queued_library_user_from_ui_state,
            text_for_box=nodes._state_payload_text_for_box,
            ordering_verified=True,
        )

    messages = "\n".join(record.getMessage() for record in caplog.records if record.name == bridge.__name__)
    assert "managed prompt queue receipt transport=v1" in messages
    assert f"snapshot_revision={persisted['library_revision']}" in messages
    assert "format='timeline'" in messages
    assert "routing='logical_chunks'" in messages
    assert "ordering_verified=True" in messages
    assert "managed Impact edge receipt transport=v1" in messages
    assert "seed_provenance=literal" in messages
    assert "expansion_state=pending_native_populate" in messages
    assert (
        "submitted_populated_sha256="
        + __import__("hashlib").sha256(submitted_populated.encode("utf-8")).hexdigest()
    ) in messages
    assert "managed consumer sidecar receipt transport=v1" in messages
    assert "queue_contract=ordered-impact-v1" in messages
    assert "PRIVATE_SHARED_SENTINEL" not in messages
    assert "PRIVATE_ONE_SENTINEL" not in messages
    assert "PRIVATE_STALE_POPULATED_SENTINEL" not in messages


def test_unknown_transform_never_receives_or_forwards_managed_provenance(
    configured_nodes, bridge, monkeypatch
):
    nodes = configured_nodes
    _install_fake_continuum_provider(monkeypatch)
    text = "[0-5s]\nONE"
    character = _persistent_character(text)
    _add_descriptor(
        nodes,
        character,
        text,
        {
            "schema_version": 1,
            "format": "timeline",
            "routing": "logical_chunks",
            "geometry": {"chunks": 1, "chunk_seconds": "5"},
        },
    )
    payload = _prompt(nodes, character)
    payload["prompt"].pop("251")
    payload["prompt"]["255"] = {
        "class_type": "Unknown String Transform",
        "inputs": {"text": ["250", 0]},
    }
    payload["prompt"]["260"] = {
        "class_type": "H3 Continuum Production",
        "inputs": {
            "sequence_prompt": ["255", 0],
            "managed_prompt_source_json": "",
        },
    }

    bridge.materialize_state_manager_impact_prompts(
        payload,
        resolve_payload=nodes._resolve_dora_state_payload,
        resolve_snapshot=nodes._resolve_dora_state_payload_snapshot,
        library_user_from_ui_state=nodes._queued_library_user_from_ui_state,
        text_for_box=nodes._state_payload_text_for_box,
        ordering_verified=True,
    )
    assert payload["prompt"]["260"]["inputs"]["managed_prompt_source_json"] == ""


def test_registered_handler_rolls_back_all_staged_mutations_on_late_failure(
    configured_nodes, bridge
):
    import copy

    nodes = configured_nodes
    character = _persistent_character("[0-5s]\nONE")
    nodes._get_state_manager_store().replace([character], 0)
    payload = _prompt(nodes, character)
    before = copy.deepcopy(payload)

    class Server:
        def __init__(self):
            self.on_prompt_handlers = []

        def add_on_prompt_handler(self, handler):
            self.on_prompt_handlers.append(handler)

    class PromptServer:
        instance = Server()

    def explode_after_snapshot(*_args):
        raise RuntimeError("late text resolution failure")

    bridge.register_prompt_bridge(
        PromptServer,
        resolve_payload=nodes._resolve_dora_state_payload,
        resolve_snapshot=nodes._resolve_dora_state_payload_snapshot,
        library_user_from_ui_state=nodes._queued_library_user_from_ui_state,
        text_for_box=explode_after_snapshot,
    )
    result = PromptServer.instance.on_prompt_handlers[0](payload)

    assert result is payload
    assert payload == before
    assert "__dsm_queue_snapshot_v1" not in payload["prompt"]["249"]["inputs"]["ui_state_json"]


def test_ordered_transport_capability_tracks_verified_handler_surface(bridge):
    class FallbackServer:
        def __init__(self):
            self.handlers = []

        def add_on_prompt_handler(self, handler):
            self.handlers.append(handler)

    class FallbackPromptServer:
        instance = FallbackServer()

    bridge.register_prompt_bridge(
        FallbackPromptServer,
        resolve_payload=lambda *_args: {},
        library_user_from_ui_state=lambda _value: "default",
        text_for_box=lambda *_args: None,
    )
    assert bridge.prompt_transport_ordering_contract() is None

    class OrderedServer:
        def __init__(self):
            self.on_prompt_handlers = []

        def add_on_prompt_handler(self, handler):
            self.on_prompt_handlers.append(handler)

    class OrderedPromptServer:
        instance = OrderedServer()

    bridge.register_prompt_bridge(
        OrderedPromptServer,
        resolve_payload=lambda *_args: {},
        library_user_from_ui_state=lambda _value: "default",
        text_for_box=lambda *_args: None,
    )
    assert bridge.prompt_transport_ordering_contract() == "ordered-impact-v1"


def test_handler_order_receipt_is_bounded_and_names_handlers(bridge):
    handlers = []
    for index in range(40):
        def handler(value, _index=index):
            return value
        handler.__name__ = f"handler_{index}"
        handlers.append(handler)

    receipt = bridge._handler_order_receipt(handlers)
    assert receipt["count"] == 40
    assert len(receipt["shown"]) == 32
    assert receipt["truncated"] is True
    assert receipt["shown"][0]["name"].endswith("handler")


def test_handler_registration_remains_first_when_impact_registers_later(bridge):
    def impact_handler(value):
        return value

    class Server:
        def __init__(self):
            self.on_prompt_handlers = []

        def add_on_prompt_handler(self, handler):
            self.on_prompt_handlers.append(handler)

    class PromptServer:
        instance = Server()

    bridge.register_prompt_bridge(
        PromptServer,
        resolve_payload=lambda *_args: {},
        resolve_snapshot=lambda *_args: {
            "version": 1,
            "library_revision": 0,
            "character_id": "",
            "prompt_id": "",
            "payload": {},
        },
        library_user_from_ui_state=lambda _value: "default",
        text_for_box=lambda *_args: None,
    )
    owned = PromptServer.instance.on_prompt_handlers[0]
    PromptServer.instance.add_on_prompt_handler(impact_handler)

    assert PromptServer.instance.on_prompt_handlers == [owned, impact_handler]


def test_handler_registration_prepends_without_reordering_other_handlers(bridge):
    calls = []

    def impact_handler(value):
        calls.append("impact")
        return value

    def other_handler(value):
        calls.append("other")
        return value

    class Server:
        def __init__(self):
            self.on_prompt_handlers = [impact_handler, other_handler]

        def add_on_prompt_handler(self, handler):
            self.on_prompt_handlers.append(handler)

    class PromptServer:
        instance = Server()

    bridge.register_prompt_bridge(
        PromptServer,
        resolve_payload=lambda *_args: {},
        resolve_snapshot=lambda *_args: {
            "version": 1,
            "library_revision": 0,
            "character_id": "",
            "prompt_id": "",
            "payload": {},
        },
        library_user_from_ui_state=lambda _value: "default",
        text_for_box=lambda *_args: None,
    )
    first_owned = PromptServer.instance.on_prompt_handlers[0]
    assert PromptServer.instance.on_prompt_handlers[1:] == [impact_handler, other_handler]

    bridge.register_prompt_bridge(
        PromptServer,
        resolve_payload=lambda *_args: {},
        resolve_snapshot=lambda *_args: {
            "version": 1,
            "library_revision": 0,
            "character_id": "",
            "prompt_id": "",
            "payload": {},
        },
        library_user_from_ui_state=lambda _value: "default",
        text_for_box=lambda *_args: None,
    )
    assert len(PromptServer.instance.on_prompt_handlers) == 3
    assert PromptServer.instance.on_prompt_handlers[0] is not first_owned
    assert PromptServer.instance.on_prompt_handlers[1:] == [impact_handler, other_handler]
