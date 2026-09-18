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
