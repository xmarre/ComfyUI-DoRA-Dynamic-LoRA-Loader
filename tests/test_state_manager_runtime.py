import json
import uuid

import pytest


def persistent_character():
    return {
        "id": str(uuid.uuid4()),
        "name": "Shared Group",
        "thumbnail": {},
        "loader_stacks": [{
            "slot": "default",
            "label": "Default loader",
            "loras": [{
                "enabled": True,
                "name": "runtime.safetensors",
                "strength_model": 0.75,
                "strength_clip": 0.5,
            }],
            "loader_globals": {"auto_strength_enabled": True},
        }],
        "loras": [],
        "loader_globals": {},
        "prompts": [{
            "id": str(uuid.uuid4()),
            "name": "Runtime Character",
            "positive": "runtime positive",
            "negative": "runtime negative",
            "text_boxes": [
                {"role": "positive", "slot": "default", "label": "Positive", "text": "runtime positive"},
                {"role": "negative", "slot": "default", "label": "Negative", "text": "runtime negative"},
            ],
            "settings": {"seed": 42},
            "reference_image": {},
            "fileimage_prefix": "runtime/output",
        }],
    }


@pytest.fixture
def configured_nodes(dora_modules, tmp_path, monkeypatch):
    nodes, _ = dora_modules
    import dora_loader_testpkg.state_manager_store as store_module

    store_module.reset_state_manager_store_for_tests()
    monkeypatch.setattr(nodes.folder_paths, "get_user_directory", lambda: str(tmp_path))
    yield nodes
    store_module.reset_state_manager_store_for_tests()


def test_runtime_resolves_persistent_library_payload(configured_nodes):
    nodes = configured_nodes
    character = persistent_character()
    store = nodes._get_state_manager_store()
    store.replace([character], 0)
    payload = nodes._resolve_dora_state_payload(
        json.dumps(nodes._state_manager_default_binding()),
        character["id"],
        character["prompts"][0]["id"],
    )
    assert payload["positive_prompt"] == "runtime positive"
    assert payload["negative_prompt"] == "runtime negative"
    assert payload["loras"][0]["name"] == "runtime.safetensors"
    assert payload["settings"]["seed"] == 42
    assert payload["fileimage_prefix"] == "runtime/output"


def test_runtime_missing_uuid_is_explicit(configured_nodes):
    nodes = configured_nodes
    character = persistent_character()
    nodes._get_state_manager_store().replace([character], 0)
    result = nodes.StateManager.VALIDATE_INPUTS(
        json.dumps(nodes._state_manager_default_binding()),
        "",
        str(uuid.uuid4()),
        character["prompts"][0]["id"],
    )
    assert isinstance(result, str)
    assert "not available locally" in result


def test_queued_runtime_seed_changes_seed_without_embedded_payload(configured_nodes):
    nodes = configured_nodes
    character = persistent_character()
    nodes._get_state_manager_store().replace([character], 0)
    manager = nodes.StateManager()
    result = manager.resolve_state(
        json.dumps(nodes._state_manager_default_binding()),
        json.dumps({"__dsm_runtime_seed": 987654, "__dsm_queued_runtime_nonce": "queue-1"}),
        character["id"],
        character["prompts"][0]["id"],
    )
    assert result[6] == 987654
    assert result[0]["positive_prompt"] == "runtime positive"


@pytest.mark.parametrize("special_seed", [-1, -2, -3])
def test_special_seed_semantics_remain_randomized(configured_nodes, special_seed):
    nodes = configured_nodes
    character = persistent_character()
    character["prompts"][0]["settings"]["seed"] = special_seed
    nodes._get_state_manager_store().replace([character], 0)
    result = nodes.StateManager().resolve_state(
        json.dumps(nodes._state_manager_default_binding()),
        "",
        character["id"],
        character["prompts"][0]["id"],
    )
    assert 1 <= result[6] <= nodes._STATE_SEED_MAX


def test_two_managers_can_resolve_same_library_selection(configured_nodes):
    nodes = configured_nodes
    character = persistent_character()
    nodes._get_state_manager_store().replace([character], 0)
    args = (
        json.dumps(nodes._state_manager_default_binding()),
        "",
        character["id"],
        character["prompts"][0]["id"],
    )
    first = nodes.StateManager().resolve_state(*args)
    second = nodes.StateManager().resolve_state(*args)
    assert first[0] == second[0]
    assert first[1:4] == second[1:4]


def test_legacy_migration_keeps_an_existing_persistent_uuid_selection(configured_nodes):
    nodes = configured_nodes
    selected = persistent_character()
    legacy = persistent_character()
    nodes._get_state_manager_store().replace([selected], 0)
    character, prompt = nodes._resolve_state_manager_selection(
        json.dumps({"version": 3, "characters": [legacy]}),
        selected["id"],
        selected["prompts"][0]["id"],
    )
    assert character["id"] == selected["id"]
    assert prompt["id"] == selected["prompts"][0]["id"]


def test_is_changed_returns_stable_marker_for_missing_preset(configured_nodes):
    nodes = configured_nodes
    marker = json.loads(nodes.StateManager.IS_CHANGED(
        json.dumps(nodes._state_manager_default_binding()),
        "",
        str(uuid.uuid4()),
        str(uuid.uuid4()),
    ))
    assert marker["library_error"] == "StatePresetNotFound"


def test_is_changed_returns_stable_marker_for_library_io_error(configured_nodes, monkeypatch):
    nodes = configured_nodes
    store = nodes._get_state_manager_store()

    def fail_resolve(*_args, **_kwargs):
        raise OSError("sharing violation")

    monkeypatch.setattr(store, "resolve", fail_resolve)
    marker = json.loads(nodes.StateManager.IS_CHANGED(
        json.dumps(nodes._state_manager_default_binding()),
        "",
        "default_character",
        "default_prompt",
    ))
    assert marker["library_error"] == "OSError"


def test_queued_user_id_selects_an_isolated_library(configured_nodes):
    nodes = configured_nodes
    first = persistent_character()
    second = persistent_character()
    nodes._get_state_manager_store("first-user").replace([first], 0)
    nodes._get_state_manager_store("second-user").replace([second], 0)
    result = nodes.StateManager().resolve_state(
        json.dumps(nodes._state_manager_default_binding()),
        json.dumps({"__dsm_library_user_id": "first-user"}),
        first["id"],
        first["prompts"][0]["id"],
    )
    assert result[0]["character"]["id"] == first["id"]


def test_explicit_library_import_kind_wins_over_stray_character_field(dora_modules):
    import dora_loader_testpkg.state_manager_api as api

    calls = []

    class Store:
        def merge_library(self, characters):
            calls.append(("library", characters))
            return {"characters": characters}

        def import_character(self, character):
            calls.append(("character", character))
            return {"character": character}

    characters = [persistent_character()]
    result = api._import_payload(Store(), {
        "kind": "dora_state_manager_library_export",
        "characters": characters,
        "character": {"id": "stray"},
    })
    assert calls == [("library", characters)]
    assert result["characters"] == characters
