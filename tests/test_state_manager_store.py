import json
import os
import sys
import threading
import uuid
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from state_manager_store import (
    InvalidStateLibrary,
    StateLibraryRevisionConflict,
    StateLibraryStore,
    StatePresetNotFound,
    UnsupportedStateLibraryVersion,
)


def normalize_state(raw):
    characters = []
    for character in raw.get("characters", []):
        value = json.loads(json.dumps(character))
        value.setdefault("thumbnail", {})
        value.setdefault("loader_stacks", [])
        value.setdefault("loras", [])
        value.setdefault("loader_globals", {})
        value.setdefault("prompts", [])
        for prompt in value["prompts"]:
            prompt.setdefault("positive", "")
            prompt.setdefault("negative", "")
            prompt.setdefault("text_boxes", [])
            prompt.setdefault("settings", {})
            prompt.setdefault("reference_image", {})
            prompt.setdefault("fileimage_prefix", "")
        characters.append(value)
    return {"version": 3, "characters": characters}


def default_state():
    return {
        "version": 3,
        "characters": [{
            "id": "default_character",
            "name": "Default Character",
            "thumbnail": {},
            "loader_stacks": [],
            "loras": [],
            "loader_globals": {},
            "prompts": [{
                "id": "default_prompt",
                "name": "Default Prompt",
                "positive": "",
                "negative": "",
                "text_boxes": [],
                "settings": {},
                "reference_image": {},
                "fileimage_prefix": "",
            }],
        }],
    }


def make_character(name="Private Character", *, character_id=None, prompt_id=None):
    return {
        "id": character_id or str(uuid.uuid4()),
        "name": name,
        "thumbnail": {"filename": "private.png", "subfolder": "dora_state_manager", "type": "input"},
        "loader_stacks": [{
            "slot": "default",
            "label": "Default loader",
            "loras": [{"enabled": True, "name": "private.safetensors", "strength_model": 0.8, "strength_clip": 0.7}],
            "loader_globals": {"auto_strength_enabled": True},
        }],
        "loras": [],
        "loader_globals": {},
        "prompts": [{
            "id": prompt_id or str(uuid.uuid4()),
            "name": f"{name} Prompt",
            "positive": "private positive prompt",
            "negative": "private negative prompt",
            "text_boxes": [{"role": "positive", "slot": "default", "label": "Main", "text": "private positive prompt"}],
            "settings": {"seed": -2, "nodes": [{"key": "sampler", "widgets": {"steps": 20}}]},
            "reference_image": {"filename": "reference.png", "subfolder": "dora_state_manager", "type": "input"},
            "fileimage_prefix": "private/output",
        }],
    }


@pytest.fixture
def store(tmp_path):
    return StateLibraryStore(str(tmp_path / "dora_state_manager" / "state-library.json"), normalize_state, default_state)


def test_persistence_survives_reinitialization_and_delete(store):
    character = make_character()
    first = store.replace([character], 0)
    assert first["revision"] == 1
    reloaded = StateLibraryStore(store.path, normalize_state, default_state)
    assert reloaded.snapshot()["characters"] == first["characters"]

    second = reloaded.replace([], 1)
    assert second == {"version": 1, "revision": 2, "characters": []}
    assert StateLibraryStore(store.path, normalize_state, default_state).snapshot() == second


def test_atomic_write_fsyncs_and_leaves_no_temporary_file(store):
    store.replace([make_character()], 0)
    parent = Path(store.path).parent
    assert list(parent.glob("*.tmp")) == []
    assert json.loads(Path(store.path).read_text(encoding="utf-8"))["revision"] == 1


def test_failed_atomic_replace_preserves_previous_document(store):
    first = store.replace([make_character("First")], 0)
    original = Path(store.path).read_bytes()
    with mock.patch("state_manager_store.os.replace", side_effect=OSError("disk full")):
        with pytest.raises(OSError, match="disk full"):
            store.replace([make_character("Second")], first["revision"])
    assert Path(store.path).read_bytes() == original
    assert list(Path(store.path).parent.glob("*.tmp")) == []


def test_malformed_file_is_quarantined_without_overwrite(store):
    path = Path(store.path)
    path.parent.mkdir(parents=True)
    path.write_text("{broken", encoding="utf-8")
    assert store.snapshot() == {"version": 1, "revision": 0, "characters": []}
    assert not path.exists()
    quarantined = list(path.parent.glob("state-library.json.corrupt-*"))
    assert len(quarantined) == 1
    assert quarantined[0].read_text(encoding="utf-8") == "{broken"
    with pytest.raises(InvalidStateLibrary, match="quarantined"):
        store.replace([], 0)
    assert store.snapshot()["characters"] == []
    assert store.replace([], 0)["revision"] == 1


def test_transient_read_error_does_not_quarantine(store):
    store.replace([make_character()], 0)
    with mock.patch("builtins.open", side_effect=PermissionError("sharing violation")):
        with pytest.raises(PermissionError, match="sharing violation"):
            store.snapshot()
    assert Path(store.path).exists()
    assert list(Path(store.path).parent.glob("*.corrupt-*")) == []


def test_future_library_version_is_left_untouched(store):
    path = Path(store.path)
    path.parent.mkdir(parents=True)
    raw = '{"version":999,"revision":1,"characters":[],"migrations":[]}'
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(UnsupportedStateLibraryVersion, match="expected 1"):
        store.snapshot()
    assert path.read_text(encoding="utf-8") == raw
    assert list(path.parent.glob("*.corrupt-*")) == []


def test_revision_conflict_prevents_stale_update_loss(store):
    first = store.replace([make_character("First")], 0)
    current = store.replace([make_character("Current")], first["revision"])
    with pytest.raises(StateLibraryRevisionConflict) as caught:
        store.replace([make_character("Stale")], first["revision"])
    assert caught.value.current == current
    assert store.snapshot() == current


def test_concurrent_writers_have_one_winner(store):
    barrier = threading.Barrier(2)
    results = []

    def writer(name):
        barrier.wait()
        try:
            results.append(("ok", store.replace([make_character(name)], 0)))
        except StateLibraryRevisionConflict as exc:
            results.append(("conflict", exc.current))

    threads = [threading.Thread(target=writer, args=(name,)) for name in ("A", "B")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(kind for kind, _ in results) == ["conflict", "ok"]
    assert store.snapshot()["revision"] == 1


def test_legacy_migration_is_complete_and_idempotent(store):
    legacy = {"version": 3, "characters": [make_character(
        character_id="legacy character",
        prompt_id="legacy prompt",
    )]}
    first = store.migrate_legacy(legacy, "legacy character", "legacy prompt")
    second = store.migrate_legacy(legacy, "legacy character", "legacy prompt")
    assert not first["already_migrated"]
    assert second["already_migrated"]
    assert first["snapshot"] == second["snapshot"]
    assert len(second["snapshot"]["characters"]) == 1
    assert uuid.UUID(first["selected_character_id"])
    assert uuid.UUID(first["selected_prompt_id"])
    migrated = second["snapshot"]["characters"][0]
    assert migrated["loader_stacks"][0]["loras"][0]["name"] == "private.safetensors"
    assert migrated["prompts"][0]["settings"]["nodes"][0]["widgets"]["steps"] == 20
    assert migrated["prompts"][0]["reference_image"]["filename"] == "reference.png"


def test_distinct_legacy_import_does_not_overwrite_colliding_uuid(store):
    shared_id = str(uuid.uuid4())
    shared_prompt_id = str(uuid.uuid4())
    store.replace([make_character("Existing", character_id=shared_id, prompt_id=shared_prompt_id)], 0)
    legacy = {"version": 3, "characters": [make_character(
        "Different",
        character_id=shared_id,
        prompt_id=shared_prompt_id,
    )]}
    result = store.migrate_legacy(legacy, shared_id, shared_prompt_id)
    assert len(result["snapshot"]["characters"]) == 2
    assert result["selected_character_id"] != shared_id
    assert result["selected_prompt_id"] != shared_prompt_id
    assert store.snapshot()["characters"][0]["name"] == "Existing"


def test_missing_selection_never_falls_back_to_unrelated_preset(store):
    character = make_character("Unrelated")
    store.replace([character], 0)
    with pytest.raises(StatePresetNotFound, match="not available locally"):
        store.resolve(str(uuid.uuid4()), character["prompts"][0]["id"])
    default_character, default_prompt = store.resolve("default_character", "default_prompt")
    assert default_character["id"] == "default_character"
    assert default_prompt["id"] == "default_prompt"


def test_character_export_import_round_trip_does_not_copy_other_characters(store):
    first = make_character("First")
    unrelated = make_character("Unrelated")
    snapshot = store.replace([first, unrelated], 0)
    exported = store.export_character(first["id"])

    target = StateLibraryStore(str(Path(store.path).parent / "target.json"), normalize_state, default_state)
    imported = target.import_character(exported["character"])
    assert len(imported["snapshot"]["characters"]) == 1
    assert imported["character"]["name"] == "First"
    assert "Unrelated" not in json.dumps(imported)
    assert snapshot["characters"][0]["prompts"][0]["positive"] == imported["character"]["prompts"][0]["positive"]


@pytest.mark.parametrize("subfolder", ["../escape", "/tmp/escape", "C:\\escape"])
def test_image_references_reject_unsafe_paths(store, subfolder):
    character = make_character()
    character["thumbnail"]["subfolder"] = subfolder
    with pytest.raises(InvalidStateLibrary, match="relative|unsafe"):
        store.replace([character], 0)
