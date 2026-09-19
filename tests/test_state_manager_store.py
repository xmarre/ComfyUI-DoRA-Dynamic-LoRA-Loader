import json
import os
import sys
import threading
import uuid
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(os.environ.get("DORA_REPO_ROOT", Path(__file__).resolve().parents[1])).resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from state_manager_store import (
    InvalidStateLibrary,
    StateLibraryRevisionConflict,
    StateLibraryStore,
    StatePresetNotFound,
    UnsupportedStateLibraryVersion,
    get_state_manager_store,
    reset_state_manager_store_for_tests,
    state_manager_library_path,
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


def test_atomic_prompt_text_box_update_preserves_unrelated_library_state(store):
    selected = make_character("Selected")
    unrelated = make_character("Unrelated")
    first = store.replace([selected, unrelated], 0)
    before_unrelated = first["characters"][1]

    timeline = "Global.\n\n[0-7s]\nOne.\n\n[7-14s]\nTwo."
    updated = store.update_prompt_text_box(
        selected["id"],
        selected["prompts"][0]["id"],
        "positive",
        "default",
        timeline,
        first["revision"],
        "Canonical Timeline",
    )

    assert updated["revision"] == first["revision"] + 1
    selected_after = updated["characters"][0]
    prompt_after = selected_after["prompts"][0]
    box_after = next(
        box for box in prompt_after["text_boxes"]
        if box["role"] == "positive" and box["slot"] == "default"
    )
    assert box_after["text"] == timeline
    assert box_after["label"] == "Canonical Timeline"
    assert prompt_after["positive"] == timeline
    assert updated["characters"][1] == before_unrelated

    with pytest.raises(StateLibraryRevisionConflict):
        store.update_prompt_text_box(
            selected["id"],
            selected["prompts"][0]["id"],
            "positive",
            "default",
            "stale overwrite",
            first["revision"],
        )
    assert store.snapshot() == updated


def test_atomic_prompt_text_box_update_rejects_missing_exact_selection(store):
    selected = make_character("Selected")
    first = store.replace([selected], 0)

    with pytest.raises(StatePresetNotFound, match="character preset"):
        store.update_prompt_text_box(
            str(uuid.uuid4()),
            selected["prompts"][0]["id"],
            "positive",
            "default",
            "must not fall back",
            first["revision"],
        )
    assert store.snapshot() == first


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


def test_malformed_character_cannot_normalize_into_default(store):
    with pytest.raises(InvalidStateLibrary, match="characters are malformed"):
        store.merge_library([None])


def test_store_cache_and_paths_are_isolated_per_comfy_user(tmp_path):
    class FolderPaths:
        @staticmethod
        def get_user_directory():
            return str(tmp_path)

        @staticmethod
        def get_public_user_directory(user_id):
            return str(tmp_path / user_id)

    reset_state_manager_store_for_tests()
    try:
        first_path = state_manager_library_path(FolderPaths, "first")
        second_path = state_manager_library_path(FolderPaths, "second")
        first = get_state_manager_store(path=first_path, normalize_state=normalize_state, default_state=default_state)
        first_again = get_state_manager_store(path=first_path, normalize_state=normalize_state, default_state=default_state)
        second = get_state_manager_store(path=second_path, normalize_state=normalize_state, default_state=default_state)
        assert first is first_again
        assert first is not second
        first.replace([make_character("First user")], 0)
        assert second.snapshot()["characters"] == []
        with pytest.raises(InvalidStateLibrary, match="user id is invalid"):
            state_manager_library_path(FolderPaths, "../second")
    finally:
        reset_state_manager_store_for_tests()


def test_prompt_document_write_migrates_v1_once_and_preserves_unrelated_state(store):
    selected = make_character("Selected")
    unrelated = make_character("Unrelated")
    first = store.replace([selected, unrelated], 0)
    assert first["version"] == 1
    before_unrelated = first["characters"][1]
    before_selected = json.loads(json.dumps(first["characters"][0]))

    descriptor = {
        "schema_version": 1,
        "format": "timeline",
        "routing": "logical_chunks",
        "geometry": {"chunks": 3, "chunk_seconds": "5.000"},
    }
    text = "Shared\n\n[0-5s]\nONE\n\n[5-10s]\nTWO\n\n[10-15s]\nTHREE"
    result = store.update_prompt_document(
        selected["id"],
        selected["prompts"][0]["id"],
        "positive",
        "default",
        text,
        descriptor,
        first["revision"],
        "Main",
    )

    assert result["snapshot"]["version"] == 2
    assert result["library_revision"] == first["revision"] + 1
    assert result["migrated_container_v2"] is True
    assert result["prompt_document"] == {
        "schema_version": 1,
        "format": "timeline",
        "routing": "logical_chunks",
        "geometry": {"chunks": 3, "chunk_seconds": "5"},
    }
    backup_path = Path(result["backup_path"])
    assert backup_path.is_file()
    backup = json.loads(backup_path.read_text(encoding="utf-8"))
    # The recoverable backup is the complete on-disk v1 container. The public
    # snapshot intentionally omits its migration ledger, so compare the public
    # state fields explicitly and keep the ledger losslessly in the backup.
    assert backup["version"] == first["version"] == 1
    assert backup["revision"] == first["revision"]
    assert backup["characters"] == first["characters"]
    assert backup.get("migrations", []) == []

    after = result["snapshot"]
    assert after["characters"][1] == before_unrelated
    selected_after = after["characters"][0]
    assert selected_after["id"] == before_selected["id"]
    assert selected_after["thumbnail"] == before_selected["thumbnail"]
    assert selected_after["loader_stacks"] == before_selected["loader_stacks"]
    assert selected_after["prompts"][0]["settings"] == before_selected["prompts"][0]["settings"]
    box = selected_after["prompts"][0]["text_boxes"][0]
    assert box["text"] == text
    assert box["prompt_document"] == result["prompt_document"]

    # v4 text-only editing stays supported and cannot erase the v5 descriptor.
    text_only = store.update_prompt_text_box(
        selected["id"],
        selected["prompts"][0]["id"],
        "positive",
        "default",
        text + " edited",
        after["revision"],
    )
    assert text_only["version"] == 2
    assert text_only["characters"][0]["prompts"][0]["text_boxes"][0]["prompt_document"] == result["prompt_document"]
    assert list(Path(store.path).parent.glob("state-library.json.v1-backup-*")) == [backup_path]


def test_stale_bulk_replace_cannot_strip_prompt_document(store):
    character = make_character("Selected")
    first = store.replace([character], 0)
    result = store.update_prompt_document(
        character["id"],
        character["prompts"][0]["id"],
        "positive",
        "default",
        "Managed timeline",
        {"schema_version": 1, "format": "fixed"},
        first["revision"],
    )
    stale = json.loads(json.dumps(result["snapshot"]["characters"]))
    stale[0]["prompts"][0]["text_boxes"][0].pop("prompt_document")

    with pytest.raises(InvalidStateLibrary, match="strip or rewrite"):
        store.replace(stale, result["library_revision"])
    assert store.snapshot() == result["snapshot"]


def test_prompt_document_write_rejects_stale_revision_without_backup_or_mutation(store):
    character = make_character("Selected")
    first = store.replace([character], 0)
    store.replace([character], first["revision"])
    before = store.snapshot()
    parent = Path(store.path).parent

    with pytest.raises(StateLibraryRevisionConflict):
        store.update_prompt_document(
            character["id"],
            character["prompts"][0]["id"],
            "positive",
            "default",
            "stale",
            {"schema_version": 1, "format": "fixed"},
            first["revision"],
        )

    assert store.snapshot() == before
    assert list(parent.glob("state-library.json.v1-backup-*")) == []


def test_descriptor_bearing_bulk_replace_promotes_v1_with_prewrite_backup(store):
    character = make_character("Descriptor Replace")
    character["prompts"][0]["text_boxes"][0]["prompt_document"] = {
        "schema_version": 1,
        "format": "fixed",
    }

    with pytest.raises(InvalidStateLibrary, match="contract v5.*prompt_document_v1"):
        store.replace([character], 0)

    result = store.replace([character], 0, document_capable=True)

    assert result["version"] == 2
    backups = list(Path(store.path).parent.glob("state-library.json.v1-backup-*"))
    assert len(backups) == 1
    backup = json.loads(backups[0].read_text(encoding="utf-8"))
    assert backup == {
        "version": 1,
        "revision": 0,
        "characters": [],
        "migrations": [],
    }


def test_descriptor_bearing_character_import_promotes_v1_but_plain_import_does_not(store):
    plain = make_character("Plain Import")
    plain_result = store.import_character(plain)
    assert plain_result["snapshot"]["version"] == 1
    assert list(Path(store.path).parent.glob("state-library.json.v1-backup-*")) == []

    descriptor = make_character("Descriptor Import")
    descriptor["prompts"][0]["text_boxes"][0]["prompt_document"] = {
        "schema_version": 1,
        "format": "fixed",
    }
    imported = store.import_character(descriptor)

    assert imported["snapshot"]["version"] == 2
    backups = list(Path(store.path).parent.glob("state-library.json.v1-backup-*"))
    assert len(backups) == 1
    backup = json.loads(backups[0].read_text(encoding="utf-8"))
    assert backup["version"] == 1
    assert backup["revision"] == plain_result["snapshot"]["revision"]
    assert [item["name"] for item in backup["characters"]] == ["Plain Import"]


def test_descriptor_bearing_library_merge_promotes_v1_before_import(store):
    descriptor = make_character("Descriptor Library Import")
    descriptor["prompts"][0]["text_boxes"][0]["prompt_document"] = {
        "schema_version": 1,
        "format": "timeline",
        "routing": "logical_chunks",
        "geometry": {"chunks": 1, "chunk_seconds": "5"},
    }

    imported = store.merge_library([descriptor])

    assert imported["snapshot"]["version"] == 2
    assert imported["snapshot"]["characters"][0]["prompts"][0]["text_boxes"][0]["prompt_document"]["format"] == "timeline"
    backups = list(Path(store.path).parent.glob("state-library.json.v1-backup-*"))
    assert len(backups) == 1
    assert json.loads(backups[0].read_text(encoding="utf-8"))["characters"] == []


def test_text_only_mutation_repairs_descriptor_bearing_v1_container(store):
    character = make_character("Partial V2")
    character["prompts"][0]["text_boxes"][0]["prompt_document"] = {
        "schema_version": 99,
        "future_mode": "opaque",
        "future_data": {"keep": True},
    }
    path = Path(store.path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = {
        "version": 1,
        "revision": 7,
        "characters": [character],
        "migrations": [],
    }
    path.write_text(json.dumps(raw), encoding="utf-8")

    result = store.update_prompt_text_box(
        character["id"],
        character["prompts"][0]["id"],
        "positive",
        "default",
        "edited",
        7,
    )

    assert result["version"] == 2
    assert result["characters"][0]["prompts"][0]["text_boxes"][0]["prompt_document"] == character["prompts"][0]["text_boxes"][0]["prompt_document"]
    backups = list(path.parent.glob("state-library.json.v1-backup-*"))
    assert len(backups) == 1
    backup = json.loads(backups[0].read_text(encoding="utf-8"))
    assert backup["version"] == 1
    assert backup["revision"] == 7
    assert backup["characters"][0]["prompts"][0]["text_boxes"][0]["text"] == "private positive prompt"
    assert backup["characters"][0]["prompts"][0]["positive"] == "private positive prompt"
    assert backup["characters"][0]["prompts"][0]["text_boxes"][0]["prompt_document"] == character["prompts"][0]["text_boxes"][0]["prompt_document"]


def test_descriptor_aware_export_advertises_support_and_preserves_future_schema(store):
    character = make_character("Selected")
    character["prompts"][0]["text_boxes"][0]["prompt_document"] = {
        "schema_version": 99,
        "future_mode": "opaque",
        "future_data": {"keep": [1, 2, 3]},
    }
    # Simulate an already descriptor-aware persisted container without routing
    # unknown future data through the explicit schema-v1 authoring method.
    path = Path(store.path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "version": 2,
        "revision": 7,
        "characters": [character],
        "migrations": [],
    }), encoding="utf-8")

    snapshot = store.snapshot()
    saved = snapshot["characters"][0]["prompts"][0]["text_boxes"][0]["prompt_document"]
    assert saved == character["prompts"][0]["text_boxes"][0]["prompt_document"]

    exported_character = store.export_character(character["id"])
    exported_library = store.export_library()
    assert exported_character["version"] == 2
    assert exported_character["capabilities"] == ["prompt_document_v1"]
    assert exported_character["character"]["prompts"][0]["text_boxes"][0]["prompt_document"] == saved
    assert exported_library["version"] == 2
    assert exported_library["capabilities"] == ["prompt_document_v1"]
    assert exported_library["characters"][0]["prompts"][0]["text_boxes"][0]["prompt_document"] == saved


@pytest.mark.parametrize(
    "descriptor",
    [
        None,
        "timeline",
        {"schema_version": 1, "format": "timeline", "routing": "logical_chunks", "geometry": {"chunks": 0, "chunk_seconds": "5"}},
        {"schema_version": 1, "format": "timeline", "routing": "logical_chunks", "geometry": {"chunks": 3, "chunk_seconds": "5e0"}},
    ],
)
def test_prompt_document_write_rejects_invalid_descriptor_without_migrating(store, descriptor):
    character = make_character("Selected")
    first = store.replace([character], 0)
    with pytest.raises((InvalidStateLibrary, ValueError)):
        store.update_prompt_document(
            character["id"],
            character["prompts"][0]["id"],
            "positive",
            "default",
            "text",
            descriptor,
            first["revision"],
        )
    assert store.snapshot() == first
    assert list(Path(store.path).parent.glob("state-library.json.v1-backup-*")) == []
