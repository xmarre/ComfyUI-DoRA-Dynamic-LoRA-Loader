import hashlib
import json
import logging
import os
import tempfile
import threading
import time
import uuid
from copy import deepcopy
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Callable, Dict, List, Optional, Tuple


LOGGER = logging.getLogger(__name__)


class InvalidStateLibrary(ValueError):
    pass


class UnsupportedStateLibraryVersion(InvalidStateLibrary):
    pass


class StatePresetNotFound(LookupError):
    pass


class StateLibraryRevisionConflict(RuntimeError):
    def __init__(self, current: Dict[str, Any]):
        super().__init__("The State Manager library changed since it was loaded.")
        self.current = current


def _is_uuid(value: Any) -> bool:
    try:
        uuid.UUID(str(value or "").strip())
        return True
    except (ValueError, AttributeError, TypeError):
        return False


def _new_uuid() -> str:
    return str(uuid.uuid4())


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _normalize_relative_component(value: Any, label: str) -> str:
    raw = str(value or "").strip().replace("\\", "/")
    if not raw:
        return ""
    if "\x00" in raw:
        raise InvalidStateLibrary(f"{label} contains a null byte.")
    posix = PurePosixPath(raw)
    windows = PureWindowsPath(raw)
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        raise InvalidStateLibrary(f"{label} must be relative.")
    if any(part in {"", ".", ".."} for part in raw.split("/")):
        raise InvalidStateLibrary(f"{label} contains an unsafe path segment.")
    return raw


def _validate_image_reference(value: Any) -> None:
    if not value:
        return
    if not isinstance(value, dict):
        raise InvalidStateLibrary("Image references must be objects.")
    filename = str(value.get("filename", "") or "").strip()
    url = str(value.get("url", "") or "").strip()
    if filename:
        if filename in {".", ".."} or "\x00" in filename or "/" in filename or "\\" in filename:
            raise InvalidStateLibrary("Image filenames must not contain a path.")
        _normalize_relative_component(value.get("subfolder", ""), "Image subfolder")
        storage_type = str(value.get("type", "input") or "input").strip().lower()
        if storage_type not in {"input", "output", "temp"}:
            raise InvalidStateLibrary("Image type must be input, output, or temp.")
    elif url and not (url.startswith("https://") or url.startswith("http://") or url.startswith("data:image/")):
        raise InvalidStateLibrary("Image URLs must use http(s) or an image data URL.")


class StateLibraryStore:
    VERSION = 1
    MAX_MIGRATIONS = 2048

    def __init__(
        self,
        path: str,
        normalize_state: Callable[[Any], Dict[str, Any]],
        default_state: Callable[[], Dict[str, Any]],
    ):
        self.path = os.path.realpath(path)
        self._normalize_state = normalize_state
        self._default_state = default_state
        self._lock = threading.RLock()
        self._recovery_pending = False

    @staticmethod
    def _empty_document() -> Dict[str, Any]:
        return {
            "version": StateLibraryStore.VERSION,
            "revision": 0,
            "characters": [],
            "migrations": [],
        }

    def _normalize_characters(self, characters: Any, *, require_uuids: bool) -> List[Dict[str, Any]]:
        if not isinstance(characters, list):
            raise InvalidStateLibrary("The State Manager library must contain a character list.")
        if not characters:
            return []
        normalized = self._normalize_state({"version": 3, "characters": characters})
        result = normalized.get("characters") if isinstance(normalized, dict) else None
        if not isinstance(result, list):
            raise InvalidStateLibrary("The State Manager character library is malformed.")
        character_ids = set()
        prompt_ids = set()
        for character in result:
            character_id = str(character.get("id", ""))
            if require_uuids and not _is_uuid(character_id):
                raise InvalidStateLibrary("Persistent character IDs must be UUIDs.")
            if character_id in character_ids:
                raise InvalidStateLibrary("The State Manager library contains duplicate character IDs.")
            character_ids.add(character_id)
            _validate_image_reference(character.get("thumbnail", {}))
            prompts = character.get("prompts")
            if not isinstance(prompts, list) or not prompts:
                raise InvalidStateLibrary("Every persistent character must contain at least one prompt.")
            for prompt in prompts:
                prompt_id = str(prompt.get("id", ""))
                if require_uuids and not _is_uuid(prompt_id):
                    raise InvalidStateLibrary("Persistent prompt IDs must be UUIDs.")
                if prompt_id in prompt_ids:
                    raise InvalidStateLibrary("Persistent prompt IDs must be globally unique.")
                prompt_ids.add(prompt_id)
                _validate_image_reference(prompt.get("reference_image", {}))
        return result

    def _normalize_document(self, raw: Any) -> Dict[str, Any]:
        if not isinstance(raw, dict):
            raise InvalidStateLibrary("The State Manager library document is malformed.")
        if raw.get("version") != self.VERSION:
            raise UnsupportedStateLibraryVersion(
                f"Unsupported State Manager library version {raw.get('version')!r}; expected {self.VERSION}."
            )
        try:
            revision = max(0, int(raw.get("revision", 0)))
        except (TypeError, ValueError) as exc:
            raise InvalidStateLibrary("The State Manager library revision is invalid.") from exc
        migrations = raw.get("migrations", [])
        if not isinstance(migrations, list):
            raise InvalidStateLibrary("The State Manager migration index is malformed.")
        normalized_migrations = []
        fingerprints = set()
        for entry in migrations[-self.MAX_MIGRATIONS :]:
            if not isinstance(entry, dict):
                raise InvalidStateLibrary("The State Manager migration index is malformed.")
            fingerprint = str(entry.get("fingerprint", "") or "").strip().lower()
            if len(fingerprint) != 64 or any(char not in "0123456789abcdef" for char in fingerprint):
                raise InvalidStateLibrary("The State Manager migration index contains an invalid fingerprint.")
            if fingerprint in fingerprints:
                continue
            fingerprints.add(fingerprint)
            normalized_migrations.append({
                "fingerprint": fingerprint,
                "character_ids": {
                    str(key): str(value)
                    for key, value in (entry.get("character_ids") or {}).items()
                    if _is_uuid(value)
                },
                "prompt_ids": {
                    str(key): str(value)
                    for key, value in (entry.get("prompt_ids") or {}).items()
                    if _is_uuid(value)
                },
                "imported_at": max(0, int(entry.get("imported_at", 0) or 0)),
            })
        return {
            "version": self.VERSION,
            "revision": revision,
            "characters": self._normalize_characters(raw.get("characters"), require_uuids=True),
            "migrations": normalized_migrations,
        }

    def _quarantine(self) -> Optional[str]:
        if not os.path.exists(self.path):
            return None
        quarantine = f"{self.path}.corrupt-{int(time.time() * 1000)}"
        try:
            os.replace(self.path, quarantine)
            return quarantine
        except OSError:
            LOGGER.exception("State Manager could not quarantine malformed library storage.")
            return None

    def _load_unlocked(self) -> Dict[str, Any]:
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                return self._normalize_document(json.load(handle))
        except FileNotFoundError:
            return self._empty_document()
        except OSError:
            LOGGER.warning("State Manager could not read library storage.", exc_info=True)
            raise
        except UnsupportedStateLibraryVersion:
            LOGGER.error("State Manager library was created by an incompatible version; leaving it untouched.")
            raise
        except (UnicodeError, json.JSONDecodeError, InvalidStateLibrary, ValueError, TypeError):
            LOGGER.warning("State Manager library storage was malformed and has been quarantined.", exc_info=True)
            self._quarantine()
            self._recovery_pending = True
            return self._empty_document()

    def _write_unlocked(self, document: Dict[str, Any]) -> None:
        parent = os.path.dirname(self.path)
        os.makedirs(parent, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=".state-library-", suffix=".tmp", dir=parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(document, handle, ensure_ascii=False, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            self._recovery_pending = False
            try:
                directory_fd = os.open(parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except OSError:
                pass
        finally:
            try:
                os.unlink(temporary)
            except OSError:
                pass

    @staticmethod
    def _public(document: Dict[str, Any]) -> Dict[str, Any]:
        return _json_copy({
            "version": document["version"],
            "revision": document["revision"],
            "characters": document["characters"],
        })

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            recovery_was_pending = self._recovery_pending
            document = self._load_unlocked()
            if recovery_was_pending and not os.path.exists(self.path):
                # A read after quarantine explicitly acknowledges the recovered
                # empty library. The discovering request itself cannot write it.
                self._recovery_pending = False
            return self._public(document)

    def revision(self) -> int:
        with self._lock:
            return int(self._load_unlocked()["revision"])

    def _require_recovery_acknowledgement(self) -> None:
        if self._recovery_pending:
            raise InvalidStateLibrary(
                "The malformed library was quarantined. Reload the empty recovered library before saving."
            )

    def replace(self, characters: Any, expected_revision: Any) -> Dict[str, Any]:
        with self._lock:
            document = self._load_unlocked()
            self._require_recovery_acknowledgement()
            try:
                expected = int(expected_revision)
            except (TypeError, ValueError) as exc:
                raise InvalidStateLibrary("An expected library revision is required.") from exc
            if expected != document["revision"]:
                raise StateLibraryRevisionConflict(self._public(document))
            document["characters"] = self._normalize_characters(characters, require_uuids=True)
            document["revision"] += 1
            self._write_unlocked(document)
            return self._public(document)

    @staticmethod
    def _without_ids(character: Dict[str, Any]) -> Dict[str, Any]:
        value = deepcopy(character)
        value.pop("id", None)
        for prompt in value.get("prompts", []):
            if isinstance(prompt, dict):
                prompt.pop("id", None)
        return value

    @staticmethod
    def _unique_name(name: str, characters: List[Dict[str, Any]]) -> str:
        used = {str(character.get("name", "")).casefold() for character in characters}
        if name.casefold() not in used:
            return name
        suffix = 2
        while f"{name} (Imported {suffix})".casefold() in used:
            suffix += 1
        return f"{name} (Imported {suffix})"

    def _merge_character(
        self,
        document: Dict[str, Any],
        raw_character: Dict[str, Any],
        *,
        deduplicate_content: bool,
    ) -> Tuple[Dict[str, Any], Dict[str, str], Dict[str, str], bool]:
        normalized = self._normalize_characters([raw_character], require_uuids=False)[0]
        old_character_id = str(normalized.get("id", ""))
        old_prompt_ids = [str(prompt.get("id", "")) for prompt in normalized.get("prompts", [])]
        content_key = _canonical(self._without_ids(normalized))
        if deduplicate_content:
            existing = next(
                (
                    character for character in document["characters"]
                    if _canonical(self._without_ids(character)) == content_key
                ),
                None,
            )
            if existing is not None:
                prompt_map = {
                    old_id: existing["prompts"][index]["id"]
                    for index, old_id in enumerate(old_prompt_ids)
                    if index < len(existing.get("prompts", []))
                }
                return existing, {old_character_id: existing["id"]}, prompt_map, False

        used_character_ids = {character["id"] for character in document["characters"]}
        used_prompt_ids = {
            prompt["id"]
            for character in document["characters"]
            for prompt in character.get("prompts", [])
        }
        character_id = old_character_id if _is_uuid(old_character_id) and old_character_id not in used_character_ids else _new_uuid()
        normalized["id"] = character_id
        normalized["name"] = self._unique_name(str(normalized.get("name") or "Imported Character"), document["characters"])
        prompt_map = {}
        for prompt, old_prompt_id in zip(normalized.get("prompts", []), old_prompt_ids):
            prompt_id = old_prompt_id if _is_uuid(old_prompt_id) and old_prompt_id not in used_prompt_ids else _new_uuid()
            used_prompt_ids.add(prompt_id)
            prompt["id"] = prompt_id
            prompt_map[old_prompt_id] = prompt_id
        document["characters"].append(normalized)
        return normalized, {old_character_id: character_id}, prompt_map, True

    def migrate_legacy(
        self,
        legacy_state: Any,
        selected_character_id: Any = "",
        selected_prompt_id: Any = "",
    ) -> Dict[str, Any]:
        if not isinstance(legacy_state, dict) or not isinstance(legacy_state.get("characters"), list):
            raise InvalidStateLibrary("Legacy State Manager data is malformed.")
        normalized = self._normalize_state(legacy_state)
        fingerprint = hashlib.sha256(_canonical(normalized).encode("utf-8")).hexdigest()
        with self._lock:
            document = self._load_unlocked()
            self._require_recovery_acknowledgement()
            existing_migration = next(
                (entry for entry in document["migrations"] if entry["fingerprint"] == fingerprint),
                None,
            )
            if existing_migration is not None:
                return {
                    "snapshot": self._public(document),
                    "fingerprint": fingerprint,
                    "already_migrated": True,
                    "selected_character_id": existing_migration["character_ids"].get(str(selected_character_id), ""),
                    "selected_prompt_id": existing_migration["prompt_ids"].get(str(selected_prompt_id), ""),
                    "character_ids": _json_copy(existing_migration["character_ids"]),
                    "prompt_ids": _json_copy(existing_migration["prompt_ids"]),
                }

            character_map: Dict[str, str] = {}
            prompt_map: Dict[str, str] = {}
            changed = False
            for character in normalized.get("characters", []):
                _, mapped_characters, mapped_prompts, inserted = self._merge_character(
                    document,
                    character,
                    deduplicate_content=True,
                )
                character_map.update(mapped_characters)
                prompt_map.update(mapped_prompts)
                changed = changed or inserted
            document["migrations"].append({
                "fingerprint": fingerprint,
                "character_ids": character_map,
                "prompt_ids": prompt_map,
                "imported_at": int(time.time() * 1000),
            })
            document["migrations"] = document["migrations"][-self.MAX_MIGRATIONS :]
            document["revision"] += 1
            self._write_unlocked(document)
            return {
                "snapshot": self._public(document),
                "fingerprint": fingerprint,
                "already_migrated": False,
                "changed": changed,
                "selected_character_id": character_map.get(str(selected_character_id), ""),
                "selected_prompt_id": prompt_map.get(str(selected_prompt_id), ""),
                "character_ids": character_map,
                "prompt_ids": prompt_map,
            }

    def import_character(self, raw_character: Any) -> Dict[str, Any]:
        if not isinstance(raw_character, dict):
            raise InvalidStateLibrary("The character import is malformed.")
        with self._lock:
            document = self._load_unlocked()
            self._require_recovery_acknowledgement()
            character, _, _, _ = self._merge_character(document, raw_character, deduplicate_content=False)
            document["revision"] += 1
            self._write_unlocked(document)
            return {"snapshot": self._public(document), "character": _json_copy(character)}

    def merge_library(self, characters: Any) -> Dict[str, Any]:
        if not isinstance(characters, list):
            raise InvalidStateLibrary("The State Manager library import is malformed.")
        with self._lock:
            document = self._load_unlocked()
            self._require_recovery_acknowledgement()
            imported_ids = []
            for character in characters:
                imported, _, _, _ = self._merge_character(document, character, deduplicate_content=True)
                imported_ids.append(imported["id"])
            document["revision"] += 1
            self._write_unlocked(document)
            return {"snapshot": self._public(document), "character_ids": imported_ids}

    def export_character(self, character_id: Any) -> Dict[str, Any]:
        character_id = str(character_id or "").strip()
        with self._lock:
            document = self._load_unlocked()
            character = next((entry for entry in document["characters"] if entry["id"] == character_id), None)
            if character is None:
                raise StatePresetNotFound(character_id)
            return {
                "version": 1,
                "kind": "dora_state_manager_character_export",
                "exported_at": int(time.time() * 1000),
                "character": _json_copy(character),
            }

    def export_library(self) -> Dict[str, Any]:
        snapshot = self.snapshot()
        return {
            "version": 1,
            "kind": "dora_state_manager_library_export",
            "exported_at": int(time.time() * 1000),
            "characters": snapshot["characters"],
        }

    def resolve(self, character_id: Any, prompt_id: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        character_id = str(character_id or "").strip()
        prompt_id = str(prompt_id or "").strip()
        if character_id in {"", "default_character"} and prompt_id in {"", "default_prompt"}:
            character = self._default_state()["characters"][0]
            return _json_copy(character), _json_copy(character["prompts"][0])
        with self._lock:
            document = self._load_unlocked()
            character = next((entry for entry in document["characters"] if entry["id"] == character_id), None)
            if character is None:
                raise StatePresetNotFound(
                    "Selected character preset is not available locally. Select or create a character."
                )
            prompt = next((entry for entry in character.get("prompts", []) if entry["id"] == prompt_id), None)
            if prompt is None:
                raise StatePresetNotFound(
                    "Selected prompt preset is not available locally for this character. Select or create a prompt."
                )
            return _json_copy(character), _json_copy(prompt)


_STORE: Optional[StateLibraryStore] = None
_STORE_LOCK = threading.Lock()


def get_state_manager_store(
    *,
    path: Optional[str] = None,
    normalize_state: Optional[Callable[[Any], Dict[str, Any]]] = None,
    default_state: Optional[Callable[[], Dict[str, Any]]] = None,
) -> StateLibraryStore:
    global _STORE
    with _STORE_LOCK:
        if _STORE is None:
            if path is None or normalize_state is None or default_state is None:
                raise RuntimeError("State Manager store has not been configured.")
            _STORE = StateLibraryStore(path, normalize_state, default_state)
        return _STORE


def reset_state_manager_store_for_tests() -> None:
    global _STORE
    with _STORE_LOCK:
        _STORE = None
