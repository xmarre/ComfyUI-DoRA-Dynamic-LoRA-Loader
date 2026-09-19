import asyncio
import hashlib
import json
import logging
import re
from typing import Any, Callable, Dict, Optional

from .state_manager_store import (
    InvalidStateLibrary,
    StateLibraryRevisionConflict,
    StatePresetNotFound,
    get_state_manager_store,
    state_manager_library_path,
)


LOGGER = logging.getLogger(__name__)
_ROUTES_REGISTERED = False
_TIMELINE_HEADER = re.compile(r"(?m)^\s*\[[0-9]+(?:\.[0-9]+)?-[0-9]+(?:\.[0-9]+)?s\]\s*$")


def configure_store(
    folder_paths_module: Any,
    normalize_state: Callable[[Any], Dict[str, Any]],
    default_state: Callable[[], Dict[str, Any]],
    user_id: Any = "default",
):
    return get_state_manager_store(
        path=state_manager_library_path(folder_paths_module, user_id),
        normalize_state=normalize_state,
        default_state=default_state,
    )


def _import_payload(store, payload: Dict[str, Any]):
    try:
        export_version = int(payload.get("version", 1) or 1)
    except (TypeError, ValueError) as exc:
        raise InvalidStateLibrary("The State Manager import version is invalid.") from exc
    if export_version >= 2:
        capabilities = payload.get("capabilities")
        if not isinstance(capabilities, list) or "prompt_document_v1" not in capabilities:
            raise InvalidStateLibrary(
                "Descriptor-aware State Manager v2 imports must advertise prompt_document_v1."
            )
    kind = str(payload.get("kind", ""))
    if kind == "dora_state_manager_library_export":
        return store.merge_library(payload.get("characters"))
    if kind == "dora_state_manager_character_export":
        return store.import_character(payload.get("character"))
    if isinstance(payload.get("character"), dict):
        return store.import_character(payload.get("character"))
    characters = payload.get("characters")
    if characters is None and isinstance(payload.get("state"), dict):
        characters = payload["state"].get("characters")
    return store.merge_library(characters)


def register_routes(
    folder_paths_module: Any,
    prompt_server: Any,
    web: Any,
    normalize_state: Callable[[Any], Dict[str, Any]],
    default_state: Callable[[], Dict[str, Any]],
    prompt_transport_provider: Optional[Callable[[], Optional[Dict[str, Any]]]] = None,
    prompt_transport_ordering_contract: Optional[Callable[[], Optional[str]]] = None,
) -> None:
    global _ROUTES_REGISTERED
    if _ROUTES_REGISTERED:
        return
    _ROUTES_REGISTERED = True
    routes = prompt_server.instance.routes
    LOGGER.info("[State Manager] managed prompt API registered contract=v5 capabilities=backend_persistent_text_write_v1,prompt_document_v1 revision=backend-document-write-v1")

    def store_for_request(request):
        user_manager = getattr(prompt_server.instance, "user_manager", None)
        user_id = user_manager.get_request_user_id(request) if user_manager is not None else "default"
        return configure_store(folder_paths_module, normalize_state, default_state, user_id), str(user_id)

    def with_user_id(payload: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        result = dict(payload)
        result["user_id"] = user_id
        if isinstance(result.get("snapshot"), dict):
            result["snapshot"] = {**result["snapshot"], "user_id": user_id}
        return result

    def error_response(exc: Exception):
        if isinstance(exc, StateLibraryRevisionConflict):
            return web.json_response(
                {"error": str(exc), "code": "revision_conflict", "snapshot": exc.current},
                status=409,
            )
        if isinstance(exc, StatePresetNotFound):
            return web.json_response({"error": str(exc) or "State Manager preset not found."}, status=404)
        if isinstance(exc, KeyError):
            return web.json_response({"error": "Invalid ComfyUI user."}, status=403)
        if isinstance(exc, (InvalidStateLibrary, json.JSONDecodeError, ValueError, TypeError)):
            return web.json_response({"error": str(exc)}, status=400)
        LOGGER.exception("State Manager library API failed.")
        return web.json_response({"error": "Unable to update the State Manager library."}, status=500)

    @routes.get("/dora_dynamic_lora/state-library")
    async def state_manager_list_library(request):
        try:
            store, user_id = store_for_request(request)
            snapshot = await asyncio.to_thread(store.snapshot)
            return web.json_response(with_user_id(snapshot, user_id))
        except Exception as exc:
            return error_response(exc)

    @routes.put("/dora_dynamic_lora/state-library")
    async def state_manager_replace_library(request):
        try:
            store, user_id = store_for_request(request)
            payload = await request.json()
            if not isinstance(payload, dict):
                raise InvalidStateLibrary("The State Manager library request is malformed.")
            try:
                contract_version = int(payload.get("contract_version", 0) or 0)
            except (TypeError, ValueError):
                contract_version = 0
            capabilities = payload.get("capabilities")
            document_capable = (
                contract_version >= 5
                and isinstance(capabilities, list)
                and "prompt_document_v1" in capabilities
            )
            snapshot = await asyncio.to_thread(
                store.replace,
                payload.get("characters"),
                payload.get("expected_revision"),
                document_capable,
            )
            return web.json_response(with_user_id(snapshot, user_id))
        except Exception as exc:
            return error_response(exc)

    @routes.put("/dora_dynamic_lora/state-library/characters/{character_id}/prompts/{prompt_id}/text-box")
    async def state_manager_update_prompt_text_box(request):
        try:
            store, user_id = store_for_request(request)
            payload = await request.json()
            if not isinstance(payload, dict):
                raise InvalidStateLibrary("The State Manager managed-text request is malformed.")

            character_id = request.match_info["character_id"]
            prompt_id = request.match_info["prompt_id"]
            role = str(payload.get("role", "positive") or "positive")
            slot = str(payload.get("slot", "default") or "default")
            text = str(payload.get("text", "") or "")
            label = str(payload.get("label", "") or "")
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            LOGGER.info(
                "[State Manager] managed prompt write request revision=backend-write-v1 character=%r prompt=%r role=%r slot=%r expected_revision=%r chars=%d timeline=%s digest=%s",
                character_id,
                prompt_id,
                role,
                slot,
                payload.get("expected_revision"),
                len(text),
                bool(_TIMELINE_HEADER.search(text)),
                digest,
            )
            snapshot = await asyncio.to_thread(
                store.update_prompt_text_box,
                character_id,
                prompt_id,
                role,
                slot,
                text,
                payload.get("expected_revision"),
                label,
            )

            LOGGER.info(
                "[State Manager] managed prompt write revision=backend-write-v1 character=%r prompt=%r role=%r slot=%r chars=%d timeline=%s digest=%s library_revision=%d",
                character_id,
                prompt_id,
                role,
                slot,
                len(text),
                bool(_TIMELINE_HEADER.search(text)),
                digest,
                int(snapshot.get("revision", -1)),
            )
            return web.json_response(with_user_id(snapshot, user_id))
        except Exception as exc:
            LOGGER.warning(
                "[State Manager] managed prompt write rejected revision=backend-write-v1 character=%r prompt=%r error=%s: %s",
                locals().get("character_id", ""),
                locals().get("prompt_id", ""),
                type(exc).__name__,
                exc,
            )
            return error_response(exc)

    @routes.get("/dora_dynamic_lora/state-library/prompt-document-provider")
    async def state_manager_prompt_document_provider(request):
        try:
            _store, user_id = store_for_request(request)
            provider = prompt_transport_provider() if callable(prompt_transport_provider) else None
            ordering_contract = (
                prompt_transport_ordering_contract()
                if callable(prompt_transport_ordering_contract)
                else None
            )
            return web.json_response({
                "contract_version": 5,
                "capability": "prompt_document_v1",
                "provider": provider,
                "ordering_contract": ordering_contract,
                "user_id": user_id,
            })
        except Exception as exc:
            return error_response(exc)

    @routes.put("/dora_dynamic_lora/state-library/characters/{character_id}/prompts/{prompt_id}/prompt-document")
    async def state_manager_update_prompt_document(request):
        try:
            store, user_id = store_for_request(request)
            payload = await request.json()
            if not isinstance(payload, dict):
                raise InvalidStateLibrary("The State Manager prompt-document request is malformed.")
            character_id = request.match_info["character_id"]
            prompt_id = request.match_info["prompt_id"]
            role = str(payload.get("role", "positive") or "positive")
            slot = str(payload.get("slot", "default") or "default")
            text = str(payload.get("text", "") or "")
            label = str(payload.get("label", "") or "")
            result = await asyncio.to_thread(
                store.update_prompt_document,
                character_id,
                prompt_id,
                role,
                slot,
                text,
                payload.get("prompt_document"),
                payload.get("expected_revision"),
                label,
            )
            snapshot = with_user_id(result["snapshot"], user_id)
            response = {
                "status": "updated",
                "role": role,
                "slot": slot,
                "character_id": character_id,
                "prompt_id": prompt_id,
                "text_sha256": result["text_sha256"],
                "prompt_document": result["prompt_document"],
                "library_revision": result["library_revision"],
                "persistent_verified": True,
                "contract_version": 5,
                "write_revision": "backend-document-write-v1",
                "migrated_container_v2": bool(result.get("migrated_container_v2")),
                "snapshot": snapshot,
                "user_id": user_id,
            }
            LOGGER.info(
                "[State Manager] prompt-document write revision=backend-document-write-v1 "
                "character=%r prompt=%r role=%r slot=%r digest=%s library_revision=%d format=%s routing=%s",
                character_id,
                prompt_id,
                role,
                slot,
                result["text_sha256"],
                int(result["library_revision"]),
                (result["prompt_document"] or {}).get("format"),
                (result["prompt_document"] or {}).get("routing"),
            )
            return web.json_response(response)
        except Exception as exc:
            LOGGER.warning(
                "[State Manager] prompt-document write rejected revision=backend-document-write-v1 "
                "character=%r prompt=%r error=%s: %s",
                locals().get("character_id", ""),
                locals().get("prompt_id", ""),
                type(exc).__name__,
                exc,
            )
            return error_response(exc)

    @routes.post("/dora_dynamic_lora/state-library/migrate")
    async def state_manager_migrate_library(request):
        try:
            store, user_id = store_for_request(request)
            payload = await request.json()
            if not isinstance(payload, dict):
                raise InvalidStateLibrary("The State Manager migration request is malformed.")
            result = await asyncio.to_thread(
                store.migrate_legacy,
                payload.get("state"),
                payload.get("selected_character_id", ""),
                payload.get("selected_prompt_id", ""),
            )
            return web.json_response(with_user_id(result, user_id))
        except Exception as exc:
            return error_response(exc)

    @routes.get("/dora_dynamic_lora/state-library/export")
    async def state_manager_export_library(request):
        try:
            store, _user_id = store_for_request(request)
            return web.json_response(await asyncio.to_thread(store.export_library))
        except Exception as exc:
            return error_response(exc)

    @routes.get("/dora_dynamic_lora/state-library/characters/{character_id}/export")
    async def state_manager_export_character(request):
        try:
            store, _user_id = store_for_request(request)
            payload = await asyncio.to_thread(
                store.export_character,
                request.match_info["character_id"],
            )
            return web.json_response(payload)
        except Exception as exc:
            return error_response(exc)

    @routes.post("/dora_dynamic_lora/state-library/import")
    async def state_manager_import_library(request):
        try:
            store, user_id = store_for_request(request)
            payload = await request.json()
            if not isinstance(payload, dict):
                raise InvalidStateLibrary("The State Manager import is malformed.")
            result = await asyncio.to_thread(_import_payload, store, payload)
            return web.json_response(with_user_id(result, user_id))
        except Exception as exc:
            return error_response(exc)
