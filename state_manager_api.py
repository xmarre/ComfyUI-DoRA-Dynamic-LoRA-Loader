import asyncio
import json
import logging
from typing import Any, Callable, Dict

from .state_manager_store import (
    InvalidStateLibrary,
    StateLibraryRevisionConflict,
    StatePresetNotFound,
    get_state_manager_store,
    state_manager_library_path,
)


LOGGER = logging.getLogger(__name__)
_ROUTES_REGISTERED = False


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
) -> None:
    global _ROUTES_REGISTERED
    if _ROUTES_REGISTERED:
        return
    _ROUTES_REGISTERED = True
    routes = prompt_server.instance.routes

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
            snapshot = await asyncio.to_thread(
                store.replace,
                payload.get("characters"),
                payload.get("expected_revision"),
            )
            return web.json_response(with_user_id(snapshot, user_id))
        except Exception as exc:
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
