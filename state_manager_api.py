import asyncio
import json
import logging
import os
from typing import Any, Callable, Dict

from .state_manager_store import (
    InvalidStateLibrary,
    StateLibraryRevisionConflict,
    StatePresetNotFound,
    get_state_manager_store,
)


LOGGER = logging.getLogger(__name__)
_ROUTES_REGISTERED = False


def _library_path(folder_paths_module: Any) -> str:
    return os.path.join(
        folder_paths_module.get_user_directory(),
        "dora_state_manager",
        "state-library.json",
    )


def configure_store(
    folder_paths_module: Any,
    normalize_state: Callable[[Any], Dict[str, Any]],
    default_state: Callable[[], Dict[str, Any]],
):
    return get_state_manager_store(
        path=_library_path(folder_paths_module),
        normalize_state=normalize_state,
        default_state=default_state,
    )


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
    store = configure_store(folder_paths_module, normalize_state, default_state)
    routes = prompt_server.instance.routes

    def error_response(exc: Exception):
        if isinstance(exc, StateLibraryRevisionConflict):
            return web.json_response(
                {"error": str(exc), "code": "revision_conflict", "snapshot": exc.current},
                status=409,
            )
        if isinstance(exc, StatePresetNotFound):
            return web.json_response({"error": str(exc) or "State Manager preset not found."}, status=404)
        if isinstance(exc, (InvalidStateLibrary, json.JSONDecodeError, ValueError, TypeError)):
            return web.json_response({"error": str(exc)}, status=400)
        LOGGER.exception("State Manager library API failed.")
        return web.json_response({"error": "Unable to update the State Manager library."}, status=500)

    @routes.get("/dora_dynamic_lora/state-library")
    async def state_manager_list_library(_request):
        try:
            return web.json_response(await asyncio.to_thread(store.snapshot))
        except Exception as exc:
            return error_response(exc)

    @routes.put("/dora_dynamic_lora/state-library")
    async def state_manager_replace_library(request):
        try:
            payload = await request.json()
            if not isinstance(payload, dict):
                raise InvalidStateLibrary("The State Manager library request is malformed.")
            snapshot = await asyncio.to_thread(
                store.replace,
                payload.get("characters"),
                payload.get("expected_revision"),
            )
            return web.json_response(snapshot)
        except Exception as exc:
            return error_response(exc)

    @routes.post("/dora_dynamic_lora/state-library/migrate")
    async def state_manager_migrate_library(request):
        try:
            payload = await request.json()
            if not isinstance(payload, dict):
                raise InvalidStateLibrary("The State Manager migration request is malformed.")
            result = await asyncio.to_thread(
                store.migrate_legacy,
                payload.get("state"),
                payload.get("selected_character_id", ""),
                payload.get("selected_prompt_id", ""),
            )
            return web.json_response(result)
        except Exception as exc:
            return error_response(exc)

    @routes.get("/dora_dynamic_lora/state-library/export")
    async def state_manager_export_library(_request):
        try:
            return web.json_response(await asyncio.to_thread(store.export_library))
        except Exception as exc:
            return error_response(exc)

    @routes.get("/dora_dynamic_lora/state-library/characters/{character_id}/export")
    async def state_manager_export_character(request):
        try:
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
            payload = await request.json()
            if not isinstance(payload, dict):
                raise InvalidStateLibrary("The State Manager import is malformed.")
            kind = str(payload.get("kind", ""))
            if kind == "dora_state_manager_character_export" or isinstance(payload.get("character"), dict):
                result = await asyncio.to_thread(store.import_character, payload.get("character"))
            else:
                characters = payload.get("characters")
                if characters is None and isinstance(payload.get("state"), dict):
                    characters = payload["state"].get("characters")
                result = await asyncio.to_thread(store.merge_library, characters)
            return web.json_response(result)
        except Exception as exc:
            return error_response(exc)
