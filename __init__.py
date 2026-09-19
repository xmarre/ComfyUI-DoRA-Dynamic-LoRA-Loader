import logging

from .nodes import (
    DoraStateManager,
    StateManager,
    StateManagerSeed,
    StateManagerTextBox,
    _normalize_state_manager_state,
    _state_manager_default_state,
    _resolve_dora_state_payload,
    _resolve_dora_state_payload_snapshot,
    _queued_library_user_from_ui_state,
    _state_payload_text_for_box,
)
from .runtime_bypass import RuntimeBypassDoraPowerLoraLoader
from .state_manager_api import register_routes as register_state_manager_routes
from .state_manager_prompt_bridge import (
    prompt_transport_ordering_contract,
    prompt_transport_provider_capabilities,
    register_prompt_bridge,
)

# Backend API for frontend LoRA dropdown (avoids relying on /object_info variants).
import folder_paths
from aiohttp import web
from server import PromptServer


@PromptServer.instance.routes.get("/dora_dynamic_lora/loras")
async def dora_dynamic_lora_list_loras(request):
    # Return plain list of filenames from ComfyUI's "loras" folder_paths category.
    # Frontend will prepend "None".
    return web.json_response(folder_paths.get_filename_list("loras"))


try:
    register_state_manager_routes(
        folder_paths,
        PromptServer,
        web,
        _normalize_state_manager_state,
        _state_manager_default_state,
        prompt_transport_provider=prompt_transport_provider_capabilities,
        prompt_transport_ordering_contract=prompt_transport_ordering_contract,
    )
except Exception:
    logging.getLogger(__name__).exception(
        "State Manager library routes could not be registered; State Manager persistence will be unavailable."
    )

try:
    register_prompt_bridge(
        PromptServer,
        resolve_payload=_resolve_dora_state_payload,
        resolve_snapshot=_resolve_dora_state_payload_snapshot,
        library_user_from_ui_state=_queued_library_user_from_ui_state,
        text_for_box=_state_payload_text_for_box,
    )
except Exception:
    logging.getLogger(__name__).exception(
        "State Manager backend prompt bridge could not be registered; managed Impact wildcard prompt routing will rely on the frontend fallback."
    )

# Tell ComfyUI to load our frontend extension.
WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS = {
    "DoRA Power LoRA Loader": RuntimeBypassDoraPowerLoraLoader,
    "State Manager": StateManager,
    "State Manager Text Box": StateManagerTextBox,
    "State Manager Seed": StateManagerSeed,
    # Backward-compatible alias for workflows made with the earlier patch.
    "DoRA State Manager": DoraStateManager,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DoRA Power LoRA Loader": "DoRA Power LoRA Loader",
    "State Manager": "State Manager",
    "State Manager Text Box": "State Text Box",
    "State Manager Seed": "State Seed",
    "DoRA State Manager": "State Manager (legacy)",
}
