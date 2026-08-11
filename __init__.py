from .frontend_guard import FrontendGuardDoraPowerLoraLoader
from .nodes import DoraStateManager, StateManager, StateManagerSeed, StateManagerTextBox

# Backend API for frontend LoRA dropdown (avoids relying on /object_info variants).
import folder_paths
from aiohttp import web
from server import PromptServer


@PromptServer.instance.routes.get("/dora_dynamic_lora/loras")
async def dora_dynamic_lora_list_loras(request):
    # Return plain list of filenames from ComfyUI's "loras" folder_paths category.
    # Frontend will prepend "None".
    return web.json_response(folder_paths.get_filename_list("loras"))

# Tell ComfyUI to load our frontend extension.
WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS = {
    "DoRA Power LoRA Loader": FrontendGuardDoraPowerLoraLoader,
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

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
