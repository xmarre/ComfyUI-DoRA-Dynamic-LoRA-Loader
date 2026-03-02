from .nodes import DoraPowerLoraLoader

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
    "DoRA Power LoRA Loader": DoraPowerLoraLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DoRA Power LoRA Loader": "DoRA Power LoRA Loader (DoRA + Flux2/OneTrainer key-fix)",
}
