from .nodes import DoraPowerLoraLoader

# Tell ComfyUI to load our frontend extension.
WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS = {
    "DoRA Power LoRA Loader": DoraPowerLoraLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DoRA Power LoRA Loader": "DoRA Power LoRA Loader (DoRA + Flux2/OneTrainer key-fix)",
}
