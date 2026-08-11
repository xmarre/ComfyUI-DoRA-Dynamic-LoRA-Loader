"""Backend-visible diagnostic for a missing/disabled custom frontend extension.

The DoRA Power LoRA Loader's LoRA rows and Add LoRA control are constructed by
``web/dora_power_lora_loader.js``. If that frontend extension is disabled,
missing from the installation, or otherwise not executed, ComfyUI falls back to
rendering the raw backend inputs. That state is functional enough to expose the
node, but it has no way to create LoRA rows.

This wrapper prepends one ordinary optional STRING widget explaining the failure
mode. The normal frontend rebuild removes all backend widgets before drawing the
real loader UI, so the diagnostic is visible only when the frontend enhancement
has not taken over.
"""

from __future__ import annotations

from .runtime_bypass import RuntimeBypassDoraPowerLoraLoader


FRONTEND_STATUS_INPUT = "frontend_ui_status"
FRONTEND_STATUS_MESSAGE = (
    "LoRA UI frontend is not active. Enable "
    "'comfyui_dora_dynamic_lora.power_lora_loader' in Settings > Extensions, "
    "then reload ComfyUI. If that extension is not listed, reinstall/update this "
    "custom node and verify that its web/ folder is present."
)


class FrontendGuardDoraPowerLoraLoader(RuntimeBypassDoraPowerLoraLoader):
    """Expose a self-diagnostic widget when the JavaScript UI is unavailable."""

    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        optional = dict(inputs.get("optional") or {})

        guarded = dict(inputs)
        guarded["optional"] = {
            FRONTEND_STATUS_INPUT: (
                "STRING",
                {
                    "default": FRONTEND_STATUS_MESSAGE,
                    "multiline": True,
                    "tooltip": (
                        "This message should disappear when the DoRA Power LoRA Loader "
                        "frontend extension is running. If it is visible, the dynamic "
                        "LoRA rows and Add LoRA control cannot be created by the browser UI."
                    ),
                },
            ),
            **optional,
        }
        return guarded
