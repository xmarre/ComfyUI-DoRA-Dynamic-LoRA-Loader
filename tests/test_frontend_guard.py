import importlib.util
import os
import pathlib
import sys


def _load_frontend_guard_module(dora_modules):
    # dora_modules creates the synthetic package and loads nodes/runtime_bypass
    # without executing the custom-node __init__.py (which needs PromptServer).
    dora_modules
    root = pathlib.Path(
        os.environ.get("DORA_REPO_ROOT", pathlib.Path(__file__).resolve().parents[1])
    ).resolve()
    package_name = "dora_loader_testpkg"
    full_name = f"{package_name}.frontend_guard"

    existing = sys.modules.get(full_name)
    if existing is not None:
        return existing

    spec = importlib.util.spec_from_file_location(full_name, root / "frontend_guard.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not build import spec for frontend_guard.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    setattr(sys.modules[package_name], "frontend_guard", module)
    spec.loader.exec_module(module)
    return module


def test_frontend_guard_explains_missing_dynamic_lora_ui(dora_modules):
    guard = _load_frontend_guard_module(dora_modules)
    inputs = guard.FrontendGuardDoraPowerLoraLoader.INPUT_TYPES()
    optional = inputs["optional"]

    assert next(iter(optional)) == guard.FRONTEND_STATUS_INPUT
    assert "runtime_bypass_lora" in optional

    kind, options = optional[guard.FRONTEND_STATUS_INPUT]
    assert kind == "STRING"
    assert options["multiline"] is True
    assert "Settings > Extensions" in options["default"]
    assert "comfyui_dora_dynamic_lora.power_lora_loader" in options["default"]
    assert "web/" in options["default"]
