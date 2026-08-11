import importlib.util
import os
import pathlib
import sys
import types

import pytest


# CI copies this test directory outside the custom-node package tree so pytest
# cannot auto-import the node's root __init__.py. DORA_REPO_ROOT points back to
# the actual checkout; local runs still fall back to the parent directory.
ROOT = pathlib.Path(
    os.environ.get("DORA_REPO_ROOT", pathlib.Path(__file__).resolve().parents[1])
).resolve()
COMFYUI_PATH = pathlib.Path(os.environ.get("COMFYUI_PATH", ROOT / "ComfyUI")).resolve()

comfy_cli_args_path = COMFYUI_PATH / "comfy" / "cli_args.py"
if not comfy_cli_args_path.is_file():
    pytest.skip(
        f"ComfyUI checkout not found at {COMFYUI_PATH}; set COMFYUI_PATH to a valid checkout.",
        allow_module_level=True,
    )

if str(COMFYUI_PATH) not in sys.path:
    sys.path.insert(0, str(COMFYUI_PATH))

# GitHub's hosted runners use the CPU-only PyTorch wheel. ComfyUI's CLI parser
# intentionally parses an empty argv when imported as a library, so its default
# device remains CUDA unless a caller changes the parsed args before importing
# comfy.model_management. Force CPU mode for these pure adapter tests; this is
# the same supported ComfyUI --cpu execution mode, without letting ComfyUI parse
# pytest's own command-line arguments.
comfy_cli_args = pytest.importorskip(
    "comfy.cli_args",
    reason=f"Could not import comfy.cli_args from ComfyUI checkout at {COMFYUI_PATH}.",
)
comfy_cli_args.args.cpu = True


@pytest.fixture(scope="session")
def dora_modules():
    """Load nodes.py/runtime_bypass.py as a package without executing __init__.py.

    The custom-node __init__ registers HTTP routes against a live PromptServer,
    which is correct inside ComfyUI but unnecessary for these pure loader tests.
    """
    package_name = "dora_loader_testpkg"
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT)]
    package.__package__ = package_name
    sys.modules[package_name] = package

    def load_module(short_name: str, filename: str):
        full_name = f"{package_name}.{short_name}"
        spec = importlib.util.spec_from_file_location(full_name, ROOT / filename)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not build import spec for {filename}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = module
        setattr(package, short_name, module)
        spec.loader.exec_module(module)
        return module

    nodes = load_module("nodes", "nodes.py")
    runtime = load_module("runtime_bypass", "runtime_bypass.py")
    return nodes, runtime
