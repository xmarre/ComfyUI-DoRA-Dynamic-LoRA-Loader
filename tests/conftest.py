import importlib.util
import os
import pathlib
import sys
import types

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
COMFYUI_PATH = pathlib.Path(os.environ.get("COMFYUI_PATH", ROOT / "ComfyUI")).resolve()

if str(COMFYUI_PATH) not in sys.path:
    sys.path.insert(0, str(COMFYUI_PATH))


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
