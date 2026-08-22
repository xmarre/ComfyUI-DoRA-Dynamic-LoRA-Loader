import pytest
import torch


def make_dora_adapter(comfy_weight_adapter):
    return comfy_weight_adapter.LoRAAdapter(
        set(),
        (
            torch.tensor([[1.0], [0.5]], dtype=torch.float32),
            torch.tensor([[0.25, -0.5, 1.0]], dtype=torch.float32),
            1.0,
            None,
            torch.ones(2, dtype=torch.float32),
            None,
        ),
    )


def test_runtime_validation_error_is_deferred_only_once(dora_modules):
    _, runtime = dora_modules
    loader = runtime.RuntimeBypassDoraPowerLoraLoader()
    loader._runtime_bypass_capture = {"model": [], "clip": []}
    adapter = make_dora_adapter(runtime.comfy.weight_adapter)
    deferred = []

    capture_add = loader._capture_add_patches(
        "model",
        {"layer.weight"},
        lambda patches, *args, **kwargs: list(patches),
        "dora.safetensors",
        deferred,
    )

    assert capture_add({"layer.weight": adapter}, 1.0) == []
    assert len(deferred) == 1
    assert isinstance(deferred[0], runtime.RuntimeBypassUnsupportedError)

    # nodes.py currently retries add_patches() after a raised exception. Once a
    # runtime validation failure has been deferred, a second invocation becomes a
    # no-op and cannot generate another copy of the same exception.
    assert capture_add({"layer.weight": adapter}, 1.0) == []
    assert len(deferred) == 1
    assert loader._runtime_bypass_capture["model"] == []


def test_runtime_load_one_raises_deferred_validation_after_base_retry_boundary(
    dora_modules,
    monkeypatch,
):
    _, runtime = dora_modules
    adapter = make_dora_adapter(runtime.comfy.weight_adapter)
    base_cls = runtime._base.DoraPowerLoraLoader
    add_attempts = []

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(3, 2, bias=False)

    class FakePatcher:
        def __init__(self):
            self.model = TinyModel()

        def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
            raise AssertionError("runtime adapter validation should intercept before materialization")

    def fake_base_load_one(self, model, clip, *args, **kwargs):
        # Mirror the current base loader's try/retry shape. With deferral, the first
        # call returns normally, so this retry path is never entered.
        try:
            add_attempts.append("first")
            model.add_patches({"layer.weight": adapter}, 1.0)
        except Exception:
            add_attempts.append("retry")
            model.add_patches({"layer.weight": adapter}, 1.0)
        return model, clip, {}

    monkeypatch.setattr(base_cls, "_load_one", fake_base_load_one)
    monkeypatch.setattr(
        runtime.RuntimeBypassDoraPowerLoraLoader,
        "_runtime_preflight",
        lambda self, lora_name: None,
    )

    loader = runtime.RuntimeBypassDoraPowerLoraLoader()
    loader._runtime_bypass_active = True
    loader._runtime_bypass_capture = {"model": [], "clip": []}

    with pytest.raises(runtime.RuntimeBypassUnsupportedError, match="DoRA magnitude scaling"):
        loader._load_one(
            FakePatcher(),
            None,
            lora_name="dora.safetensors",
        )

    assert add_attempts == ["first"]
    assert loader._runtime_bypass_capture["model"] == []
