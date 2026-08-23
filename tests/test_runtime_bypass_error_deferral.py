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


def make_lora_adapter(comfy_weight_adapter):
    return comfy_weight_adapter.LoRAAdapter(
        set(),
        (
            torch.tensor([[1.0], [0.5]], dtype=torch.float32),
            torch.tensor([[0.25, -0.5, 1.0]], dtype=torch.float32),
            1.0,
            None,
            None,
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


def test_runtime_capture_is_transactional_across_base_retry(dora_modules):
    _, runtime = dora_modules
    loader = runtime.RuntimeBypassDoraPowerLoraLoader()
    loader._runtime_bypass_capture = {"model": [], "clip": []}
    adapter = make_lora_adapter(runtime.comfy.weight_adapter)
    attempts = []

    def materialize_regular(patches, *args, **kwargs):
        attempts.append(dict(patches))
        if len(attempts) == 1:
            raise RuntimeError("transient materialization failure")
        return list(patches)

    capture_add = loader._capture_add_patches(
        "model",
        {"layer.weight", "layer.bias"},
        materialize_regular,
        "mixed.safetensors",
    )
    patches = {
        "layer.weight": adapter,
        "layer.bias": torch.tensor([0.25, -0.5], dtype=torch.float32),
    }

    with pytest.raises(RuntimeError, match="transient materialization failure"):
        capture_add(patches, 0.75)
    assert loader._runtime_bypass_capture["model"] == []

    assert capture_add(patches, 0.75) == ["layer.weight", "layer.bias"]
    assert len(attempts) == 2
    assert len(loader._runtime_bypass_capture["model"]) == 1
    assert loader._runtime_bypass_capture["model"][0]["adapter"] is adapter


def test_deferred_validation_does_not_commit_partial_capture(dora_modules):
    _, runtime = dora_modules
    loader = runtime.RuntimeBypassDoraPowerLoraLoader()
    loader._runtime_bypass_capture = {"model": [], "clip": []}
    deferred = []
    patches = {
        "first.weight": make_lora_adapter(runtime.comfy.weight_adapter),
        "second.weight": make_dora_adapter(runtime.comfy.weight_adapter),
    }

    capture_add = loader._capture_add_patches(
        "model",
        set(patches),
        lambda regular, *args, **kwargs: list(regular),
        "partially-supported.safetensors",
        deferred,
    )

    assert capture_add(patches, 1.0) == []
    assert len(deferred) == 1
    assert isinstance(deferred[0], runtime.RuntimeBypassUnsupportedError)
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
