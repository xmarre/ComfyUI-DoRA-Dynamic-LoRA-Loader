import json

import pytest
import torch
import torch.nn.functional as F


def make_adapter(comfy_weight_adapter, up, down, *, alpha=1.0, dora_scale=None, reshape=None):
    up_t = torch.tensor(up, dtype=torch.float32)
    down_t = torch.tensor(down, dtype=torch.float32)
    return comfy_weight_adapter.LoRAAdapter(
        set(),
        (up_t, down_t, alpha, None, dora_scale, reshape),
    )


def test_switch_defaults_off_and_invalidates_cache(dora_modules):
    _, runtime = dora_modules
    cls = runtime.RuntimeBypassDoraPowerLoraLoader

    spec = cls.INPUT_TYPES()["optional"]["runtime_bypass_lora"]
    assert spec[0] == "BOOLEAN"
    assert spec[1]["default"] is False

    off_key = json.loads(cls.IS_CHANGED(model=None, clip=None, runtime_bypass_lora=False))
    on_key = json.loads(cls.IS_CHANGED(model=None, clip=None, runtime_bypass_lora=True))
    assert off_key["runtime_bypass_lora"] is False
    assert on_key["runtime_bypass_lora"] is True
    assert off_key != on_key


def test_dora_key_detection_is_fail_closed(dora_modules):
    _, runtime = dora_modules

    assert runtime._is_dora_key("foo.dora_scale")
    assert runtime._is_dora_key("foo.lora_magnitude_vector.default.weight")
    assert runtime._is_dora_key("foo.w_norm")
    assert runtime._is_dora_key("foo.b_norm")
    assert not runtime._is_dora_key("foo.lora_up.weight")
    assert not runtime._is_dora_key("foo.lora_down.weight")


def test_dora_adapter_is_refused(dora_modules):
    _, runtime = dora_modules
    comfy_weight_adapter = runtime.comfy.weight_adapter
    adapter = make_adapter(
        comfy_weight_adapter,
        [[1.0], [0.5]],
        [[0.25, -0.5, 1.0]],
        dora_scale=torch.ones(2, dtype=torch.float32),
    )

    with pytest.raises(runtime.RuntimeBypassUnsupportedError, match="DoRA magnitude scaling"):
        runtime._validate_lora_adapter(adapter, "character.safetensors", "layer.weight")


def test_reshape_adapter_is_refused(dora_modules):
    _, runtime = dora_modules
    comfy_weight_adapter = runtime.comfy.weight_adapter
    adapter = make_adapter(
        comfy_weight_adapter,
        [[1.0], [0.5]],
        [[0.25, -0.5, 1.0]],
        reshape=[2, 3],
    )

    with pytest.raises(runtime.RuntimeBypassUnsupportedError, match="reshape metadata"):
        runtime._validate_lora_adapter(adapter, "reshape.safetensors", "layer.weight")


def test_preflight_cache_keeps_multiple_stacked_loras(dora_modules, monkeypatch, tmp_path):
    _, runtime = dora_modules
    loader = runtime.RuntimeBypassDoraPowerLoraLoader()
    loader._runtime_dora_scan_cache = {}

    paths = {}
    for name in ("a.safetensors", "b.safetensors"):
        path = tmp_path / name
        path.write_bytes(b"test")
        paths[name] = str(path)

    scans = []
    monkeypatch.setattr(
        runtime.folder_paths,
        "get_full_path",
        lambda category, name: paths.get(name),
    )

    def fake_raw_dora_keys(name):
        scans.append(name)
        return []

    monkeypatch.setattr(runtime, "_raw_dora_keys", fake_raw_dora_keys)

    loader._runtime_preflight("a.safetensors")
    loader._runtime_preflight("b.safetensors")
    loader._runtime_preflight("a.safetensors")

    assert scans == ["a.safetensors", "b.safetensors"]
    assert len(loader._runtime_dora_scan_cache) == 2


def test_standard_adapter_is_captured_without_materializing_patch(dora_modules):
    _, runtime = dora_modules
    comfy_weight_adapter = runtime.comfy.weight_adapter
    loader = runtime.RuntimeBypassDoraPowerLoraLoader()
    loader._runtime_bypass_capture = {"model": [], "clip": []}

    adapter = make_adapter(
        comfy_weight_adapter,
        [[1.0], [0.5]],
        [[0.25, -0.5, 1.0]],
    )
    materialized_calls = []

    def original_add_patches(patches, *args, **kwargs):
        materialized_calls.append((patches, args, kwargs))
        return list(patches)

    capture_add = loader._capture_add_patches(
        "model",
        {"layer.weight"},
        original_add_patches,
        "standard.safetensors",
    )
    applied = capture_add({"layer.weight": adapter}, 0.65)

    assert applied == ["layer.weight"]
    assert materialized_calls == []
    assert len(loader._runtime_bypass_capture["model"]) == 1
    captured = loader._runtime_bypass_capture["model"][0]
    assert captured["key"] == "layer.weight"
    assert captured["adapter"] is adapter
    assert captured["strength"] == pytest.approx(0.65)


def test_non_adapter_patch_preserves_native_materialized_semantics(dora_modules):
    _, runtime = dora_modules
    loader = runtime.RuntimeBypassDoraPowerLoraLoader()
    loader._runtime_bypass_capture = {"model": [], "clip": []}
    materialized_calls = []

    def original_add_patches(patches, *args, **kwargs):
        materialized_calls.append((patches, args, kwargs))
        return list(patches)

    capture_add = loader._capture_add_patches(
        "model",
        {"layer.weight"},
        original_add_patches,
        "mixed.safetensors",
    )
    patch = ("diff", torch.ones(2, 3))
    applied = capture_add({"layer.weight": patch}, 0.4)

    assert applied == ["layer.weight"]
    assert len(materialized_calls) == 1
    assert materialized_calls[0][0] == {"layer.weight": patch}
    assert loader._runtime_bypass_capture["model"] == []


def test_offset_adapter_is_refused_instead_of_approximated(dora_modules):
    _, runtime = dora_modules
    comfy_weight_adapter = runtime.comfy.weight_adapter
    loader = runtime.RuntimeBypassDoraPowerLoraLoader()
    loader._runtime_bypass_capture = {"model": [], "clip": []}
    adapter = make_adapter(
        comfy_weight_adapter,
        [[1.0], [0.5]],
        [[0.25, -0.5, 1.0]],
    )

    capture_add = loader._capture_add_patches(
        "model",
        {"layer.weight"},
        lambda patches, *args, **kwargs: list(patches),
        "offset.safetensors",
    )

    with pytest.raises(runtime.RuntimeBypassUnsupportedError, match="sliced/offset"):
        capture_add({("layer.weight", (0, 2), None): adapter}, 1.0)


def test_stacked_runtime_loras_match_additive_lora_math_and_restore_forward(dora_modules):
    _, runtime = dora_modules
    comfy_weight_adapter = runtime.comfy.weight_adapter

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(3, 2, bias=False)

    model = TinyModel()
    with torch.no_grad():
        model.layer.weight.copy_(
            torch.tensor(
                [
                    [0.20, -0.40, 0.10],
                    [0.50, 0.30, -0.20],
                ],
                dtype=torch.float32,
            )
        )

    adapter_a = make_adapter(
        comfy_weight_adapter,
        [[0.8], [-0.3]],
        [[0.4, -0.2, 0.6]],
        alpha=1.0,
    )
    adapter_b = make_adapter(
        comfy_weight_adapter,
        [[-0.5], [0.7]],
        [[0.1, 0.5, -0.4]],
        alpha=1.0,
    )
    strength_a = 0.65
    strength_b = -0.30
    x = torch.tensor(
        [
            [0.2, -0.7, 1.1],
            [1.3, 0.4, -0.2],
        ],
        dtype=torch.float32,
    )

    base = model.layer(x).detach().clone()
    expected_a = F.linear(F.linear(x, adapter_a.weights[1]), adapter_a.weights[0]) * strength_a
    expected_b = F.linear(F.linear(x, adapter_b.weights[1]), adapter_b.weights[0]) * strength_b
    expected = base + expected_a + expected_b

    injections, hook_count = runtime._make_stacked_injection(
        model,
        [
            {"key": "layer.weight", "adapter": adapter_a, "strength": strength_a, "lora_name": "a"},
            {"key": "layer.weight", "adapter": adapter_b, "strength": strength_b, "lora_name": "b"},
        ],
    )
    assert hook_count == 2
    assert len(injections) == 1

    # Direct repeated injection/ejection should be harmless as well as the normal
    # ModelPatcher-managed single inject/eject lifecycle.
    injections[0].inject(None)
    injections[0].inject(None)
    torch.testing.assert_close(model.layer(x), expected)
    injections[0].eject(None)
    injections[0].eject(None)

    torch.testing.assert_close(model.layer(x), base)


def test_load_loras_intercepts_cloned_patcher_end_to_end(dora_modules, monkeypatch):
    _, runtime = dora_modules
    comfy_weight_adapter = runtime.comfy.weight_adapter
    adapter = make_adapter(
        comfy_weight_adapter,
        [[0.8], [-0.3]],
        [[0.4, -0.2, 0.6]],
        alpha=1.0,
    )

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(3, 2, bias=False)

    class FakePatcher:
        def __init__(self, model):
            self.model = model
            self.materialized_calls = []
            self.injections = {}
            self.clone_source = None

        def clone(self):
            clone = FakePatcher(self.model)
            clone.clone_source = self
            return clone

        def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
            self.materialized_calls.append((patches, strength_patch, strength_model))
            return list(patches)

        def set_injections(self, key, injections):
            self.injections[key] = injections

    base_cls = runtime._base.DoraPowerLoraLoader

    def fake_base_load_one(self, model, clip, *args, **kwargs):
        model.add_patches(
            {"layer.weight": adapter},
            kwargs.get("strength_model", 1.0),
        )
        return model, clip, {}

    def fake_base_load_loras(self, model, clip, **kwargs):
        cloned = model.clone()
        new_model, new_clip, _ = self._load_one(
            cloned,
            clip,
            lora_name="standard.safetensors",
            strength_model=0.75,
            strength_clip=0.0,
        )
        return {"result": (new_model, new_clip, "{}", "", {})}

    monkeypatch.setattr(base_cls, "_load_one", fake_base_load_one)
    monkeypatch.setattr(base_cls, "load_loras", fake_base_load_loras)
    monkeypatch.setattr(
        runtime.RuntimeBypassDoraPowerLoraLoader,
        "_runtime_preflight",
        lambda self, lora_name: None,
    )

    model = TinyModel()
    original = FakePatcher(model)
    loader = runtime.RuntimeBypassDoraPowerLoraLoader()
    output = loader.load_loras(original, None, runtime_bypass_lora=True)
    result_model = output["result"][0]

    assert result_model is not original
    assert result_model.clone_source is original
    assert original.materialized_calls == []
    assert result_model.materialized_calls == []
    assert len(result_model.injections) == 1

    x = torch.tensor([[0.2, -0.7, 1.1]], dtype=torch.float32)
    base = model.layer(x).detach().clone()
    expected_delta = F.linear(F.linear(x, adapter.weights[1]), adapter.weights[0]) * 0.75
    injection = next(iter(result_model.injections.values()))[0]
    injection.inject(result_model)
    try:
        torch.testing.assert_close(model.layer(x), base + expected_delta)
    finally:
        injection.eject(result_model)
    torch.testing.assert_close(model.layer(x), base)


def test_load_loras_fails_closed_if_captured_adapters_cannot_be_injected(dora_modules, monkeypatch):
    _, runtime = dora_modules
    comfy_weight_adapter = runtime.comfy.weight_adapter
    adapter = make_adapter(
        comfy_weight_adapter,
        [[1.0], [0.5]],
        [[0.25, -0.5, 1.0]],
    )
    base_cls = runtime._base.DoraPowerLoraLoader

    def fake_base_load_loras(self, model, clip, **kwargs):
        self._runtime_bypass_capture["model"].append(
            {
                "key": "layer.weight",
                "adapter": adapter,
                "strength": 1.0,
                "lora_name": "standard.safetensors",
            }
        )
        return (model, clip)

    monkeypatch.setattr(base_cls, "load_loras", fake_base_load_loras)
    loader = runtime.RuntimeBypassDoraPowerLoraLoader()

    with pytest.raises(runtime.RuntimeBypassUnsupportedError, match="output shape is unsupported"):
        loader.load_loras(None, None, runtime_bypass_lora=True)


def test_runtime_module_uses_comfy_bypass_primitive(dora_modules):
    _, runtime = dora_modules
    assert hasattr(runtime.comfy.weight_adapter, "BypassForwardHook")
