import pytest
import torch
import torch.nn.functional as F


def make_lora_adapter(comfy_weight_adapter, up, down, *, alpha=1.0):
    return comfy_weight_adapter.LoRAAdapter(
        set(),
        (
            torch.tensor(up, dtype=torch.float32),
            torch.tensor(down, dtype=torch.float32),
            alpha,
            None,
            None,
            None,
        ),
    )


def make_lokr_adapter(
    comfy_weight_adapter,
    w1,
    w2,
    *,
    alpha=1.0,
    dora_scale=None,
    require_bypass=True,
):
    lokr_type = getattr(comfy_weight_adapter, "LoKrAdapter", None)
    if lokr_type is None:
        pytest.skip("This ComfyUI revision does not expose LoKrAdapter")
    if require_bypass:
        base_h = getattr(comfy_weight_adapter.WeightAdapterBase, "h", None)
        lokr_h = getattr(lokr_type, "h", None)
        if not callable(lokr_h) or lokr_h is base_h:
            pytest.skip("This ComfyUI revision does not implement LoKr bypass math")
    return lokr_type(
        set(),
        (
            torch.tensor(w1, dtype=torch.float32),
            torch.tensor(w2, dtype=torch.float32),
            alpha,
            None,
            None,
            None,
            None,
            None,
            dora_scale,
        ),
    )


def test_plain_lokr_adapter_is_accepted_when_comfy_has_native_bypass(dora_modules):
    _, runtime = dora_modules
    adapter = make_lokr_adapter(
        runtime.comfy.weight_adapter,
        [[1.0, -0.25], [0.5, 0.75]],
        [[0.3, -0.2, 0.1], [0.4, 0.6, -0.5]],
    )

    runtime._validate_lora_adapter(adapter, "plain_lokr.safetensors", "layer.weight")


def test_lokr_dora_scale_is_refused(dora_modules):
    _, runtime = dora_modules
    adapter = make_lokr_adapter(
        runtime.comfy.weight_adapter,
        [[1.0]],
        [[0.3, -0.2, 0.1], [0.4, 0.6, -0.5]],
        dora_scale=torch.ones(2, dtype=torch.float32),
    )

    with pytest.raises(runtime.RuntimeBypassUnsupportedError, match="LoKr DoRA magnitude scaling"):
        runtime._validate_lora_adapter(adapter, "dora_lokr.safetensors", "layer.weight")


def test_lokr_without_real_comfy_bypass_math_is_refused(dora_modules, monkeypatch):
    _, runtime = dora_modules
    comfy_weight_adapter = runtime.comfy.weight_adapter
    adapter = make_lokr_adapter(
        comfy_weight_adapter,
        [[1.0]],
        [[0.3, -0.2, 0.1], [0.4, 0.6, -0.5]],
        require_bypass=False,
    )

    monkeypatch.setattr(
        type(adapter),
        "h",
        comfy_weight_adapter.WeightAdapterBase.h,
    )

    with pytest.raises(runtime.RuntimeBypassUnsupportedError, match="does not implement LoKr bypass math"):
        runtime._validate_lora_adapter(adapter, "old_comfy_lokr.safetensors", "layer.weight")


def test_lokr_is_captured_without_materializing_patch(dora_modules):
    _, runtime = dora_modules
    loader = runtime.RuntimeBypassDoraPowerLoraLoader()
    loader._runtime_bypass_capture = {"model": [], "clip": []}
    adapter = make_lokr_adapter(
        runtime.comfy.weight_adapter,
        [[1.0, -0.25], [0.5, 0.75]],
        [[0.3, -0.2, 0.1], [0.4, 0.6, -0.5]],
    )
    materialized_calls = []

    def original_add_patches(patches, *args, **kwargs):
        materialized_calls.append((patches, args, kwargs))
        return list(patches)

    capture_add = loader._capture_add_patches(
        "model",
        {"layer.weight"},
        original_add_patches,
        "plain_lokr.safetensors",
    )
    applied = capture_add({"layer.weight": adapter}, 0.65)

    assert applied == ["layer.weight"]
    assert materialized_calls == []
    captured = loader._runtime_bypass_capture["model"]
    assert len(captured) == 1
    assert captured[0]["key"] == "layer.weight"
    assert captured[0]["adapter"] is not adapter
    assert captured[0]["adapter"].weights[2] == pytest.approx(1.0)
    assert captured[0]["strength"] == pytest.approx(0.65)
    assert captured[0]["lora_name"] == "plain_lokr.safetensors"


def test_direct_lokr_nonfinite_alpha_is_normalized_to_materialized_semantics(dora_modules):
    _, runtime = dora_modules
    comfy_weight_adapter = runtime.comfy.weight_adapter
    loader = runtime.RuntimeBypassDoraPowerLoraLoader()
    loader._runtime_bypass_capture = {"model": [], "clip": []}

    w1 = torch.tensor([[0.5, -0.25], [0.75, 0.2]], dtype=torch.float32)
    w2 = torch.tensor([[0.3, -0.2, 0.1], [0.4, 0.6, -0.5]], dtype=torch.float32)
    adapter = make_lokr_adapter(
        comfy_weight_adapter,
        w1.tolist(),
        w2.tolist(),
        alpha=float("inf"),
    )

    capture_add = loader._capture_add_patches(
        "model",
        {"layer.weight"},
        lambda patches, *args, **kwargs: list(patches),
        "inf_alpha_lokr.safetensors",
    )
    capture_add({"layer.weight": adapter}, 0.7)
    captured_adapter = loader._runtime_bypass_capture["model"][0]["adapter"]

    assert captured_adapter is not adapter
    assert adapter.weights[2] == float("inf")
    assert captured_adapter.weights[2] == pytest.approx(1.0)

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(6, 4, bias=False)

    model = TinyModel()
    x = torch.tensor([[0.2, -0.7, 1.1, 0.3, -0.4, 0.9]], dtype=torch.float32)
    base = model.layer(x).detach().clone()
    expected_delta = F.linear(x, torch.kron(w1, w2)) * 0.7

    injections, _ = runtime._make_stacked_injection(
        model,
        [{"key": "layer.weight", "adapter": captured_adapter, "strength": 0.7, "lora_name": "inf_alpha"}],
    )
    injections[0].inject(None)
    try:
        actual = model.layer(x)
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, base + expected_delta)
    finally:
        injections[0].eject(None)


def test_unequal_decomposed_lokr_ranks_are_normalized_to_materialized_scale(dora_modules):
    _, runtime = dora_modules
    comfy_weight_adapter = runtime.comfy.weight_adapter
    lokr_type = getattr(comfy_weight_adapter, "LoKrAdapter", None)
    if lokr_type is None:
        pytest.skip("This ComfyUI revision does not expose LoKrAdapter")
    base_h = getattr(comfy_weight_adapter.WeightAdapterBase, "h", None)
    lokr_h = getattr(lokr_type, "h", None)
    if not callable(lokr_h) or lokr_h is base_h:
        pytest.skip("This ComfyUI revision does not implement LoKr bypass math")

    w1_a = torch.tensor([[0.5], [0.3]], dtype=torch.float32)
    w1_b = torch.tensor([[0.8, -0.4]], dtype=torch.float32)  # rank_w1 = 1
    w2_a = torch.tensor([[0.4, -0.3], [0.2, 0.9]], dtype=torch.float32)
    w2_b = torch.tensor([[0.5, -0.1, 0.7], [-0.6, 0.8, 0.2]], dtype=torch.float32)  # rank_w2 = 2
    alpha = 1.5
    adapter = lokr_type(
        set(),
        (None, None, alpha, w1_a, w1_b, w2_a, w2_b, None, None),
    )

    runtime_adapter = runtime._runtime_adapter_for_bypass(
        adapter,
        "unequal_rank_lokr.safetensors",
        "layer.weight",
    )
    assert runtime_adapter is not adapter
    assert adapter.weights[2] == pytest.approx(alpha)
    assert runtime_adapter.weights[2] == pytest.approx(alpha * 1.0 / 2.0)

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(6, 4, bias=False)

    model = TinyModel()
    x = torch.tensor([[0.2, -0.7, 1.1, 0.3, -0.4, 0.9]], dtype=torch.float32)
    base = model.layer(x).detach().clone()
    materialized_delta = torch.kron(w1_a @ w1_b, w2_a @ w2_b) * (alpha / w2_b.shape[0])
    strength = 0.45

    injections, _ = runtime._make_stacked_injection(
        model,
        [{"key": "layer.weight", "adapter": runtime_adapter, "strength": strength, "lora_name": "unequal"}],
    )
    injections[0].inject(None)
    try:
        torch.testing.assert_close(
            model.layer(x),
            base + F.linear(x, materialized_delta) * strength,
        )
    finally:
        injections[0].eject(None)


def test_runtime_lokr_matches_materialized_kronecker_math_and_restores_forward(dora_modules):
    _, runtime = dora_modules
    comfy_weight_adapter = runtime.comfy.weight_adapter

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(6, 4, bias=False)

    model = TinyModel()
    with torch.no_grad():
        model.layer.weight.copy_(torch.arange(24, dtype=torch.float32).reshape(4, 6) / 50.0)

    w1 = torch.tensor([[0.5, -0.25], [0.75, 0.2]], dtype=torch.float32)
    w2 = torch.tensor([[0.3, -0.2, 0.1], [0.4, 0.6, -0.5]], dtype=torch.float32)
    adapter = make_lokr_adapter(comfy_weight_adapter, w1.tolist(), w2.tolist())
    strength = 0.7
    x = torch.tensor(
        [
            [0.2, -0.7, 1.1, 0.3, -0.4, 0.9],
            [1.3, 0.4, -0.2, 0.8, -0.6, 0.1],
        ],
        dtype=torch.float32,
    )

    base = model.layer(x).detach().clone()
    expected_delta = F.linear(x, torch.kron(w1, w2)) * strength

    injections, hook_count = runtime._make_stacked_injection(
        model,
        [
            {
                "key": "layer.weight",
                "adapter": adapter,
                "strength": strength,
                "lora_name": "plain_lokr.safetensors",
            }
        ],
    )
    assert hook_count == 1

    injections[0].inject(None)
    try:
        torch.testing.assert_close(model.layer(x), base + expected_delta)
    finally:
        injections[0].eject(None)

    torch.testing.assert_close(model.layer(x), base)


def test_decomposed_lokr_bypass_matches_materialized_scale(dora_modules):
    _, runtime = dora_modules
    comfy_weight_adapter = runtime.comfy.weight_adapter
    lokr_type = getattr(comfy_weight_adapter, "LoKrAdapter", None)
    if lokr_type is None:
        pytest.skip("This ComfyUI revision does not expose LoKrAdapter")
    base_h = getattr(comfy_weight_adapter.WeightAdapterBase, "h", None)
    lokr_h = getattr(lokr_type, "h", None)
    if not callable(lokr_h) or lokr_h is base_h:
        pytest.skip("This ComfyUI revision does not implement LoKr bypass math")

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(6, 4, bias=False)

    w1_a = torch.tensor([[0.5, -0.2], [0.3, 0.7]], dtype=torch.float32)
    w1_b = torch.tensor([[0.8, -0.4], [0.1, 0.6]], dtype=torch.float32)
    w2_a = torch.tensor([[0.4, -0.3], [0.2, 0.9]], dtype=torch.float32)
    w2_b = torch.tensor([[0.5, -0.1, 0.7], [-0.6, 0.8, 0.2]], dtype=torch.float32)
    alpha = 1.5
    rank = w2_b.shape[0]
    adapter = lokr_type(
        set(),
        (None, None, alpha, w1_a, w1_b, w2_a, w2_b, None, None),
    )

    model = TinyModel()
    x = torch.tensor([[0.2, -0.7, 1.1, 0.3, -0.4, 0.9]], dtype=torch.float32)
    base = model.layer(x).detach().clone()
    materialized_delta = torch.kron(w1_a @ w1_b, w2_a @ w2_b) * (alpha / rank)
    strength = 0.45

    injections, _ = runtime._make_stacked_injection(
        model,
        [{"key": "layer.weight", "adapter": adapter, "strength": strength, "lora_name": "factorized"}],
    )
    injections[0].inject(None)
    try:
        torch.testing.assert_close(
            model.layer(x),
            base + F.linear(x, materialized_delta) * strength,
        )
    finally:
        injections[0].eject(None)

    torch.testing.assert_close(model.layer(x), base)


def test_mixed_lora_and_lokr_stack_matches_materialized_math(dora_modules):
    _, runtime = dora_modules
    comfy_weight_adapter = runtime.comfy.weight_adapter

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(6, 4, bias=False)

    model = TinyModel()
    with torch.no_grad():
        model.layer.weight.copy_(torch.arange(24, dtype=torch.float32).reshape(4, 6) / 40.0)

    lora_up = torch.tensor([[0.8], [-0.3], [0.25], [0.6]], dtype=torch.float32)
    lora_down = torch.tensor([[0.4, -0.2, 0.6, 0.1, -0.5, 0.3]], dtype=torch.float32)
    lora = make_lora_adapter(comfy_weight_adapter, lora_up.tolist(), lora_down.tolist())

    lokr_w1 = torch.tensor([[0.5, -0.25], [0.75, 0.2]], dtype=torch.float32)
    lokr_w2 = torch.tensor([[0.3, -0.2, 0.1], [0.4, 0.6, -0.5]], dtype=torch.float32)
    lokr = make_lokr_adapter(comfy_weight_adapter, lokr_w1.tolist(), lokr_w2.tolist())

    lora_strength = 0.55
    lokr_strength = -0.35
    x = torch.tensor([[0.2, -0.7, 1.1, 0.3, -0.4, 0.9]], dtype=torch.float32)
    base = model.layer(x).detach().clone()
    expected_lora = F.linear(F.linear(x, lora_down), lora_up) * lora_strength
    expected_lokr = F.linear(x, torch.kron(lokr_w1, lokr_w2)) * lokr_strength

    injections, hook_count = runtime._make_stacked_injection(
        model,
        [
            {"key": "layer.weight", "adapter": lora, "strength": lora_strength, "lora_name": "a"},
            {"key": "layer.weight", "adapter": lokr, "strength": lokr_strength, "lora_name": "b"},
        ],
    )
    assert hook_count == 2

    injections[0].inject(None)
    try:
        torch.testing.assert_close(model.layer(x), base + expected_lora + expected_lokr)
    finally:
        injections[0].eject(None)

    torch.testing.assert_close(model.layer(x), base)
