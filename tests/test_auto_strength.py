import math

import pytest
import torch


def _linear_pair(magnitude, out_features=2, in_features=2):
    return (
        torch.full((out_features, 1), float(magnitude), dtype=torch.float32),
        torch.ones((1, in_features), dtype=torch.float32),
    )


def _add_linear(lora_sd, key_map, state_dict, base, destination, magnitude, *, shape=(2, 2), mapping=None):
    up, down = _linear_pair(magnitude, out_features=shape[0], in_features=shape[1])
    lora_sd[f"{base}.lora_up.weight"] = up
    lora_sd[f"{base}.lora_down.weight"] = down
    key_map[base] = destination if mapping is None else mapping
    state_dict[destination] = torch.ones(shape, dtype=torch.float32)


def _analyze(
    nodes,
    lora_sd,
    key_map,
    model_state_dict,
    clip_state_dict=None,
    *,
    strength_model=1.0,
    strength_clip=1.0,
    ratio_floor=0.1,
    ratio_ceiling=10.0,
    logical_groups=None,
):
    return nodes._auto_strength_analyze_base_targets(
        lora_sd=lora_sd,
        lora_bases=sorted(nodes._extract_lora_bases(lora_sd)),
        key_map=key_map,
        model_state_dict=model_state_dict,
        clip_state_dict=clip_state_dict,
        analysis_device_mode="cpu",
        analysis_load_device=None,
        strength_model=strength_model,
        strength_clip=strength_clip,
        ratio_floor=ratio_floor,
        ratio_ceiling=ratio_ceiling,
        logical_groups=logical_groups,
    )


def _logical_report(report, logical_id):
    return next(item for item in report["logical_groups"] if item["logical_id"] == logical_id)


def test_minimax_projection_regimes_are_redistributed_by_uniform_role_gain(dora_modules):
    nodes, _ = dora_modules
    lora_sd = {}
    key_map = {}
    model_state = {}
    regimes = {
        "attn.qkv_proj": 4.0,
        "attn.out_proj": 2.0,
        "mlp.fc1": 5.0,
        "mlp.fc2": 1.0,
    }

    for block in range(50):
        for role, magnitude in regimes.items():
            base = f"h3_blocks_{block}_{role.replace('.', '_')}"
            destination = f"diffusion_model.transformer.blocks.{block}.{role}.weight"
            _add_linear(lora_sd, key_map, model_state, base, destination, magnitude)

    targets, report = _analyze(nodes, lora_sd, key_map, model_state)

    expected_family_reference = 3.0
    expected_gains = {
        "attn.qkv_proj": 0.75,
        "attn.out_proj": 1.5,
        "mlp.fc1": 0.6,
        "mlp.fc2": 3.0,
    }
    assert len(targets) == 200
    for block in range(50):
        for role, expected_gain in expected_gains.items():
            base = f"h3_blocks_{block}_{role.replace('.', '_')}"
            assert targets[base] == pytest.approx(expected_gain)

    assert report["schema"] == 2
    assert report["scoring_basis"] == "absolute_update_rms"
    assert report["base_relative_scoring"] is False
    assert report["family_references"]["model/linear"] == pytest.approx(expected_family_reference)
    assert report["cohort_reference_policy"] == {
        "repeated_block_linear": "family_arithmetic_role_gain_plus_log_median_mad_single_outlier_gate",
        "other_tensor_families": "arithmetic_mean",
    }
    assert {cohort["semantic_role"] for cohort in report["cohorts"]} == set(regimes)
    for cohort in report["cohorts"]:
        assert cohort["measured_logical_count"] == 50
        assert cohort["role_gain_raw"] == pytest.approx(expected_gains[cohort["semantic_role"]])
        assert cohort["family_reference_score"] == pytest.approx(expected_family_reference)


def test_role_gain_preserves_depth_profile_exactly(dora_modules):
    nodes, _ = dora_modules
    lora_sd = {}
    key_map = {}
    model_state = {}

    role_a = (1.0, 2.0, 3.0, 4.0, 5.0)
    role_b = (0.5, 1.0, 1.5, 2.0, 2.5)
    for block, magnitude in enumerate(role_a):
        _add_linear(
            lora_sd,
            key_map,
            model_state,
            f"a_{block}",
            f"diffusion_model.blocks.{block}.attn.qkv_proj.weight",
            magnitude,
        )
    for block, magnitude in enumerate(role_b):
        _add_linear(
            lora_sd,
            key_map,
            model_state,
            f"b_{block}",
            f"diffusion_model.blocks.{block}.mlp.fc2.weight",
            magnitude,
        )

    targets, report = _analyze(nodes, lora_sd, key_map, model_state)

    # Family mean is 2.25; role means are 3.0 and 1.5.
    assert all(targets[f"a_{block}"] == pytest.approx(0.75) for block in range(5))
    assert all(targets[f"b_{block}"] == pytest.approx(1.5) for block in range(5))

    gains = {cohort["semantic_role"]: cohort["role_gain_raw"] for cohort in report["cohorts"]}
    assert gains["attn.qkv_proj"] == pytest.approx(0.75)
    assert gains["mlp.fc2"] == pytest.approx(1.5)

    # The same scalar applies to every member in each role, so trained depth ratios survive.
    assert _logical_report(report, "a_0")["role_gain_raw"] == pytest.approx(
        _logical_report(report, "a_4")["role_gain_raw"]
    )
    assert _logical_report(report, "b_0")["role_gain_raw"] == pytest.approx(
        _logical_report(report, "b_4")["role_gain_raw"]
    )


def test_weak_projection_composes_role_gain_with_isolated_outlier_correction(dora_modules):
    nodes, _ = dora_modules
    lora_sd = {}
    key_map = {}
    model_state = {}

    for block in range(5):
        for role, magnitude in {"attn.qkv_proj": 4.0, "mlp.fc2": (0.2 if block == 2 else 1.0)}.items():
            base = f"blocks_{block}_{role.replace('.', '_')}"
            destination = f"diffusion_model.blocks.{block}.{role}.weight"
            _add_linear(lora_sd, key_map, model_state, base, destination, magnitude)

    targets, report = _analyze(nodes, lora_sd, key_map, model_state)

    weak = "blocks_2_mlp_fc2"
    family_reference = (5 * 4.0 + 0.2 + 4 * 1.0) / 10.0
    qkv_gain = family_reference / 4.0
    fc2_gain = family_reference / ((0.2 + 4.0) / 5.0)

    assert targets[weak] == pytest.approx(10.0)
    assert all(
        targets[f"blocks_{block}_attn_qkv_proj"] == pytest.approx(qkv_gain)
        for block in range(5)
    )
    assert all(
        targets[f"blocks_{block}_mlp_fc2"] == pytest.approx(fc2_gain)
        for block in (0, 1, 3, 4)
    )
    weak_report = _logical_report(report, weak)
    assert weak_report["semantic_role"] == "mlp.fc2"
    assert weak_report["block_identity"] == "blocks.2"
    assert weak_report["cohort_reference_score"] == pytest.approx(1.0)
    assert weak_report["role_gain_raw"] == pytest.approx(fc2_gain)
    assert weak_report["anomaly_gain_raw"] == pytest.approx(5.0)
    assert weak_report["ratio_raw"] == pytest.approx(fc2_gain * 5.0)
    assert weak_report["ratio_applied"] == pytest.approx(10.0)


def test_strong_projection_is_pulled_back_within_its_role(dora_modules):
    nodes, _ = dora_modules
    lora_sd = {}
    key_map = {}
    model_state = {}

    for block in range(5):
        base = f"blocks_{block}_mlp_fc2"
        destination = f"diffusion_model.blocks.{block}.mlp.fc2.weight"
        _add_linear(lora_sd, key_map, model_state, base, destination, 5.0 if block == 3 else 1.0)

    targets, report = _analyze(nodes, lora_sd, key_map, model_state)

    strong = "blocks_3_mlp_fc2"
    assert targets[strong] == pytest.approx(0.2)
    assert _logical_report(report, strong)["ratio_raw"] == pytest.approx(0.2)


def test_mad_gate_detects_an_isolated_outlier_in_a_noisy_cohort(dora_modules):
    nodes, _ = dora_modules
    lora_sd = {}
    key_map = {}
    model_state = {}
    for block, magnitude in enumerate((0.9, 1.0, 1.05, 1.1, 0.2)):
        _add_linear(
            lora_sd,
            key_map,
            model_state,
            f"block_{block}",
            f"diffusion_model.blocks.{block}.mlp.fc2.weight",
            magnitude,
        )

    targets, report = _analyze(nodes, lora_sd, key_map, model_state)

    assert targets["block_4"] == pytest.approx(5.0)
    assert all(targets[f"block_{block}"] == pytest.approx(1.0) for block in range(4))
    cohort = report["cohorts"][0]
    assert cohort["log_mad"] > 0.0
    assert cohort["outlier_candidate_count"] == 1
    assert cohort["corrected_logical_count"] == 1


def test_bimodal_depth_regime_is_preserved(dora_modules):
    nodes, _ = dora_modules
    lora_sd = {}
    key_map = {}
    model_state = {}

    for block in range(50):
        magnitude = 0.5 if block < 25 else 1.0
        _add_linear(
            lora_sd,
            key_map,
            model_state,
            f"block_{block}",
            f"diffusion_model.blocks.{block}.attn.out_proj.weight",
            magnitude,
        )

    targets, report = _analyze(nodes, lora_sd, key_map, model_state)

    assert all(target == pytest.approx(1.0) for target in targets.values())
    cohort = report["cohorts"][0]
    assert cohort["redistribution_eligible"] is True
    assert cohort["outlier_candidate_count"] == 0
    assert cohort["corrected_logical_count"] == 0
    assert all(
        item["decision_reason"] == "role_redistribution"
        for item in report["logical_groups"]
    )


def test_smooth_depth_profile_is_preserved(dora_modules):
    nodes, _ = dora_modules
    lora_sd = {}
    key_map = {}
    model_state = {}

    for block in range(50):
        magnitude = 0.45 + (0.55 * block / 49.0)
        _add_linear(
            lora_sd,
            key_map,
            model_state,
            f"block_{block}",
            f"diffusion_model.blocks.{block}.attn.out_proj.weight",
            magnitude,
        )

    targets, report = _analyze(nodes, lora_sd, key_map, model_state)

    assert all(target == pytest.approx(1.0) for target in targets.values())
    assert report["cohorts"][0]["outlier_candidate_count"] == 0


def test_populous_tail_is_treated_as_a_coherent_regime(dora_modules):
    nodes, _ = dora_modules
    lora_sd = {}
    key_map = {}
    model_state = {}

    for block in range(50):
        magnitude = 0.5 if block < 20 else 1.0
        _add_linear(
            lora_sd,
            key_map,
            model_state,
            f"block_{block}",
            f"diffusion_model.blocks.{block}.attn.out_proj.weight",
            magnitude,
        )

    targets, report = _analyze(nodes, lora_sd, key_map, model_state)

    assert all(target == pytest.approx(1.0) for target in targets.values())
    cohort = report["cohorts"][0]
    assert cohort["redistribution_eligible"] is True
    assert cohort["outlier_candidate_count"] == 20
    assert cohort["outlier_candidate_limit"] == 1
    assert cohort["corrected_logical_count"] == 0
    assert cohort["fallback_reason"] is None
    assert cohort["anomaly_fallback_reason"] == "multiple_outlier_candidates"
    assert all(
        _logical_report(report, f"block_{block}")["anomaly_fallback_reason"]
        == "multiple_outlier_candidates"
        for block in range(20)
    )


def test_multiple_outlier_candidates_are_left_unchanged(dora_modules):
    nodes, _ = dora_modules
    lora_sd = {}
    key_map = {}
    model_state = {}

    for block, magnitude in enumerate((0.2, 1.0, 1.0, 1.0, 1.0, 5.0)):
        _add_linear(
            lora_sd,
            key_map,
            model_state,
            f"block_{block}",
            f"diffusion_model.blocks.{block}.attn.out_proj.weight",
            magnitude,
        )

    targets, report = _analyze(nodes, lora_sd, key_map, model_state)

    assert all(target == pytest.approx(1.0) for target in targets.values())
    cohort = report["cohorts"][0]
    assert cohort["outlier_candidate_count"] == 2
    assert cohort["outlier_candidate_limit"] == 1
    assert cohort["corrected_logical_count"] == 0
    assert cohort["fallback_reason"] is None
    assert cohort["anomaly_fallback_reason"] == "multiple_outlier_candidates"


def test_tiny_linear_cohort_cannot_infer_anomalies(dora_modules):
    nodes, _ = dora_modules
    lora_sd = {}
    key_map = {}
    model_state = {}
    for block, magnitude in enumerate((0.1, 10.0)):
        _add_linear(
            lora_sd,
            key_map,
            model_state,
            f"block_{block}",
            f"diffusion_model.blocks.{block}.mlp.fc2.weight",
            magnitude,
        )

    targets, report = _analyze(nodes, lora_sd, key_map, model_state)

    assert targets == {"block_0": 1.0, "block_1": 1.0}
    assert report["cohorts"][0]["fallback_reason"] is None
    assert report["cohorts"][0]["anomaly_fallback_reason"] == "insufficient_outlier_sample"
    assert all(item["ratio_applied"] == pytest.approx(1.0) for item in report["logical_groups"])


@pytest.mark.parametrize(
    ("destination", "expected_role", "expected_block"),
    [
        ("diffusion_model.double_blocks.7.img_attn.qkv.weight", "img_attn.qkv", "double_blocks.7"),
        (
            "diffusion_model.input_blocks.4.1.transformer_blocks.0.attn1.to_q.weight",
            "attn1.to_q",
            "input_blocks.4/transformer_blocks.0",
        ),
        (
            "clip.transformer.text_model.encoder.layers.11.self_attn.q_proj.weight",
            "self_attn.q_proj",
            "layers.11",
        ),
    ],
)
def test_projection_role_inference_uses_generic_repeated_paths(
    dora_modules, destination, expected_role, expected_block
):
    nodes, _ = dora_modules
    base = "adapter"
    metadata = nodes._auto_strength_projection_metadata(
        base,
        {base: destination},
        torch.ones((2, 2), dtype=torch.float32),
    )

    assert metadata["semantic_role"] == expected_role
    assert metadata["block_identity"] == expected_block
    assert metadata["cohort_eligible"] is True


def test_linear_cohort_shape_uses_logical_adapter_shape_not_packed_storage(dora_modules):
    nodes, _ = dora_modules
    lora_sd = {}
    key_map = {}
    model_state = {}

    for block in range(5):
        base = f"packed_{block}"
        destination = f"diffusion_model.blocks.{block}.mlp.fc1.weight"
        up, down = _linear_pair(1.0, out_features=4, in_features=4)
        lora_sd[f"{base}.lora_up.weight"] = up
        lora_sd[f"{base}.lora_down.weight"] = down
        key_map[base] = destination
        # Alternate physical storage shapes to emulate packed vs ordinary checkpoint tensors.
        model_state[destination] = torch.ones((4, 2 if block % 2 else 4), dtype=torch.float32)

    targets, report = _analyze(nodes, lora_sd, key_map, model_state)

    assert all(value == pytest.approx(1.0) for value in targets.values())
    assert len(report["cohorts"]) == 1
    cohort = report["cohorts"][0]
    assert cohort["destination_shape"] == "4x4"
    assert cohort["destination_shape_source"] == "adapter_update"
    assert cohort["measured_logical_count"] == 5


def test_unclassifiable_and_non_transformer_linear_layers_fall_back_safely(dora_modules):
    nodes, _ = dora_modules
    lora_sd = {}
    key_map = {}
    model_state = {}

    _add_linear(lora_sd, key_map, model_state, "head_a", "model.classifier.0.weight", 0.1)
    _add_linear(lora_sd, key_map, model_state, "head_b", "model.classifier.2.weight", 10.0)
    targets, report = _analyze(nodes, lora_sd, key_map, model_state)

    assert targets == {"head_a": 1.0, "head_b": 1.0}
    for logical_id in targets:
        item = _logical_report(report, logical_id)
        assert item["semantic_role"] is None
        assert item["ratio_applied"] is None
        assert item["fallback_to_global"] is True
        assert item["fallback_reason"] == "semantic_role_unresolved"


def test_conv_family_keeps_family_cohorting(dora_modules):
    nodes, _ = dora_modules
    lora_sd = {}
    key_map = {}
    model_state = {}
    for index, magnitude in enumerate((1.0, 4.0)):
        base = f"conv_{index}"
        destination = f"diffusion_model.convs.{index}.weight"
        lora_sd[f"{base}.lora_up.weight"] = torch.full((2, 1, 1, 1), magnitude)
        lora_sd[f"{base}.lora_down.weight"] = torch.ones((1, 2, 3, 3))
        key_map[base] = destination
        model_state[destination] = torch.ones((2, 2, 3, 3))

    targets, report = _analyze(nodes, lora_sd, key_map, model_state)

    assert targets["conv_0"] == pytest.approx(2.5)
    assert targets["conv_1"] == pytest.approx(0.625)
    assert len(report["cohorts"]) == 1
    assert report["cohorts"][0]["family"] == "conv:3x3"
    assert report["cohorts"][0]["reference_statistic"] == "arithmetic_mean"
    assert report["cohorts"][0]["reference_score"] == pytest.approx(2.5)


def test_model_and_clip_roles_never_share_a_cohort(dora_modules):
    nodes, _ = dora_modules
    lora_sd = {}
    key_map = {}
    model_state = {}
    clip_state = {}

    for block, magnitude in enumerate((0.2, 1.0, 1.0, 1.0, 1.0)):
        _add_linear(
            lora_sd,
            key_map,
            model_state,
            f"model_{block}",
            f"diffusion_model.blocks.{block}.attn.q_proj.weight",
            magnitude,
        )
    for block in range(5):
        _add_linear(
            lora_sd,
            key_map,
            clip_state,
            f"clip_{block}",
            f"clip.encoder.layers.{block}.self_attn.q_proj.weight",
            100.0,
        )

    targets, report = _analyze(nodes, lora_sd, key_map, model_state, clip_state)

    assert targets["model_0"] == pytest.approx(5.0)
    assert all(targets[f"model_{block}"] == pytest.approx(1.0) for block in range(1, 5))
    assert all(targets[f"clip_{block}"] == pytest.approx(1.0) for block in range(5))
    assert {cohort["group"] for cohort in report["cohorts"]} == {"model", "clip"}


def test_same_named_roles_with_different_shapes_do_not_share_a_cohort(dora_modules):
    nodes, _ = dora_modules
    lora_sd = {}
    key_map = {}
    model_state = {}
    specs = [
        *((block, (2, 2), 0.2 if block == 0 else 1.0) for block in range(5)),
        *((block + 5, (3, 3), 1.0) for block in range(5)),
    ]
    for block, shape, magnitude in specs:
        _add_linear(
            lora_sd,
            key_map,
            model_state,
            f"shape_{block}",
            f"diffusion_model.blocks.{block}.attn.q_proj.weight",
            magnitude,
            shape=shape,
        )

    targets, report = _analyze(nodes, lora_sd, key_map, model_state)

    assert targets["shape_0"] == pytest.approx(5.0)
    assert all(targets[f"shape_{block}"] == pytest.approx(1.0) for block in range(1, 10))
    assert {cohort["destination_shape"] for cohort in report["cohorts"]} == {"2x2", "3x3"}
    assert len(report["cohorts"]) == 2


def test_logical_fanout_and_sliced_targets_are_not_double_counted(dora_modules):
    nodes, _ = dora_modules
    lora_sd = {}
    key_map = {}
    model_state = {}
    logical_groups = {}

    for logical_index in range(5):
        logical_id = f"source_{logical_index}"
        magnitude = 0.1 if logical_index == 0 else 0.5
        for clone in range(2):
            block = logical_index * 2 + clone
            base = f"slice_{block}"
            destination = f"diffusion_model.blocks.{block}.attn.qkv.weight"
            _add_linear(
                lora_sd,
                key_map,
                model_state,
                base,
                destination,
                magnitude,
                shape=(2, 2),
                mapping=(destination, (0, 0, 2)),
            )
            logical_groups[base] = (logical_id, 0.5)

    targets, report = _analyze(
        nodes,
        lora_sd,
        key_map,
        model_state,
        logical_groups=logical_groups,
    )

    assert report["logical_groups_total"] == 5
    assert report["logical_groups_measured"] == 5
    assert all(_logical_report(report, f"source_{index}")["fanout"] == 2 for index in range(5))
    assert targets["slice_0"] == pytest.approx(5.0)
    assert targets["slice_1"] == pytest.approx(5.0)
    assert all(targets[f"slice_{block}"] == pytest.approx(1.0) for block in range(2, 10))


def test_alpha_and_no_alpha_scaling_preserve_source_tensors_and_dora_magnitude(dora_modules):
    nodes, _ = dora_modules
    up = torch.tensor([[1.0], [2.0]])
    down = torch.tensor([[3.0, 4.0]])
    alpha = torch.tensor(2.0)
    dora_scale = torch.tensor([1.0, 1.0])
    with_alpha = {
        "layer.lora_up.weight": up,
        "layer.lora_down.weight": down,
        "layer.alpha": alpha,
        "layer.dora_scale": dora_scale,
    }

    scaled_alpha, changed = nodes._apply_base_strength_ratios(with_alpha, {"layer": 2.5})

    assert changed is True
    assert scaled_alpha["layer.alpha"].item() == pytest.approx(5.0)
    assert with_alpha["layer.alpha"].item() == pytest.approx(2.0)
    assert scaled_alpha["layer.lora_up.weight"] is up
    assert scaled_alpha["layer.lora_down.weight"] is down
    assert scaled_alpha["layer.dora_scale"] is dora_scale

    without_alpha = {
        "layer.lora_up.weight": up,
        "layer.lora_down.weight": down,
        "layer.dora_scale": dora_scale,
    }
    scaled_up, changed = nodes._apply_base_strength_ratios(without_alpha, {"layer": 0.25})

    assert changed is True
    torch.testing.assert_close(scaled_up["layer.lora_up.weight"], up * 0.25)
    torch.testing.assert_close(without_alpha["layer.lora_up.weight"], up)
    assert scaled_up["layer.lora_down.weight"] is down
    assert scaled_up["layer.dora_scale"] is dora_scale

    no_op, changed = nodes._apply_base_strength_ratios(with_alpha, {"layer": 1.0})
    assert changed is False
    assert no_op["layer.alpha"] is alpha
    assert no_op["layer.lora_up.weight"] is up
    assert no_op["layer.lora_down.weight"] is down
    assert no_op["layer.dora_scale"] is dora_scale


def test_dora_measurement_remains_post_normalization_update_rms(dora_modules):
    nodes, _ = dora_modules
    weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    delta = torch.tensor([[0.0, 0.25], [0.5, 0.0]])
    dora_scale = torch.ones(2)

    measured = nodes._auto_strength_measure_dora_effect(weight, delta, dora_scale)

    weight_calc = weight + delta
    expected_weight = weight_calc * (
        dora_scale.reshape(2, 1)
        / (weight_calc.reshape(2, -1).norm(dim=1, keepdim=True) + torch.finfo(torch.float32).eps)
    )
    expected = (expected_weight - weight).norm().item() / math.sqrt(weight.numel())
    assert measured == pytest.approx(expected)


def test_zero_global_strength_and_ratio_clamps_are_safe(dora_modules):
    nodes, _ = dora_modules
    lora_sd = {}
    key_map = {}
    model_state = {}
    for role, magnitudes in {
        "mlp.fc1": (0.01, 1.0, 1.0, 1.0, 1.0),
        "mlp.fc2": (100.0, 1.0, 1.0, 1.0, 1.0),
    }.items():
        for block, magnitude in enumerate(magnitudes):
            base = f"{role.replace('.', '_')}_{block}"
            _add_linear(
                lora_sd,
                key_map,
                model_state,
                base,
                f"diffusion_model.blocks.{block}.{role}.weight",
                magnitude,
            )

    zero_targets, zero_report = _analyze(
        nodes,
        lora_sd,
        key_map,
        model_state,
        strength_model=0.0,
    )
    zero_ratios = nodes._auto_strength_targets_to_ratios(
        zero_targets, key_map, model_state, None, strength_model=0.0, strength_clip=0.0
    )

    assert all(target == 0.0 for target in zero_targets.values())
    assert all(ratio == 1.0 for ratio in zero_ratios.values())
    assert zero_report["measured_bases"] == 0

    clamped_targets, report = _analyze(
        nodes,
        lora_sd,
        key_map,
        model_state,
        ratio_floor=0.5,
        ratio_ceiling=2.0,
    )
    assert clamped_targets["mlp_fc1_0"] == pytest.approx(2.0)
    assert clamped_targets["mlp_fc2_0"] == pytest.approx(0.5)
    assert _logical_report(report, "mlp_fc1_0")["anomaly_gain_raw"] == pytest.approx(100.0)
    assert _logical_report(report, "mlp_fc2_0")["anomaly_gain_raw"] == pytest.approx(0.01)
    assert _logical_report(report, "mlp_fc1_0")["ratio_raw"] > 100.0
    assert _logical_report(report, "mlp_fc2_0")["ratio_raw"] < 0.01


def test_standard_lora_scoring_is_checkpoint_weight_independent(dora_modules):
    nodes, _ = dora_modules
    lora_sd = {}
    key_map = {}
    model_state_a = {}
    for block, magnitude in enumerate((0.2, 1.0, 1.0, 1.0, 1.0)):
        _add_linear(
            lora_sd,
            key_map,
            model_state_a,
            f"block_{block}",
            f"diffusion_model.blocks.{block}.mlp.fc2.weight",
            magnitude,
        )
    model_state_b = {key: value * 1000.0 for key, value in model_state_a.items()}

    targets_a, report_a = _analyze(nodes, lora_sd, key_map, model_state_a)
    targets_b, report_b = _analyze(nodes, lora_sd, key_map, model_state_b)

    assert targets_a == pytest.approx(targets_b)
    assert [item["update_rms"] for item in report_a["logical_groups"]] == pytest.approx(
        [item["update_rms"] for item in report_b["logical_groups"]]
    )
    assert all(item["base_weight_rms"] is None for item in report_a["logical_groups"])
    assert all(item["relative_perturbation"] is None for item in report_a["logical_groups"])
