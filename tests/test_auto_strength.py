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
    residual_beta=0.0,
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
        residual_beta=residual_beta,
        logical_groups=logical_groups,
    )


def _geomean(values):
    values = list(values)
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _family_reference(local_references, scores, beta):
    """Scalar that makes ordinary gains ``F / L * (L / x) ** beta`` multiply to 1."""
    pairs = list(zip(local_references, scores))
    return math.exp(
        sum((1.0 - beta) * math.log(local) + beta * math.log(score) for local, score in pairs)
        / len(pairs)
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

    # Every helper update is 2x2, so each score carries the same sqrt(fan_in=2)
    # output-gain factor; gains are unaffected by it.
    expected_family_reference = _geomean(regimes.values())
    expected_gains = {
        role: expected_family_reference / magnitude
        for role, magnitude in regimes.items()
    }
    assert len(targets) == 200
    for block in range(50):
        for role, expected_gain in expected_gains.items():
            base = f"h3_blocks_{block}_{role.replace('.', '_')}"
            assert targets[base] == pytest.approx(expected_gain)

    assert report["schema"] == 2
    assert report["scoring_basis"] == "linear=fan_in_normalized_update_rms;other=absolute_update_rms"
    assert report["base_relative_scoring"] is False
    assert report["family_references"]["model/linear"] == pytest.approx(
        expected_family_reference * math.sqrt(2.0)
    )
    assert report["cohort_reference_policy"] == {
        "repeated_block_linear": "gain_normalized_family_local_depth_log_median_plus_single_outlier_gate",
        "other_tensor_families": "arithmetic_mean",
    }
    assert {cohort["semantic_role"] for cohort in report["cohorts"]} == set(regimes)
    for cohort in report["cohorts"]:
        assert cohort["measured_logical_count"] == 50
        assert cohort["role_gain_raw"] == pytest.approx(expected_gains[cohort["semantic_role"]])
        assert cohort["family_reference_score"] == pytest.approx(
            expected_family_reference * math.sqrt(2.0)
        )



def test_residual_beta_endpoints_and_log_midpoint(dora_modules):
    nodes, _ = dora_modules
    lora_sd = {}
    key_map = {}
    model_state = {}

    # Seven points keep the local reference stable at 1.0 for the center block,
    # while the center layer itself is 0.25.
    magnitudes = (1.0, 1.0, 1.0, 0.25, 1.0, 1.0, 1.0)
    for block, magnitude in enumerate(magnitudes):
        _add_linear(
            lora_sd,
            key_map,
            model_state,
            f"block_{block}",
            f"diffusion_model.blocks.{block}.mlp.fc2.weight",
            magnitude,
        )

    center = "block_3"
    local_only, local_report = _analyze(
        nodes, lora_sd, key_map, model_state, residual_beta=0.0
    )
    hybrid, hybrid_report = _analyze(
        nodes, lora_sd, key_map, model_state, residual_beta=0.5
    )
    direct, direct_report = _analyze(
        nodes, lora_sd, key_map, model_state, residual_beta=1.0
    )

    # The center is the cohort's single statistical outlier, so it is deliberately
    # promoted to full residual correction at every beta.
    local_references = [1.0] * len(magnitudes)
    for targets, beta in ((local_only, 0.0), (hybrid, 0.5), (direct, 1.0)):
        family_reference = _family_reference(local_references, magnitudes, beta)
        assert targets[center] == pytest.approx(family_reference / 0.25)
    assert _logical_report(hybrid_report, center)["residual_exponent_applied"] == pytest.approx(1.0)

    # A non-outlier layer with score 1.0 has no residual and therefore stays at the
    # local depth gain for all beta values.
    for report in (local_report, hybrid_report, direct_report):
        item = _logical_report(report, "block_2")
        assert item["residual_gain_raw"] == pytest.approx(1.0)


def test_residual_beta_partially_restores_non_outlier_per_layer_correction(dora_modules):
    nodes, _ = dora_modules
    lora_sd = {}
    key_map = {}
    model_state = {}

    # Smooth local variation: block 3 is not an outlier, local median is 1.0, and
    # the direct old-style gain differs from the local-depth gain.
    magnitudes = (0.8, 0.9, 1.0, 0.7, 1.0, 1.1, 1.2)
    for block, magnitude in enumerate(magnitudes):
        _add_linear(
            lora_sd,
            key_map,
            model_state,
            f"block_{block}",
            f"diffusion_model.blocks.{block}.mlp.fc2.weight",
            magnitude,
        )

    local_references = [1.0] * len(magnitudes)
    local_gain = _family_reference(local_references, magnitudes, 0.0) / 1.0
    half_reference = _family_reference(local_references, magnitudes, 0.5)
    direct_gain = _family_reference(local_references, magnitudes, 1.0) / 0.7

    local, _ = _analyze(nodes, lora_sd, key_map, model_state, residual_beta=0.0)
    half, half_report = _analyze(nodes, lora_sd, key_map, model_state, residual_beta=0.5)
    direct, _ = _analyze(nodes, lora_sd, key_map, model_state, residual_beta=1.0)

    expected_half = half_reference * math.sqrt(1.0 / 0.7)
    assert local["block_3"] == pytest.approx(local_gain)
    assert half["block_3"] == pytest.approx(expected_half)
    assert direct["block_3"] == pytest.approx(direct_gain)
    item = _logical_report(half_report, "block_3")
    assert item["residual_beta"] == pytest.approx(0.5)
    assert item["residual_exponent_applied"] == pytest.approx(0.5)
    assert item["residual_gain_raw"] == pytest.approx(math.sqrt(1.0 / 0.7))
    assert item["direct_family_gain_raw"] == pytest.approx(half_reference / 0.7)


def test_analyzer_default_residual_beta_is_half(dora_modules):
    nodes, _ = dora_modules
    lora_sd = {}
    key_map = {}
    model_state = {}
    for block, magnitude in enumerate((0.8, 0.9, 1.0, 0.7, 1.0, 1.1, 1.2)):
        _add_linear(
            lora_sd,
            key_map,
            model_state,
            f"block_{block}",
            f"diffusion_model.blocks.{block}.mlp.fc2.weight",
            magnitude,
        )

    _, report = nodes._auto_strength_analyze_base_targets(
        lora_sd=lora_sd,
        lora_bases=sorted(nodes._extract_lora_bases(lora_sd)),
        key_map=key_map,
        model_state_dict=model_state,
        clip_state_dict=None,
        analysis_device_mode="cpu",
        analysis_load_device=None,
        strength_model=1.0,
        strength_clip=1.0,
        ratio_floor=0.1,
        ratio_ceiling=10.0,
    )
    assert report["residual_beta"] == pytest.approx(0.5)
    assert report["linear_depth_redistribution_policy"]["residual_beta"] == pytest.approx(0.5)


def test_h3_like_weak_fc2_depth_region_recovers_large_boosts(dora_modules):
    nodes, _ = dora_modules
    lora_sd = {}
    key_map = {}
    model_state = {}

    for block in range(50):
        _add_linear(
            lora_sd,
            key_map,
            model_state,
            f"qkv_{block}",
            f"diffusion_model.blocks.{block}.attn.qkv_proj.weight",
            1.0,
        )
        fc2_magnitude = 0.25 if block < 35 else 2.0
        _add_linear(
            lora_sd,
            key_map,
            model_state,
            f"fc2_{block}",
            f"diffusion_model.blocks.{block}.mlp.fc2.weight",
            fc2_magnitude,
        )

    targets, report = _analyze(nodes, lora_sd, key_map, model_state)

    # Helper updates are 2x2: reported scores carry a sqrt(2) output-gain factor.
    family_reference = report["family_references"]["model/linear"] / math.sqrt(2.0)
    assert _geomean(targets.values()) == pytest.approx(1.0)
    early_gain = family_reference / 0.25
    late_gain = family_reference / 2.0
    assert early_gain > 2.5 and late_gain < 0.5
    assert all(targets[f"fc2_{block}"] == pytest.approx(early_gain) for block in range(0, 32))
    assert all(targets[f"fc2_{block}"] == pytest.approx(late_gain) for block in range(38, 50))

    fc2_cohort = next(c for c in report["cohorts"] if c["semantic_role"] == "mlp.fc2")
    assert fc2_cohort["depth_profile_eligible"] is True
    assert fc2_cohort["depth_gain_max_raw"] == pytest.approx(early_gain)
    assert fc2_cohort["depth_gain_min_raw"] == pytest.approx(late_gain)


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

    # Five members form one depth window, so each role's local reference is its
    # median (3.0 and 1.5). The role fallback reference is its geometric mean.
    family_reference = _family_reference([3.0] * 5 + [1.5] * 5, role_a + role_b, 0.0)
    assert all(targets[f"a_{block}"] == pytest.approx(family_reference / 3.0) for block in range(5))
    assert all(targets[f"b_{block}"] == pytest.approx(family_reference / 1.5) for block in range(5))

    gains = {cohort["semantic_role"]: cohort["role_gain_raw"] for cohort in report["cohorts"]}
    assert gains["attn.qkv_proj"] == pytest.approx(family_reference / _geomean(role_a))
    assert gains["mlp.fc2"] == pytest.approx(family_reference / _geomean(role_b))
    assert _geomean(targets.values()) == pytest.approx(1.0)

    # The same scalar applies to every member in each role, so trained depth ratios survive.
    assert _logical_report(report, "a_0")["role_gain_raw"] == pytest.approx(
        _logical_report(report, "a_4")["role_gain_raw"]
    )
    assert _logical_report(report, "b_0")["role_gain_raw"] == pytest.approx(
        _logical_report(report, "b_4")["role_gain_raw"]
    )


def test_weak_projection_composes_depth_gain_with_isolated_outlier_correction(dora_modules):
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
    family_reference = _family_reference([4.0] * 5 + [1.0] * 5, [4.0] * 5 + [0.2] + [1.0] * 4, 0.0)
    qkv_gain = family_reference / 4.0
    fc2_depth_gain = family_reference / 1.0

    assert targets[weak] == pytest.approx(fc2_depth_gain * 5.0)
    assert all(
        targets[f"blocks_{block}_attn_qkv_proj"] == pytest.approx(qkv_gain)
        for block in range(5)
    )
    assert all(
        targets[f"blocks_{block}_mlp_fc2"] == pytest.approx(fc2_depth_gain)
        for block in (0, 1, 3, 4)
    )
    weak_report = _logical_report(report, weak)
    assert weak_report["semantic_role"] == "mlp.fc2"
    assert weak_report["block_identity"] == "blocks.2"
    assert weak_report["depth_reference_score"] == pytest.approx(math.sqrt(2.0))
    assert weak_report["depth_gain_raw"] == pytest.approx(fc2_depth_gain)
    assert weak_report["anomaly_gain_raw"] == pytest.approx(5.0)
    assert weak_report["ratio_raw"] == pytest.approx(fc2_depth_gain * 5.0)
    assert weak_report["ratio_applied"] == pytest.approx(fc2_depth_gain * 5.0)


def test_strong_projection_uses_family_over_local_depth_reference(dora_modules):
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
    family_reference = _family_reference([1.0] * 5, (1.0, 1.0, 1.0, 5.0, 1.0), 0.0)
    assert targets[strong] == pytest.approx(family_reference / 5.0)
    assert all(
        targets[f"blocks_{block}_mlp_fc2"] == pytest.approx(family_reference)
        for block in (0, 1, 2, 4)
    )
    strong_report = _logical_report(report, strong)
    assert strong_report["depth_reference_score"] == pytest.approx(math.sqrt(2.0))
    assert strong_report["anomaly_gain_raw"] == pytest.approx(0.2)
    assert strong_report["ratio_raw"] == pytest.approx(family_reference / 5.0)


def test_mad_gate_composes_with_robust_local_depth_baseline(dora_modules):
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

    family_reference = _family_reference([1.0] * 5, (0.9, 1.0, 1.05, 1.1, 0.2), 0.0)
    assert targets["block_4"] == pytest.approx(family_reference / 0.2)
    assert all(targets[f"block_{block}"] == pytest.approx(family_reference) for block in range(4))
    cohort = report["cohorts"][0]
    assert cohort["log_mad"] > 0.0
    assert cohort["depth_profile_eligible"] is True
    assert cohort["outlier_candidate_count"] == 1
    assert cohort["corrected_logical_count"] == 1


def test_bimodal_depth_regime_receives_coherent_local_gains(dora_modules):
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

    family_reference = _geomean([0.5] * 25 + [1.0] * 25)
    assert all(targets[f"block_{block}"] == pytest.approx(family_reference / 0.5) for block in range(25))
    assert all(targets[f"block_{block}"] == pytest.approx(family_reference) for block in range(25, 50))
    cohort = report["cohorts"][0]
    assert cohort["depth_profile_eligible"] is True
    assert cohort["depth_profile_window_size"] == 7
    assert cohort["outlier_candidate_count"] == 0
    assert cohort["corrected_logical_count"] == 0


def test_smooth_depth_profile_uses_smooth_local_baseline(dora_modules):
    nodes, _ = dora_modules
    lora_sd = {}
    key_map = {}
    model_state = {}

    magnitudes = []
    for block in range(50):
        magnitude = 0.45 + (0.55 * block / 49.0)
        magnitudes.append(magnitude)
        _add_linear(
            lora_sd,
            key_map,
            model_state,
            f"block_{block}",
            f"diffusion_model.blocks.{block}.attn.out_proj.weight",
            magnitude,
        )

    targets, report = _analyze(nodes, lora_sd, key_map, model_state)

    local_references = [magnitudes[min(max(block, 3), 46)] for block in range(50)]
    family_reference = _family_reference(local_references, magnitudes, 0.0)
    for block in (3, 10, 24, 40, 46):
        assert targets[f"block_{block}"] == pytest.approx(
            family_reference / magnitudes[block]
        )
    assert targets["block_0"] == pytest.approx(family_reference / magnitudes[3])
    assert targets["block_49"] == pytest.approx(family_reference / magnitudes[46])
    assert report["cohorts"][0]["depth_profile_eligible"] is True


def test_populous_tail_is_redistributed_as_a_coherent_depth_regime(dora_modules):
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

    family_reference = _geomean([0.5] * 20 + [1.0] * 30)
    assert all(targets[f"block_{block}"] == pytest.approx(family_reference / 0.5) for block in range(20))
    assert all(targets[f"block_{block}"] == pytest.approx(family_reference) for block in range(20, 50))
    cohort = report["cohorts"][0]
    assert cohort["redistribution_eligible"] is True
    assert cohort["depth_profile_eligible"] is True
    assert cohort["outlier_candidate_count"] == 20
    assert cohort["outlier_candidate_limit"] == 1
    assert cohort["corrected_logical_count"] == 0
    assert cohort["anomaly_fallback_reason"] == "multiple_outlier_candidates"


def test_multiple_outlier_candidates_suppress_only_individual_correction(dora_modules):
    nodes, _ = dora_modules
    lora_sd = {}
    key_map = {}
    model_state = {}

    magnitudes = (0.2, 1.0, 1.0, 1.0, 1.0, 5.0)
    for block, magnitude in enumerate(magnitudes):
        _add_linear(
            lora_sd,
            key_map,
            model_state,
            f"block_{block}",
            f"diffusion_model.blocks.{block}.attn.out_proj.weight",
            magnitude,
        )

    targets, report = _analyze(nodes, lora_sd, key_map, model_state)

    family_reference = _geomean(magnitudes)
    assert all(target == pytest.approx(family_reference) for target in targets.values())
    cohort = report["cohorts"][0]
    assert cohort["outlier_candidate_count"] == 2
    assert cohort["outlier_candidate_limit"] == 1
    assert cohort["corrected_logical_count"] == 0
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
        if block % 2:
            model_state[destination] = torch.ones((4, 2), dtype=torch.uint8)
        else:
            model_state[destination] = torch.ones((4, 4), dtype=torch.float32)

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

    model_family_reference = _family_reference([1.0] * 5, (0.2, 1.0, 1.0, 1.0, 1.0), 0.0)
    assert targets["model_0"] == pytest.approx(model_family_reference * 5.0)
    assert all(targets[f"model_{block}"] == pytest.approx(model_family_reference) for block in range(1, 5))
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

    # Scores are output gains: element RMS times sqrt(fan_in) (2 or 3 here).
    scores_2 = [math.sqrt(2.0) * (0.2 if block == 0 else 1.0) for block in range(5)]
    scores_3 = [math.sqrt(3.0)] * 5
    family_reference = _family_reference(
        [math.sqrt(2.0)] * 5 + [math.sqrt(3.0)] * 5, scores_2 + scores_3, 0.0
    )
    assert targets["shape_0"] == pytest.approx(family_reference / math.sqrt(2.0) * 5.0)
    assert all(
        targets[f"shape_{block}"] == pytest.approx(family_reference / math.sqrt(2.0))
        for block in range(1, 5)
    )
    assert all(
        targets[f"shape_{block}"] == pytest.approx(family_reference / math.sqrt(3.0))
        for block in range(5, 10)
    )
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
    # The geometric family reference equals every local depth median here.
    assert _logical_report(report, "mlp_fc1_0")["depth_gain_raw"] == pytest.approx(1.0)
    assert _logical_report(report, "mlp_fc2_0")["depth_gain_raw"] == pytest.approx(1.0)
    assert _logical_report(report, "mlp_fc1_0")["ratio_raw"] == pytest.approx(100.0)
    assert _logical_report(report, "mlp_fc2_0")["ratio_raw"] == pytest.approx(0.01)


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
    assert [item["score"] for item in report_a["logical_groups"]] == pytest.approx(
        [item["score"] for item in report_b["logical_groups"]]
    )
    assert all(item["base_weight_rms"] is None for item in report_a["logical_groups"])
    assert all(item["relative_perturbation"] is None for item in report_a["logical_groups"])


def test_linear_scores_are_comparable_across_projection_input_widths(dora_modules):
    nodes, _ = dora_modules
    lora_sd = {}
    key_map = {}
    model_state = {}
    # Same output RMS change for unit-RMS input in every projection, but element
    # RMS scales with 1/sqrt(fan_in) exactly like a uniformly trained adapter.
    roles = {
        "attn.qkv_proj": (12, 4),
        "mlp.fc2": (4, 16),
        "adaln_proj.linear": (24, 2),
    }
    for block in range(6):
        for role, (out_features, in_features) in roles.items():
            base = f"b{block}_{role.replace('.', '_')}"
            _add_linear(
                lora_sd,
                key_map,
                model_state,
                base,
                f"diffusion_model.blocks.{block}.{role}.weight",
                1.0 / math.sqrt(in_features),
                shape=(out_features, in_features),
            )

    targets, report = _analyze(nodes, lora_sd, key_map, model_state, residual_beta=0.5)

    assert len(targets) == 18
    assert all(value == pytest.approx(1.0) for value in targets.values())
    for item in report["logical_groups"]:
        assert item["scoring_basis"] == "fan_in_normalized_update_rms"
        assert item["score"] == pytest.approx(1.0)
        assert item["update_rms"] is None


def _two_role_imbalanced_lora(role_a, role_b):
    lora_sd = {}
    key_map = {}
    model_state = {}
    for role, values in (("attn.qkv_proj", role_a), ("mlp.fc2", role_b)):
        for block, magnitude in enumerate(values):
            _add_linear(
                lora_sd,
                key_map,
                model_state,
                f"b{block}_{role.replace('.', '_')}",
                f"diffusion_model.blocks.{block}.{role}.weight",
                magnitude,
            )
    return lora_sd, key_map, model_state


@pytest.mark.parametrize("beta", [0.0, 0.25, 0.5, 1.0])
@pytest.mark.parametrize(
    "role_a, role_b",
    [
        ((1.0, 2.0, 3.0, 4.0, 5.0), (0.5, 1.0, 1.5, 2.0, 2.5)),
        ((1.0, 1.3, 0.8, 1.1, 0.9, 1.2, 1.0), (0.2, 0.25, 0.3, 0.2, 0.35, 0.25, 0.3)),
    ],
)
def test_linear_redistribution_is_gain_neutral_for_every_beta(dora_modules, beta, role_a, role_b):
    nodes, _ = dora_modules
    lora_sd, key_map, model_state = _two_role_imbalanced_lora(role_a, role_b)

    _, report = _analyze(
        nodes, lora_sd, key_map, model_state,
        ratio_floor=0.0, ratio_ceiling=1e6, residual_beta=beta,
    )

    items = report["logical_groups"]
    assert len(items) == len(role_a) + len(role_b)
    assert not any(item["outlier_corrected"] for item in items)
    # Unclamped ordinary gains multiply to exactly 1 across the family, so the
    # redistribution neither strengthens nor weakens the adapter in log space.
    assert _geomean(item["ratio_raw"] for item in items) == pytest.approx(1.0)


def test_clamping_is_applied_after_gain_normalization(dora_modules):
    nodes, _ = dora_modules
    role_a = (1.0, 2.0, 3.0, 4.0, 5.0)
    role_b = (0.5, 1.0, 1.5, 2.0, 2.5)
    lora_sd, key_map, model_state = _two_role_imbalanced_lora(role_a, role_b)

    _, free_report = _analyze(
        nodes, lora_sd, key_map, model_state,
        ratio_floor=0.0, ratio_ceiling=1e6, residual_beta=0.5,
    )
    targets, report = _analyze(
        nodes, lora_sd, key_map, model_state,
        ratio_floor=0.8, ratio_ceiling=1.2, residual_beta=0.5,
    )

    free_raw = {item["logical_id"]: item["ratio_raw"] for item in free_report["logical_groups"]}
    clipped_any = False
    for item in report["logical_groups"]:
        # The ratio window only clips; it never renormalizes the remaining members.
        assert item["ratio_raw"] == pytest.approx(free_raw[item["logical_id"]])
        assert item["ratio_applied"] == pytest.approx(min(1.2, max(0.8, item["ratio_raw"])))
        assert targets[item["logical_id"]] == pytest.approx(item["ratio_applied"])
        clipped_any = clipped_any or not (0.8 < item["ratio_raw"] < 1.2)
    assert clipped_any


def test_isolated_outlier_correction_is_outside_gain_normalization(dora_modules):
    nodes, _ = dora_modules
    lora_sd = {}
    key_map = {}
    model_state = {}
    magnitudes = (0.9, 1.0, 1.05, 1.1, 0.2)
    for block, magnitude in enumerate(magnitudes):
        _add_linear(
            lora_sd,
            key_map,
            model_state,
            f"block_{block}",
            f"diffusion_model.blocks.{block}.mlp.fc2.weight",
            magnitude,
        )

    _, report = _analyze(
        nodes, lora_sd, key_map, model_state,
        ratio_floor=0.0, ratio_ceiling=1e6, residual_beta=0.5,
    )

    outlier = _logical_report(report, "block_4")
    assert outlier["outlier_corrected"] is True
    assert outlier["residual_exponent_applied"] == pytest.approx(1.0)
    # Ordinary composition (exponent beta for everyone) is what is normalized; the
    # outlier's promotion to exponent 1 multiplies its gain on top of that.
    ordinary = [
        item["depth_gain_raw"] * item["residual_gain_raw"] ** (0.5 / item["residual_exponent_applied"])
        for item in report["logical_groups"]
    ]
    assert _geomean(ordinary) == pytest.approx(1.0)
    assert outlier["ratio_raw"] == pytest.approx(outlier["depth_gain_raw"] * (1.0 / 0.2))


def _five_fc1_bases(extra=None):
    lora_sd = {}
    key_map = {}
    model_state = {}
    for block, magnitude in enumerate((0.8, 1.0, 1.2, 1.0, 0.9)):
        base = f"fc1_{block}"
        _add_linear(
            lora_sd,
            key_map,
            model_state,
            base,
            f"diffusion_model.blocks.{block}.mlp.fc1.weight",
            magnitude,
        )
        if extra is not None:
            extra(lora_sd, base, block)
    return lora_sd, key_map, model_state


@pytest.mark.parametrize("bias_blocks", [(2,), (0, 1, 2, 3, 4)])
def test_bias_patches_do_not_change_weight_shape_scores_or_targets(dora_modules, bias_blocks):
    nodes, _ = dora_modules
    baseline_targets, baseline_report = _analyze(
        nodes, *_five_fc1_bases(), residual_beta=0.5
    )

    def add_bias(lora_sd, base, block):
        if block in bias_blocks:
            lora_sd[f"{base}.diff_b"] = torch.full((2,), 7.0, dtype=torch.float32)

    targets, report = _analyze(nodes, *_five_fc1_bases(add_bias), residual_beta=0.5)

    assert report["shape_mismatch_bases"] == []
    assert report["measured_bases"] == 5
    assert targets == pytest.approx(baseline_targets)
    for item in report["logical_groups"]:
        base = item["logical_id"]
        block = int(base.split("_")[-1])
        assert item["destination_shape"] == "2x2"
        assert item["score"] == pytest.approx(_logical_report(baseline_report, base)["score"])
        bias = item["bases"][0]["bias_update_rms"]
        assert bias == (pytest.approx(7.0) if block in bias_blocks else None)


def test_direct_diff_requires_exact_destination_shape(dora_modules):
    nodes, _ = dora_modules

    def build(diff_shape):
        lora_sd = {}
        key_map = {}
        model_state = {}
        for block in range(5):
            base = f"diff_{block}"
            destination = f"diffusion_model.blocks.{block}.attn.out_proj.weight"
            key_map[base] = destination
            model_state[destination] = torch.ones((4, 4), dtype=torch.float32)
            lora_sd[f"{base}.diff"] = torch.full(diff_shape, 1.0 if block else 0.2)
        return _analyze(nodes, lora_sd, key_map, model_state)

    targets, report = build((2, 8))
    assert report["shape_mismatch_bases"] == sorted(f"diff_{block}" for block in range(5))
    assert all(value == pytest.approx(1.0) for value in targets.values())

    targets, report = build((4, 4))
    assert report["shape_mismatch_bases"] == []
    assert targets["diff_0"] > 1.0


def test_factorized_update_is_applicable_when_element_counts_match(dora_modules):
    nodes, _ = dora_modules
    lora_sd = {}
    key_map = {}
    model_state = {}
    for block in range(5):
        base = f"lora_{block}"
        destination = f"diffusion_model.blocks.{block}.attn.out_proj.weight"
        _add_linear(
            lora_sd, key_map, model_state, base, destination,
            1.0 if block else 0.2, shape=(2, 8),
        )
        # ComfyUI reshapes a factorized update to the destination shape.
        model_state[destination] = torch.ones((4, 4), dtype=torch.float32)

    targets, report = _analyze(nodes, lora_sd, key_map, model_state)

    assert report["shape_mismatch_bases"] == []
    assert targets["lora_0"] > 1.0


def test_updates_that_cannot_patch_their_destination_do_not_shift_references(dora_modules):
    nodes, _ = dora_modules

    def build(include_reduced_basis_updates):
        lora_sd = {}
        key_map = {}
        model_state = {}
        for block, magnitude in enumerate((0.8, 1.0, 1.2, 1.0, 0.9)):
            _add_linear(
                lora_sd,
                key_map,
                model_state,
                f"qkv_{block}",
                f"diffusion_model.blocks.{block}.attn.qkv_proj.weight",
                magnitude,
            )
            if include_reduced_basis_updates:
                # Full-width update (8x4) against a dense destination whose input was
                # reduced to a small basis (8x2): ComfyUI cannot reshape and skips it.
                base = f"adaln_{block}"
                _add_linear(
                    lora_sd,
                    key_map,
                    model_state,
                    base,
                    f"diffusion_model.blocks.{block}.adaln_proj.linear.weight",
                    50.0,
                    shape=(8, 4),
                )
                model_state[key_map[base]] = torch.ones((8, 2), dtype=torch.float32)
        return _analyze(nodes, lora_sd, key_map, model_state, residual_beta=0.5)

    baseline_targets, _ = build(False)
    targets, report = build(True)

    for block in range(5):
        assert targets[f"qkv_{block}"] == pytest.approx(baseline_targets[f"qkv_{block}"])
        assert targets[f"adaln_{block}"] == pytest.approx(1.0)
        item = _logical_report(report, f"adaln_{block}")
        assert item["fallback_to_global"] is True
        assert item["fallback_reason"] == "destination_shape_mismatch"
    assert report["shape_mismatch_bases"] == sorted(f"adaln_{block}" for block in range(5))
