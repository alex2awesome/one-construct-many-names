"""Selection-aware lockbox readout tests."""

import numpy as np

from methods.codability.experiments.summarize_policy_isomorphism_lockbox import (
    _gap_closure,
    _paired_comparison,
    _ratio_summary,
)


def test_paired_gain_and_gap_closure_reward_direct_reconstruction():
    q = np.tile(np.linspace(0.05, 0.95, 120), 2)
    target = {"canonical": q, "question": np.clip(q + 0.01, 0, 1)}
    name = {"canonical": np.clip(q + 0.18, 0, 1),
            "question": np.clip(q + 0.20, 0, 1)}
    candidate = {"canonical": np.clip(q + 0.01, 0, 1),
                 "question": np.clip(q + 0.02, 0, 1)}
    rng = np.random.default_rng(11)
    samples = rng.integers(0, len(q), size=(500, len(q)))

    paired = _paired_comparison(
        target, name, candidate, samples=samples, confidence=0.95)
    closure = _gap_closure(
        target, name, candidate, samples=samples, confidence=0.95)

    assert paired["gates"]["mae_superior"]
    assert paired["estimates"]["mae_gain"]["point"] > 0.1
    assert closure["direct_target_quotient"]["mae_gap_closed"]["point"] > 0.9
    assert closure["adverse_form_identity_band"]["mae_excess_removed"]["point"] > 0.9


def test_ratio_summary_does_not_invert_a_nonpositive_reference_gap():
    result = _ratio_summary(
        1.0, -1.0, np.array([1.0, 2.0]), np.array([-1.0, -2.0]), confidence=0.95)
    assert result["point"] is None
    assert result["CI"] is None
    assert result["valid_bootstrap_fraction"] == 1.0
    assert result["nonpositive_denominator_fraction"] == 1.0
    assert not result["denominator_stable_positive"]


def test_ratio_summary_does_not_condition_away_nonpositive_draws():
    result = _ratio_summary(
        0.5,
        1.0,
        np.array([0.5, 0.5, 0.5, 0.5]),
        np.array([1.0, 1.0, -0.1, 1.0]),
        confidence=0.95,
    )
    assert result["nonpositive_denominator_fraction"] == 0.25
    assert result["CI"] is None
    assert result["inference_status"].startswith("ratio_interval_undefined")
