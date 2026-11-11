"""Tests for the additive target-indexed articulation layer."""

import numpy as np
import pytest

from methods.codability.experiments.target_articulation_frontier import (
    SCORE_KEY,
    bootstrap_orbit_values,
    direct_orbit_fidelity,
    dose_record,
    manifest_sha256,
    monotone_frontier,
    orbit_recovery,
    paired_substitution_test,
    recovery_point,
    select_minimal_cost,
    target_orbit_mean,
)


def test_manifest_is_present_and_content_addressed():
    assert len(manifest_sha256()) == 64


def test_fixed_target_recovery_orients_an_inverted_policy():
    q = np.tile(np.array([0.05, 0.20, 0.80, 0.95]), 20)
    aligned = recovery_point(q, q)
    inverted = recovery_point(q, 1.0 - q)
    assert aligned["recovery_fraction"] == pytest.approx(inverted["recovery_fraction"])
    assert aligned[SCORE_KEY] > 0
    assert inverted[SCORE_KEY] < 0
    assert aligned["positive_polarity"]
    assert not inverted["positive_polarity"]


def test_vectorized_bootstrap_matches_pointwise_tvd_recovery():
    q = np.tile(np.array([0.05, 0.20, 0.80, 0.95]), 12)
    p1 = np.clip(q + 0.03, 0, 1)
    p2 = np.clip(q * 0.8 + 0.1, 0, 1)
    rng = np.random.default_rng(22)
    samples = rng.integers(0, len(q), size=(25, len(q)))
    result = bootstrap_orbit_values(q, {"a": p1, "b": p2}, samples,
                                    divergence="tvd", min_target_information=1e-6)
    expected = [orbit_recovery(q[idx], {"a": p1[idx], "b": p2[idx]})["robust"][SCORE_KEY]
                for idx in samples]
    assert result[SCORE_KEY] == pytest.approx(expected)


def test_target_quotient_and_candidate_frontier_use_adverse_form():
    q1 = np.tile(np.array([0.1, 0.2, 0.8, 0.9]), 12)
    q2 = np.clip(q1 + 0.02, 0, 1)
    q = target_orbit_mean({"canonical": q1, "question": q2})
    good = np.clip(q + 0.01, 0, 1)
    bad = np.full_like(q, 0.5)
    row = orbit_recovery(q, {"canonical": good, "question": bad})
    assert row["valid"]
    assert row["robust"]["adverse_form"] == "question"
    assert row["robust"][SCORE_KEY] == pytest.approx(0.0)
    assert not row["robust"]["all_positive_polarity"]


def test_direct_signature_is_not_implied_by_equal_target_recovery():
    q = np.tile(np.array([0.05, 0.20, 0.80, 0.95]), 20)
    # These have equal unsigned target recovery but opposite item-level policies.
    result = direct_orbit_fidelity({"canonical": q}, {"canonical": 1.0 - q})
    assert result["valid"]
    assert result["robust"]["spearman"] == pytest.approx(-1.0)


def _candidate(candidate_id, cost, score_source, q):
    return {
        "dose": dose_record(candidate_id, "declarative", word_count=int(cost),
                            interaction_degree=1, scalar_cost=cost,
                            cost_basis="synthetic_words"),
        "recovery": orbit_recovery(q, {"canonical": score_source}),
    }


def test_frontier_is_free_disposal_and_selection_is_minimal_cost():
    q = np.tile(np.array([0.05, 0.15, 0.85, 0.95]), 20)
    weak = np.clip(0.45 + 0.10 * q, 0, 1)
    strong = q.copy()
    medium = np.clip(0.20 + 0.60 * q, 0, 1)
    rows = [_candidate("weak", 5, weak, q), _candidate("strong", 20, strong, q),
            _candidate("medium", 30, medium, q)]
    frontier = monotone_frontier(rows)
    scores = [point["frontier_score"] for point in frontier["points"]]
    assert scores == sorted(scores)
    assert frontier["points"][-1]["frontier_candidate"] == "strong"
    selected = select_minimal_cost(rows, target_score=rows[1]["recovery"]["robust"][SCORE_KEY] - 1e-9,
                                   min_signature_rho=0.5)
    assert selected["target_attained"]
    assert selected["candidate_id"] == "strong"


def test_paired_substitution_requires_gap_gain_equivalence_and_fidelity():
    rng = np.random.default_rng(4)
    q = np.tile(np.linspace(0.03, 0.97, 80), 3)
    big = np.clip(q + rng.normal(0, 0.015, len(q)), 0.001, 0.999)
    weak = np.clip(0.5 + rng.normal(0, 0.08, len(q)), 0.001, 0.999)
    rich = big.copy()
    result = paired_substitution_test(
        q,
        small_sparse_orbit={"canonical": weak, "question": weak},
        big_sparse_orbit={"canonical": big, "question": big},
        articulated_orbit={"canonical": rich, "question": rich},
        gap_delta=0.02,
        equivalence_delta=0.03,
        min_signature_rho=0.8,
        signature_equivalence_delta=0.05,
        n_boot=250,
        seed=9,
    )
    assert result["valid"]
    assert result["gates"]["baseline_gap_confirmed"]
    assert result["gates"]["articulation_improvement_confirmed"]
    assert result["gates"]["equivalent_to_big_sparse"]
    assert result["gates"]["signature_improved"]
    assert result["gates"]["direct_signature_floor"]
    assert result["gates"]["direct_signature_improved"]
    assert result["methodological_substitution"]
    assert result["articulation_specific_substitution"] is None
    assert not result["paper_grade_substitution"]


def test_dose_rejects_undeclared_channels_and_negative_costs():
    with pytest.raises(ValueError, match="unknown articulation channel"):
        dose_record("x", "magic")
    with pytest.raises(ValueError, match="cannot be negative"):
        dose_record("x", "declarative", scalar_cost=-1)


def test_matched_control_upgrades_only_when_articulation_beats_it():
    rng = np.random.default_rng(17)
    q = np.tile(np.linspace(0.02, 0.98, 90), 3)
    big = np.clip(q + rng.normal(0, 0.01, len(q)), 0.001, 0.999)
    weak = np.clip(0.5 + rng.normal(0, 0.08, len(q)), 0.001, 0.999)
    result = paired_substitution_test(
        q,
        small_sparse_orbit={"canonical": weak},
        big_sparse_orbit={"canonical": big},
        articulated_orbit={"canonical": big},
        control_orbit={"canonical": weak},
        equivalence_delta=0.03,
        min_signature_rho=0.8,
        n_boot=200,
        seed=2,
    )
    assert result["methodological_substitution"]
    assert result["gates"]["articulation_specificity"]
    assert result["articulation_specific_substitution"]
