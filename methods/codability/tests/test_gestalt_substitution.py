"""Planted tests for the target-indexed gestalt adapter."""

import numpy as np

from methods.codability.experiments.gestalt_substitution import (
    gestalt_diagnostics,
    gestalt_substitution_report,
)


def test_xor_is_localized_as_within_span_interaction_and_composed_value():
    rng = np.random.default_rng(8)
    n = 400
    u0 = (rng.random(n) > 0.5).astype(float)
    u1 = (rng.random(n) > 0.5).astype(float)
    target = np.logical_xor(u0 > 0.5, u1 > 0.5).astype(float)
    target = np.where(target > 0.5, 0.97, 0.03)
    noise = (rng.random((3, n)) > 0.5).astype(float)
    units = np.vstack([u0, u1, noise])
    composed = np.clip(target + rng.normal(0, 0.01, n), 0.001, 0.999)
    result = gestalt_diagnostics(
        target_orbit={"canonical": target, "question": target},
        unit_signals=units,
        candidate_orbits={"composed": {"canonical": composed, "question": composed}},
        unit_ids=["u0", "u1", "n0", "n1", "n2"],
        seed=3,
    )
    profile = result["gestalt_profile"]
    assert result["joint_combiner_ladder"]["dpi_ok"]
    assert profile["interaction_within_unit_span"]
    assert profile["interaction_bits"] > 0.3
    assert result["candidate_recovery"]["composed"]["robust"][
        "oriented_recovery_fraction"] > 0.8
    assert result["certified"] is False


def test_gestalt_report_keeps_target_provenance_and_substitution_separate():
    rng = np.random.default_rng(19)
    n = 300
    q = np.tile(np.linspace(0.03, 0.97, 100), 3)
    big = np.clip(q + rng.normal(0, 0.01, n), 0.001, 0.999)
    weak = np.clip(0.5 + rng.normal(0, 0.08, n), 0.001, 0.999)
    units = np.vstack([q, rng.random(n)])
    spec = {
        "target_id": "gestalt:synthetic:0",
        "target_view": "gestalt",
        "community_or_frame": "synthetic-practice",
        "informant_or_source": "large-reader",
        "probe_set_id": "synthetic-lockbox",
        "frozen_before_candidate_evaluation": True,
    }
    result = gestalt_substitution_report(
        target_spec=spec,
        target_orbit={"canonical": big, "question": big},
        unit_signals=units,
        candidate_orbits={"explicit-teaching": {"canonical": big, "question": big}},
        selected_candidate_id="explicit-teaching",
        small_sparse_orbit={"canonical": weak, "question": weak},
        n_boot=200,
        equivalence_delta=0.03,
        min_signature_rho=0.8,
        seed=4,
    )
    assert result["target_spec"]["target_view"] == "gestalt"
    assert result["substitution"]["methodological_substitution"]
    assert result["diagnostics"]["certified"] is False
    assert not result["paper_grade_claim_eligible"]

