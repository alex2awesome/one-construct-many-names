"""Tests for the additive lexical fixed-target adapter."""

import json

import numpy as np

from methods.codability.experiments.fixed_target_name_substitution import (
    alignment_report,
    analyze_metric,
    load_grid_orbits,
    resolve_grid_dir,
    soft_stratified_split,
    summarize_metrics,
)


def test_soft_target_split_preserves_both_halves_and_is_deterministic():
    q = np.linspace(0.01, 0.99, 100)
    train1, test1 = soft_stratified_split(q, seed=7)
    train2, test2 = soft_stratified_split(q, seed=7)
    assert np.array_equal(train1, train2)
    assert np.array_equal(test1, test2)
    assert not set(train1) & set(test1)
    assert sorted(np.r_[train1, test1]) == list(range(100))
    assert 0.4 < q[train1].mean() < 0.6
    assert 0.4 < q[test1].mean() < 0.6


def test_grid_templates_keep_cross_family_sources_separate(tmp_path):
    default = resolve_grid_dir(tmp_path, "math")
    qwen = resolve_grid_dir(tmp_path, "math", "r3_{domain}/grid_{domain}_qwen")
    absolute = resolve_grid_dir(tmp_path, "math", str(tmp_path / "external"))
    assert default == tmp_path / "r3_math/grid_math_v1"
    assert qwen == tmp_path / "r3_math/grid_math_qwen"
    assert absolute == tmp_path / "external"


def test_summary_separates_ineligible_cells_from_errors():
    summary = summarize_metrics([
        {"domain": "x", "gi": 0, "ineligible": "target_information_below_floor"},
        {"domain": "x", "gi": 1, "error": "corrupt tensor"},
    ])
    assert summary["n_evaluable"] == 0
    assert summary["n_ineligible"] == 1
    assert summary["n_errors"] == 1
    assert summary["ineligible_reasons"] == {"target_information_below_floor": 1}


def test_loader_preserves_form_orbits_and_alignment_is_conservative(tmp_path):
    meta = np.array([
        json.dumps({"gi": 0, "rung": "name", "form": "canonical"}),
        json.dumps({"gi": 0, "rung": "name", "form": "question"}),
        json.dumps({"gi": 0, "rung": "definition", "form": "canonical"}),
    ], dtype=object)
    scores = np.array([[0.1, 0.9, 0.2, 0.8],
                       [0.2, 0.8, 0.3, 0.7],
                       [0.1, 0.9, 0.1, 0.9]])
    paths = []
    for reader in ("small", "big"):
        path = tmp_path / f"grid_{reader}.npz"
        np.savez(path, scores=scores, meta=meta, reader=reader)
        paths.append(path)
    small, big = map(load_grid_orbits, paths)
    assert set(small["orbits"][0]["name"]) == {"canonical", "question"}
    report = alignment_report(small, big)
    assert report["shape_equal"] and report["row_metadata_equal"]
    assert not report["cryptographically_verified"]
    assert report["status"] == "row_metadata_and_shape_only"


def test_metric_adapter_selects_on_development_and_recovers_on_heldout():
    rng = np.random.default_rng(33)
    q = np.tile(np.linspace(0.03, 0.97, 100), 3)
    big_c = np.clip(q + rng.normal(0, 0.01, len(q)), 0.001, 0.999)
    big_q = np.clip(q + rng.normal(0, 0.01, len(q)), 0.001, 0.999)
    weak_c = np.clip(0.5 + rng.normal(0, 0.08, len(q)), 0.001, 0.999)
    weak_q = np.clip(0.5 + rng.normal(0, 0.08, len(q)), 0.001, 0.999)
    small = {
        "name": {"canonical": weak_c, "question": weak_q},
        "definition": {"canonical": big_c.copy(), "question": big_q.copy()},
        "explanation": {"canonical": weak_c.copy(), "question": weak_q.copy()},
    }
    big = {"name": {"canonical": big_c, "question": big_q}}
    message = {
        "name": "synthetic norm",
        "rungs": {"definition": "a synthetic definition", "explanation": "filler"},
        "word_len": {"definition": 8, "explanation": 20},
        "exemplar_idx": {"pos": [], "neg": []},
    }
    row = analyze_metric(
        domain="synthetic", gi=0, small_rungs=small, big_rungs=big, target_rungs=big,
        message=message, small_reader="small", big_reader="big", target_reader="big",
        rungs=["definition", "explanation"],
        control_rungs=[], sparse="name", divergence="tvd", min_target_information=0.01,
        train_frac=0.5, gap_delta=0.02, equivalence_delta=0.03,
        min_signature_rho=0.8, signature_equivalence_delta=0.05,
        n_boot=200, seed=5)
    assert row["development"]["selection"]["target_attained"]
    assert row["selected_rung"] == "definition"
    assert row["heldout"]["methodological_substitution"]
    assert row["heldout"]["equivalent_methodological_substitution"]
    assert row["heldout"]["articulation_specific_substitution"] is None
    assert row["claim_grade"] == "diagnostic_reanalysis_of_legacy_artifacts"
    summary = summarize_metrics([row])
    specificity = summary["articulation_specific_substitution_among_confirmed_gaps"]
    assert specificity["n_available"] == 0
    assert specificity["n_unavailable"] == 1
    assert summary["articulation_debt"]["finite"] == 1
    assert summary["posthoc_margin_sensitivity"]["status"] == "diagnostic_only_not_preregistered"
