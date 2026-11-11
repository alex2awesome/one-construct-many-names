"""Regression tests for the strengthened scale--articulation estimand and its controls."""
from types import SimpleNamespace

import numpy as np
import pytest

from methods.codability import unit_count_grid as ucg
from methods.codability.eval_prereg_70b import verify_frozen_hash
from methods.codability.run_expansion_chain import triangle_slack
from methods.codability.scale_articulation_substitution import (
    analyze_raw_grid,
    classify_point,
    crossfit_metric,
    stratified_split,
    summarize_point_records,
)
from methods.codability.stipulation_probe import NONCE, nonce_labeled_definition
from methods.codability.unit_deficit_report import direct_rescue, summarize_direct_rescue


def _rungs(name, rich):
    return {"name": {"auc": name}, "definition": {"auc": rich}}


def test_point_rescue_is_conditioned_on_a_real_sparse_gap():
    no_gap = classify_point(_rungs(0.70, 0.82), _rungs(0.71, 0.80), delta=0.02)
    assert no_gap["status"] == "no_baseline_gap"
    assert no_gap["small_best_noninferior_big_sparse"]
    assert not no_gap["rescue_big_sparse"]

    rescued = classify_point(_rungs(0.58, 0.79), _rungs(0.78, 0.80), delta=0.02)
    assert rescued["baseline_gap_present"]
    assert rescued["rescue_big_sparse"]
    summary = summarize_point_records([no_gap, rescued], n_boot=0)
    assert summary["rescue_big_sparse_among_gaps"] == {
        "success": 1, "n": 1, "rate": 1.0, "CI95_metric_bootstrap": None
    }


def test_crossfit_can_certify_heldout_form_matched_segment_substitution():
    rng = np.random.default_rng(12)
    labels = np.repeat([False, True], 120)
    small_sparse = labels.astype(float) * 0.15 + rng.normal(0, 0.7, len(labels))
    big_sparse = labels.astype(float) + rng.normal(0, 0.22, len(labels))
    # Exact equality is intentional: it makes the synthetic isomorphism ground truth unambiguous.
    small = {"u0": small_sparse, "u1": big_sparse.copy(),
             "u2": big_sparse.copy(), "f1": small_sparse.copy(), "f2": small_sparse.copy()}
    big = {"u0": big_sparse, "u1": big_sparse.copy(), "u2": big_sparse.copy()}
    result = crossfit_metric(
        small, big, labels, rungs=["u0", "u1", "u2"], sparse="u0",
        placebo_rungs=["f1", "f2"], delta=0.03, floor=0.55, seed=7, n_boot=300,
        min_signature_rho=0.8, selection_policy="minimal_cost_noninferior")

    assert result["selected_small_arm"] == "u1"
    assert result["selected_articulation_segments"] == 1
    assert result["placebo"]["selected_arm"] == "f1"
    assert result["placebo"]["selection_rule"] == "matched_to_selected_content_k"
    assert result["baseline_gap_confirmed"]
    assert result["equivalent_big_sparse"]
    assert result["signature_gate"]
    assert result["articulation_specific"]
    assert result["articulation_control"]["selected_arm"] == "f1"
    assert result["content_specific"]
    assert result["certified_substitution"]


def test_crossfit_split_refuses_unusable_or_degenerate_splits():
    with pytest.raises(ValueError, match="strictly between"):
        stratified_split(np.array([0, 0, 1, 1]), train_frac=1)
    with pytest.raises(ValueError, match="at least two"):
        stratified_split(np.array([0, 0, 1]))


def test_triangle_tolerance_accumulates_across_two_hops():
    strong = [0.80, 0.80, 0.80]
    mid = [0.70, 0.751, 0.80]
    weak = [0.60, 0.702, 0.752]
    result = triangle_slack(weak, mid, strong, delta=0.05)
    assert result["h_mid"] == 1
    assert result["composed"] == 1
    assert result["direct"] == 1
    assert result["slack"] == 0
    assert result["direct_delta"] == pytest.approx(0.10)


def test_nonce_definition_removes_all_literal_real_name_occurrences():
    text = "Comic timing depends on pauses; COMIC TIMING also depends on release."
    result = nonce_labeled_definition("Comic Timing", text)
    assert "comic timing" not in result.lower()
    assert result.lower().count(NONCE) == 3  # label plus two body occurrences


def _curve(raw):
    best, env = None, []
    for value in raw:
        best = value if best is None else max(best, value)
        env.append(best)
    return list(range(len(raw))), env, raw, {}


def test_segment_rescue_report_does_not_count_trivial_parity():
    no_gap = direct_rescue(_curve([0.70, 0.80]), _curve([0.71, 0.79]), delta=0.02)
    rescued = direct_rescue(_curve([0.55, 0.78]), _curve([0.76, 0.79]), delta=0.02)
    assert not no_gap["rescue_big_sparse"]
    assert rescued["rescue_big_sparse"]
    summary = summarize_direct_rescue([no_gap, rescued])
    assert summary["rescue_big_sparse_among_gaps"]["n"] == 1
    assert summary["rescue_big_sparse_among_gaps"]["success"] == 1


def test_unit_score_applies_same_form_orbit_to_content_and_filler(tmp_path, monkeypatch):
    monkeypatch.setattr(ucg, "make_judge_backend", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        "methods.metric_implementer.experiments.alpha_probe.signature",
        lambda *_args, **_kwargs: np.array([0.1, 0.9]),
    )
    monkeypatch.setattr(
        "methods.metric_implementer.experiments.alpha_probe._reformulations",
        lambda prompt: [("question", prompt + " q"),
                        ("boilerplate", "expert " + prompt)],
    )
    args = SimpleNamespace(task="humor", fake=False, forms=3, out_dir=str(tmp_path),
                           ref_dir="refs")
    cfg = SimpleNamespace(max_text_chars=1000)
    msgs = {"0": {"name": "timing", "rungs": {"u0": "timing", "u1": "timing. pause",
                                                    "f1": "timing. indeed"}}}
    path = ucg.score_reader(args, cfg, "org/reader", msgs, ["a", "b"])
    z = np.load(path, allow_pickle=True)
    meta = [__import__("json").loads(x) for x in z["meta"]]
    counts = {r: sum(m["rung"] == r for m in meta) for r in ("u0", "u1", "f1")}
    assert counts == {"u0": 3, "u1": 3, "f1": 3}
    assert str(z["protocol_schema"]) == "address_segment_grid/v2_form_matched"
    assert len(z["probe_sha256"]) == 2
    assert len(str(z["probe_set_sha256"])) == 64


def test_raw_placebo_certificate_rejects_legacy_form_mismatched_grids(tmp_path):
    import json

    meta = np.array([json.dumps({"gi": 0, "rung": rung, "form": "canonical"})
                     for rung in ("u0", "u1", "f1")], dtype=object)
    for tag in ("small", "big"):
        np.savez(tmp_path / f"grid_{tag}.npz", scores=np.ones((3, 8)), meta=meta,
                 reader=tag)
    with pytest.raises(ValueError, match="v2 form-matched"):
        analyze_raw_grid(
            grid_dir=str(tmp_path), ref_dir=str(tmp_path / "refs"), small_tag="small",
            big_tag="big", rungs=["u0", "u1"], sparse="u0", placebo_rungs=["f1"],
            delta=0.02, floor=0.55, train_frac=0.5, seed=0, n_boot=10,
            min_signature_rho=0.5)


def test_frozen_hash_verification_detects_mutation():
    import hashlib
    import json

    payload = {"frozen": "before data", "prediction": [1, 2]}
    payload = {"sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
               **payload}
    assert verify_frozen_hash(payload, "test")["verified"]
    payload["prediction"].append(3)
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_frozen_hash(payload, "test")
