"""Rank-transplant invariants for comparative policy reconstruction."""

import json

import numpy as np
import pytest

from methods.codability.experiments.analyze_pairwise_policy_isomorphism import (
    _load_pairwise,
    percentile_rank,
    quantile_reorder,
    recipe_family,
    transplant_orbit,
)


def test_quantile_reorder_preserves_probability_multiset_and_follows_signal():
    base = np.asarray([0.8, 0.1, 0.4, 0.6])
    signal = np.asarray([0.0, 3.0, 1.0, 2.0])
    observed = quantile_reorder(base, signal)
    assert np.array_equal(np.sort(observed), np.sort(base))
    assert np.array_equal(np.argsort(observed), np.argsort(signal))


def test_zero_weight_replays_base_and_full_weight_follows_comparisons():
    base = {form: np.asarray([0.7, 0.1, 0.9, 0.3]) for form in (
        "canonical", "question", "boilerplate")}
    pair = {form: np.asarray([0.0, 0.9, 0.2, 0.7]) for form in base}
    replay = transplant_orbit(base, pair, aggregation="matching_form", alpha=0.0)
    full = transplant_orbit(base, pair, aggregation="mean_rank", alpha=1.0)
    assert all(np.array_equal(replay[form], base[form]) for form in base)
    assert all(np.array_equal(np.argsort(full[form]), np.argsort(pair[form]))
               for form in base)
    assert np.allclose(percentile_rank(base["canonical"]), [2 / 3, 0, 1, 1 / 3])


def test_fold_specific_arm_ids_share_one_recipe_family():
    assert recipe_family("behavior_contrastive_from_prompt_selection") == (
        "behavior_contrastive")
    assert recipe_family("behavior_contrastive_from_unit_certification") == (
        "behavior_contrastive")
    assert recipe_family("self_contrastive") == "self_contrastive"


def test_pairwise_loader_rejects_duplicate_probe_hashes(tmp_path):
    path = tmp_path / "pairwise.npz"
    meta = [{"arm_id": "name", "form": "canonical"}]
    np.savez_compressed(
        path,
        probe_sha256=np.array(["same", "same"]),
        meta=np.array([json.dumps(row) for row in meta], dtype=object),
        borda_scores=np.array([[0.1, 0.2]]),
        bradley_terry_scores=np.array([[0.1, 0.2]]),
    )
    with pytest.raises(ValueError, match="duplicate probe hashes"):
        _load_pairwise(str(path), ["same"])


def test_pairwise_loader_rejects_duplicate_target_hashes_before_alignment(tmp_path):
    path = tmp_path / "pairwise.npz"
    meta = [{"arm_id": "name", "form": "canonical"}]
    np.savez_compressed(
        path,
        probe_sha256=np.array(["a", "b"]),
        meta=np.array([json.dumps(row) for row in meta], dtype=object),
        borda_scores=np.array([[0.1, 0.2]]),
        bradley_terry_scores=np.array([[0.1, 0.2]]),
    )
    with pytest.raises(ValueError, match="target shard contains duplicate probe hashes"):
        _load_pairwise(str(path), ["a", "a"])
