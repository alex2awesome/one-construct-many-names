"""Convex multi-articulation portfolio tests."""

import numpy as np
import pytest

from methods.codability.experiments.prompt_portfolio_crossfit import (
    fit_simplex,
    normalize_arm_id,
    portfolio_loss,
    run,
)


def test_normalized_counterpart_recipe_removes_only_public_fold_suffix():
    assert normalize_arm_id("rule_v1_from_prompt_selection") == "rule_v1"
    assert normalize_arm_id("rule_v1_from_unit_certification") == "rule_v1"
    assert normalize_arm_id("rule_v1_from_self") == "rule_v1_from_self"


def test_simplex_recovers_convex_target_and_sparse_refit():
    features = np.asarray([
        [0.0, 1.0, 0.2],
        [1.0, 0.0, 0.8],
        [0.2, 0.8, 0.3],
        [0.7, 0.3, 0.6],
    ])
    target = 0.7 * features[:, 0] + 0.3 * features[:, 1]
    weights, report = fit_simplex(features, target, loss="mse")
    assert np.allclose(features @ weights, target, atol=1e-5)
    assert np.isclose(weights.sum(), 1.0)
    assert np.all(weights >= 0)
    sparse, sparse_report = fit_simplex(features, target, loss="mse", top_k=2)
    assert np.allclose(features @ sparse, target, atol=1e-5)
    assert report["success"] and sparse_report["success"]


def test_portfolio_rejects_unknown_feature_scope_before_loading_data():
    with pytest.raises(ValueError, match="unknown portfolio feature scope"):
        run(data_root="missing", feature_scope="all_the_things")


def test_rank_loss_rewards_correct_order_despite_level_shift():
    target = np.array([0.1, 0.2, 0.8, 0.9])
    ordered = np.array([0.5, 0.6, 0.7, 0.8])
    reversed_values = ordered[::-1]
    assert portfolio_loss(ordered, target, loss="rank") < 1e-10
    assert portfolio_loss(reversed_values, target, loss="rank") > 1.9
