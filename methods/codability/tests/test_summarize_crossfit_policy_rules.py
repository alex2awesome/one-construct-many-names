from methods.codability.experiments.summarize_crossfit_policy_rules import (
    _margin_frontier,
    identity_excess,
    recipe_id,
)


def test_recipe_id_joins_opposite_fold_realizations_only_at_suffix():
    assert recipe_id("source_rule_gestalt_from_prompt_selection") == (
        "source_rule_gestalt_from_crossfit_source")
    assert recipe_id("rule_gestalt_v0_from_self") == "rule_gestalt_v0_from_self"


def test_identity_excess_puts_all_axes_on_lower_is_better_scale():
    target = {"mae_tvd": 0.1, "spearman": 0.9,
              "binary_flip_rate": 0.1, "absolute_bias": 0.05}
    candidate = {"mae_tvd": 0.14, "spearman": 0.8,
                 "binary_flip_rate": 0.08, "absolute_bias": 0.07}
    assert identity_excess(candidate, target) == {
        "mae_tvd": 0.04000000000000001,
        "spearman": 0.09999999999999998,
        "binary_flip_rate": 0.0,
        "absolute_bias": 0.020000000000000004,
    }


def test_margin_frontier_uses_frozen_secondary_equivalence_but_requires_mae_gain():
    incumbent = {"mae_tvd": 0.05, "spearman": 0.30,
                 "binary_flip_rate": 0.07, "absolute_bias": 0.02}
    challenger = {"mae_tvd": 0.04, "spearman": 0.304,
                  "binary_flip_rate": 0.05, "absolute_bias": 0.0}
    assert _margin_frontier(challenger, incumbent)
    assert not _margin_frontier({**challenger, "mae_tvd": 0.051}, incumbent)
    zero_excess = {key: 0.0 for key in incumbent}
    assert _margin_frontier(challenger, incumbent, challenger_excess=zero_excess,
                            incumbent_excess=zero_excess)
