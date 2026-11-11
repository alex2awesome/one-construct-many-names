"""Adaptive-ostension analysis recipe matching tests."""

from methods.codability.experiments.analyze_adaptive_ostensive_isomorphism import (
    recipe_family,
)


def test_crossfit_arm_suffixes_map_to_one_stable_recipe():
    prompt = "self_contrastive_plus_hybrid_residual_k2_from_prompt_selection"
    unit = "self_contrastive_plus_hybrid_residual_k2_from_unit_certification"
    expected = "self_contrastive_plus_hybrid_residual_k2"
    assert recipe_family(prompt) == expected
    assert recipe_family(unit) == expected
