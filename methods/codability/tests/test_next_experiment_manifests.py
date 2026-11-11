"""Structural closure tests for the fresh residual-teaching and gestalt protocols."""

import json
from pathlib import Path


ROOT = Path(__file__).parents[1] / "experiments"


def test_residual_manifest_separates_provenance_and_requires_controls():
    data = json.loads((ROOT / "residual_teaching_manifest_v1.json").read_text())
    assert len(data["priority_cells"]) == 5
    assert {"source_telling", "residual_teaching", "fitted_optimizer"} <= \
           set(data["provenance_arms"])
    assert any("wrong-construct" in control for control in data["controls"])
    assert data["inference"]["bootstrap_draws_minimum"] >= 5000
    assert len(data["unit_requirements"]) == 5


def test_gestalt_manifest_has_model_and_practice_targets_and_measured_locations():
    data = json.loads((ROOT / "gestalt_execution_manifest_v1.json").read_text())
    assert set(data["target_views"]) == {"G", "P"}
    assert len(data["pilot_domains"]) == 4
    assert set(data["gestalt_location"]) >= {"within_unit_span", "composition", "outside_span"}
    assert data["inference"]["paired_bootstrap_draws_minimum"] >= 5000

