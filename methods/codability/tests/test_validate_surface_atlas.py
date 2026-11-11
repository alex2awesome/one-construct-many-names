"""Tests for persisted surface-atlas integrity validation."""

import hashlib
import json

import numpy as np

from methods.codability.experiments.fixed_target_surface import save_surface
from methods.codability.experiments.validate_surface_atlas import validate_atlas


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_validate_minimal_surface_atlas(tmp_path):
    atlas = tmp_path / "atlas"
    surface_path = atlas / "surfaces" / "s.npz"
    report = {
        "schema": "fixed_target_reader_surface/v1", "n_arm_rows": 1,
        "n_metric_cells": 1, "n_ineligible_metrics": 0, "n_errors": 0,
        "config": {"n_boot": 2},
    }
    arrays = {
        "heldout_score": np.array([0.4]), "dev_score": np.array([0.3]),
        "heldout_rho": np.array([0.7]), "heldout_mae": np.array([0.2]),
        "score_draws": np.array([[0.3, 0.5]]),
        "rho_draws": np.array([[0.6, 0.8]]),
        "mae_draws": np.array([[0.1, 0.3]]),
    }
    save_surface({"report": report, "meta": [{"domain": "d", "gi": 0,
                                               "rung": "name"}],
                  "arrays": arrays}, surface_path)
    comparison = {
        "schema": "fixed_target_surface_comparison/v2", "validation": {"valid": True},
        "config": {"family_size": 1},
        "small_surface": {"path": str(surface_path), "sha256": _sha256(surface_path)},
        "big_surface": {"path": str(surface_path), "sha256": _sha256(surface_path)},
        "familywise_substitution_count": 1,
        "by_domain": {"d": {"per_metric": [{
            "heldout": {"valid": True, "gates": {"baseline_gap_confirmed": True},
                        "methodological_substitution": True,
                        "familywise": {"methodological_substitution": True}}
        }]}},
    }
    comparison_path = atlas / "comparisons" / "c.json"
    comparison_path.parent.mkdir(parents=True)
    comparison_path.write_text(json.dumps(comparison))
    summary = {
        "schema": "name_surface_atlas/v1", "manifest_sha256": "not-checked",
        "surfaces": [{"id": "s", "status": "complete", "sha256": _sha256(surface_path),
                      "n_metric_cells": 1, "n_ineligible_metrics": 0, "n_errors": 0}],
        "comparisons": [{"id": "c", "status": "complete",
                         "sha256": _sha256(comparison_path), "n_evaluable": 1,
                         "n_confirmed_gaps": 1, "n_substitutions": 1,
                         "n_familywise_substitutions": 1}],
        "coverage": {"surfaces_complete": 1, "surfaces_total": 1,
                     "comparisons_complete": 1, "comparisons_total": 1},
        "totals_over_comparisons_not_unique_metrics": {
            "evaluable": 1, "confirmed_gaps": 1, "cellwise_substitutions": 1,
            "familywise_substitutions": 1},
    }
    (atlas / "atlas_summary.json").write_text(json.dumps(summary))

    certificate = validate_atlas(atlas, check_manifest=False)

    assert certificate["valid"]
    assert certificate["counts"]["familywise_substitutions"] == 1
