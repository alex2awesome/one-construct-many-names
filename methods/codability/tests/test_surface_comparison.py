"""Tests for pairwise substitution derived from aligned fixed-target surfaces."""

import json

import numpy as np
import pytest

from methods.codability.experiments.fixed_target_surface import (
    build_fixed_target_surface,
    save_surface,
)
from methods.codability.experiments.surface_comparison import compare_surfaces


def _write_grid(path, reader, name, definition):
    rows, meta = [], []
    for rung, values in (("name", name), ("definition", definition)):
        for form in ("canonical", "question", "boilerplate"):
            rows.append(values)
            meta.append(json.dumps({"gi": 0, "rung": rung, "form": form}))
    np.savez(path, scores=np.stack(rows), meta=np.asarray(meta, dtype=object), reader=reader)


def _surfaces(tmp_path):
    grid_dir = tmp_path / "r3_humor/grid_humor_v1"
    grid_dir.mkdir(parents=True)
    rng = np.random.default_rng(14)
    q = np.tile(np.linspace(0.03, 0.97, 80), 3)
    weak = np.clip(0.5 + rng.normal(0, 0.08, len(q)), 0.001, 0.999)
    _write_grid(grid_dir / "grid_big.npz", "big", q, q)
    _write_grid(grid_dir / "grid_small.npz", "small", weak, q)
    (grid_dir / "messages.json").write_text(json.dumps({
        "0": {"name": "synthetic norm",
              "rungs": {"name": "synthetic norm", "definition": "explicit rule"},
              "word_len": {"name": 2, "definition": 8},
              "exemplar_idx": {"pos": [], "neg": []}}}))
    kwargs = dict(data_dir=str(tmp_path), domains=["humor"], target_tag="big",
                  rungs=["name", "definition"], n_boot=150, seed=7)
    small = build_fixed_target_surface(executor_tag="small", **kwargs)
    big = build_fixed_target_surface(executor_tag="big", **kwargs)
    sp, _ = save_surface(small, tmp_path / "small.npz")
    bp, _ = save_surface(big, tmp_path / "big.npz")
    from methods.codability.experiments.fixed_target_surface import load_surface
    return load_surface(sp), load_surface(bp)


def test_surface_comparison_recovers_planted_substitution(tmp_path):
    small, big = _surfaces(tmp_path)
    result = compare_surfaces(small, big, equivalence_delta=0.03,
                              min_signature_rho=0.8)
    assert result["validation"]["valid"]
    assert result["pooled"]["baseline_gap_confirmed"] == 1
    assert result["pooled"]["methodological_substitution_among_confirmed_gaps"]["success"] == 1
    row = result["by_domain"]["humor"]["per_metric"][0]
    assert row["selected_rung"] == "definition"
    assert row["heldout"]["equivalent_methodological_substitution"]
    assert row["heldout"]["familywise"]["methodological_substitution"]
    assert result["familywise_substitution_count"] == 1


def test_surface_comparison_rejects_target_mismatch(tmp_path):
    small, big = _surfaces(tmp_path)
    big["report"]["config"]["target_tag"] = "other"
    with pytest.raises(ValueError, match="not comparable"):
        compare_surfaces(small, big)
