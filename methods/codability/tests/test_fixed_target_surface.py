"""Tests for reusable fixed-target reader surfaces."""

import json

import numpy as np

from methods.codability.experiments.fixed_target_surface import (
    build_fixed_target_surface,
    load_surface,
    save_surface,
)


def _write_grid(path, reader, name, definition):
    forms = ["canonical", "question", "boilerplate"]
    rows, meta = [], []
    for rung, values in (("name", name), ("definition", definition)):
        for form in forms:
            rows.append(values)
            meta.append(json.dumps({"gi": 0, "rung": rung, "form": form}))
    np.savez(path, scores=np.stack(rows), meta=np.asarray(meta, dtype=object), reader=reader)


def test_surface_persists_all_arms_and_raw_paired_draws(tmp_path):
    grid_dir = tmp_path / "r3_humor/grid_humor_v1"
    grid_dir.mkdir(parents=True)
    rng = np.random.default_rng(3)
    q = np.tile(np.linspace(0.03, 0.97, 40), 3)
    weak = np.clip(0.5 + rng.normal(0, 0.08, len(q)), 0.001, 0.999)
    _write_grid(grid_dir / "grid_big.npz", "big", q, q)
    _write_grid(grid_dir / "grid_small.npz", "small", weak, q)
    messages = {"0": {"name": "synthetic norm",
                       "rungs": {"name": "synthetic norm", "definition": "explicit rule"},
                       "word_len": {"name": 2, "definition": 8},
                       "exemplar_idx": {"pos": [], "neg": []}}}
    (grid_dir / "messages.json").write_text(json.dumps(messages))
    bundle = build_fixed_target_surface(
        data_dir=str(tmp_path), domains=["humor"], executor_tag="small", target_tag="big",
        rungs=["name", "definition"], n_boot=20, seed=7)
    assert bundle["report"]["n_metric_cells"] == 1
    assert bundle["report"]["n_arm_rows"] == 2
    assert bundle["arrays"]["score_draws"].shape == (2, 20)
    rows = {row["rung"]: i for i, row in enumerate(bundle["meta"])}
    assert bundle["arrays"]["heldout_score"][rows["definition"]] > \
           bundle["arrays"]["heldout_score"][rows["name"]]
    path, report_path = save_surface(bundle, tmp_path / "surface.npz")
    loaded = load_surface(path)
    assert report_path.exists()
    assert loaded["report"]["n_arm_rows"] == 2
    assert np.array_equal(loaded["arrays"]["score_draws"],
                          bundle["arrays"]["score_draws"], equal_nan=True)

