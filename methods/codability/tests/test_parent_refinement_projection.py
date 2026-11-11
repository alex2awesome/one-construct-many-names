import json

import pytest

from methods.codability.lexicon.parent_refinement_projection import (
    project_parent_refinement,
)


def _write(path, payload):
    path.write_text(json.dumps(payload) + "\n")
    return path


def test_projects_llm_upper_labels_through_pure_refinement(tmp_path):
    old = _write(tmp_path / "old.json", {
        "a": "old_1", "b": "old_1", "c": "old_2", "d": "old_2",
    })
    new = _write(tmp_path / "new.json", {
        "a": "new_1a", "b": "new_1b", "c": "new_2", "d": "new_2",
    })
    upper = _write(tmp_path / "upper.json", {
        "old_1": "construct_x", "old_2": "construct_y",
    })
    output, manifest = tmp_path / "projected.json", tmp_path / "manifest.json"

    report = project_parent_refinement(
        "demo", "R1", old, new, upper, output, manifest,
    )

    assert json.loads(output.read_text()) == {
        "new_1a": "construct_x", "new_1b": "construct_x", "new_2": "construct_y",
    }
    assert report["n_leaves"] == 4
    assert report["n_new_parent_nodes"] == 3
    assert report["n_upper_groups"] == 2
    assert report["n_non_refinement_nodes"] == 0
    assert json.loads(manifest.read_text())["output_partition_sha256"] == report[
        "output_partition_sha256"
    ]


def test_rejects_new_node_that_crosses_old_parent_boundary(tmp_path):
    old = _write(tmp_path / "old.json", {"a": "old_1", "b": "old_2"})
    new = _write(tmp_path / "new.json", {"a": "new_mixed", "b": "new_mixed"})
    upper = _write(tmp_path / "upper.json", {
        "old_1": "construct_x", "old_2": "construct_y",
    })

    with pytest.raises(ValueError, match="not a pure refinement"):
        project_parent_refinement(
            "demo", "R1", old, new, upper,
            tmp_path / "projected.json", tmp_path / "manifest.json",
        )


def test_rejects_incomplete_semantic_source(tmp_path):
    old = _write(tmp_path / "old.json", {"a": "old_1", "b": "old_2"})
    new = _write(tmp_path / "new.json", {"a": "new_1", "b": "new_2"})
    upper = _write(tmp_path / "upper.json", {"old_1": "construct_x"})

    with pytest.raises(ValueError, match="does not exactly cover"):
        project_parent_refinement(
            "demo", "R1", old, new, upper,
            tmp_path / "projected.json", tmp_path / "manifest.json",
        )
