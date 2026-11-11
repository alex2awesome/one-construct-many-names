"""Schema and sealed-item loader tests for fresh target-view execution."""

import json

from methods.codability.experiments.score_fresh_target_views import (
    _by_id,
    load_domain_items,
    load_manifest,
)


def test_target_view_manifest_is_closed_and_non_name_gestalt_is_distinct():
    manifest = load_manifest()
    cells = _by_id(manifest["cells"])
    jobs = _by_id(manifest["model_jobs"])
    assert len(cells) == 8 and len(jobs) == 3
    for job in jobs.values():
        for domain in job["domains"]:
            assert all(cell_id in cells for cell_id in domain["cells"])
    for cell in cells.values():
        assert len(cell["forms"]) == 3
        if cell["view"] == "G":
            assert cell["construct"] is None
            assert all("checklist" not in form["prompt"].lower()
                       or "without reducing" in form["prompt"].lower()
                       for form in cell["forms"])


def test_load_domain_items_preserves_partition_and_hashes(tmp_path):
    root = tmp_path / "pack" / "humor" / "items"
    root.mkdir(parents=True)
    from methods.codability.experiments.build_fresh_item_partitions import text_sha256
    for partition, text in [("a", "first"), ("b", "second")]:
        row = {"item_id": partition, "text": text, "text_sha256": text_sha256(text),
               "source_group": partition, "source_split": None}
        (root / f"{partition}.jsonl").write_text(json.dumps(row) + "\n")

    result = load_domain_items(tmp_path / "pack", "humor")

    assert result["texts"] == ["first", "second"]
    assert result["partitions"] == ["a", "b"]

    selected = load_domain_items(tmp_path / "pack", "humor", partitions=["a"])
    assert selected["texts"] == ["first"]
    assert selected["partitions"] == ["a"]
