"""Manifest and aggregation tests for the frozen within-family surface atlas."""

from methods.codability.experiments.name_surface_atlas import (
    _by_id,
    load_atlas_manifest,
    manifest_sha256,
)


def test_atlas_manifest_ids_and_references_are_closed():
    manifest = load_atlas_manifest()
    surfaces = _by_id(manifest["surfaces"])
    comparisons = _by_id(manifest["comparisons"])
    assert len(manifest_sha256()) == 64
    assert len(surfaces) == 13
    assert len(comparisons) == 14
    for comparison in comparisons.values():
        assert comparison["small"] in surfaces
        assert comparison["big"] in surfaces
    qwen = [row for row in surfaces.values() if row["family"] == "qwen2.5"]
    assert all(row.get("executor_grid_template") for row in qwen)
    assert manifest["ladder_caveats"]["llama_target70"]

