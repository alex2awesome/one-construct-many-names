"""Closure checks for the reciprocal cross-family atlas manifest."""

from methods.codability.experiments.cross_family_surface_atlas import (
    _map,
    load_manifest,
    manifest_sha256,
)


def test_cross_family_manifest_is_closed_and_reciprocal():
    manifest = load_manifest()
    existing = _map(manifest["existing_surfaces"])
    surfaces = _map(manifest["surfaces"])
    comparisons = _map(manifest["comparisons"])
    all_ids = set(existing) | set(surfaces)
    assert len(manifest_sha256()) == 64
    assert len(surfaces) == 18
    assert len(comparisons) == 24
    assert all(row["small"] in all_ids and row["big"] in all_ids
               for row in comparisons.values())
    directions = {row["direction"] for row in comparisons.values()}
    assert "qwen_to_llama8_self" in directions and "llama_to_qwen7_self" in directions
    assert "gemma_to_llama70_self" in directions and "llama_to_gemma31_self" in directions

