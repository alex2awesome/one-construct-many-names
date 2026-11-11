"""Within-Llama adjacent-scale bank tests."""

import json

import pytest

from methods.codability.experiments.compile_adjacent_scale_isomorphism_bank import compile_bank
from methods.codability.experiments.score_adjacent_scale_isomorphism import run


def test_adjacent_bank_selects_only_declared_humor_cell_and_preserves_arms(tmp_path):
    arms = [{"id": "name"}, {"id": "iso_explanation"}]
    source = {"cells": [
        {"id": "N_humor_49", "domain": "humor", "arms": arms},
        {"id": "N_cw_27", "domain": "cw", "arms": [{"id": "name"}]},
    ]}
    path = tmp_path / "source.json"
    path.write_text(json.dumps(source))
    observed = compile_bank(str(path))
    assert observed["status"] == "frozen-before-1b-executor-public-scoring"
    assert observed["model_family"] == "Llama-3.2 only"
    assert observed["cells"] == [source["cells"][0]]
    assert observed["cells"][0]["arms"] == arms
    assert len(observed["bank_content_sha256"]) == 64


def test_scale_pair_scorer_rejects_nonpublic_partition_before_backend_start(tmp_path):
    bank = tmp_path / "bank.json"
    bank.write_text(json.dumps({
        "status": "frozen-before-1b-executor-public-scoring",
        "partitions": ["residual_lockbox"],
    }))
    with pytest.raises(ValueError, match="does not authorize partition"):
        run(
            bank_path=str(bank),
            packet_root="missing",
            target_manifest_path="missing",
            out_root=str(tmp_path / "out"),
            fake=True,
        )
