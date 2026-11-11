import json
from pathlib import Path

import pytest

from methods.codability.lexicon import r1_truth_reaudit as r


def test_vote_loader_strict_and_complete(tmp_path: Path):
    path = tmp_path / "votes.jsonl"
    path.write_text(json.dumps({"pair_id": "x", "score": 2}) + "\n")
    assert r._load(path, {"x"}) == {"x": 2}
    path.write_text(json.dumps({"pair_id": "x", "score": True}) + "\n")
    with pytest.raises(ValueError):
        r._load(path, {"x"})


def test_vote_loader_rejects_missing(tmp_path: Path):
    path = tmp_path / "votes.jsonl"
    path.write_text("")
    with pytest.raises(ValueError):
        r._load(path, {"x"})
