import json
from pathlib import Path

import pytest

from methods.codability.lexicon import upper_precision_audit as u


def test_wilson_bounds_and_empty():
    assert u._wilson(0, 0) is None
    lo, hi = u._wilson(80, 100)
    assert 0.70 < lo < 0.80 < hi < 0.90


def test_vote_loader_is_strict_and_complete(tmp_path: Path):
    path = tmp_path / "votes.jsonl"
    path.write_text(json.dumps({"pair_id": "a", "score": 2}) + "\n")
    assert u._load_votes(str(path), {"a"}) == {"a": 2}
    path.write_text(json.dumps({"pair_id": "a", "score": 2.0}) + "\n")
    with pytest.raises(ValueError):
        u._load_votes(str(path), {"a"})


@pytest.mark.parametrize("field", ["partition", "assignment"])
def test_candidate_loader_accepts_canonical_and_llm_shapes(tmp_path: Path, field: str):
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps({field: {"a": "g1", "b": 2}}))
    assert u._load_candidate_partition(path) == {"a": "g1", "b": "2"}


def test_prepare_can_freeze_a_versioned_protocol(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(u, "ROOT", tmp_path / "audit")
    monkeypatch.setattr(u, "OUT", str(tmp_path))
    monkeypatch.setattr(u, "_excluded_pairs", lambda task, level: set())
    monkeypatch.setattr(u, "nodes_from_level", lambda task, level: (
        [{"node_id": "a", "name": "A", "gloss": ""},
         {"node_id": "b", "name": "B", "gloss": ""}], {}))
    monkeypatch.setattr(u, "rep_text", lambda node: node["name"])
    partition = tmp_path / "partition.json"
    partition.write_text(json.dumps({"assignment": {"a": "g", "b": "g"}}))
    protocol = tmp_path / "R2_V2_1_PROTOCOL.md"
    protocol.write_text("focused R2 protocol")
    manifest = u.prepare("t", "R2", str(partition), protocol_path=str(protocol))
    assert manifest["protocol_path"] == str(protocol.resolve())
    assert manifest["protocol_sha256"] == u._file_sha256(str(protocol))


def test_prepare_accepts_and_freezes_custom_node_inventory(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(u, "ROOT", tmp_path / "audit")
    monkeypatch.setattr(u, "OUT", str(tmp_path))
    monkeypatch.setattr(u, "_excluded_pairs", lambda task, level: set())
    partition = tmp_path / "partition.json"
    partition.write_text(json.dumps({"assignment": {"x": "g", "y": "g"}}))
    protocol = tmp_path / "R3_PROTOCOL.md"
    protocol.write_text("top-level category")
    nodes = tmp_path / "variant_nodes.jsonl"
    nodes.write_text("".join(json.dumps(row) + "\n" for row in (
        {"node_id": "x", "name": "X", "gloss": "one"},
        {"node_id": "y", "name": "Y", "gloss": "two"})))
    manifest = u.prepare("t", "R3", str(partition), protocol_path=str(protocol),
                         nodes_path=str(nodes), exclude_prior=False)
    assert manifest["nodes_path"] == str(nodes.resolve())
    assert manifest["nodes_sha256"] == u._file_sha256(str(nodes))
    assert manifest["n_nodes"] == 2


def test_excluded_pairs_are_canonical(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(u, "OUT", str(tmp_path))
    (tmp_path / "level_arbiter").mkdir()
    (tmp_path / "level_eval_t_R1.jsonl").write_text(
        json.dumps({"node_a": "b", "node_b": "a"}) + "\n")
    (tmp_path / "level_arbiter" / "t_R1_verify_000.jsonl").write_text(
        json.dumps({"node_a": "c", "node_b": "a"}) + "\n")
    assert u._excluded_pairs("t", "R1") == {("a", "b"), ("a", "c")}
