import json
from pathlib import Path

import numpy as np
import pytest

from methods.codability.lexicon import semantic_group_merge as sgm
from methods.codability.lexicon import semantic_reconsider as sr


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _toy(tmp_path, monkeypatch):
    monkeypatch.setattr(sgm, "OUT", str(tmp_path))
    monkeypatch.setattr(sr, "OUT", str(tmp_path))
    monkeypatch.setattr(sr, "ROOT", tmp_path / "semantic_group_merge")
    task, level = "toy", "R1"
    nodes = [{"node_id": x, "name": f"concept {x}", "gloss": f"quality {x}"}
             for x in ("a", "b", "c", "d")]
    partition = {x: f"g{x}" for x in ("a", "b", "c", "d")}
    partition_path = tmp_path / "partition_toy_R1.json"
    partition_path.write_text(json.dumps(partition) + "\n")
    (tmp_path / "ARBITER_PROTOCOL_R1.txt").write_text("frozen protocol\n")
    _write_jsonl(tmp_path / "level_eval_toy_R1.jsonl",
                 [{"pair_id": "eval", "node_a": "c", "node_b": "d"}])
    monkeypatch.setattr(sgm, "nodes_from_level", lambda *_: (nodes, {}))
    vectors = np.array([[1, 0, 0], [.99, .1, 0], [.98, .2, 0], [.7, .7, .1]],
                       dtype=np.float32)
    model = {"model_id": sgm.MODEL_ID, "model_revision": "test",
             "model_sha256": "a" * 64, "pooling": "cls", "normalized": True}
    monkeypatch.setattr(sgm, "_embed_bge", lambda texts: (vectors[:len(texts)], model))
    manifest = sgm.prepare(task, level, str(partition_path), k=3, cap=20, tag="x")
    rows = [row for path in manifest["payload_paths"] for row in sgm._jsonl_rows(Path(path))]
    return manifest, rows, partition


def test_reconsider_adds_only_two_fresh_score2_votes(tmp_path, monkeypatch):
    manifest, rows, source = _toy(tmp_path, monkeypatch)
    target = rows[0]["pair_id"]
    screen_dir = tmp_path / "screen"
    _write_jsonl(screen_dir / "votes.jsonl", [
        {"pair_id": row["pair_id"], "score": 1 if row["pair_id"] == target else 0}
        for row in rows
    ])
    staged = sr.stage("toy", "R1", "x", str(screen_dir),
                      source_manifest_path=str(sgm._manifest_path("toy", "R1", "x")))
    assert staged["selected_pair_ids"] == [target]
    base = tmp_path / "base.json"
    base.write_text(json.dumps(source) + "\n")
    b_dir, c_dir = tmp_path / "b", tmp_path / "c"
    _write_jsonl(b_dir / "votes.jsonl", [{"pair_id": target, "score": 2}])
    _write_jsonl(c_dir / "votes.jsonl", [{"pair_id": target, "score": 2}])
    output = tmp_path / "out.json"
    report = sr.apply("toy", "R1", "x", "score1", str(base),
                      str(b_dir), str(c_dir), str(output))
    result = json.loads(output.read_text())
    ga, gb = rows[0]["group_a"]["group_id"], rows[0]["group_b"]["group_id"]
    na = next(node for node, group in source.items() if group == ga)
    nb = next(node for node, group in source.items() if group == gb)
    assert result[na] == result[nb]
    assert report["n_double_score2_added"] == 1


def test_reconsider_requires_exact_fresh_vote_coverage(tmp_path, monkeypatch):
    manifest, rows, source = _toy(tmp_path, monkeypatch)
    selected = rows[:2]
    screen_dir = tmp_path / "screen"
    _write_jsonl(screen_dir / "votes.jsonl", [
        {"pair_id": row["pair_id"], "score": 1 if row in selected else 0} for row in rows
    ])
    sr.stage("toy", "R1", "x", str(screen_dir),
             source_manifest_path=str(sgm._manifest_path("toy", "R1", "x")))
    base = tmp_path / "base.json"; base.write_text(json.dumps(source) + "\n")
    b_dir, c_dir = tmp_path / "b", tmp_path / "c"
    _write_jsonl(b_dir / "votes.jsonl", [{"pair_id": selected[0]["pair_id"], "score": 2}])
    _write_jsonl(c_dir / "votes.jsonl", [
        {"pair_id": row["pair_id"], "score": 2} for row in selected])
    with pytest.raises(sgm.VoteCoverageError, match="coverage is not exact"):
        sr.apply("toy", "R1", "x", "score1", str(base), str(b_dir), str(c_dir),
                 str(tmp_path / "out.json"))
