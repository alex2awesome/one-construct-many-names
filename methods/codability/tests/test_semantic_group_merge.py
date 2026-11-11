import json
from pathlib import Path

import numpy as np
import pytest

from methods.codability.lexicon import semantic_group_merge as sgm


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _prepare_toy(tmp_path, monkeypatch, *, eval_pair=("a", "b"), lexical_pairs=()):
    monkeypatch.setattr(sgm, "OUT", str(tmp_path))
    task, level = "toy", "R1"
    nodes = [
        {"node_id": "a", "name": "alpha construct", "gloss": "first quality"},
        {"node_id": "b", "name": "beta construct", "gloss": "second quality"},
        {"node_id": "c", "name": "gamma construct", "gloss": "third quality"},
        {"node_id": "d", "name": "delta construct", "gloss": "fourth quality"},
    ]
    partition = {"a": "ga", "b": "gb", "c": "gc", "d": "gd"}
    partition_path = tmp_path / "partition_toy_R1.json"
    partition_path.write_text(json.dumps(partition) + "\n")
    (tmp_path / "ARBITER_PROTOCOL_R1.txt").write_text("frozen narrow-construct protocol\n")
    _write_jsonl(tmp_path / "level_eval_toy_R1.jsonl", [
        {"pair_id": "eval", "node_a": eval_pair[0], "node_b": eval_pair[1]},
    ])
    if lexical_pairs:
        _write_jsonl(tmp_path / "level_arbiter" / "toy_R1_verify_000.jsonl", [
            {"pair_id": f"lex-{i}", "node_a": a, "node_b": b}
            for i, (a, b) in enumerate(lexical_pairs)
        ])
    monkeypatch.setattr(sgm, "nodes_from_level", lambda _task, _level: (nodes, {}))
    # All vectors are normalized by prepare.  Every node can retrieve every other node with k=3.
    vectors = np.array([
        [1.0, 0.0, 0.0],
        [0.99, 0.10, 0.0],
        [0.98, 0.20, 0.0],
        [0.70, 0.70, 0.1],
    ], dtype=np.float32)
    model = {
        "model_id": sgm.MODEL_ID,
        "model_revision": "test-revision",
        "model_sha256": "a" * 64,
        "pooling": "cls",
        "normalized": True,
    }
    monkeypatch.setattr(sgm, "_embed_bge", lambda texts: (vectors[:len(texts)], model))
    manifest = sgm.prepare(task, level, str(partition_path), k=3, cap=20)
    payload_rows = [
        row
        for path in manifest["payload_paths"]
        for row in sgm._jsonl_rows(Path(path))
    ]
    return manifest, payload_rows, partition


def _write_votes(directory: Path, rows):
    _write_jsonl(directory / "votes.jsonl", rows)


def test_prepare_excludes_eval_and_dispatched_lexical_pairs(tmp_path, monkeypatch):
    manifest, payload_rows, _ = _prepare_toy(
        tmp_path, monkeypatch, eval_pair=("a", "b"), lexical_pairs=[("a", "c")])

    evidence_pairs = {
        tuple(sorted((evidence["node_a"], evidence["node_b"])))
        for row in payload_rows
        for evidence in row["evidence"]
    }
    assert ("a", "b") not in evidence_pairs
    assert ("a", "c") not in evidence_pairs
    assert manifest["n_eval_pairs_excluded"] == 1
    assert manifest["n_lexical_verify_pairs_excluded"] == 1
    assert manifest["model"]["model_sha256"] == "a" * 64
    assert all(row["group_a"]["all_members"] and row["group_b"]["all_members"]
               for row in payload_rows)
    assert all(row["group_a"]["representative_members"]
               and row["group_b"]["representative_members"] for row in payload_rows)


def test_prepare_excludes_group_pairs_from_prior_frozen_manifest(tmp_path, monkeypatch):
    first, first_rows, _ = _prepare_toy(tmp_path, monkeypatch)
    second = sgm.prepare(
        "toy", "R1", str(tmp_path / "partition_toy_R1.json"), k=3, cap=20,
        tag="fresh", exclude_manifest_paths=[str(sgm._manifest_path("toy", "R1"))])
    second_rows = [
        row for path in second["payload_paths"] for row in sgm._jsonl_rows(Path(path))
    ]

    def group_pairs(rows):
        return {
            tuple(sorted((row["group_a"]["group_id"], row["group_b"]["group_id"])))
            for row in rows
        }

    assert group_pairs(first_rows).isdisjoint(group_pairs(second_rows))
    assert second["n_prior_group_pairs_listed"] == len(group_pairs(first_rows))
    assert second["n_prior_group_pairs_excluded"] == len(group_pairs(first_rows))
    assert second["exclude_manifest_sources"][0]["sha256"] == sgm._file_sha256(
        str(sgm._manifest_path("toy", "R1")))


def test_apply_fails_closed_on_inexact_vote_coverage(tmp_path, monkeypatch):
    manifest, payload_rows, _ = _prepare_toy(tmp_path, monkeypatch)
    expected = [row["pair_id"] for row in payload_rows]
    assert len(expected) >= 2
    screen_dir, confirm_dir = tmp_path / "screen", tmp_path / "confirm"
    _write_votes(screen_dir, [{"pair_id": pair_id, "score": 2}
                              for pair_id in expected[:-1]])
    _write_votes(confirm_dir, [{"pair_id": pair_id, "score": 2}
                               for pair_id in expected])
    output = tmp_path / "should_not_exist.json"

    with pytest.raises(sgm.VoteCoverageError, match="coverage is not exact"):
        sgm.apply("toy", "R1", str(screen_dir), str(confirm_dir),
                  manifest_path=str(sgm._manifest_path("toy", "R1")),
                  output_path=str(output))
    assert not output.exists()
    assert manifest["n_group_pair_candidates"] == len(expected)


def test_apply_requires_dual_score2_before_composing_groups(tmp_path, monkeypatch):
    # Freeze a different eval pair so ga/gb remains an eligible semantic-recall candidate.
    _, payload_rows, source = _prepare_toy(tmp_path, monkeypatch, eval_pair=("c", "d"))
    by_groups = {
        tuple(sorted((row["group_a"]["group_id"], row["group_b"]["group_id"]))): row["pair_id"]
        for row in payload_rows
    }
    target = by_groups[("ga", "gb")]
    screen_dir, confirm_dir = tmp_path / "screen", tmp_path / "confirm"
    _write_votes(screen_dir, [{"pair_id": row["pair_id"], "score": 2}
                              for row in payload_rows])
    _write_votes(confirm_dir, [
        {"pair_id": row["pair_id"], "score": 2 if row["pair_id"] == target else 1}
        for row in payload_rows
    ])
    output = tmp_path / "composed.json"

    report = sgm.apply("toy", "R1", str(screen_dir), str(confirm_dir),
                       manifest_path=str(sgm._manifest_path("toy", "R1")),
                       output_path=str(output))
    composed = json.loads(output.read_text())
    assert composed["a"] == composed["b"]
    assert composed["c"] != composed["a"] and composed["d"] != composed["a"]
    assert composed["c"] == source["c"] and composed["d"] == source["d"]
    assert report["n_dual_score2_edges"] == 1
    assert report["groups_before"] == 4 and report["groups_after"] == 3


def test_staged_confirm_routes_only_screen_positives_and_applies_fail_closed(tmp_path, monkeypatch):
    _, payload_rows, source = _prepare_toy(tmp_path, monkeypatch, eval_pair=("c", "d"))
    expected = [row["pair_id"] for row in payload_rows]
    target = expected[0]
    screen_dir, confirm_dir = tmp_path / "screen", tmp_path / "confirm"
    _write_votes(screen_dir, [
        {"pair_id": pair_id, "score": 2 if pair_id == target else 1}
        for pair_id in expected
    ])
    staged = sgm.stage_confirm(
        "toy", "R1", str(screen_dir),
        manifest_path=str(sgm._manifest_path("toy", "R1")))
    assert staged["confirm_pair_ids"] == [target]
    assert sum(1 for path in staged["payload_paths"] for _ in Path(path).open()) == 1
    _write_votes(confirm_dir, [{"pair_id": target, "score": 2}])
    output = tmp_path / "staged-composed.json"

    report = sgm.apply(
        "toy", "R1", str(screen_dir), str(confirm_dir),
        manifest_path=str(sgm._manifest_path("toy", "R1")),
        confirm_manifest_path=str(sgm._confirm_manifest_path("toy", "R1")),
        output_path=str(output))
    composed = json.loads(output.read_text())
    target_row = next(row for row in payload_rows if row["pair_id"] == target)
    ga, gb = target_row["group_a"]["group_id"], target_row["group_b"]["group_id"]
    members = {group: [node for node, current in source.items() if current == group]
               for group in (ga, gb)}
    assert composed[members[ga][0]] == composed[members[gb][0]]
    assert report["n_confirm_judgments"] == 1


def test_staged_confirm_rejects_missing_positive_vote(tmp_path, monkeypatch):
    _, payload_rows, _ = _prepare_toy(tmp_path, monkeypatch, eval_pair=("c", "d"))
    expected = [row["pair_id"] for row in payload_rows]
    screen_dir, confirm_dir = tmp_path / "screen", tmp_path / "confirm"
    _write_votes(screen_dir, [{"pair_id": pair_id, "score": 2} for pair_id in expected])
    sgm.stage_confirm(
        "toy", "R1", str(screen_dir),
        manifest_path=str(sgm._manifest_path("toy", "R1")))
    _write_votes(confirm_dir, [{"pair_id": pair_id, "score": 2}
                               for pair_id in expected[:-1]])

    with pytest.raises(sgm.VoteCoverageError, match="coverage is not exact"):
        sgm.apply(
            "toy", "R1", str(screen_dir), str(confirm_dir),
            manifest_path=str(sgm._manifest_path("toy", "R1")),
            confirm_manifest_path=str(sgm._confirm_manifest_path("toy", "R1")),
            output_path=str(tmp_path / "should-not-exist.json"))
