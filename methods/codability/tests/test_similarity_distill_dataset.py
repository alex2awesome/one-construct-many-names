from __future__ import annotations

import json
from pathlib import Path

import pytest

from methods.codability.lexicon_distill.dataset import (
    PROTOCOL_IDS,
    SourceSpec,
    Vote,
    balanced_target,
    build_dataset,
    classify_score_source,
    infer_level,
    infer_task,
)


def test_context_and_source_classification() -> None:
    path = "level_votes/vrf_math-stackexchange_R2_007.jsonl"
    assert infer_task(path) == "math-stackexchange"
    assert infer_level(path) == "R2"
    assert classify_score_source(path) == SourceSpec(
        "sonnet", "train", "workflow_inferred"
    )
    assert classify_score_source("repair_votes/screen_humor_widen_001.jsonl") is None
    assert classify_score_source(
        "r2_recluster_v2/humor_v2_1_comparison_votes_agent.jsonl"
    ).role == "reserved"


def _vote(family: str, score: int, line: int) -> Vote:
    return Vote(
        pair_id="p",
        score=score,
        task="humor",
        level="R1",
        protocol_id=PROTOCOL_IDS["R1"],
        teacher_family=family,
        role="train",
        provenance_strength="workflow_inferred",
        label_kind="independent",
        source_path=f"{family}.jsonl",
        source_sha256=family,
        source_line=line,
        base_weight=1.0,
    )


def test_balanced_teacher_consensus_is_not_volume_weighted() -> None:
    votes = [_vote("sonnet", 2, index) for index in range(10)] + [_vote("gpt5", 0, 1)]
    target, weight, distributions = balanced_target(votes)
    assert target == [0.5, 0.0, 0.5]
    assert weight == 1.0
    assert distributions["sonnet"] == [0.0, 0.0, 1.0]


def _write_protocols(root: Path) -> None:
    for relative in (
        "ARBITER_PROTOCOL_R1.txt",
        "ARBITER_PROTOCOL_R2.txt",
        "ARBITER_PROTOCOL_R3.txt",
        "r2_recluster_v2/R2_V2_PROTOCOL.md",
        "r2_recluster_v2/R2_V2_1_PROTOCOL.md",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")


def test_builder_joins_scores_reserves_eval_and_freezes_hashes(tmp_path: Path) -> None:
    root = tmp_path / "lexicon"
    _write_protocols(root)
    payload = root / "level_arbiter" / "humor_R1_verify_000.jsonl"
    payload.parent.mkdir(parents=True)
    rows = [
        {"pair_id": "p1", "task": "humor", "level": "R1", "node_a": "a", "node_b": "b", "canonical_a": "Timing", "canonical_b": "Pacing"},
        {"pair_id": "p2", "task": "humor", "level": "R1", "node_a": "c", "node_b": "d", "canonical_a": "Surprise", "canonical_b": "Wordplay"},
    ]
    payload.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    votes = root / "level_votes" / "vrf_humor_R1_000.jsonl"
    votes.parent.mkdir()
    votes.write_text('{"pair_id":"p1","score":2}\n{"pair_id":"p2","score":1}\n', encoding="utf-8")
    reserved = root / "r1_truth_reaudit" / "final_votes" / "arb_humor_R1.jsonl"
    reserved.parent.mkdir(parents=True)
    reserved.write_text('{"pair_id":"p1","score":2}\n', encoding="utf-8")
    output = tmp_path / "frozen"
    inventory = build_dataset(root, output)
    all_rows = [json.loads(line) for line in (output / "all.jsonl").read_text().splitlines()]
    assert inventory["counts"]["rows"] == 2
    p1 = next(row for row in all_rows if "p1" in row["source_pair_ids"])
    assert p1["reserved"] is True
    assert p1["split"] == "external_test"
    assert all(
        source["role"] == "reserved"
        for source in p1["sources"] if source["used_in_target"]
    )
    assert (output / "manifest.json").is_file()
    assert json.loads((output / "manifest.json").read_text())["artifacts"]["all.jsonl"]["sha256"]


def test_builder_refuses_overwrite(tmp_path: Path) -> None:
    root = tmp_path / "lexicon"
    _write_protocols(root)
    output = tmp_path / "frozen"
    output.mkdir()
    with pytest.raises(FileExistsError):
        build_dataset(root, output)


def test_reservation_crosses_protocol_variants(tmp_path: Path) -> None:
    root = tmp_path / "lexicon"
    _write_protocols(root)
    legacy_payload = root / "level_arbiter" / "humor_R2_verify_000.jsonl"
    legacy_payload.parent.mkdir(parents=True)
    legacy_payload.write_text(
        json.dumps({"pair_id": "legacy", "task": "humor", "level": "R2", "node_a": "a", "node_b": "b", "canonical_a": "Delivery", "canonical_b": "Timing"}) + "\n",
        encoding="utf-8",
    )
    train_votes = root / "level_votes" / "vrf_humor_R2_000.jsonl"
    train_votes.parent.mkdir()
    train_votes.write_text('{"pair_id":"legacy","score":2}\n', encoding="utf-8")
    current_payload = root / "r2_recluster_v2" / "humor_v2_1_comparison_blind.jsonl"
    current_payload.parent.mkdir(exist_ok=True)
    current_payload.write_text(
        json.dumps({"pair_id": "current", "task": "humor", "level": "R2", "node_a": "a", "node_b": "b", "concept_a": "Delivery", "concept_b": "Timing"}) + "\n",
        encoding="utf-8",
    )
    current_votes = root / "r2_recluster_v2" / "humor_v2_1_comparison_votes_agent.jsonl"
    current_votes.write_text('{"pair_id":"current","score":1}\n', encoding="utf-8")
    output = tmp_path / "frozen"
    build_dataset(root, output)
    rows = [json.loads(line) for line in (output / "all.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["protocol_id"] == PROTOCOL_IDS["R2_V2_1"]
    assert rows[0]["split"] == "external_test"
