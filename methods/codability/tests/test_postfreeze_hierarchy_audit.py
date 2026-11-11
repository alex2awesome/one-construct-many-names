import json
from pathlib import Path

import pytest

from methods.codability.lexicon.postfreeze_hierarchy_audit import prepare, summarize


def _json(path: Path, value) -> None:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _jsonl(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _fixture(tmp_path: Path):
    nodes = tmp_path / "nodes.jsonl"
    _jsonl(
        nodes,
        [
            {"node_id": node, "name": f"Concept {node}", "gloss": f"Meaning of {node}"}
            for node in "abcdef"
        ],
    )
    candidate = tmp_path / "candidate.json"
    reference = tmp_path / "reference.json"
    _json(candidate, {node: "left" if node in "abc" else "right" for node in "abcdef"})
    _json(reference, {node: "left" if node in "abd" else "right" for node in "abcdef"})
    protocol = tmp_path / "protocol.txt"
    protocol.write_text("Score 2 only for the same construct.\n", encoding="utf-8")
    scores = tmp_path / "scores.jsonl"
    _jsonl(
        scores,
        [
            {"node_a": "a", "node_b": "c", "probabilities": {"DIFFERENT": 0.1, "RELATED": 0.1, "SAME": 0.8}},
            {"node_a": "a", "node_b": "f", "probabilities": [0.1, 0.2, 0.7]},
        ],
    )
    return nodes, candidate, reference, protocol, scores


def test_prepare_is_postfreeze_blind_and_stratified(tmp_path):
    nodes, candidate, reference, protocol, scores = _fixture(tmp_path)
    out = tmp_path / "audit"
    manifest = prepare(
        task="demo",
        level="R2",
        candidate_path=candidate,
        reference_path=reference,
        nodes_path=nodes,
        protocol_path=protocol,
        gemma_scores_path=scores,
        output_dir=out,
        sample_per_stratum=100,
    )
    assert manifest["n_pairs"] == 15
    assert set(manifest["strata"]) >= {"both_same", "candidate_only", "reference_only"}
    blind = [json.loads(line) for line in (out / "audit.jsonl").read_text().splitlines()]
    assert blind
    forbidden = {"stratum", "candidate_same", "reference_same", "weight", "node_a", "node_b"}
    assert all(not forbidden.intersection(row) for row in blind)
    with pytest.raises(FileExistsError):
        prepare(
            task="demo",
            level="R2",
            candidate_path=candidate,
            reference_path=reference,
            nodes_path=nodes,
            protocol_path=protocol,
            output_dir=out,
        )


def test_two_family_adjudication_and_paired_metrics(tmp_path):
    nodes, candidate, reference, protocol, _scores = _fixture(tmp_path)
    out = tmp_path / "audit"
    prepare(
        task="demo",
        level="R1",
        candidate_path=candidate,
        reference_path=reference,
        nodes_path=nodes,
        protocol_path=protocol,
        output_dir=out,
        sample_per_stratum=100,
    )
    key = [json.loads(line) for line in (out / "key.jsonl").read_text().splitlines()]
    truth = {row["pair_id"]: 2 if row["candidate_same"] else 0 for row in key}
    first = next(iter(truth))
    votes_a = tmp_path / "a.jsonl"
    votes_b = tmp_path / "b.jsonl"
    tie = tmp_path / "tie.jsonl"
    _jsonl(votes_a, [{"pair_id": pair_id, "score": score} for pair_id, score in truth.items()])
    _jsonl(
        votes_b,
        [
            {"pair_id": pair_id, "score": (1 if pair_id == first else score)}
            for pair_id, score in truth.items()
        ],
    )
    _jsonl(tie, [{"pair_id": first, "score": truth[first]}])
    report = summarize(
        manifest_path=out / "manifest.json",
        votes_a_path=votes_a,
        votes_b_path=votes_b,
        tiebreak_votes_path=tie,
        report_path=out / "report.json",
        bootstrap_samples=200,
    )
    assert report["adjudication"]["n_disagreements"] == 1
    assert report["candidate"]["precision"] == 1
    assert report["candidate"]["recall"] == 1
    assert report["paired_comparison"]["delta_cohen_kappa"] > 0
    assert report["paired_comparison"]["delta_same_f1"] > 0
    # Every stratum is fully enumerated in this fixture, so the finite-population
    # bootstrap must not invent uncertainty by resampling census rows.
    assert report["paired_comparison"]["delta_cohen_kappa_ci95"] == [
        report["paired_comparison"]["delta_cohen_kappa"]
    ] * 2
    assert report["paired_comparison"]["delta_same_f1_ci95"] == [
        report["paired_comparison"]["delta_same_f1"]
    ] * 2
    artifacts = report["adjudication"]["vote_artifacts"]
    assert artifacts["judge_a"]["path"] == str(votes_a.resolve())
    assert artifacts["judge_b"]["path"] == str(votes_b.resolve())
    assert artifacts["tiebreak"]["path"] == str(tie.resolve())
    assert all(len(artifacts[name]["sha256"]) == 64 for name in artifacts)


def test_frozen_input_drift_and_boolean_votes_fail_closed(tmp_path):
    nodes, candidate, _reference, protocol, _scores = _fixture(tmp_path)
    out = tmp_path / "audit"
    prepare(
        task="demo",
        level="R3",
        candidate_path=candidate,
        nodes_path=nodes,
        protocol_path=protocol,
        output_dir=out,
        sample_per_stratum=100,
    )
    keys = [json.loads(line) for line in (out / "key.jsonl").read_text().splitlines()]
    bad = tmp_path / "bad.jsonl"
    bad_b = tmp_path / "bad_b.jsonl"
    _jsonl(bad, [{"pair_id": row["pair_id"], "score": True} for row in keys])
    _jsonl(bad_b, [{"pair_id": row["pair_id"], "score": True} for row in keys])
    with pytest.raises(ValueError, match="invalid vote"):
        summarize(
            manifest_path=out / "manifest.json",
            votes_a_path=bad,
            votes_b_path=bad_b,
            tiebreak_votes_path=None,
            report_path=out / "report.json",
            bootstrap_samples=100,
        )
    protocol.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="frozen protocol changed"):
        summarize(
            manifest_path=out / "manifest.json",
            votes_a_path=bad,
            votes_b_path=bad_b,
            tiebreak_votes_path=None,
            report_path=out / "report2.json",
            bootstrap_samples=100,
        )


def test_same_vote_file_cannot_impersonate_two_independent_families(tmp_path):
    nodes, candidate, _reference, protocol, _scores = _fixture(tmp_path)
    out = tmp_path / "audit"
    prepare(
        task="demo",
        level="R2",
        candidate_path=candidate,
        nodes_path=nodes,
        protocol_path=protocol,
        output_dir=out,
        sample_per_stratum=100,
    )
    keys = [json.loads(line) for line in (out / "key.jsonl").read_text().splitlines()]
    votes = tmp_path / "one_judge.jsonl"
    _jsonl(votes, [{"pair_id": row["pair_id"], "score": 0} for row in keys])
    with pytest.raises(ValueError, match="independent judge families"):
        summarize(
            manifest_path=out / "manifest.json",
            votes_a_path=votes,
            votes_b_path=votes,
            tiebreak_votes_path=None,
            report_path=out / "report.json",
            bootstrap_samples=100,
        )
