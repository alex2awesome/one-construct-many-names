from __future__ import annotations

import json
from pathlib import Path

import pytest

from methods.codability.lexicon import partition_ledger as ledger


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _fixture(tmp_path: Path) -> dict:
    evaluation = []
    votes = []
    candidate_partition = {}
    reference_partition = {}
    # Forty truth-SAME pairs.  The reference misses ten of them.
    for index in range(40):
        node_a, node_b = f"same_{index}_a", f"same_{index}_b"
        pair_id = f"same_{index}"
        evaluation.append({"pair_id": pair_id, "node_a": node_a, "node_b": node_b,
                           "stratum": "neighbor"})
        votes.append({"pair_id": pair_id, "score": 2})
        candidate_partition[node_a] = candidate_partition[node_b] = f"same_{index}"
        reference_partition[node_a] = f"reference_same_{index}_a"
        reference_partition[node_b] = (f"reference_same_{index}_b" if index < 10
                                       else f"reference_same_{index}_a")
    # Eighty truth-DIFFERENT pairs.  The reference falsely merges ten of them.
    for index in range(80):
        node_a, node_b = f"diff_{index}_a", f"diff_{index}_b"
        pair_id = f"diff_{index}"
        evaluation.append({"pair_id": pair_id, "node_a": node_a, "node_b": node_b,
                           "stratum": "random"})
        votes.append({"pair_id": pair_id, "score": 0})
        candidate_partition[node_a] = f"candidate_diff_{index}_a"
        candidate_partition[node_b] = f"candidate_diff_{index}_b"
        reference_partition[node_a] = f"reference_diff_{index}_a"
        reference_partition[node_b] = (f"reference_diff_{index}_a" if index < 10
                                       else f"reference_diff_{index}_b")

    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text("".join(json.dumps(row) + "\n" for row in evaluation),
                         encoding="utf-8")
    vote_path = tmp_path / "votes.jsonl"
    vote_path.write_text("".join(json.dumps(row) + "\n" for row in votes), encoding="utf-8")
    protocol_path = _write_json(tmp_path / "protocol.json", {"relation": "same construct"})
    parent_path = _write_json(tmp_path / "parent.json", {"sha256": "frozen-parent"})
    certificates = []
    for certificate_type in ledger.DEFAULT_CERTIFICATE_TYPES:
        certificates.append(_write_json(
            tmp_path / f"{certificate_type}.json",
            {"certificate_type": certificate_type, "passed": True}))
    return {
        "evaluation": eval_path,
        "votes": vote_path,
        "protocol": protocol_path,
        "parent": parent_path,
        "certificates": certificates,
        "candidate_partition": _write_json(tmp_path / "candidate.json", candidate_partition),
        "reference_partition": _write_json(tmp_path / "reference.json", reference_partition),
    }


def _score(inputs: dict, partition_path: Path) -> dict:
    evaluation = ledger._load_eval(inputs["evaluation"])
    votes = ledger._load_votes([str(inputs["votes"])], set(evaluation))
    partition = ledger._load_partition(partition_path)
    truth = [votes[pair_id] for pair_id in evaluation]
    metrics = ledger._metrics(truth, ledger._predictions(partition, evaluation))
    return {
        "task": "demo",
        "level": "R2",
        "relation": "same focused operational theme",
        "complete": True,
        "partition_path": str(partition_path.resolve()),
        "partition_sha256": ledger.file_sha256(partition_path),
        "arbiter_vote_paths": [str(inputs["votes"].resolve())],
        "arbiter_vote_sha256": [ledger.file_sha256(inputs["votes"])],
        # build_level.score reports these endpoints rounded to three decimals.
        "cohen_kappa_same_binary_eval": round(metrics["cohen_kappa"], 3),
        "precision": round(metrics["same_precision"], 3),
        "recall": round(metrics["same_recall"], 3),
    }


def _append(tmp_path: Path, inputs: dict, name: str, *, cold: float = 0.7) -> dict:
    partition_path = inputs[f"{name}_partition"]
    score_path = _write_json(tmp_path / f"{name}_score.json", _score(inputs, partition_path))
    return ledger.append_candidate(
        tmp_path / "ledger", score_path,
        eval_path=inputs["evaluation"], protocol_path=inputs["protocol"],
        parent_path=inputs["parent"], integrity_cert_paths=inputs["certificates"],
        cold_metrics={"cohen_kappa": cold})


def test_append_authenticates_score_and_is_content_addressed(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    first = _append(tmp_path, inputs, "candidate")
    second = ledger.append_candidate(
        tmp_path / "ledger", tmp_path / "candidate_score.json",
        eval_path=inputs["evaluation"], protocol_path=inputs["protocol"],
        parent_path=inputs["parent"], integrity_cert_paths=inputs["certificates"],
        cold_metrics={"cohen_kappa": 0.7})

    assert first["candidate_id"] == second["candidate_id"]
    assert first["metrics"]["cohen_kappa"] == 1.0
    assert first["metrics"]["same_precision"] == 1.0
    assert first["metrics"]["same_recall"] == 1.0
    assert first["metrics"]["same_f1"] == 1.0
    assert first["selection_metric"] == "cohen_kappa"
    assert first["canonical_write_authorized"] is False
    assert len(list((tmp_path / "ledger" / "candidates").glob("*.json"))) == 1


def test_append_rejects_bare_recall_or_changed_input(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    score = _score(inputs, inputs["candidate_partition"])
    score["cohen_kappa_same_binary_eval"] = None
    with pytest.raises(ledger.LedgerIntegrityError, match="Cohen kappa"):
        ledger.append_candidate(
            tmp_path / "ledger", score, eval_path=inputs["evaluation"],
            protocol_path=inputs["protocol"], parent_path=inputs["parent"],
            integrity_cert_paths=inputs["certificates"])

    score = _score(inputs, inputs["candidate_partition"])
    inputs["candidate_partition"].write_text("{}", encoding="utf-8")
    with pytest.raises(ledger.LedgerIntegrityError, match="SHA-256"):
        ledger.append_candidate(
            tmp_path / "ledger", score, eval_path=inputs["evaluation"],
            protocol_path=inputs["protocol"], parent_path=inputs["parent"],
            integrity_cert_paths=inputs["certificates"])


def test_append_requires_all_passing_integrity_certificates(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    score = _score(inputs, inputs["candidate_partition"])
    with pytest.raises(ledger.LedgerIntegrityError, match="missing required"):
        ledger.append_candidate(
            tmp_path / "ledger", score, eval_path=inputs["evaluation"],
            protocol_path=inputs["protocol"], parent_path=inputs["parent"],
            integrity_cert_paths=inputs["certificates"][:-1])


def test_paired_promotion_applies_every_gate_without_canonical_write(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    reference = _append(tmp_path, inputs, "reference", cold=0.6)
    candidate = _append(tmp_path, inputs, "candidate", cold=0.7)

    decision = ledger.decide_promotion(
        tmp_path / "ledger", reference["candidate_id"], candidate["candidate_id"],
        bootstrap_samples=500, seed=19)

    assert decision["promote"] is True
    assert decision["paired_bootstrap"]["delta_cohen_kappa_ci95"][0] > 0
    assert decision["delta"]["same_f1"] > 0
    assert decision["candidate_metrics"]["same_precision"] >= 0.5
    assert decision["candidate_metrics"]["same_recall"] >= 0.5
    assert all(decision["gates"].values())
    assert decision["canonical_write_authorized"] is False
    assert not (tmp_path / "ledger" / "canonical.json").exists()


def test_load_fails_closed_after_authenticated_artifact_drift(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    candidate = _append(tmp_path, inputs, "candidate")
    inputs["protocol"].write_text('{"relation":"changed"}', encoding="utf-8")
    with pytest.raises(ledger.LedgerIntegrityError, match="frozen artifact changed"):
        ledger.load_candidate(tmp_path / "ledger", candidate["candidate_id"])


def test_promotion_rejects_one_sided_cold_metrics(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    reference = _append(tmp_path, inputs, "reference")
    candidate_score = _write_json(
        tmp_path / "candidate_score.json",
        _score(inputs, inputs["candidate_partition"]))
    candidate = ledger.append_candidate(
        tmp_path / "ledger", candidate_score,
        eval_path=inputs["evaluation"], protocol_path=inputs["protocol"],
        parent_path=inputs["parent"], integrity_cert_paths=inputs["certificates"])
    with pytest.raises(ledger.PromotionComparisonError, match="both paired candidates"):
        ledger.decide_promotion(
            tmp_path / "ledger", reference["candidate_id"], candidate["candidate_id"],
            bootstrap_samples=100)


def test_promotion_denies_cold_concept_regression(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    reference = _append(tmp_path, inputs, "reference", cold=0.8)
    candidate = _append(tmp_path, inputs, "candidate", cold=0.7)
    decision = ledger.decide_promotion(
        tmp_path / "ledger", reference["candidate_id"], candidate["candidate_id"],
        bootstrap_samples=200, seed=23)
    assert decision["gates"]["cold_non_regression"] is False
    assert decision["promote"] is False
