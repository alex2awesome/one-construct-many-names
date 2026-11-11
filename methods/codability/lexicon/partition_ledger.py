"""Immutable, authenticated ledger for LLM-judged partition evaluations.

This module is deliberately downstream of :func:`build_level.score`.  It does not
construct partitions, call a judge, reinterpret a semantic label, or update a
canonical partition.  It authenticates the frozen inputs to a complete score,
recomputes the binary SAME metrics from those inputs, and stores one immutable
candidate record.  Promotion decisions compare two candidates on the same
LLM-judged pairs with a deterministic paired bootstrap.

The ledger is a directory of content-addressed JSON records rather than a mutable
JSON array.  Adding a record is therefore atomic, idempotent, and safe when several
offline scoring jobs finish at once.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence


SCHEMA_VERSION = "partition-evaluation-ledger-v1"
DEFAULT_CERTIFICATE_TYPES = (
    "parent_integrity",
    "coverage",
    "naming",
    "large_clusters",
)


class LedgerIntegrityError(ValueError):
    """A score or one of its frozen source artifacts failed authentication."""


class PromotionComparisonError(ValueError):
    """Two candidate records cannot support a paired promotion decision."""


def file_sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(payload: object) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False) + "\n").encode("utf-8")


def _artifact_ref(path: str | os.PathLike[str]) -> dict:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise LedgerIntegrityError(f"artifact is not a file: {resolved}")
    return {"path": str(resolved), "sha256": file_sha256(resolved)}


def _atomic_content_write(directory: Path, payload: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    content = _canonical_json(payload)
    record_id = hashlib.sha256(content).hexdigest()
    destination = directory / f"{record_id}.json"
    try:
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        if destination.read_bytes() != content:
            raise LedgerIntegrityError(f"content-address collision at {destination}")
        return destination
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    return destination


def _load_json(path: str | os.PathLike[str]) -> object:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LedgerIntegrityError(f"cannot read JSON artifact {path}: {exc}") from exc


def _load_jsonl(path: str | os.PathLike[str]) -> list[dict]:
    rows: list[dict] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LedgerIntegrityError(f"malformed JSON at {path}:{line_number}") from exc
        if not isinstance(row, dict):
            raise LedgerIntegrityError(f"non-object JSON at {path}:{line_number}")
        rows.append(row)
    return rows


def _load_partition(path: str | os.PathLike[str]) -> dict[str, str]:
    payload = _load_json(path)
    if isinstance(payload, dict) and isinstance(payload.get("assignment"), dict):
        payload = payload["assignment"]
    if (not isinstance(payload, dict) or not payload
            or any(not isinstance(node, str) or not isinstance(group, str)
                   or not node or not group for node, group in payload.items())):
        raise LedgerIntegrityError(f"invalid partition artifact: {path}")
    return dict(payload)


def _load_eval(path: str | os.PathLike[str]) -> dict[str, dict]:
    rows = _load_jsonl(path)
    result: dict[str, dict] = {}
    for row in rows:
        pair_id, node_a, node_b = row.get("pair_id"), row.get("node_a"), row.get("node_b")
        if (not isinstance(pair_id, str) or not pair_id
                or not isinstance(node_a, str) or not isinstance(node_b, str)
                or node_a == node_b or pair_id in result):
            raise LedgerIntegrityError(f"invalid or duplicate evaluation pair in {path}")
        result[pair_id] = row
    if not result:
        raise LedgerIntegrityError(f"empty evaluation artifact: {path}")
    return result


def _load_votes(paths: Sequence[str], expected: set[str]) -> dict[str, bool]:
    votes: dict[str, bool] = {}
    for path in paths:
        for row in _load_jsonl(path):
            pair_id, score = row.get("pair_id"), row.get("score")
            # bool is an int subclass: exact type checking is intentional.
            if (not isinstance(pair_id, str) or type(score) is not int
                    or score not in (0, 1, 2) or pair_id in votes
                    or pair_id not in expected):
                raise LedgerIntegrityError(f"malformed, duplicate, or unexpected LLM vote in {path}")
            votes[pair_id] = score == 2
    missing = expected - set(votes)
    if missing:
        raise LedgerIntegrityError(
            f"LLM vote artifacts miss {len(missing)} evaluation pairs; sample={sorted(missing)[:5]}")
    return votes


def _cohen_kappa(truth: Sequence[bool], prediction: Sequence[bool]) -> float | None:
    if not truth or len(truth) != len(prediction):
        return None
    observed = sum(a == b for a, b in zip(truth, prediction)) / len(truth)
    truth_rate = sum(truth) / len(truth)
    prediction_rate = sum(prediction) / len(prediction)
    expected = (truth_rate * prediction_rate
                + (1.0 - truth_rate) * (1.0 - prediction_rate))
    return (observed - expected) / (1.0 - expected) if expected < 1.0 else None


def _metrics(truth: Sequence[bool], prediction: Sequence[bool]) -> dict[str, float | int]:
    if not truth or len(truth) != len(prediction):
        raise LedgerIntegrityError("metrics require non-empty aligned truth and predictions")
    true_positive = sum(a and b for a, b in zip(truth, prediction))
    predicted_positive = sum(prediction)
    truth_positive = sum(truth)
    precision = true_positive / predicted_positive if predicted_positive else 0.0
    recall = true_positive / truth_positive if truth_positive else 0.0
    same_f1 = (2 * precision * recall / (precision + recall)
               if precision + recall else 0.0)
    kappa = _cohen_kappa(truth, prediction)
    if kappa is None:
        raise LedgerIntegrityError("Cohen kappa is undefined for this evaluation")
    return {
        "n": len(truth),
        "n_truth_same": truth_positive,
        "n_predicted_same": predicted_positive,
        "cohen_kappa": kappa,
        "same_precision": precision,
        "same_recall": recall,
        "same_f1": same_f1,
    }


def _predictions(partition: Mapping[str, str], evaluation: Mapping[str, dict]) -> list[bool]:
    predictions = []
    for row in evaluation.values():
        node_a, node_b = row["node_a"], row["node_b"]
        if node_a not in partition or node_b not in partition:
            raise LedgerIntegrityError(
                f"stale evaluation pair references absent node: {node_a}, {node_b}")
        predictions.append(partition[node_a] == partition[node_b])
    return predictions


def _validate_score_shape(score: dict) -> None:
    required = ("task", "level", "complete", "partition_path", "partition_sha256",
                "arbiter_vote_paths", "arbiter_vote_sha256",
                "cohen_kappa_same_binary_eval", "precision", "recall")
    missing = [key for key in required if key not in score]
    if missing:
        raise LedgerIntegrityError(f"not a build_level.score artifact; missing={missing}")
    if (not isinstance(score["task"], str) or not score["task"]
            or not isinstance(score["level"], str) or not score["level"]
            or not isinstance(score["partition_path"], str)
            or not isinstance(score["partition_sha256"], str)):
        raise LedgerIntegrityError("invalid task, level, or partition reference in score artifact")
    vote_paths, vote_hashes = score["arbiter_vote_paths"], score["arbiter_vote_sha256"]
    if (not isinstance(vote_paths, list) or not vote_paths
            or not isinstance(vote_hashes, list) or len(vote_paths) != len(vote_hashes)
            or any(not isinstance(value, str) or not value
                   for value in [*vote_paths, *vote_hashes])):
        raise LedgerIntegrityError("invalid arbiter vote paths/SHA-256s in score artifact")
    if score["complete"] is not True:
        raise LedgerIntegrityError("only complete build_level.score artifacts enter the ledger")
    if score["cohen_kappa_same_binary_eval"] is None:
        raise LedgerIntegrityError("a real Cohen kappa is required; bare recall cannot enter the ledger")


def _validate_score(score: dict, metrics: Mapping[str, float | int],
                    partition_ref: dict, vote_refs: Sequence[dict]) -> None:
    _validate_score_shape(score)
    _validate_score_source_refs(score, partition_ref, vote_refs)

    comparisons = {
        "cohen_kappa_same_binary_eval": metrics["cohen_kappa"],
        "precision": metrics["same_precision"],
        "recall": metrics["same_recall"],
    }
    for score_key, recomputed in comparisons.items():
        claimed = score[score_key]
        if (isinstance(claimed, bool) or not isinstance(claimed, (int, float))
                or not math.isfinite(float(claimed))
                or abs(float(claimed) - float(recomputed)) > 0.00051):
            raise LedgerIntegrityError(
                f"score {score_key}={claimed!r} disagrees with frozen LLM inputs "
                f"({recomputed:.9f})")


def _validate_score_source_refs(score: dict, partition_ref: dict,
                                vote_refs: Sequence[dict]) -> None:
    """Fail on source drift before attempting to parse the changed artifact."""
    if Path(score["partition_path"]).expanduser().resolve() != Path(partition_ref["path"]):
        raise LedgerIntegrityError("score partition path does not match authenticated partition")
    if score["partition_sha256"] != partition_ref["sha256"]:
        raise LedgerIntegrityError("score partition SHA-256 does not match partition bytes")
    claimed_votes = sorted(
        (str(Path(path).expanduser().resolve()), digest)
        for path, digest in zip(score["arbiter_vote_paths"], score["arbiter_vote_sha256"])
    )
    observed_votes = sorted((ref["path"], ref["sha256"]) for ref in vote_refs)
    if claimed_votes != observed_votes:
        raise LedgerIntegrityError("score vote paths/SHA-256s do not match authenticated vote inputs")


def _validate_certificates(paths: Iterable[str], required_types: Sequence[str]) -> list[dict]:
    references = []
    found: set[str] = set()
    for path in paths:
        reference = _artifact_ref(path)
        payload = _load_json(reference["path"])
        certificate_type = payload.get("certificate_type") if isinstance(payload, dict) else None
        if (not isinstance(certificate_type, str) or not certificate_type
                or payload.get("passed") is not True or certificate_type in found):
            raise LedgerIntegrityError(
                f"certificate must have unique certificate_type and passed=true: {path}")
        found.add(certificate_type)
        references.append({**reference, "certificate_type": certificate_type})
    absent = sorted(set(required_types) - found)
    if absent:
        raise LedgerIntegrityError(f"missing required integrity certificates: {absent}")
    return sorted(references, key=lambda item: item["certificate_type"])


def _validate_cold_metrics(cold_metrics: Mapping[str, float] | None) -> dict[str, float] | None:
    if cold_metrics is None:
        return None
    if not cold_metrics:
        raise LedgerIntegrityError("cold metrics cannot be empty")
    result = {}
    for name, value in cold_metrics.items():
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(float(value))):
            raise LedgerIntegrityError(f"invalid cold metric {name}={value!r}")
        result[str(name)] = float(value)
    return dict(sorted(result.items()))


def append_candidate(
    ledger_dir: str | os.PathLike[str],
    score_artifact: dict | str | os.PathLike[str],
    *,
    eval_path: str | os.PathLike[str],
    protocol_path: str | os.PathLike[str],
    parent_path: str | os.PathLike[str],
    integrity_cert_paths: Sequence[str | os.PathLike[str]],
    required_certificate_types: Sequence[str] = DEFAULT_CERTIFICATE_TYPES,
    cold_metrics: Mapping[str, float] | None = None,
) -> dict:
    """Authenticate and append one immutable candidate evaluation.

    ``score_artifact`` is the return value of ``build_level.score`` or a JSON file
    containing that return value.  All semantic truth is loaded exclusively from
    its frozen arbiter vote files; the ledger never manufactures labels.
    """
    if isinstance(score_artifact, (str, os.PathLike)):
        score_ref = _artifact_ref(score_artifact)
        score = _load_json(score_ref["path"])
    else:
        score_ref = None
        score = score_artifact
    if not isinstance(score, dict):
        raise LedgerIntegrityError("score artifact must be a JSON object")
    _validate_score_shape(score)
    partition_ref = _artifact_ref(str(score.get("partition_path", "")))
    eval_ref = _artifact_ref(eval_path)
    protocol_ref = _artifact_ref(protocol_path)
    parent_ref = _artifact_ref(parent_path)
    vote_paths = score.get("arbiter_vote_paths")
    if not isinstance(vote_paths, list) or not vote_paths:
        raise LedgerIntegrityError("score artifact has no arbiter vote files")
    vote_refs = sorted((_artifact_ref(path) for path in vote_paths), key=lambda ref: ref["path"])
    # Authenticate the exact bytes named by build_level.score before parsing them.
    _validate_score_source_refs(score, partition_ref, vote_refs)
    certificates = _validate_certificates(
        [str(path) for path in integrity_cert_paths], required_certificate_types)

    evaluation = _load_eval(eval_ref["path"])
    votes = _load_votes([ref["path"] for ref in vote_refs], set(evaluation))
    partition = _load_partition(partition_ref["path"])
    truth = [votes[pair_id] for pair_id in evaluation]
    metrics = _metrics(truth, _predictions(partition, evaluation))
    _validate_score(score, metrics, partition_ref, vote_refs)

    record = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "candidate_evaluation",
        "task": score["task"],
        "level": score["level"],
        "relation": score.get("relation"),
        "artifacts": {
            "score": score_ref,
            "partition": partition_ref,
            "evaluation": eval_ref,
            "arbiter_votes": vote_refs,
            "protocol": protocol_ref,
            "parent": parent_ref,
            "integrity_certificates": certificates,
        },
        "required_certificate_types": sorted(set(required_certificate_types)),
        "build_level_score": score,
        "build_level_score_sha256": hashlib.sha256(_canonical_json(score)).hexdigest(),
        "metrics": metrics,
        "cold_metrics": _validate_cold_metrics(cold_metrics),
        "selection_metric": "cohen_kappa",
        "canonical_write_authorized": False,
    }
    path = _atomic_content_write(Path(ledger_dir).expanduser().resolve() / "candidates", record)
    return {"candidate_id": path.stem, "record_path": str(path), **record}


def _authenticate_ref(reference: dict) -> None:
    if not isinstance(reference, dict) or not {"path", "sha256"} <= set(reference):
        raise LedgerIntegrityError(f"frozen artifact changed: {reference!r}")
    try:
        observed = file_sha256(reference["path"])
    except OSError as exc:
        raise LedgerIntegrityError(f"frozen artifact changed: {reference!r}") from exc
    if observed != reference["sha256"]:
        raise LedgerIntegrityError(f"frozen artifact changed: {reference!r}")


def load_candidate(ledger_dir: str | os.PathLike[str], candidate_id: str) -> dict:
    path = Path(ledger_dir).expanduser().resolve() / "candidates" / f"{candidate_id}.json"
    payload = _load_json(path)
    if not isinstance(payload, dict) or payload.get("record_type") != "candidate_evaluation":
        raise LedgerIntegrityError(f"invalid candidate ledger record: {path}")
    if hashlib.sha256(_canonical_json(payload)).hexdigest() != candidate_id:
        raise LedgerIntegrityError(f"candidate record is not content-addressed: {path}")
    embedded_score = payload.get("build_level_score")
    if (not isinstance(embedded_score, dict)
            or hashlib.sha256(_canonical_json(embedded_score)).hexdigest()
            != payload.get("build_level_score_sha256")):
        raise LedgerIntegrityError(f"embedded build_level.score changed: {path}")
    artifacts = payload.get("artifacts", {})
    for name in ("partition", "evaluation", "protocol", "parent"):
        _authenticate_ref(artifacts.get(name))
    if artifacts.get("score") is not None:
        _authenticate_ref(artifacts["score"])
    for reference in artifacts.get("arbiter_votes", []):
        _authenticate_ref(reference)
    for reference in artifacts.get("integrity_certificates", []):
        _authenticate_ref(reference)
        certificate = _load_json(reference["path"])
        if certificate.get("passed") is not True:
            raise LedgerIntegrityError(f"integrity certificate no longer passes: {reference['path']}")
    return payload


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _paired_kappa_interval(truth: Sequence[bool], reference: Sequence[bool],
                           candidate: Sequence[bool], *, samples: int,
                           seed: int) -> tuple[float, float]:
    if samples < 100:
        raise PromotionComparisonError("at least 100 bootstrap samples are required")
    rng = random.Random(seed)
    deltas = []
    for _ in range(samples):
        indices = [rng.randrange(len(truth)) for _ in truth]
        sampled_truth = [truth[index] for index in indices]
        reference_kappa = _cohen_kappa(sampled_truth, [reference[index] for index in indices])
        candidate_kappa = _cohen_kappa(sampled_truth, [candidate[index] for index in indices])
        if reference_kappa is not None and candidate_kappa is not None:
            deltas.append(candidate_kappa - reference_kappa)
    if len(deltas) < math.ceil(samples * 0.9):
        raise PromotionComparisonError(
            "too many bootstrap replicates have undefined Cohen kappa")
    return _percentile(deltas, 0.025), _percentile(deltas, 0.975)


def decide_promotion(
    ledger_dir: str | os.PathLike[str],
    reference_id: str,
    candidate_id: str,
    *,
    bootstrap_samples: int = 2000,
    seed: int = 0,
) -> dict:
    """Record an immutable paired decision; never update a canonical pointer."""
    root = Path(ledger_dir).expanduser().resolve()
    reference = load_candidate(root, reference_id)
    candidate = load_candidate(root, candidate_id)
    if (reference["task"], reference["level"]) != (candidate["task"], candidate["level"]):
        raise PromotionComparisonError("paired candidates must have the same task and level")
    for artifact_name in ("evaluation", "arbiter_votes", "protocol", "parent"):
        if reference["artifacts"][artifact_name] != candidate["artifacts"][artifact_name]:
            raise PromotionComparisonError(
                f"paired candidates do not share frozen {artifact_name} inputs")

    evaluation = _load_eval(candidate["artifacts"]["evaluation"]["path"])
    votes = _load_votes(
        [item["path"] for item in candidate["artifacts"]["arbiter_votes"]], set(evaluation))
    truth = [votes[pair_id] for pair_id in evaluation]
    reference_prediction = _predictions(
        _load_partition(reference["artifacts"]["partition"]["path"]), evaluation)
    candidate_prediction = _predictions(
        _load_partition(candidate["artifacts"]["partition"]["path"]), evaluation)
    reference_metrics = _metrics(truth, reference_prediction)
    candidate_metrics = _metrics(truth, candidate_prediction)
    kappa_interval = _paired_kappa_interval(
        truth, reference_prediction, candidate_prediction,
        samples=bootstrap_samples, seed=seed)

    reference_cold = reference.get("cold_metrics")
    candidate_cold = candidate.get("cold_metrics")
    if (reference_cold is None) != (candidate_cold is None):
        raise PromotionComparisonError(
            "cold metrics must be supplied for both paired candidates or neither")
    cold_gate = True
    cold_delta = None
    if reference_cold is not None:
        if set(reference_cold) != set(candidate_cold):
            raise PromotionComparisonError("paired candidates have different cold metric endpoints")
        cold_delta = {name: candidate_cold[name] - reference_cold[name]
                      for name in reference_cold}
        cold_gate = all(delta >= 0.0 for delta in cold_delta.values())

    certificate_types = {
        item["certificate_type"]
        for item in candidate["artifacts"]["integrity_certificates"]
    }
    integrity_gate = set(candidate["required_certificate_types"]) <= certificate_types
    delta = {
        key: candidate_metrics[key] - reference_metrics[key]
        for key in ("cohen_kappa", "same_precision", "same_recall", "same_f1")
    }
    gates = {
        "delta_cohen_kappa_ci95_above_zero": kappa_interval[0] > 0.0,
        "positive_delta_same_f1": delta["same_f1"] > 0.0,
        "candidate_same_precision_at_least_0_50": candidate_metrics["same_precision"] >= 0.50,
        "candidate_same_recall_at_least_0_50": candidate_metrics["same_recall"] >= 0.50,
        "same_precision_regression_at_most_0_02": delta["same_precision"] >= -0.02,
        "same_recall_regression_at_most_0_02": delta["same_recall"] >= -0.02,
        "cold_non_regression": cold_gate,
        "integrity_certificates_complete": integrity_gate,
    }
    decision = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "paired_promotion_decision",
        "task": candidate["task"],
        "level": candidate["level"],
        "reference_id": reference_id,
        "candidate_id": candidate_id,
        "n_pairs": len(truth),
        "reference_metrics": reference_metrics,
        "candidate_metrics": candidate_metrics,
        "delta": delta,
        "paired_bootstrap": {
            "samples": bootstrap_samples,
            "seed": seed,
            "delta_cohen_kappa_ci95": list(kappa_interval),
        },
        "cold_delta": cold_delta,
        "gates": gates,
        "promote": all(gates.values()),
        "selection_metric": "cohen_kappa",
        "canonical_write_authorized": False,
    }
    path = _atomic_content_write(root / "decisions", decision)
    return {"decision_id": path.stem, "record_path": str(path), **decision}


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    append = subparsers.add_parser("append", help="append an authenticated score")
    append.add_argument("--ledger", required=True)
    append.add_argument("--score", required=True)
    append.add_argument("--eval", required=True)
    append.add_argument("--protocol", required=True)
    append.add_argument("--parent", required=True)
    append.add_argument("--certificate", action="append", default=[])
    append.add_argument("--cold-metrics", help="optional JSON object path")
    promote = subparsers.add_parser("promote", help="record a paired promotion decision")
    promote.add_argument("--ledger", required=True)
    promote.add_argument("--reference-id", required=True)
    promote.add_argument("--candidate-id", required=True)
    promote.add_argument("--bootstrap-samples", type=int, default=2000)
    promote.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()
    if arguments.command == "append":
        cold = _load_json(arguments.cold_metrics) if arguments.cold_metrics else None
        result = append_candidate(
            arguments.ledger, arguments.score, eval_path=arguments.eval,
            protocol_path=arguments.protocol, parent_path=arguments.parent,
            integrity_cert_paths=arguments.certificate, cold_metrics=cold)
    else:
        result = decide_promotion(
            arguments.ledger, arguments.reference_id, arguments.candidate_id,
            bootstrap_samples=arguments.bootstrap_samples, seed=arguments.seed)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    _main()
