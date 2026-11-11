"""Post-freeze, paired LLM audit for immutable hierarchy candidates.

Construction must finish before this module is invoked.  ``prepare`` authenticates the
candidate (and optional reference) partition, then draws a deterministic stratified sample from
the complete node-pair population.  The blind payload contains no partition assignments or
stratum names.  Two independent LLM families label every pair; a third vote is required only
where the first two disagree.

Code never supplies semantic truth.  It performs sampling, integrity validation, weighted
arithmetic, and a paired bootstrap over labels supplied by LLM judges.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .build_level import _file_sha256


VERSION = "postfreeze-hierarchy-audit-v1"
VALID_LEVELS = {"R1", "R2", "R3"}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(value)
    return rows


def _load_partition(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        raw = payload.get("partition", payload.get("assignment", payload))
    else:
        raw = None
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"partition must be a nonempty JSON object: {path}")
    result = {str(node): str(group) for node, group in raw.items()}
    if any(not node or not group for node, group in result.items()):
        raise ValueError(f"partition contains empty node/group identifiers: {path}")
    return result


def _node_text(row: Mapping[str, Any]) -> str:
    parts = []
    for key in ("name", "gloss"):
        value = str(row.get(key) or "").strip()
        if value and value not in parts:
            parts.append(value)
    examples = row.get("member_examples") or []
    if isinstance(examples, list):
        for value in examples[:4]:
            text = str(value or "").strip()
            if text and text not in parts:
                parts.append(text)
    rendered = "\n".join(parts)
    if not rendered:
        raise ValueError(f"node {row.get('node_id')!r} has no semantic representation")
    return rendered


def _pair(a: str, b: str) -> tuple[str, str]:
    if a == b:
        raise ValueError("self-pairs are not auditable")
    return tuple(sorted((a, b)))


def _pair_id(task: str, level: str, candidate_sha: str, a: str, b: str) -> str:
    material = f"{VERSION}|{task}|{level}|{candidate_sha}|{a}|{b}"
    return hashlib.sha256(material.encode()).hexdigest()[:24]


def _gemma_high_pairs(path: Path | None, threshold: float) -> set[tuple[str, str]]:
    if path is None:
        return set()
    result: set[tuple[str, str]] = set()
    for line_number, row in enumerate(_load_jsonl(path), 1):
        if "node_a" not in row or "node_b" not in row:
            raise ValueError(f"{path}:{line_number}: Gemma row lacks node_a/node_b")
        probabilities = row.get("probabilities")
        if isinstance(probabilities, dict):
            same = probabilities.get("SAME")
        elif isinstance(probabilities, list) and len(probabilities) == 3:
            same = probabilities[2]
        else:
            raise ValueError(f"{path}:{line_number}: invalid probabilities")
        if isinstance(same, bool) or not isinstance(same, (int, float)) or not math.isfinite(same):
            raise ValueError(f"{path}:{line_number}: invalid SAME probability")
        if float(same) >= threshold:
            result.add(_pair(str(row["node_a"]), str(row["node_b"])))
    return result


def _stratum(candidate_same: bool, reference_same: bool | None, high_same: bool) -> str:
    if reference_same is None:
        base = "candidate_same" if candidate_same else "candidate_different"
    elif candidate_same and reference_same:
        base = "both_same"
    elif candidate_same:
        base = "candidate_only"
    elif reference_same:
        base = "reference_only"
    else:
        base = "both_different"
    return base + ("_gemma_high" if high_same and not candidate_same else "")


def prepare(
    *,
    task: str,
    level: str,
    candidate_path: str | Path,
    nodes_path: str | Path,
    protocol_path: str | Path,
    output_dir: str | Path,
    reference_path: str | Path | None = None,
    gemma_scores_path: str | Path | None = None,
    gemma_same_threshold: float = 0.5,
    sample_per_stratum: int = 200,
    per_shard: int = 50,
    seed: int = 20260714,
) -> dict[str, Any]:
    """Freeze a blind audit sampled only after the candidate partition exists."""
    if level not in VALID_LEVELS:
        raise ValueError(f"invalid hierarchy level: {level}")
    if sample_per_stratum < 1 or per_shard < 1:
        raise ValueError("sample sizes must be positive")
    if not 0 <= gemma_same_threshold <= 1:
        raise ValueError("Gemma SAME threshold must be in [0, 1]")
    candidate_file = Path(candidate_path).expanduser().resolve()
    nodes_file = Path(nodes_path).expanduser().resolve()
    protocol_file = Path(protocol_path).expanduser().resolve()
    reference_file = Path(reference_path).expanduser().resolve() if reference_path else None
    scores_file = Path(gemma_scores_path).expanduser().resolve() if gemma_scores_path else None
    for path in (candidate_file, nodes_file, protocol_file, reference_file, scores_file):
        if path is not None and not path.is_file():
            raise FileNotFoundError(path)
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)

    candidate = _load_partition(candidate_file)
    reference = _load_partition(reference_file) if reference_file else None
    node_rows = _load_jsonl(nodes_file)
    nodes = {str(row.get("node_id")): row for row in node_rows if row.get("node_id") is not None}
    if len(nodes) != len(node_rows):
        raise ValueError("node inventory has duplicate or missing node_id")
    expected = set(nodes)
    if set(candidate) != expected:
        raise ValueError("candidate partition does not exactly cover the frozen node inventory")
    if reference is not None and set(reference) != expected:
        raise ValueError("reference partition does not exactly cover the frozen node inventory")

    candidate_sha = _file_sha256(str(candidate_file))
    high_pairs = _gemma_high_pairs(scores_file, gemma_same_threshold)
    unknown = {node for pair in high_pairs for node in pair} - expected
    if unknown:
        raise ValueError(f"Gemma scores reference unknown nodes: {sorted(unknown)[:5]}")

    populations: dict[str, list[tuple[str, str, bool, bool | None]]] = {}
    ordered = sorted(expected)
    for index, a in enumerate(ordered):
        for b in ordered[index + 1 :]:
            candidate_same = candidate[a] == candidate[b]
            reference_same = None if reference is None else reference[a] == reference[b]
            stratum = _stratum(candidate_same, reference_same, (a, b) in high_pairs)
            populations.setdefault(stratum, []).append((a, b, candidate_same, reference_same))

    selected: list[dict[str, Any]] = []
    stratum_counts: dict[str, dict[str, int | float]] = {}
    for stratum, population in sorted(populations.items()):
        ranked = sorted(
            population,
            key=lambda row: hashlib.sha256(
                f"{seed}|{candidate_sha}|{stratum}|{row[0]}|{row[1]}".encode()
            ).hexdigest(),
        )
        sample = ranked[: min(sample_per_stratum, len(ranked))]
        stratum_counts[stratum] = {
            "population": len(population),
            "sample": len(sample),
            "inclusion_probability": len(sample) / len(population),
        }
        for a, b, candidate_same, reference_same in sample:
            selected.append(
                {
                    "pair_id": _pair_id(task, level, candidate_sha, a, b),
                    "node_a": a,
                    "node_b": b,
                    "stratum": stratum,
                    "candidate_same": candidate_same,
                    "reference_same": reference_same,
                    "weight": len(population) / len(sample),
                }
            )
    selected.sort(key=lambda row: hashlib.sha256(f"blind|{seed}|{row['pair_id']}".encode()).hexdigest())

    destination.mkdir(parents=True)
    payload_dir = destination / "payloads"
    payload_dir.mkdir()
    blind_rows = [
        {
            "pair_id": row["pair_id"],
            "task": task,
            "level": level,
            "concept_a": _node_text(nodes[row["node_a"]]),
            "concept_b": _node_text(nodes[row["node_b"]]),
        }
        for row in selected
    ]
    audit_path = destination / "audit.jsonl"
    key_path = destination / "key.jsonl"
    audit_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in blind_rows))
    key_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in selected))
    shards = []
    for start in range(0, len(blind_rows), per_shard):
        path = payload_dir / f"audit_{start // per_shard:03d}.jsonl"
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in blind_rows[start : start + per_shard])
        )
        shards.append({"path": str(path), "sha256": _file_sha256(str(path))})

    manifest: dict[str, Any] = {
        "schema_version": VERSION,
        "task": task,
        "level": level,
        "seed": seed,
        "candidate": {"path": str(candidate_file), "sha256": candidate_sha},
        "nodes": {"path": str(nodes_file), "sha256": _file_sha256(str(nodes_file))},
        "protocol": {"path": str(protocol_file), "sha256": _file_sha256(str(protocol_file))},
        "reference": (
            {"path": str(reference_file), "sha256": _file_sha256(str(reference_file))}
            if reference_file
            else None
        ),
        "gemma_scores": (
            {"path": str(scores_file), "sha256": _file_sha256(str(scores_file)), "same_threshold": gemma_same_threshold}
            if scores_file
            else None
        ),
        "audit": {"path": str(audit_path), "sha256": _file_sha256(str(audit_path))},
        "key": {"path": str(key_path), "sha256": _file_sha256(str(key_path))},
        "payloads": shards,
        "strata": stratum_counts,
        "n_pairs": len(selected),
        "sampling": "deterministic uniform-within-stratum sample drawn after candidate freeze",
        "blindness": "judge payload omits partition IDs, assignments, predictions, strata, and weights",
        "judge_policy": "independent Sonnet and GPT-5; third frontier pass only on disagreements",
        "semantic_truth": "LLM judgments only; code performs no semantic labeling",
        "vote_schema": {"pair_id": "string", "score": "strict integer 0|1|2"},
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _votes(path: Path, expected: set[str], *, allow_subset: bool = False) -> dict[str, int]:
    result: dict[str, int] = {}
    malformed = 0
    for row in _load_jsonl(path):
        pair_id, score = row.get("pair_id"), row.get("score")
        if (
            set(row) != {"pair_id", "score"}
            or pair_id not in expected
            or pair_id in result
            or type(score) is not int
            or score not in (0, 1, 2)
        ):
            malformed += 1
            continue
        result[str(pair_id)] = score
    if malformed or (not allow_subset and set(result) != expected):
        raise ValueError(
            f"invalid vote file {path}: malformed={malformed} missing={len(expected - set(result))}"
        )
    return result


def _weighted_metrics(rows: Sequence[Mapping[str, Any]], prediction_key: str) -> dict[str, float | None]:
    cells = Counter()
    for row in rows:
        truth = bool(row["truth_same"])
        predicted = bool(row[prediction_key])
        cells[(truth, predicted)] += float(row["weight"])
    tp, fp = cells[(True, True)], cells[(False, True)]
    fn, tn = cells[(True, False)], cells[(False, False)]
    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    observed = (tp + tn) / total if total else 0.0
    truth_pos = (tp + fn) / total if total else 0.0
    pred_pos = (tp + fp) / total if total else 0.0
    expected = truth_pos * pred_pos + (1 - truth_pos) * (1 - pred_pos)
    # Cohen kappa is undefined, not zero, when both marginals are constant.  Treating that
    # degenerate audit as chance-level agreement would make an uninformative cell look scored.
    kappa = (observed - expected) / (1 - expected) if expected < 1 else None
    return {
        "precision": precision,
        "recall": recall,
        "same_f1": f1,
        "accuracy": observed,
        "cohen_kappa": kappa,
        "estimated_tp": tp,
        "estimated_fp": fp,
        "estimated_fn": fn,
        "estimated_tn": tn,
    }


def _percentile(values: Sequence[float], p: float) -> float:
    ordered = sorted(values)
    location = p * (len(ordered) - 1)
    low, high = math.floor(location), math.ceil(location)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - location) + ordered[high] * (location - low)


def summarize(
    *,
    manifest_path: str | Path,
    votes_a_path: str | Path,
    votes_b_path: str | Path,
    tiebreak_votes_path: str | Path | None,
    report_path: str | Path,
    bootstrap_samples: int = 2000,
    seed: int = 20260714,
) -> dict[str, Any]:
    """Adjudicate the two-family panel and compute paired weighted map metrics."""
    if bootstrap_samples < 100:
        raise ValueError("at least 100 bootstrap samples are required")
    manifest_file = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != VERSION:
        raise ValueError("unsupported audit manifest")
    for field in ("candidate", "nodes", "protocol", "audit", "key"):
        ref = manifest[field]
        if _file_sha256(ref["path"]) != ref["sha256"]:
            raise ValueError(f"frozen {field} changed")
    for optional in ("reference", "gemma_scores"):
        ref = manifest.get(optional)
        if ref and _file_sha256(ref["path"]) != ref["sha256"]:
            raise ValueError(f"frozen {optional} changed")
    for ref in manifest["payloads"]:
        if _file_sha256(ref["path"]) != ref["sha256"]:
            raise ValueError("frozen audit payload changed")

    keys = _load_jsonl(Path(manifest["key"]["path"]))
    by_id = {str(row["pair_id"]): row for row in keys}
    if len(by_id) != len(keys):
        raise ValueError("duplicate pair IDs in audit key")
    key_by_stratum: dict[str, list[dict[str, Any]]] = {}
    for row in keys:
        stratum = row.get("stratum")
        weight = row.get("weight")
        if (not isinstance(stratum, str) or not stratum
                or type(row.get("candidate_same")) is not bool
                or (manifest.get("reference") is None and row.get("reference_same") is not None)
                or (manifest.get("reference") is not None
                    and type(row.get("reference_same")) is not bool)
                or isinstance(weight, bool) or not isinstance(weight, (int, float))
                or not math.isfinite(float(weight)) or float(weight) <= 0):
            raise ValueError("invalid prediction, stratum, or weight in frozen audit key")
        key_by_stratum.setdefault(stratum, []).append(row)
    strata_manifest = manifest.get("strata")
    if not isinstance(strata_manifest, dict) or set(key_by_stratum) != set(strata_manifest):
        raise ValueError("audit key strata do not match frozen sampling manifest")
    for stratum, rows in key_by_stratum.items():
        specification = strata_manifest[stratum]
        population, sample = specification.get("population"), specification.get("sample")
        if (type(population) is not int or type(sample) is not int
                or population < sample or sample < 1 or len(rows) != sample):
            raise ValueError(f"invalid frozen sample counts for stratum {stratum}")
        expected_weight = population / sample
        if any(not math.isclose(float(row["weight"]), expected_weight,
                                rel_tol=1e-12, abs_tol=1e-12) for row in rows):
            raise ValueError(f"invalid sampling weight for stratum {stratum}")
    expected = set(by_id)
    votes_a_file = Path(votes_a_path).expanduser().resolve()
    votes_b_file = Path(votes_b_path).expanduser().resolve()
    if votes_a_file == votes_b_file:
        raise ValueError("independent judge families cannot use the same vote file")
    for vote_file in (votes_a_file, votes_b_file):
        if not vote_file.is_file():
            raise FileNotFoundError(vote_file)
    votes_a_sha = _file_sha256(str(votes_a_file))
    votes_b_sha = _file_sha256(str(votes_b_file))
    votes_a = _votes(votes_a_file, expected)
    votes_b = _votes(votes_b_file, expected)
    # Fail closed if a concurrently running writer changed either vote file while it was read.
    if (_file_sha256(str(votes_a_file)) != votes_a_sha
            or _file_sha256(str(votes_b_file)) != votes_b_sha):
        raise ValueError("judge vote file changed while it was being read")
    disagreements = {pair_id for pair_id in expected if votes_a[pair_id] != votes_b[pair_id]}
    tiebreak_file = None
    tiebreak_sha = None
    if disagreements:
        if not tiebreak_votes_path:
            raise ValueError(f"third-judge votes required for {len(disagreements)} disagreements")
        tiebreak_file = Path(tiebreak_votes_path).expanduser().resolve()
        if not tiebreak_file.is_file():
            raise FileNotFoundError(tiebreak_file)
        if tiebreak_file in (votes_a_file, votes_b_file):
            raise ValueError("tiebreak judge cannot reuse a primary judge vote file")
        tiebreak_sha = _file_sha256(str(tiebreak_file))
        tiebreak = _votes(tiebreak_file, expected, allow_subset=True)
        if _file_sha256(str(tiebreak_file)) != tiebreak_sha:
            raise ValueError("tiebreak vote file changed while it was being read")
        if set(tiebreak) != disagreements:
            raise ValueError(
                f"tiebreak must cover exactly disagreements: missing={len(disagreements-set(tiebreak))} "
                f"extra={len(set(tiebreak)-disagreements)}"
            )
    else:
        tiebreak = {}
        if tiebreak_votes_path:
            tiebreak_file = Path(tiebreak_votes_path).expanduser().resolve()
            if not tiebreak_file.is_file():
                raise FileNotFoundError(tiebreak_file)
            tiebreak_sha = _file_sha256(str(tiebreak_file))
            supplied = _votes(tiebreak_file, expected, allow_subset=True)
            if _file_sha256(str(tiebreak_file)) != tiebreak_sha:
                raise ValueError("tiebreak vote file changed while it was being read")
            if supplied:
                raise ValueError("tiebreak votes supplied despite no disagreements")

    scored = []
    for pair_id, row in by_id.items():
        final_score = votes_a[pair_id] if votes_a[pair_id] == votes_b[pair_id] else tiebreak[pair_id]
        scored.append({**row, "truth_same": final_score == 2, "adjudicated_score": final_score})
    candidate = _weighted_metrics(scored, "candidate_same")
    has_reference = manifest.get("reference") is not None
    reference = _weighted_metrics(scored, "reference_same") if has_reference else None
    if candidate["cohen_kappa"] is None or (
            reference is not None and reference["cohen_kappa"] is None):
        raise ValueError("Cohen kappa is undefined for the adjudicated audit")

    comparison = None
    if reference is not None:
        rng = random.Random(seed)
        by_stratum: dict[str, list[dict[str, Any]]] = {}
        for row in scored:
            by_stratum.setdefault(str(row["stratum"]), []).append(row)
        delta_kappa, delta_f1 = [], []
        for _ in range(bootstrap_samples):
            sample = []
            for stratum, rows in by_stratum.items():
                population = strata_manifest[stratum]["population"]
                if len(rows) == population:
                    # This stratum is a census of its finite pair population.  Resampling it
                    # would manufacture sampling variance where none exists.
                    sample.extend(rows)
                else:
                    sample.extend(rows[rng.randrange(len(rows))] for _ in rows)
            cm = _weighted_metrics(sample, "candidate_same")
            rm = _weighted_metrics(sample, "reference_same")
            if cm["cohen_kappa"] is not None and rm["cohen_kappa"] is not None:
                delta_kappa.append(cm["cohen_kappa"] - rm["cohen_kappa"])
            delta_f1.append(cm["same_f1"] - rm["same_f1"])
        if len(delta_kappa) < math.ceil(bootstrap_samples * 0.9):
            raise ValueError("too many paired bootstrap replicates have undefined Cohen kappa")
        dk = candidate["cohen_kappa"] - reference["cohen_kappa"]
        df = candidate["same_f1"] - reference["same_f1"]
        dp = candidate["precision"] - reference["precision"]
        dr = candidate["recall"] - reference["recall"]
        kappa_ci = [_percentile(delta_kappa, 0.025), _percentile(delta_kappa, 0.975)]
        f1_ci = [_percentile(delta_f1, 0.025), _percentile(delta_f1, 0.975)]
        promoted = (
            kappa_ci[0] > 0
            and df > 0
            and candidate["precision"] >= 0.5
            and candidate["recall"] >= 0.5
            and dp >= -0.02
            and dr >= -0.02
        )
        comparison = {
            "delta_cohen_kappa": dk,
            "delta_cohen_kappa_ci95": kappa_ci,
            "delta_same_f1": df,
            "delta_same_f1_ci95": f1_ci,
            "delta_precision": dp,
            "delta_recall": dr,
            "promotion_gate": {
                "passes_metric_gates": promoted,
                "requires_integrity_and_large_group_certificates": True,
                "thresholds": {
                    "delta_kappa_ci95_lower_strictly_positive": True,
                    "delta_same_f1_positive": True,
                    "minimum_candidate_precision": 0.5,
                    "minimum_candidate_recall": 0.5,
                    "maximum_precision_regression": 0.02,
                    "maximum_recall_regression": 0.02,
                },
            },
        }

    n = len(expected)
    agreement = sum(votes_a[pair_id] == votes_b[pair_id] for pair_id in expected) / n if n else 0.0
    judge_rows = [
        {
            **by_id[pair_id],
            "truth_same": votes_a[pair_id] == 2,
            "judge_b_same": votes_b[pair_id] == 2,
        }
        for pair_id in expected
    ]
    judge_reliability = _weighted_metrics(judge_rows, "judge_b_same")
    reliability_kappa = judge_reliability["cohen_kappa"]
    at_measurable_ceiling = bool(
        reliability_kappa is not None
        and candidate["precision"] >= 0.5
        and candidate["recall"] >= 0.5
        and candidate["cohen_kappa"] >= reliability_kappa - 0.03
    )
    report = {
        "schema_version": "postfreeze-hierarchy-audit-report-v1",
        "manifest": {"path": str(manifest_file), "sha256": _file_sha256(str(manifest_file))},
        "n_pairs": n,
        "adjudication": {
            "judge_a_family": "sonnet",
            "judge_b_family": "gpt5",
            "vote_artifacts": {
                "judge_a": {"path": str(votes_a_file), "sha256": votes_a_sha},
                "judge_b": {"path": str(votes_b_file), "sha256": votes_b_sha},
                "tiebreak": (
                    {"path": str(tiebreak_file), "sha256": tiebreak_sha}
                    if tiebreak_file is not None
                    else None
                ),
            },
            "agreement": agreement,
            "binary_same_reliability": judge_reliability,
            "n_disagreements": len(disagreements),
            "tiebreak_used_only_for_disagreements": True,
        },
        "candidate": candidate,
        "ceiling_assessment": {
            "at_measurable_ceiling": at_measurable_ceiling,
            "noninferiority_margin_kappa": 0.03,
            "benchmark": "weighted Sonnet-vs-GPT-5 binary SAME Cohen kappa on the same audit",
            "requires_candidate_precision_and_recall_at_least": 0.5,
        },
        "reference": reference,
        "paired_comparison": comparison,
        "semantic_truth": "independent frontier-LLM judgments only",
    }
    destination = Path(report_path).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--task", required=True)
    prep.add_argument("--level", required=True, choices=sorted(VALID_LEVELS))
    prep.add_argument("--candidate", required=True)
    prep.add_argument("--reference")
    prep.add_argument("--nodes", required=True)
    prep.add_argument("--protocol", required=True)
    prep.add_argument("--gemma-scores")
    prep.add_argument("--gemma-same-threshold", type=float, default=0.5)
    prep.add_argument("--sample-per-stratum", type=int, default=200)
    prep.add_argument("--per-shard", type=int, default=50)
    prep.add_argument("--output-dir", required=True)
    prep.add_argument("--seed", type=int, default=20260714)
    score = sub.add_parser("summarize")
    score.add_argument("--manifest", required=True)
    score.add_argument("--votes-a", required=True)
    score.add_argument("--votes-b", required=True)
    score.add_argument("--tiebreak-votes")
    score.add_argument("--report", required=True)
    score.add_argument("--bootstrap-samples", type=int, default=2000)
    score.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(
            task=args.task,
            level=args.level,
            candidate_path=args.candidate,
            reference_path=args.reference,
            nodes_path=args.nodes,
            protocol_path=args.protocol,
            gemma_scores_path=args.gemma_scores,
            gemma_same_threshold=args.gemma_same_threshold,
            sample_per_stratum=args.sample_per_stratum,
            per_shard=args.per_shard,
            output_dir=args.output_dir,
            seed=args.seed,
        )
    else:
        result = summarize(
            manifest_path=args.manifest,
            votes_a_path=args.votes_a,
            votes_b_path=args.votes_b,
            tiebreak_votes_path=args.tiebreak_votes,
            report_path=args.report,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
