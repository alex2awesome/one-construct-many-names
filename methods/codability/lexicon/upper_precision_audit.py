"""Blind LLM precision gate for candidate R1/R2/R3 partitions.

The audit population is the set of node pairs that a candidate partition co-labels.  Code draws a
deterministic uniform sample from that population, excluding frozen evaluation and build-verifier
pairs.  It never assigns semantic truth.  Two independent LLM judges supply 0/1/2 relation scores;
the conservative precision estimate counts a pair as SAME only when both judges score it 2.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .build_level import OUT, _file_sha256, nodes_from_level, rep_text


ROOT = Path(OUT) / "upper_precision_audit"
VERSION = "upper-precision-audit-v1"


def _rows(paths: Iterable[Path]):
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if line.strip():
                yield json.loads(line)


def _pair(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((str(a), str(b))))


def _excluded_pairs(task: str, level: str) -> set[tuple[str, str]]:
    paths = [Path(OUT) / f"level_eval_{task}_{level}.jsonl"]
    paths.extend(sorted((Path(OUT) / "level_arbiter").glob(
        f"{task}_{level}_verify_*.jsonl")))
    excluded = set()
    for row in _rows(paths):
        if row.get("node_a") is not None and row.get("node_b") is not None:
            excluded.add(_pair(row["node_a"], row["node_b"]))
    return excluded


def _pid(task: str, level: str, partition_sha: str, a: str, b: str) -> str:
    return hashlib.sha1(
        f"{VERSION}||{task}||{level}||{partition_sha}||{a}||{b}".encode()).hexdigest()[:16]


def _wilson(successes: int, n: int, z: float = 1.959963984540054) -> list[float] | None:
    if not n:
        return None
    p = successes / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(max(0.0, center - half), 3), round(min(1.0, center + half), 3)]


def _stem(task: str, level: str, tag: str = "") -> str:
    return f"{task}_{level}" + (f"_{tag}" if tag else "")


def _load_candidate_partition(path: Path) -> dict[str, str]:
    """Load either a canonical ``partition`` artifact or an LLM ``assignment`` artifact."""
    payload = json.loads(path.read_text())
    raw = payload.get("partition") if isinstance(payload, dict) else None
    if raw is None and isinstance(payload, dict):
        raw = payload.get("assignment")
    if raw is None:
        raw = payload
    if not isinstance(raw, dict):
        raise ValueError(f"candidate partition must be a JSON object: {path}")
    return {str(k): str(v) for k, v in raw.items()}


def prepare(task: str, level: str, partition_path: str, n_pairs: int = 300,
            per_agent: int = 100, tag: str = "", exclude_prior: bool = True,
            protocol_path: str | None = None,
            nodes_path: str | None = None) -> dict:
    """Stage a uniform sample of predicted-positive pairs for blind LLM judgment."""
    if n_pairs < 1 or per_agent < 1:
        raise ValueError("n_pairs and per_agent must be positive")
    path = Path(partition_path).expanduser().resolve()
    partition_sha = _file_sha256(str(path))
    partition = _load_candidate_partition(path)
    custom_nodes = Path(nodes_path).expanduser().resolve() if nodes_path else None
    if custom_nodes is not None:
        if not custom_nodes.exists():
            raise FileNotFoundError(custom_nodes)
        nodes = list(_rows([custom_nodes]))
    else:
        nodes, _ = nodes_from_level(task, level)
    by_id = {str(node["node_id"]): node for node in nodes}
    if len(by_id) != len(nodes):
        raise ValueError(f"[{task}/{level}] duplicate or missing node_id in audit inventory")
    if set(partition) != set(by_id):
        raise ValueError(f"[{task}/{level}] partition/node coverage mismatch")

    members: dict[str, list[str]] = defaultdict(list)
    for node_id, group_id in partition.items():
        members[group_id].append(node_id)
    excluded = _excluded_pairs(task, level) if exclude_prior else set()
    candidates = []
    n_population = 0
    for group_id, ids in sorted(members.items()):
        ids = sorted(ids)
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                pair = (a, b)
                if pair in excluded:
                    continue
                n_population += 1
                rank = hashlib.sha256(
                    f"sample||{VERSION}||{task}||{level}||{partition_sha}||{a}||{b}".encode()
                ).hexdigest()
                candidates.append((rank, group_id, a, b))
    chosen = sorted(candidates)[:min(n_pairs, len(candidates))]
    protocol = (Path(protocol_path).expanduser().resolve() if protocol_path else
                (Path(OUT) / f"ARBITER_PROTOCOL_{level}.txt").resolve())
    if not protocol.exists():
        raise FileNotFoundError(protocol)

    ROOT.mkdir(parents=True, exist_ok=True)
    payload_dir = ROOT / "payloads"
    payload_dir.mkdir(exist_ok=True)
    stem = _stem(task, level, tag)
    for old in payload_dir.glob(f"{stem}_*.jsonl"):
        old.unlink()
    audit_path = ROOT / f"{stem}_audit.jsonl"
    key_path = ROOT / f"{stem}_key.json"
    rows = []
    key = {}
    for _, group_id, a, b in chosen:
        pid = _pid(task, level, partition_sha, a, b)
        rows.append({"pair_id": pid, "concept_a": rep_text(by_id[a]),
                     "concept_b": rep_text(by_id[b])})
        key[pid] = {"node_a": a, "node_b": b, "group_id": group_id}
    audit_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    key_path.write_text(json.dumps(key, indent=2) + "\n")
    payload_paths = []
    for start in range(0, len(rows), per_agent):
        payload = payload_dir / f"{stem}_{start // per_agent:03d}.jsonl"
        payload.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                   for row in rows[start:start + per_agent]))
        payload_paths.append(str(payload.resolve()))
    manifest = {
        "version": VERSION, "task": task, "level": level, "tag": tag,
        "partition_path": str(path), "partition_sha256": partition_sha,
        "protocol_path": str(protocol), "protocol_sha256": _file_sha256(str(protocol)),
        "n_nodes": len(partition), "n_groups": len(members),
        "n_colabeled_pair_population_excluding_prior_measurement": n_population,
        "n_pairs": len(rows), "n_shards": len(payload_paths),
        "n_prior_pairs_excluded": len(excluded),
        "exclude_prior_measurement_pairs": exclude_prior,
        "audit_path": str(audit_path.resolve()),
        "audit_sha256": _file_sha256(str(audit_path)),
        "key_path": str(key_path.resolve()), "key_sha256": _file_sha256(str(key_path)),
        "payload_paths": payload_paths,
        "sampling": "uniform deterministic hash sample of co-labeled pairs",
        "semantic_truth": "independent LLM judges only; code performs sampling and arithmetic",
        "vote_schema": {"pair_id": "string", "score": "strict integer 0|1|2"},
    }
    if custom_nodes is not None:
        manifest["nodes_path"] = str(custom_nodes)
        manifest["nodes_sha256"] = _file_sha256(str(custom_nodes))
    (ROOT / f"{stem}_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _load_votes(path: str, expected: set[str]) -> dict[str, int]:
    votes = {}
    malformed = 0
    for row in _rows([Path(path)]):
        pid, score = row.get("pair_id"), row.get("score")
        if (set(row) != {"pair_id", "score"} or pid not in expected or pid in votes
                or type(score) is not int or score not in (0, 1, 2)):
            malformed += 1
            continue
        votes[pid] = score
    missing = expected - set(votes)
    if missing or malformed:
        raise ValueError(f"invalid precision votes: missing={len(missing)} malformed={malformed}")
    return votes


def summarize(task: str, level: str, votes_a_path: str, votes_b_path: str,
              precision_floor: float = .80, tag: str = "") -> dict:
    """Report judge-specific and dual-confirmed precision; fail closed on incomplete votes."""
    stem = _stem(task, level, tag)
    manifest = json.loads((ROOT / f"{stem}_manifest.json").read_text())
    for field in ("partition", "protocol", "audit", "key"):
        if _file_sha256(manifest[f"{field}_path"]) != manifest[f"{field}_sha256"]:
            raise ValueError(f"[{task}/{level}] frozen {field} input changed")
    if manifest.get("nodes_path") and _file_sha256(manifest["nodes_path"]) != manifest.get(
            "nodes_sha256"):
        raise ValueError(f"[{task}/{level}] frozen custom node inventory changed")
    expected = set(json.loads(Path(manifest["key_path"]).read_text()))
    a = _load_votes(votes_a_path, expected)
    b = _load_votes(votes_b_path, expected)
    n = len(expected)

    def stats(values: list[bool]) -> dict:
        same = sum(values)
        return {"same": same, "n": n, "precision": round(same / n, 3) if n else None,
                "ci95": _wilson(same, n)}

    av = [a[pid] == 2 for pid in expected]
    bv = [b[pid] == 2 for pid in expected]
    dual = [x and y for x, y in zip(av, bv)]
    po = sum(x == y for x, y in zip(av, bv)) / n if n else None
    pa = sum(av) / n if n else None
    pb = sum(bv) / n if n else None
    pe = pa * pb + (1 - pa) * (1 - pb) if n else None
    kappa = (po - pe) / (1 - pe) if n and pe < 1 else None
    result = {
        "task": task, "level": level, "tag": tag, "precision_floor": precision_floor,
        "judge_a": stats(av), "judge_b": stats(bv), "dual_confirmed": stats(dual),
        "binary_same_agreement": round(po, 3) if po is not None else None,
        "binary_same_kappa": round(kappa, 3) if kappa is not None else None,
        "passes_floor": bool(n and min(pa, pb, sum(dual) / n) >= precision_floor),
        "promotion_rule": "both individual and dual-confirmed SAME rates meet precision_floor",
        "semantic_truth": "independent LLM votes; no similarity-based labels",
    }
    (ROOT / f"{stem}_report.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
