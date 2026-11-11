"""Replicated LLM truth and recall gates for versioned upper-level inventories.

Canonical ``build_level`` evaluations are tied to canonical parent artifacts.  R2/R3 variants
instead carry their own node inventories, so this module freezes an explicit inventory, draws the
same neighbor/random evaluation mixture, and requires two complete blind LLM passes plus a third
blind adjudication of every ordinal disagreement.  Code only samples and computes statistics; it
never supplies a semantic score.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from .build_level import OUT, _file_sha256, rep_text
from .upper_precision_audit import _load_candidate_partition, _wilson


ROOT = Path(OUT) / "versioned_level_eval"
VERSION = "versioned-level-eval-v1"


def _stem(task: str, level: str, tag: str) -> str:
    return f"{task}_{level}_{tag}"


def _jsonl(path: Path) -> list[dict]:
    rows = []
    for index, line in enumerate(path.read_text().splitlines()):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"malformed JSON at {path}:{index + 1}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"non-object row at {path}:{index + 1}")
        rows.append(row)
    return rows


def _load_nodes(path: Path) -> list[dict]:
    rows = _jsonl(path)
    ids = [row.get("node_id") for row in rows]
    if (not rows or any(not isinstance(node_id, str) or not node_id for node_id in ids)
            or len(ids) != len(set(ids))):
        raise ValueError(f"invalid or duplicate node_id in {path}")
    return rows


def _pid(task: str, level: str, tag: str, inventory_sha: str, a: str, b: str) -> str:
    left, right = sorted((a, b))
    return hashlib.sha1(
        f"{VERSION}||{task}||{level}||{tag}||{inventory_sha}||{left}||{right}".encode()
    ).hexdigest()[:16]


def _vote_files(path: str) -> list[Path]:
    source = Path(path).expanduser().resolve()
    if source.is_file():
        return [source]
    if source.is_dir():
        return sorted(item for item in source.iterdir()
                      if item.is_file() and item.suffix in (".json", ".jsonl"))
    raise ValueError(f"vote source does not exist: {source}")


def _load_votes(path: str | None, expected: set[str]) -> dict[str, int]:
    if not expected and path is None:
        return {}
    files = _vote_files(path or "")
    votes: dict[str, int] = {}
    malformed = 0
    for file_path in files:
        payload = _jsonl(file_path) if file_path.suffix == ".jsonl" else json.loads(
            file_path.read_text())
        rows = payload if isinstance(payload, list) else [payload]
        for row in rows:
            if not isinstance(row, dict):
                malformed += 1
                continue
            pair_id, score = row.get("pair_id"), row.get("score")
            if (set(row) != {"pair_id", "score"} or pair_id not in expected
                    or pair_id in votes or type(score) is not int or score not in (0, 1, 2)):
                malformed += 1
                continue
            votes[pair_id] = score
    missing = expected - set(votes)
    if missing or malformed:
        raise ValueError(f"invalid LLM votes: missing={len(missing)} malformed={malformed}")
    return votes


def _vote_sha(votes: dict[str, int]) -> str:
    frozen = json.dumps(dict(sorted(votes.items())), separators=(",", ":"))
    return hashlib.sha256(frozen.encode()).hexdigest()


def _validate_manifest_inputs(manifest: dict) -> None:
    for field in ("partition", "nodes", "protocol", "audit", "key"):
        if _file_sha256(manifest[f"{field}_path"]) != manifest[f"{field}_sha256"]:
            raise ValueError(f"frozen {field} input changed")


def prepare(task: str, level: str, tag: str, partition_path: str, nodes_path: str,
            protocol_path: str, n_pairs: int = 900, per_agent: int = 150) -> dict:
    """Freeze a 50% representation-neighbor / 50% random pair sample."""
    if n_pairs < 1 or per_agent < 1:
        raise ValueError("n_pairs and per_agent must be positive")
    partition_file = Path(partition_path).expanduser().resolve()
    nodes_file = Path(nodes_path).expanduser().resolve()
    protocol_file = Path(protocol_path).expanduser().resolve()
    partition = _load_candidate_partition(partition_file)
    nodes = _load_nodes(nodes_file)
    by_id = {row["node_id"]: row for row in nodes}
    if set(partition) != set(by_id):
        raise ValueError(f"[{task}/{level}/{tag}] partition/node coverage mismatch")
    ids = list(by_id)
    max_pairs = len(ids) * (len(ids) - 1) // 2
    if len(ids) < 2 or n_pairs > max_pairs:
        raise ValueError(f"requested {n_pairs} pairs from {len(ids)} nodes ({max_pairs} maximum)")
    inventory_sha = _file_sha256(str(nodes_file))

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors

    reps = [rep_text(by_id[node_id]) for node_id in ids]
    matrix = TfidfVectorizer(min_df=1, max_features=40000,
                             sublinear_tf=True).fit_transform(reps)
    k = min(11, len(ids))
    dist, neighbors = NearestNeighbors(n_neighbors=k, metric="cosine").fit(
        matrix).kneighbors(matrix)
    pool: dict[tuple[str, str], dict] = {}
    for i, node_id in enumerate(ids):
        for position in range(1, k):
            other = ids[int(neighbors[i][position])]
            pair = tuple(sorted((node_id, other)))
            similarity = 1.0 - float(dist[i][position])
            # Route the fixed neighbor half even when an abstract level is lexically diffuse.
            # A cosine threshold would silently turn R3 into a mostly-random battery and destroy
            # the matched 50/50 design; similarity is routing metadata, never semantic truth.
            if pair not in pool:
                pool[pair] = {"stratum": "highsim", "tfidf_cos": round(similarity, 3)}
    highsim = sorted(pool, key=lambda pair: hashlib.sha256(
        f"highsim||{VERSION}||{task}||{level}||{tag}||{pair[0]}||{pair[1]}".encode()
    ).hexdigest())[:n_pairs // 2]
    selected = set(highsim)
    seed = int(hashlib.sha256(
        f"random||{VERSION}||{task}||{level}||{tag}||{inventory_sha}".encode()
    ).hexdigest()[:16], 16)
    rng = random.Random(seed)
    while len(selected) < n_pairs:
        a, b = rng.sample(ids, 2)
        selected.add(tuple(sorted((a, b))))

    rows = []
    key = {}
    highsim_set = set(highsim)
    for a, b in selected:
        pair_id = _pid(task, level, tag, inventory_sha, a, b)
        stratum = "highsim" if (a, b) in highsim_set else "random"
        row = {"pair_id": pair_id, "concept_a": rep_text(by_id[a]),
               "concept_b": rep_text(by_id[b])}
        rows.append(row)
        key[pair_id] = {"node_a": a, "node_b": b, "stratum": stratum,
                        "tfidf_cos": pool.get((a, b), {}).get("tfidf_cos")}
    rows.sort(key=lambda row: hashlib.sha256(
        f"order||{VERSION}||{row['pair_id']}".encode()).hexdigest())

    ROOT.mkdir(parents=True, exist_ok=True)
    payload_dir = ROOT / "payloads"
    payload_dir.mkdir(exist_ok=True)
    stem = _stem(task, level, tag)
    for old in payload_dir.glob(f"{stem}_[0-9][0-9][0-9].jsonl"):
        old.unlink()
    audit_path = ROOT / f"{stem}_audit.jsonl"
    key_path = ROOT / f"{stem}_key.json"
    audit_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    key_path.write_text(json.dumps(key, indent=2) + "\n")
    payload_paths = []
    for start in range(0, len(rows), per_agent):
        output = payload_dir / f"{stem}_{start // per_agent:03d}.jsonl"
        output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                  for row in rows[start:start + per_agent]))
        payload_paths.append(str(output.resolve()))
    manifest = {
        "version": VERSION, "task": task, "level": level, "tag": tag,
        "partition_path": str(partition_file),
        "partition_sha256": _file_sha256(str(partition_file)),
        "nodes_path": str(nodes_file), "nodes_sha256": inventory_sha,
        "protocol_path": str(protocol_file),
        "protocol_sha256": _file_sha256(str(protocol_file)),
        "audit_path": str(audit_path.resolve()), "audit_sha256": _file_sha256(str(audit_path)),
        "key_path": str(key_path.resolve()), "key_sha256": _file_sha256(str(key_path)),
        "n_nodes": len(ids), "n_pairs": len(rows), "n_highsim": len(highsim),
        "n_random": len(rows) - len(highsim), "payload_paths": payload_paths,
        "sampling": "deterministic 50% TF-IDF-neighbor / 50% random pair mixture",
        "semantic_truth": "two independent LLM passes plus blind third-LLM disagreement adjudication",
        "vote_schema": {"pair_id": "string", "score": "strict integer 0|1|2"},
    }
    (ROOT / f"{stem}_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def stage_adjudication(task: str, level: str, tag: str, votes_a_path: str,
                       votes_b_path: str, per_agent: int = 150) -> dict:
    stem = _stem(task, level, tag)
    manifest = json.loads((ROOT / f"{stem}_manifest.json").read_text())
    _validate_manifest_inputs(manifest)
    key = json.loads(Path(manifest["key_path"]).read_text())
    expected = set(key)
    votes_a = _load_votes(votes_a_path, expected)
    votes_b = _load_votes(votes_b_path, expected)
    disagreements = sorted(pair_id for pair_id in expected
                           if votes_a[pair_id] != votes_b[pair_id])
    audit = {row["pair_id"]: row for row in _jsonl(Path(manifest["audit_path"]))}
    payload_dir = ROOT / "adjudication_payloads"
    payload_dir.mkdir(exist_ok=True)
    for old in payload_dir.glob(f"{stem}_[0-9][0-9][0-9].jsonl"):
        old.unlink()
    paths = []
    rows = [audit[pair_id] for pair_id in disagreements]
    for start in range(0, len(rows), per_agent):
        output = payload_dir / f"{stem}_{start // per_agent:03d}.jsonl"
        output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                  for row in rows[start:start + per_agent]))
        paths.append(str(output.resolve()))
    result = {
        "version": VERSION, "task": task, "level": level, "tag": tag,
        "base_manifest_sha256": _file_sha256(str(ROOT / f"{stem}_manifest.json")),
        "votes_a_path": str(Path(votes_a_path).expanduser().resolve()),
        "votes_a_sha256": _vote_sha(votes_a),
        "votes_b_path": str(Path(votes_b_path).expanduser().resolve()),
        "votes_b_sha256": _vote_sha(votes_b),
        "n_disagreements": len(disagreements), "pair_ids": disagreements,
        "payload_paths": paths,
        "instruction": "third LLM judges blind concepts; prior scores are not exposed",
    }
    (ROOT / f"{stem}_adjudication_manifest.json").write_text(
        json.dumps(result, indent=2) + "\n")
    return result


def _binary_kappa(left: list[bool], right: list[bool]) -> float | None:
    if not left:
        return None
    observed = sum(a == b for a, b in zip(left, right)) / len(left)
    pa, pb = sum(left) / len(left), sum(right) / len(right)
    expected = pa * pb + (1 - pa) * (1 - pb)
    return (observed - expected) / (1 - expected) if expected < 1 else None


def summarize(task: str, level: str, tag: str, votes_a_path: str,
              votes_b_path: str, adjudication_votes_path: str | None) -> dict:
    stem = _stem(task, level, tag)
    manifest_path = ROOT / f"{stem}_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    _validate_manifest_inputs(manifest)
    adjudication = json.loads((ROOT / f"{stem}_adjudication_manifest.json").read_text())
    if (_file_sha256(str(manifest_path)) != adjudication["base_manifest_sha256"]
            or adjudication.get("version") != VERSION):
        raise ValueError("adjudication manifest no longer matches frozen evaluation")
    key = json.loads(Path(manifest["key_path"]).read_text())
    expected = set(key)
    votes_a = _load_votes(votes_a_path, expected)
    votes_b = _load_votes(votes_b_path, expected)
    if (_vote_sha(votes_a) != adjudication["votes_a_sha256"]
            or _vote_sha(votes_b) != adjudication["votes_b_sha256"]):
        raise ValueError("judge A/B votes changed after adjudication was staged")
    disagreement_ids = set(adjudication["pair_ids"])
    votes_c = _load_votes(adjudication_votes_path, disagreement_ids)
    final = {}
    for pair_id in expected:
        if votes_a[pair_id] == votes_b[pair_id]:
            final[pair_id] = votes_a[pair_id]
        else:
            final[pair_id] = sorted((votes_a[pair_id], votes_b[pair_id],
                                     votes_c[pair_id]))[1]

    partition = _load_candidate_partition(Path(manifest["partition_path"]))
    n_same = true_positive = n_colabeled = colabeled_same = 0
    truth_binary = []
    pred_binary = []
    strata: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    for pair_id, row in key.items():
        same = final[pair_id] == 2
        colabeled = partition[row["node_a"]] == partition[row["node_b"]]
        truth_binary.append(same)
        pred_binary.append(colabeled)
        strata[str(row["stratum"])].append((same, colabeled))
        n_same += int(same)
        true_positive += int(same and colabeled)
        n_colabeled += int(colabeled)
        colabeled_same += int(same and colabeled)
    recall = true_positive / n_same if n_same else None
    mixture_precision = colabeled_same / n_colabeled if n_colabeled else None
    sizes = Counter(partition.values())
    n_nodes = sum(sizes.values())
    p0 = (sum(math.comb(size, 2) for size in sizes.values()) /
          math.comb(n_nodes, 2)) if n_nodes >= 2 else None
    chance_corrected = ((recall - p0) / (1 - p0)
                        if recall is not None and p0 not in (None, 1) else None)
    recall_ci = _wilson(true_positive, n_same)
    cc_ci = ([round((bound - p0) / (1 - p0), 3) for bound in recall_ci]
             if recall_ci is not None and p0 not in (None, 1) else None)
    by_stratum = {}
    for stratum, pairs in sorted(strata.items()):
        ns = sum(same for same, _ in pairs)
        nc = sum(co for _, co in pairs)
        tp = sum(same and co for same, co in pairs)
        by_stratum[stratum] = {
            "n": len(pairs), "n_same": ns, "n_colabeled": nc,
            "recall": round(tp / ns, 3) if ns else None,
            "precision_mixture": round(tp / nc, 3) if nc else None,
        }
    a_binary = [votes_a[pair_id] == 2 for pair_id in sorted(expected)]
    b_binary = [votes_b[pair_id] == 2 for pair_id in sorted(expected)]
    result = {
        "version": VERSION, "task": task, "level": level, "tag": tag,
        "n_pairs": len(expected), "n_truth_same": n_same,
        "truth_rule": "ordinal median of two blind LLM votes plus blind third vote on every disagreement",
        "judge_ordinal_exact_agreement": round(sum(votes_a[p] == votes_b[p] for p in expected) /
                                                 len(expected), 3),
        "judge_binary_same_agreement": round(sum(a == b for a, b in zip(a_binary, b_binary)) /
                                               len(expected), 3),
        "judge_binary_same_kappa": (round(value, 3) if
                                      (value := _binary_kappa(a_binary, b_binary)) is not None
                                      else None),
        "n_adjudicated": len(disagreement_ids),
        "recall": round(recall, 3) if recall is not None else None,
        "recall_ci95": recall_ci,
        "p0": round(p0, 4) if p0 is not None else None,
        "chance_corrected_recall": (round(chance_corrected, 3)
                                      if chance_corrected is not None else None),
        "chance_corrected_recall_ci95": cc_ci,
        "precision_mixture": (round(mixture_precision, 3)
                               if mixture_precision is not None else None),
        "precision_scope": "fixed 50% neighbor / 50% random mixture; use upper_precision_audit for global precision",
        "by_stratum": by_stratum,
        "semantic_truth": "LLM judgments only; code performs frozen sampling and arithmetic",
    }
    report_path = ROOT / f"{stem}_report.json"
    report_path.write_text(json.dumps(result, indent=2) + "\n")
    return result
