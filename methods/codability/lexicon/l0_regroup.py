"""Strict-L0 regrouping of v6 source clusters currently coalesced inside L0v3.

This is a precision-repair path for the historical R1-prompt-at-L0 mismatch. Code composes and
validates inventories only. An LLM partitions each current cluster's v6 source clusters under the
frozen L0 same-criterion protocol. Historical L0v3 is never overwritten; apply writes L0v4.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict

from .audit import CACHE, load_partition
from .build_level import OUT, _file_sha256, _load_partition
from .judge import canon_map


ROOT = Path(OUT) / "l0_regroup"
PROTOCOL = Path(OUT) / "CONFIRM_PROTOCOL_L0_V2.txt"


def prepare(task: str, per_agent: int = 25, examples_per_source: int = 4) -> dict:
    ROOT.mkdir(parents=True, exist_ok=True)
    base_path = Path(CACHE) / "match_out" / f"clusters_{task}.json"
    base = {str(k): str(v) for k, v in load_partition(task, str(base_path)).items()}
    current_path = Path(OUT) / f"partition_{task}_L0v3.json"
    current = {str(k): str(v) for k, v in _load_partition(str(current_path)).items()}
    if set(base) != set(current):
        raise ValueError(f"[{task}] v6/current key mismatch")
    base_members: Dict[str, list[str]] = defaultdict(list)
    current_of_base: Dict[str, set[str]] = defaultdict(set)
    for key, source in base.items():
        base_members[source].append(key); current_of_base[source].add(current[key])
    if any(len(x) != 1 for x in current_of_base.values()):
        raise ValueError(f"[{task}] current partition split at least one v6 source cluster")
    sources: Dict[str, list[str]] = defaultdict(list)
    for source, targets in current_of_base.items():
        sources[next(iter(targets))].append(source)
    cmap = canon_map(task)
    rows = []
    for current_id, source_ids in sorted(sources.items()):
        if len(source_ids) < 2:
            continue
        source_rows = []
        for source in sorted(source_ids):
            keys = sorted(base_members[source], key=lambda key: hashlib.sha256(
                f"{task}||{source}||{key}".encode()).hexdigest())[:examples_per_source]
            source_rows.append({"source_cluster_id": source,
                                "criteria": [cmap[k] for k in keys if k in cmap]})
        rows.append({"current_cluster_id": current_id, "n_source_clusters": len(source_rows),
                     "source_clusters": source_rows})
    payload_dir = ROOT / "payloads"; payload_dir.mkdir(exist_ok=True)
    for old in payload_dir.glob(f"{task}_*.jsonl"):
        old.unlink()
    paths = []
    for start in range(0, len(rows), per_agent):
        path = payload_dir / f"{task}_{start//per_agent:03d}.jsonl"
        with path.open("w") as out:
            for row in rows[start:start + per_agent]:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
        paths.append(str(path))
    manifest = {"task": task, "version": "l0-strict-regroup-v1",
                "n_current_clusters_to_regroup": len(rows),
                "n_source_clusters_in_scope": sum(r["n_source_clusters"] for r in rows),
                "n_payload_shards": len(paths), "per_agent": per_agent,
                "protocol_path": str(PROTOCOL), "protocol_sha256": _file_sha256(str(PROTOCOL)),
                "v6_partition_path": str(base_path),
                "v6_partition_sha256": _file_sha256(str(base_path)),
                "current_partition_path": str(current_path),
                "current_partition_sha256": _file_sha256(str(current_path)),
                "output_schema": {"current_cluster_id": "string",
                                  "groups": "list[list[source_cluster_id]]"},
                "semantic_decider": "LLM only; code validates exact nested coverage"}
    (ROOT / f"{task}_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def apply(task: str, votes_glob: str | None = None) -> dict:
    manifest = json.loads((ROOT / f"{task}_manifest.json").read_text())
    if (_file_sha256(manifest["v6_partition_path"]) != manifest["v6_partition_sha256"]
            or _file_sha256(manifest["current_partition_path"]) != manifest["current_partition_sha256"]
            or _file_sha256(manifest["protocol_path"]) != manifest["protocol_sha256"]):
        raise ValueError(f"[{task}] frozen regroup input changed")
    payload = {}
    for path in sorted((ROOT / "payloads").glob(f"{task}_*.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line); payload[str(row["current_cluster_id"])] = row
    vote_paths = sorted(Path().glob(votes_glob)) if votes_glob else sorted(
        (ROOT / "votes").glob(f"{task}_*.jsonl"))
    decisions = {}
    malformed = 0
    for path in vote_paths:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1; continue
            current_id, groups = str(row.get("current_cluster_id")), row.get("groups")
            if current_id not in payload or current_id in decisions or not isinstance(groups, list):
                malformed += 1; continue
            expected = {str(x["source_cluster_id"]) for x in payload[current_id]["source_clusters"]}
            flat = [str(x) for group in groups if isinstance(group, list) for x in group]
            if (any(not isinstance(group, list) or not group for group in groups)
                    or len(flat) != len(set(flat)) or set(flat) != expected):
                malformed += 1; continue
            decisions[current_id] = [list(map(str, group)) for group in groups]
    missing = set(payload) - set(decisions)
    if missing or malformed:
        raise ValueError(f"[{task}] regroup incomplete: missing={len(missing)} malformed={malformed}")

    base = {str(k): str(v) for k, v in load_partition(task, manifest["v6_partition_path"]).items()}
    current = {str(k): str(v) for k, v in _load_partition(
        str(Path(OUT) / f"partition_{task}_L0v3.json")).items()}
    new_of_source = {}
    for current_id, groups in decisions.items():
        for group in groups:
            new_id = current_id if current_id in group else sorted(group)[0]
            for source in group:
                new_of_source[source] = new_id
    # Current clusters containing one v6 source need no LLM regroup row and retain that source ID.
    for source in set(base.values()):
        new_of_source.setdefault(source, source)
    partition = {key: new_of_source[source] for key, source in base.items()}
    out_path = Path(OUT) / f"partition_{task}_L0v4.json"
    out_path.write_text(json.dumps(partition) + "\n")
    from . import repair
    score_v6 = repair.score_vs_truth(task, base)
    score_l0v3 = repair.score_vs_truth(task, current)
    score_l0v4 = repair.score_vs_truth(task, partition)
    result = {"task": task, "version": manifest["version"], "n_keys": len(partition),
              "clusters_v6": len(set(base.values())), "clusters_l0v3": len(set(current.values())),
              "clusters_l0v4": len(set(partition.values())),
              "current_groups_regrouped": len(decisions),
              "source_clusters_remerged": len(set(base.values())) - len(set(partition.values())),
              "partition_path": str(out_path), "partition_sha256": _file_sha256(str(out_path)),
              "score_v6": score_v6, "score_l0v3": score_l0v3, "score_l0v4": score_l0v4,
              "semantic_quality": "frozen LLM truth; regroup decisions use independent non-eval LLM evidence"}
    (ROOT / f"{task}_apply_report.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def emit_l0v4_names(task: str, per_agent: int = 40) -> dict:
    """Rename every multi-member L0v4 subgroup created inside a regrouped L0v3 cluster."""
    from .levels import emit_rename_batches
    v3 = {str(k): str(v) for k, v in _load_partition(
        str(Path(OUT) / f"partition_{task}_L0v3.json")).items()}
    v4 = {str(k): str(v) for k, v in _load_partition(
        str(Path(OUT) / f"partition_{task}_L0v4.json")).items()}
    regrouped = set()
    for path in (ROOT / "payloads").glob(f"{task}_*.jsonl"):
        for line in path.read_text().splitlines():
            regrouped.add(str(json.loads(line)["current_cluster_id"]))
    affected = {v4[key] for key in v4 if v3[key] in regrouped}
    paths = emit_rename_batches(task, v4, per_agent=per_agent, k=8, level="L0v4",
                                cluster_ids=affected)
    report = {"task": task, "n_affected_l0v4_clusters": len(affected),
              "n_rename_payload_shards": len(paths), "payload_paths": paths}
    (ROOT / f"{task}_rename_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def ingest_l0v4_names(task: str) -> dict:
    """Carry unaffected L0v3 names forward; require fresh names for affected multi-member groups."""
    from .levels import ingest_names, members
    v3 = {str(k): str(v) for k, v in _load_partition(
        str(Path(OUT) / f"partition_{task}_L0v3.json")).items()}
    v4 = {str(k): str(v) for k, v in _load_partition(
        str(Path(OUT) / f"partition_{task}_L0v4.json")).items()}
    regrouped = set()
    for path in (ROOT / "payloads").glob(f"{task}_*.jsonl"):
        for line in path.read_text().splitlines():
            regrouped.add(str(json.loads(line)["current_cluster_id"]))
    affected = {v4[key] for key in v4 if v3[key] in regrouped}
    base_path = Path(OUT) / f"cluster_names_{task}_L0v3.json"
    if not base_path.exists():
        base_path = Path(OUT) / f"cluster_names_{task}_L0v2.json"
    base = json.loads(base_path.read_text()) if base_path.exists() else {}
    safe_base = {cluster: row for cluster, row in base.items() if cluster not in affected}
    names = ingest_names(task, v4, level="L0v4", base_names=safe_base)
    multi = members(v4)
    fallback_affected = [cluster for cluster in affected
                         if len(multi[cluster]) > 1 and names[cluster].get("source") != "fleet"]
    if fallback_affected:
        raise ValueError(f"[{task}] {len(fallback_affected)} affected L0v4 multis lack fresh LLM names")
    return {"task": task, "n_names": len(names), "n_affected": len(affected),
            "n_fleet_affected": sum(names[x].get("source") == "fleet" for x in affected),
            "n_singleton_affected": sum(names[x].get("source") == "singleton" for x in affected),
            "names_path": str(Path(OUT) / f"cluster_names_{task}_L0v4.json"),
            "names_sha256": _file_sha256(str(Path(OUT) / f"cluster_names_{task}_L0v4.json"))}
