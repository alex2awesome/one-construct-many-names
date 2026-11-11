"""LLM-only coherence refinement of provisional R1 Louvain communities.

Louvain supplies candidate supersets from independently judged SAME edges.  An LLM then partitions
every non-singleton community under the frozen R1 relation.  Code only stages semantic records,
validates exact coverage, and composes the resulting partition.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .build_level import OUT, _file_sha256, _load_partition, nodes_from_level

ROOT = Path(OUT) / "r1_refine"
PROTOCOL = Path(OUT) / "STRICT_BUILD_PROTOCOL_R1.txt"


def prepare(task: str) -> dict:
    part_path = Path(OUT) / f"partition_{task}_R1.json"
    part = {str(k): str(v) for k, v in _load_partition(str(part_path)).items()}
    nodes, _ = nodes_from_level(task, "R1")
    by_id = {str(n["node_id"]): n for n in nodes}
    members: dict[str, list[str]] = defaultdict(list)
    for node, group in part.items():
        members[group].append(node)
    payload_dir = ROOT / "payloads"; payload_dir.mkdir(parents=True, exist_ok=True)
    for old in payload_dir.glob(f"{task}_*.json"):
        old.unlink()
    paths = []
    for i, (group, ids) in enumerate(sorted(members.items())):
        if len(ids) < 2:
            continue
        row = {"provisional_group_id": group, "n_nodes": len(ids),
               "nodes": [{"node_id": node,
                          "name": str(by_id[node].get("name") or node),
                          "gloss": str(by_id[node].get("gloss") or "")}
                         for node in sorted(ids)]}
        path = payload_dir / f"{task}_{len(paths):03d}.json"
        path.write_text(json.dumps(row, ensure_ascii=False, indent=1) + "\n")
        paths.append(str(path))
    manifest = {"task": task, "version": "r1-coherence-refine-v1",
                "source_partition_path": str(part_path),
                "source_partition_sha256": _file_sha256(str(part_path)),
                "protocol_path": str(PROTOCOL),
                "protocol_sha256": _file_sha256(str(PROTOCOL)),
                "n_nodes": len(part), "n_source_groups": len(members),
                "n_groups_to_refine": len(paths), "payload_paths": paths,
                "output_schema": {"provisional_group_id": "string",
                                  "groups": "list[list[node_id]]"},
                "semantic_decider": "LLM only; code validates exact within-community coverage"}
    ROOT.mkdir(exist_ok=True)
    (ROOT / f"{task}_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def apply(task: str) -> dict:
    manifest = json.loads((ROOT / f"{task}_manifest.json").read_text())
    if (_file_sha256(manifest["source_partition_path"]) != manifest["source_partition_sha256"]
            or _file_sha256(manifest["protocol_path"]) != manifest["protocol_sha256"]):
        raise ValueError(f"[{task}] frozen R1 refinement input changed")
    source = {str(k): str(v) for k, v in _load_partition(
        manifest["source_partition_path"]).items()}
    decisions = {}
    malformed = 0
    for payload_path in manifest["payload_paths"]:
        payload = json.loads(Path(payload_path).read_text())
        gid = str(payload["provisional_group_id"])
        vote_path = ROOT / "votes" / (Path(payload_path).stem + ".json")
        if not vote_path.exists():
            continue
        try:
            vote = json.loads(vote_path.read_text())
        except Exception:
            malformed += 1; continue
        groups = vote.get("groups")
        expected = {str(n["node_id"]) for n in payload["nodes"]}
        if str(vote.get("provisional_group_id")) != gid or not isinstance(groups, list):
            malformed += 1; continue
        flat = [str(x) for group in groups if isinstance(group, list) for x in group]
        if (any(not isinstance(group, list) or not group for group in groups)
                or len(flat) != len(set(flat)) or set(flat) != expected):
            malformed += 1; continue
        decisions[gid] = [list(map(str, group)) for group in groups]
    if len(decisions) != manifest["n_groups_to_refine"] or malformed:
        raise ValueError(f"[{task}] incomplete R1 refine: decisions={len(decisions)}/"
                         f"{manifest['n_groups_to_refine']} malformed={malformed}")
    refined = {}
    for node, gid in source.items():
        if gid not in decisions:  # provisional singleton
            refined[node] = gid
    for gid, groups in decisions.items():
        for i, group in enumerate(groups):
            new_gid = gid if len(groups) == 1 else f"{gid}_r{i}"
            for node in group:
                refined[node] = new_gid
    if set(refined) != set(source):
        raise ValueError(f"[{task}] refined partition coverage mismatch")
    out = Path(OUT) / f"partition_{task}_R1_refined.json"
    out.write_text(json.dumps(refined) + "\n")
    report = {"task": task, "source_groups": len(set(source.values())),
              "refined_groups": len(set(refined.values())), "n_nodes": len(refined),
              "partition_path": str(out), "partition_sha256": _file_sha256(str(out))}
    (ROOT / f"{task}_apply_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
