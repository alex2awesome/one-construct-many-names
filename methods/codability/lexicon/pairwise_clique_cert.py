"""Split proposed groups into cliques of independently dual-certified SAME pairs."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import networkx as nx

from .build_level import OUT, _file_sha256, _load_partition
from .upper_precision_audit import ROOT as AUDIT_ROOT, _load_votes, _stem


ROOT = Path(OUT) / "pairwise_clique_cert"
VERSION = "pairwise-clique-cert-v1"


def _clique_cover(members: list[str], allowed_edges: set[tuple[str, str]]) -> list[list[str]]:
    """Deterministic largest-maximal-clique cover; every output group is pairwise certified."""
    remaining = set(map(str, members))
    groups = []
    while remaining:
        graph = nx.Graph()
        graph.add_nodes_from(sorted(remaining))
        graph.add_edges_from((a, b) for a, b in allowed_edges if a in remaining and b in remaining)
        cliques = [tuple(sorted(c)) for c in nx.find_cliques(graph)]
        chosen = min(cliques, key=lambda c: (-len(c), c))
        groups.append(list(chosen))
        remaining.difference_update(chosen)
    return groups


def apply(task: str, level: str, tag: str, votes_a_path: str,
          votes_b_path: str) -> dict:
    stem = _stem(task, level, tag)
    manifest_path = AUDIT_ROOT / f"{stem}_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("exclude_prior_measurement_pairs") is not False:
        raise ValueError("clique certification requires exhaustive, non-excluding audit preparation")
    for field in ("partition", "protocol", "audit", "key"):
        if _file_sha256(manifest[f"{field}_path"]) != manifest[f"{field}_sha256"]:
            raise ValueError(f"[{task}/{level}/{tag}] frozen {field} changed")
    source = {str(k): str(v) for k, v in _load_partition(manifest["partition_path"]).items()}
    key = json.loads(Path(manifest["key_path"]).read_text())
    expected = set(key)
    a = _load_votes(votes_a_path, expected)
    b = _load_votes(votes_b_path, expected)

    members: dict[str, list[str]] = defaultdict(list)
    for node, group in source.items():
        members[group].append(node)
    expected_pairs = {tuple(sorted(pair)) for ids in members.values()
                      for pair in combinations(sorted(ids), 2)}
    keyed_pairs = {tuple(sorted((str(row["node_a"]), str(row["node_b"]))))
                   for row in key.values()}
    if keyed_pairs != expected_pairs or len(key) != len(expected_pairs):
        raise ValueError(f"audit is not exhaustive over source co-pairs: "
                         f"{len(keyed_pairs)}/{len(expected_pairs)}")

    allowed = set()
    for pid, row in key.items():
        if a[pid] == 2 and b[pid] == 2:
            allowed.add(tuple(sorted((str(row["node_a"]), str(row["node_b"])))))
    out = {}
    source_groups_split = 0
    for source_group, ids in sorted(members.items()):
        groups = _clique_cover(sorted(ids), allowed)
        source_groups_split += len(groups) > 1
        for index, group in enumerate(groups):
            if len(groups) == 1:
                new_group = source_group
            else:
                digest = hashlib.sha1(
                    f"{VERSION}||{task}||{level}||{source_group}||{index}".encode()
                ).hexdigest()[:10]
                new_group = f"{task}_{level}_clique_{digest}"
            for node in group:
                out[node] = new_group
    if set(out) != set(source):
        raise ValueError("clique certification changed node coverage")
    # Defense in depth: every output co-pair must be a dual-score-2 allowed edge.
    output_members: dict[str, list[str]] = defaultdict(list)
    for node, group in out.items():
        output_members[group].append(node)
    certified_pairs = {tuple(sorted(pair)) for ids in output_members.values()
                       for pair in combinations(sorted(ids), 2)}
    if not certified_pairs <= allowed:
        raise ValueError("output contains a pair without dual score-2 certification")
    ROOT.mkdir(parents=True, exist_ok=True)
    destination = Path(OUT) / f"partition_{task}_{level}_{tag}_clique_certified.json"
    destination.write_text(json.dumps(out, sort_keys=True) + "\n")
    report = {
        "version": VERSION, "task": task, "level": level, "tag": tag,
        "source_partition_path": manifest["partition_path"],
        "source_partition_sha256": manifest["partition_sha256"],
        "votes_a_path": str(Path(votes_a_path).resolve()),
        "votes_a_sha256": _file_sha256(votes_a_path),
        "votes_b_path": str(Path(votes_b_path).resolve()),
        "votes_b_sha256": _file_sha256(votes_b_path),
        "n_nodes": len(source), "groups_before": len(members),
        "groups_after": len(output_members), "source_groups_split": source_groups_split,
        "n_source_colabeled_pairs": len(expected_pairs),
        "n_dual_score2_edges": len(allowed),
        "n_output_colabeled_pairs": len(certified_pairs),
        "partition_path": str(destination.resolve()),
        "partition_sha256": _file_sha256(str(destination)),
        "guarantee": "every output co-pair received score 2 from both independent LLM judges",
    }
    (ROOT / f"{stem}_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
