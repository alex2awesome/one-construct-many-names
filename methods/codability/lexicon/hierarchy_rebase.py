#!/usr/bin/env python
"""Audit and rebase an existing R1 partition after an append-only L0 repair.

L0v3 can move rubric items between surviving L0 cluster IDs or retire one ID into another.  Merely
filtering stale R1 keys is structurally complete, but can hide a semantic conflict when the affected
L0 clusters belonged to different R1 constructs.  This module emits those cross-construct bridge
pairs for LLM adjudication and applies only strict score==2 decisions.  Candidate discovery and
bookkeeping are deterministic; semantic sameness is never inferred by code.

The historical R1 partition is never overwritten.  The derived artifact is
``partition_<task>_R1_rebased_L0v3.json`` with a hash manifest beside it.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict

from .build_level import OUT, _file_sha256, _load_partition


def _sha_id(*parts: str) -> str:
    return hashlib.sha1("||".join(parts).encode()).hexdigest()[:16]


def _name_map(task: str, level: str) -> dict:
    if level.startswith("L0"):
        path = os.path.join(OUT, f"cluster_names_{task}_{level}.json")
        if not os.path.exists(path):
            path = os.path.join(OUT, f"cluster_names_{task}_L0v2.json")
    else:
        path = os.path.join(OUT, f"node_names_{task}_{level}.json")
    return json.load(open(path)) if os.path.exists(path) else {}


def _rep(names: dict, node_id: str) -> str:
    row = names.get(str(node_id)) or {}
    name = str(row.get("name") or node_id).strip()
    gloss = str(row.get("gloss") or "").strip()
    return f"{name}. {gloss}".strip()


def emit_r1_transition_audit(task: str, old_l0: str = "L0v2", new_l0: str = "L0v3") -> dict:
    """Emit one LLM pair per distinct active R1-group bridge induced by changed L0 membership."""
    old_path = os.path.join(OUT, f"partition_{task}_{old_l0}.json")
    new_path = os.path.join(OUT, f"partition_{task}_{new_l0}.json")
    r1_path = os.path.join(OUT, f"partition_{task}_R1.json")
    old, new, r1 = map(_load_partition, (old_path, new_path, r1_path))
    old = {str(k): str(v) for k, v in old.items()}
    new = {str(k): str(v) for k, v in new.items()}
    r1 = {str(k): str(v) for k, v in r1.items()}

    active_nodes = set(new.values())
    active_groups = {r1[n] for n in active_nodes if n in r1}
    transitions: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    n_changed_keys = 0
    n_unmapped = 0
    for key in old.keys() & new.keys():
        a, b = old[key], new[key]
        if a == b:
            continue
        n_changed_keys += 1
        if a not in r1 or b not in r1:
            n_unmapped += 1
            continue
        ga, gb = r1[a], r1[b]
        if ga == gb or ga not in active_groups or gb not in active_groups:
            continue
        transitions[tuple(sorted((ga, gb)))].add((a, b))

    r1_names = _name_map(task, "R1")
    old_names, new_names = _name_map(task, old_l0), _name_map(task, new_l0)
    rows = []
    for ga, gb in sorted(transitions):
        evidence = []
        for a, b in sorted(transitions[(ga, gb)]):
            evidence.append({"old_l0_id": a, "new_l0_id": b,
                             "old_l0_rep": _rep(old_names, a),
                             "new_l0_rep": _rep(new_names, b)})
        rows.append({"pair_id": _sha_id(task, old_l0, new_l0, ga, gb),
                     "group_a": ga, "group_b": gb,
                     "canonical_a": _rep(r1_names, ga),
                     "canonical_b": _rep(r1_names, gb),
                     "transition_evidence": evidence})

    payload_dir = os.path.join(OUT, "rebase_payloads")
    os.makedirs(payload_dir, exist_ok=True)
    payload_path = os.path.join(payload_dir, f"{task}_R1_{old_l0}_to_{new_l0}.jsonl")
    with open(payload_path, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    manifest = {"task": task, "level": "R1", "old_l0": old_l0, "new_l0": new_l0,
                "old_l0_path": os.path.relpath(old_path, OUT),
                "old_l0_sha256": _file_sha256(old_path),
                "new_l0_path": os.path.relpath(new_path, OUT),
                "new_l0_sha256": _file_sha256(new_path),
                "source_r1_path": os.path.relpath(r1_path, OUT),
                "source_r1_sha256": _file_sha256(r1_path),
                "n_changed_keys": n_changed_keys, "n_unmapped_changed_keys": n_unmapped,
                "n_bridge_pairs": len(rows), "payload_sha256": _file_sha256(payload_path)}
    manifest_path = os.path.join(payload_dir, f"{task}_R1_{old_l0}_to_{new_l0}_manifest.json")
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=1)
    return manifest


class IncompleteRebaseVotesError(RuntimeError):
    pass


def apply_r1_transition_audit(task: str, old_l0: str = "L0v2", new_l0: str = "L0v3") -> dict:
    """Filter R1 to current L0 nodes and union only bridge pairs receiving strict LLM score 2."""
    stem = f"{task}_R1_{old_l0}_to_{new_l0}"
    payload_path = os.path.join(OUT, "rebase_payloads", f"{stem}.jsonl")
    vote_path = os.path.join(OUT, "rebase_votes", f"{stem}.jsonl")
    payload = {r["pair_id"]: r for r in (json.loads(x) for x in open(payload_path) if x.strip())}
    votes: dict[str, int] = {}
    malformed = 0
    if payload and not os.path.exists(vote_path):
        raise IncompleteRebaseVotesError(f"[{task}] missing rebase vote file: {vote_path}")
    vote_lines = open(vote_path) if os.path.exists(vote_path) else ()
    for line in vote_lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            continue
        pid, score = row.get("pair_id"), row.get("score")
        if (set(row) != {"pair_id", "score"} or pid not in payload or type(score) is not int
                or score not in (0, 1, 2) or pid in votes):
            malformed += 1
            continue
        votes[pid] = score
    missing = set(payload) - set(votes)
    if missing or malformed:
        raise IncompleteRebaseVotesError(
            f"[{task}] rebase votes incomplete: missing={len(missing)} malformed_or_duplicate={malformed}")

    new = {str(k): str(v) for k, v in _load_partition(
        os.path.join(OUT, f"partition_{task}_{new_l0}.json")).items()}
    source = {str(k): str(v) for k, v in _load_partition(
        os.path.join(OUT, f"partition_{task}_R1.json")).items()}
    active_nodes = set(new.values())
    missing_nodes = active_nodes - set(source)
    if missing_nodes:
        raise IncompleteRebaseVotesError(
            f"[{task}] source R1 misses {len(missing_nodes)} active L0 nodes: {sorted(missing_nodes)[:5]}")

    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            lo, hi = sorted((ra, rb))
            parent[hi] = lo

    for group in {source[n] for n in active_nodes}:
        find(group)
    for pid, score in votes.items():
        if score == 2:
            row = payload[pid]
            union(str(row["group_a"]), str(row["group_b"]))
    rebased = {n: find(source[n]) for n in sorted(active_nodes)}
    out_path = os.path.join(OUT, f"partition_{task}_R1_rebased_{new_l0}.json")
    with open(out_path, "w") as fh:
        json.dump(rebased, fh)
    result = {"task": task, "source_nodes": len(source), "active_nodes": len(active_nodes),
              "stale_nodes_removed": len(set(source) - active_nodes),
              "source_groups": len(set(source.values())), "rebased_groups": len(set(rebased.values())),
              "bridge_pairs": len(payload), "bridge_same": sum(s == 2 for s in votes.values()),
              "partition_path": os.path.relpath(out_path, OUT),
              "partition_sha256": _file_sha256(out_path),
              "vote_path": os.path.relpath(vote_path, OUT) if os.path.exists(vote_path) else None,
              "vote_sha256": _file_sha256(vote_path) if os.path.exists(vote_path) else None}
    with open(os.path.join(OUT, "rebase_payloads", f"{stem}_apply_manifest.json"), "w") as fh:
        json.dump(result, fh, indent=1)
    return result


def emit_rebased_r1_names(task: str, new_l0: str = "L0v3", per_agent: int = 40) -> dict:
    """Emit semantic rename rows only for rebased R1 groups that combine old named groups."""
    source = {str(k): str(v) for k, v in _load_partition(
        os.path.join(OUT, f"partition_{task}_R1.json")).items()}
    rebased = {str(k): str(v) for k, v in _load_partition(
        os.path.join(OUT, f"partition_{task}_R1_rebased_{new_l0}.json")).items()}
    old_names = _name_map(task, "R1")
    components: dict[str, set[str]] = defaultdict(set)
    for node, group in rebased.items():
        components[group].add(source[node])
    rows = [{"cluster_id": group, "n_members": len(parts),
             "members": [_rep(old_names, part) for part in sorted(parts)]}
            for group, parts in sorted(components.items()) if len(parts) > 1]
    payload_dir = os.path.join(OUT, "rename_payloads")
    os.makedirs(payload_dir, exist_ok=True)
    stem = f"{task}_R1_rebased_{new_l0}_gname"
    for name in os.listdir(payload_dir):
        if name.startswith(stem + "_") and name.endswith(".jsonl"):
            os.remove(os.path.join(payload_dir, name))
    for start in range(0, len(rows), per_agent):
        path = os.path.join(payload_dir, f"{stem}_{start // per_agent:03d}.jsonl")
        with open(path, "w") as fh:
            for row in rows[start:start + per_agent]:
                fh.write(json.dumps(row) + "\n")
    return {"task": task, "n_rebased_groups": len(components),
            "n_groups_requiring_rename": len(rows),
            "n_payload_shards": (len(rows) + per_agent - 1) // per_agent}


def ingest_rebased_r1_names(task: str, new_l0: str = "L0v3") -> dict:
    """Combine LLM names for merged groups with exact historical names for unchanged groups."""
    source = {str(k): str(v) for k, v in _load_partition(
        os.path.join(OUT, f"partition_{task}_R1.json")).items()}
    rebased = {str(k): str(v) for k, v in _load_partition(
        os.path.join(OUT, f"partition_{task}_R1_rebased_{new_l0}.json")).items()}
    old_names = _name_map(task, "R1")
    components: dict[str, set[str]] = defaultdict(set)
    for node, group in rebased.items():
        components[group].add(source[node])
    vote_dir = os.path.join(OUT, "rename_votes")
    stem = f"{task}_R1_rebased_{new_l0}_gname"
    votes = {}
    if os.path.exists(vote_dir):
        for name in sorted(os.listdir(vote_dir)):
            if not (name.startswith(stem + "_") and name.endswith(".jsonl")):
                continue
            for line in open(os.path.join(vote_dir, name)):
                if not line.strip():
                    continue
                row = json.loads(line)
                if set(row) != {"cluster_id", "name", "gloss"}:
                    raise IncompleteRebaseVotesError(f"[{task}] malformed rebased name row")
                cid = str(row["cluster_id"])
                if cid in votes or not str(row["name"]).strip() or not str(row["gloss"]).strip():
                    raise IncompleteRebaseVotesError(f"[{task}] duplicate/empty rebased name for {cid}")
                votes[cid] = {"name": str(row["name"])[:90], "gloss": str(row["gloss"])}
    required = {group for group, parts in components.items() if len(parts) > 1}
    if required - set(votes) or set(votes) - required:
        raise IncompleteRebaseVotesError(
            f"[{task}] rebased names mismatch: missing={len(required-set(votes))} "
            f"unexpected={len(set(votes)-required)}")
    names = {}
    for group, parts in components.items():
        if group in votes:
            names[group] = votes[group]
        else:
            part = next(iter(parts))
            row = old_names.get(part) or {"name": part, "gloss": ""}
            names[group] = {"name": str(row.get("name") or part)[:90],
                            "gloss": str(row.get("gloss") or "")}
    path = os.path.join(OUT, f"node_names_{task}_R1_rebased_{new_l0}.json")
    with open(path, "w") as fh:
        json.dump(names, fh)
    return {"task": task, "n_names": len(names), "n_llm_renamed": len(votes),
            "path": os.path.relpath(path, OUT), "sha256": _file_sha256(path)}
