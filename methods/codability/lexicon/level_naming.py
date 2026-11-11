#!/usr/bin/env python
"""Name the GROUPS of a level so the next level can build on them (R-level analogue of the L0
cluster rename). emit_group_names(task, 'R1') writes payloads of each R1 group's member constructs;
a Sonnet fleet names them; ingest_group_names writes node_names_<task>_R1.json (what
nodes_from_level(task,'R2') reads). Same for R2 -> node_names_<task>_R2.json for R3.
"""
import glob
import json
import os
from collections import defaultdict

from .build_level import OUT, _load_partition, nodes_from_level, rep_text


def emit_group_names(task, level, per_agent=40, k=8):
    """One payload line per multi-member group of partition_<task>_<level>.json:
    {group_id, n_members, members:[<=k member construct reps]}."""
    nodes, _ = nodes_from_level(task, level)
    by_id = {n["node_id"]: n for n in nodes}
    part = _load_partition(os.path.join(OUT, f"partition_{task}_{level}.json"))
    groups = defaultdict(list)
    for nid, g in part.items():
        if nid in by_id:
            groups[g].append(by_id[nid])
    # key is "cluster_id" so the existing rename fleet/prompt works unchanged
    rows = [{"cluster_id": str(g), "n_members": len(mem), "members": [rep_text(n) for n in mem[:k]]}
            for g, mem in sorted(groups.items()) if len(mem) >= 2]
    pd = os.path.join(OUT, "rename_payloads")
    os.makedirs(pd, exist_ok=True)
    for f in glob.glob(os.path.join(pd, f"{task}_{level}_gname_*.jsonl")):
        os.remove(f)
    outs = []
    for a in range(0, len(rows), per_agent):
        p = os.path.join(pd, f"{task}_{level}_gname_{a // per_agent:03d}.jsonl")
        with open(p, "w") as fh:
            for r in rows[a:a + per_agent]:
                fh.write(json.dumps(r) + "\n")
        outs.append(p)
    return outs, len(rows), len(groups)


def ingest_group_names(task, level):
    """Fleet names (group_id -> name+gloss) for multi-member groups; singletons/unnamed fall back to
    their first member's name. Writes node_names_<task>_<level>.json."""
    nodes, _ = nodes_from_level(task, level)
    by_id = {n["node_id"]: n for n in nodes}
    part = _load_partition(os.path.join(OUT, f"partition_{task}_{level}.json"))
    groups = defaultdict(list)
    for nid, g in part.items():
        if nid in by_id:
            groups[g].append(nid)
    names = {}
    for f in sorted(glob.glob(os.path.join(OUT, "rename_votes", f"{task}_{level}_gname_*.jsonl"))):
        for l in open(f):
            if not l.strip():
                continue
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                continue
            if r.get("cluster_id") is not None and r.get("name"):
                names[str(r["cluster_id"])] = {"name": r["name"][:90], "gloss": r.get("gloss", "")}
    n_fleet = len(names)
    for g, mem in groups.items():
        if str(g) not in names:
            m0 = by_id.get(mem[0], {})
            names[str(g)] = {"name": (m0.get("name") or str(g))[:90], "gloss": m0.get("gloss", ""),
                             "source": "singleton"}
    json.dump(names, open(os.path.join(OUT, f"node_names_{task}_{level}.json"), "w"))
    return names, n_fleet
