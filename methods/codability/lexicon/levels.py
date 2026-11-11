#!/usr/bin/env python
"""Level rebuild + canonical naming + per-level precision/recall.

Each level has its OWN "same" relation; prec/recall are measured against THAT relation, never L0's:
  L0  SAME CRITERION  — two rubric items express the same criterion (adjudicated_truth_<task>.json).
  R1  SAME CONSTRUCT  — two L0 concepts are facets of the same evaluated quality.
  R2  SAME THEME      — two R1 constructs belong to the same evaluative theme.
  R3  SAME CATEGORY   — two R2 themes belong to the same top-level category (subsumption at top).

Naming: singletons inherit their single member's canonical; multi-member clusters are named by a
Sonnet-5/Opus fleet from their members (a short noun-phrase for the shared criterion). Names + reps
drive the next level's grouping and its eval payloads.

Per-level metric (level partition P_L, frozen level-L pair truth T_L):
  recall_L    = P(co-clustered in P_L | T_L = same-at-L)
  precision_L = P(T_L = same-at-L      | co-clustered in P_L)
Identical shape to repair.score_vs_truth, but T_L is the LEVEL-L relation, judged at node grain.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional

from .judge import canon_map
from .sources import ROOT

OUT = os.path.join(ROOT, "outputs", "lexicon")


def _h(*p: str) -> str:
    return hashlib.sha1("||".join(p).encode()).hexdigest()


def members(partition: Dict[str, str]) -> Dict[str, List[str]]:
    m: Dict[str, List[str]] = defaultdict(list)
    for k, c in partition.items():
        m[str(c)].append(k)
    return m


def rep_texts(task: str, partition: Dict[str, str], k: int = 8) -> Dict[str, List[str]]:
    """Deterministic member-canonical sample per cluster (hash-ordered, so stable across runs)."""
    cmap = canon_map(task)
    out: Dict[str, List[str]] = {}
    for c, ks in members(partition).items():
        ks = sorted(ks, key=lambda x: _h(c, x))[:k]
        out[c] = [cmap[x] for x in ks if x in cmap]
    return out


def emit_rename_batches(task: str, partition: Dict[str, str], per_agent: int = 40,
                        k: int = 8, level: str = "L0v2",
                        cluster_ids: Optional[set[str]] = None) -> List[str]:
    """One line per MULTI-member cluster: {cluster_id, n_members, members:[<=k canonicals]}."""
    reps = rep_texts(task, partition, k)
    mem = members(partition)
    rows = [{"cluster_id": c, "n_members": len(mem[c]), "members": reps[c]}
            for c in sorted(reps) if len(mem[c]) >= 2
            and (cluster_ids is None or c in cluster_ids)]
    pd = os.path.join(OUT, "rename_payloads")
    os.makedirs(pd, exist_ok=True)
    for f in glob.glob(os.path.join(pd, f"{task}_{level}_rename_*.jsonl")):
        os.remove(f)
    outs = []
    for a in range(0, len(rows), per_agent):
        i = a // per_agent
        p = os.path.join(pd, f"{task}_{level}_rename_{i:03d}.jsonl")
        with open(p, "w") as fh:
            for r in rows[a: a + per_agent]:
                fh.write(json.dumps(r) + "\n")
        outs.append(p)
    print(f"[{task}/{level}] {len(rows)} multi-member clusters -> {len(outs)} rename shards @ {per_agent}")
    return outs


def ingest_names(task: str, partition: Dict[str, str], level: str = "L0v2",
                 vote_glob: Optional[str] = None,
                 base_names: Optional[Dict[str, dict]] = None) -> Dict[str, dict]:
    """cluster_id -> {name, gloss?, source}. Multi from the fleet; singletons from their member."""
    cmap = canon_map(task)
    mem = members(partition)
    # A repaired partition can preserve most cluster IDs and need LLM renaming only for a small
    # subset.  Carry forward exact prior names for active IDs, then let fresh votes override them;
    # never import names for retired IDs.
    names: Dict[str, dict] = {str(c): dict(row) for c, row in (base_names or {}).items()
                              if str(c) in mem}
    vote_glob = vote_glob or os.path.join(OUT, "rename_votes", f"{task}_{level}_rename_*.jsonl")
    for f in sorted(glob.glob(vote_glob)):
        for line in open(f):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid, nm = str(r.get("cluster_id")), (r.get("name") or "").strip()
            if cid and nm:
                names[cid] = {"name": nm, "gloss": (r.get("gloss") or "").strip(), "source": "fleet"}
    # singletons + any un-named multi -> member canonical (first, hash-stable)
    for c, ks in mem.items():
        if c not in names:
            ks = sorted(ks, key=lambda x: _h(c, x))
            txt = cmap.get(ks[0], "")
            names[c] = {"name": txt[:90], "gloss": "", "source": "singleton" if len(ks) == 1 else "fallback"}
    op = os.path.join(OUT, f"cluster_names_{task}_{level}.json")
    json.dump(names, open(op, "w"), indent=0)
    n_fleet = sum(1 for v in names.values() if v["source"] == "fleet")
    print(f"[{task}/{level}] named {len(names)} clusters ({n_fleet} fleet, "
          f"{sum(1 for v in names.values() if v['source']=='singleton')} singleton, "
          f"{sum(1 for v in names.values() if v['source']=='fallback')} fallback) -> {op}")
    return names


def score_level(partition: Dict[str, str], truth_eval_path: str, truth_path: str) -> dict:
    """recall=P(co|same-at-L), precision=P(same-at-L|co) for a level partition vs its level truth.
    truth_eval_path: jsonl with pair_id + node_a/node_b OR key_a/key_b; truth_path: pid->bool."""
    truth = {k: bool(v) for k, v in json.load(open(truth_path)).items()}
    ev = {r["pair_id"]: r for r in (json.loads(l) for l in open(truth_eval_path))}
    n_same = same_co = n_co = co_same = miss = 0
    for pid, same in truth.items():
        r = ev.get(pid)
        if not r:
            miss += 1
            continue
        a = r.get("node_a") or r.get("key_a")
        b = r.get("node_b") or r.get("key_b")
        ca, cb = partition.get(a), partition.get(b)
        if ca is None or cb is None:
            miss += 1
            continue
        co = ca == cb
        if same:
            n_same += 1
            same_co += int(co)
        if co:
            n_co += 1
            co_same += int(same)
    rec = same_co / n_same if n_same else None
    prec = co_same / n_co if n_co else None
    f1 = (2 * rec * prec / (rec + prec)) if (rec and prec) else None
    return {"n_same": n_same, "recall": round(rec, 3) if rec is not None else None,
            "n_colabeled": n_co, "precision": round(prec, 3) if prec is not None else None,
            "f1": round(f1, 3) if f1 is not None else None, "unmapped": miss}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["rename_emit", "rename_ingest"])
    ap.add_argument("--task", required=True)
    ap.add_argument("--partition", required=True, help="path to partition json (key->cluster)")
    ap.add_argument("--level", default="L0v2")
    ap.add_argument("--per-agent", type=int, default=40)
    a = ap.parse_args()
    d = json.load(open(a.partition))
    part = {k: str(v) for k, v in (d.get("partition", d)).items()}
    if a.cmd == "rename_emit":
        emit_rename_batches(a.task, part, per_agent=a.per_agent, level=a.level)
    else:
        ingest_names(a.task, part, level=a.level)
