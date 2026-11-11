#!/usr/bin/env python
"""GEPA dev-set builder for the cluster-NAMING prompt (RENAME_PROTOCOL.txt).

Objective GEPA hill-climbs (user 2026-07-06, "both 1/2"): a name is scored on
  (A) DISCRIMINATIVE RECONSTRUCTION — a blind reader, given only the NAME + a lineup of the
      cluster's members mixed with HARD distractors (TF-IDF-near items from OTHER clusters),
      recovers the members. metric = F1(picked, true members). Rewards distinct + faithful names.
  (B) ROUND-TRIP RECOVERY — a blind reader, given only the NAME, reconstructs the criterion; a
      judge rates its fidelity to the members. Rewards completeness/specificity.
  composite = 0.5*F1 + 0.5*fidelity.

This module only BUILDS the frozen dev/test sets (deterministic; hard distractors via TF-IDF).
The GEPA loop itself (Sonnet driver, Opus proposer) is orchestrated separately and consumes these
files. Reconstruction-only: no human labels; names scored by their sufficiency, on-philosophy.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from typing import Dict, List

from .judge import canon_map
from .sources import ROOT

OUT = os.path.join(ROOT, "outputs", "lexicon")
GDIR = os.path.join(OUT, "gepa_rename")


def _h(*p: str) -> str:
    return hashlib.sha1("||".join(p).encode()).hexdigest()


def build_devset(task: str, partition_path: str, n_dev: int = 40, n_test: int = 40,
                 max_members: int = 6, n_distract: int = 6, min_size: int = 3,
                 max_size: int = 15) -> dict:
    os.makedirs(GDIR, exist_ok=True)
    part = {k: str(v) for k, v in json.load(open(partition_path)).items()}
    cmap = canon_map(task)
    mem: Dict[str, List[str]] = defaultdict(list)
    for k, c in part.items():
        if k in cmap:
            mem[c].append(k)
    # eligible multi-member clusters (not the giant catch-alls)
    elig = [c for c, ks in mem.items() if min_size <= len(ks) <= max_size]
    elig.sort(key=lambda c: _h(c, "gepadev"))
    picked = elig[: n_dev + n_test]

    keys = sorted(k for k in cmap if k in part)
    texts = [cmap[k] for k in keys]
    ki = {k: i for i, k in enumerate(keys)}
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors
    X = TfidfVectorizer(min_df=2, max_features=40000, sublinear_tf=True).fit_transform(texts)
    nn = NearestNeighbors(n_neighbors=40, metric="cosine").fit(X)

    def make(c: str) -> dict:
        ks = sorted(mem[c], key=lambda x: _h(c, x))
        mtexts = [cmap[x] for x in ks[:max_members]]
        # hard distractors: TF-IDF neighbors of members that live in OTHER clusters
        dist = []
        seen = set(ks)
        for x in ks[:max_members]:
            _, idx = nn.kneighbors(X[ki[x]])
            for j in idx[0][1:]:
                dk = keys[int(j)]
                if part.get(dk) != c and dk not in seen:
                    dist.append(cmap[dk]); seen.add(dk)
                    break
        dist = dist[:n_distract]
        # lineup = members(as-shown) + distractors, deterministically shuffled with truth flags
        pool = [(t, 1) for t in mtexts] + [(t, 0) for t in dist]
        pool.sort(key=lambda tf: _h(c, tf[0]))
        return {"cluster_id": c, "n_members": len(mem[c]),
                "members": mtexts,
                "lineup": [{"item_id": f"{c[:8]}_{i}", "text": t} for i, (t, _) in enumerate(pool)],
                "lineup_truth": {f"{c[:8]}_{i}": fl for i, (t, fl) in enumerate(pool)}}

    dev = [make(c) for c in picked[:n_dev]]
    test = [make(c) for c in picked[n_dev:n_dev + n_test]]
    # agent-facing (no truth) vs scoring (truth) split
    def strip(rows):
        return [{"cluster_id": r["cluster_id"], "members": r["members"],
                 "lineup": r["lineup"]} for r in rows]
    truth = {r["cluster_id"]: r["lineup_truth"] for r in dev + test}
    json.dump(strip(dev), open(os.path.join(GDIR, "dev.json"), "w"), indent=1)
    json.dump(strip(test), open(os.path.join(GDIR, "test.json"), "w"), indent=1)
    json.dump(truth, open(os.path.join(GDIR, "truth.json"), "w"), indent=1)
    # batch for fan-out, split so the READER never sees which lineup items are members:
    #   <split>_drv_batch_NN.json  = {cluster_id, members}  (driver names; judge reads members)
    #   <split>_rdr_batch_NN.json  = {cluster_id, lineup}   (reader reconstructs, BLIND to members)
    def batch(rows, name, per=10):
        n = 0
        for a in range(0, len(rows), per):
            grp = rows[a:a + per]
            json.dump([{"cluster_id": r["cluster_id"], "members": r["members"]} for r in grp],
                      open(os.path.join(GDIR, f"{name}_drv_batch_{a//per:02d}.json"), "w"), indent=1)
            json.dump([{"cluster_id": r["cluster_id"], "lineup": r["lineup"]} for r in grp],
                      open(os.path.join(GDIR, f"{name}_rdr_batch_{a//per:02d}.json"), "w"), indent=1)
            n += 1
        return n
    n_dbatch = batch(dev, "dev")
    n_tbatch = batch(test, "test")
    dbatches = [os.path.join(GDIR, f"dev_drv_batch_{i:02d}.json") for i in range(n_dbatch)]
    tbatches = [os.path.join(GDIR, f"test_drv_batch_{i:02d}.json") for i in range(n_tbatch)]
    # seed prompt copy
    seed = open(os.path.join(OUT, "RENAME_PROTOCOL.txt")).read()
    json.dump({"round": 0, "prompt": seed}, open(os.path.join(GDIR, "seed_prompt.json"), "w"), indent=1)
    manifest = {"task": task, "n_dev": len(dev), "n_test": len(test),
                "dev_batches": dbatches, "test_batches": tbatches,
                "avg_lineup": round(sum(len(r["lineup"]) for r in dev) / len(dev), 1),
                "avg_members_shown": round(sum(len(r["members"]) for r in dev) / len(dev), 1)}
    json.dump(manifest, open(os.path.join(GDIR, "manifest.json"), "w"), indent=1)
    print(json.dumps(manifest, indent=1))
    return manifest


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="humor")
    ap.add_argument("--partition", default=os.path.join(OUT, "partition_humor_L0v2.json"))
    a = ap.parse_args()
    build_devset(a.task, a.partition)
