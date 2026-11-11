"""P0 of the undermerge program: FROZEN, stratified arbiter-eval pair sets.

Strata per task (~1.5K pairs, stable-hash seeded, frozen before any repair round runs):
  - TF-IDF cosine spectrum: kNN pool for high-sim bins, random pairs for low-sim bins
    (TF-IDF = deterministic, embedding-free pre-filter; never in a reported number)
  - shared-rare-name-token pairs (surface-name net)
  - within-cluster pairs (precision arm) + cross-cluster T3 sample (the disputed region)
  - blinded anchors: trusted positives (triangle + cos>=0.97) and score-0 negatives

Output: arbiter_eval_<task>.jsonl (+ per-agent payload chunks, cluster info stripped).
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from collections import defaultdict
from typing import Dict, List

from . import audit
from .judge import canon_map
from .sources import ROOT

OUT = os.path.join(ROOT, "outputs", "lexicon")


def _h(*parts: str) -> str:
    return hashlib.sha1("||".join(parts).encode()).hexdigest()


def _pid(a: str, b: str) -> str:
    return _h(*sorted((a, b)))[:16]


def build_eval(task: str, n_per_bin: int = 90, n_name: int = 120, n_within: int = 180,
               n_t3: int = 220, n_anchor: int = 12) -> str:
    path = os.path.join(OUT, f"arbiter_eval_{task}.jsonl")
    if os.path.exists(path):
        print(f"[{task}] FROZEN eval exists — not rebuilding: {path}")
        return path
    cmap = canon_map(task)
    keys = sorted(cmap)
    texts = [cmap[k] for k in keys]
    part = audit.load_partition(task)
    verd = audit.load_verdicts(task)
    T = audit.trust_edges(verd)

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors
    import numpy as np
    X = TfidfVectorizer(min_df=2, max_features=40000, sublinear_tf=True).fit_transform(texts)
    nn = NearestNeighbors(n_neighbors=11, metric="cosine").fit(X)
    dist, idx = nn.kneighbors(X)

    pool: Dict[str, dict] = {}

    def add(i, j, stratum, sim=None):
        a, b = keys[i], keys[j]
        if a == b:
            return
        pid = _pid(a, b)
        if pid not in pool:
            pool[pid] = {"pair_id": pid, "task": task, "stratum": stratum,
                         "key_a": a, "key_b": b, "tfidf_cos": sim,
                         "canonical_a": cmap[a], "canonical_b": cmap[b]}

    # high-sim bins from kNN pool
    bins: Dict[int, List] = defaultdict(list)
    for i in range(len(keys)):
        for j_pos in range(1, 11):
            j, sim = int(idx[i][j_pos]), 1.0 - float(dist[i][j_pos])
            if i < j:
                bins[min(9, int(sim * 10))].append((i, j, sim))
    rng = random.Random(0)
    for b in range(3, 10):
        cand = sorted(bins.get(b, []), key=lambda t: _pid(keys[t[0]], keys[t[1]]))
        for i, j, sim in cand[:n_per_bin]:
            add(i, j, f"spectrum_{b/10:.1f}", round(sim, 3))
    # low-sim random
    for b in range(0, 3):
        got = 0
        while got < n_per_bin // 2:
            i, j = rng.randrange(len(keys)), rng.randrange(len(keys))
            if i == j:
                continue
            add(i, j, "random_low")
            got += 1
    # shared rare name token
    tok2keys = defaultdict(list)
    from .census import norm_term
    for k in keys:
        name = cmap[k].split(".")[0][:80]
        for w in set(norm_term(name).split()):
            if len(w) > 5:
                tok2keys[w].append(k)
    name_pairs = []
    for w, ks in tok2keys.items():
        if 2 <= len(ks) <= 8:
            for x in range(len(ks) - 1):
                name_pairs.append((ks[x], ks[x + 1]))
    ki = {k: n for n, k in enumerate(keys)}
    for a, b in sorted(name_pairs, key=lambda p: _pid(*p))[:n_name]:
        add(ki[a], ki[b], "shared_name")
    # within-cluster (precision arm)
    members = defaultdict(list)
    for k, c in part.items():
        if k in ki:
            members[c].append(k)
    wc = []
    for c, ms in members.items():
        if len(ms) >= 2:
            ms = sorted(ms, key=lambda k: _h(k, "w"))
            wc.append((ms[0], ms[1]))
    for a, b in sorted(wc, key=lambda p: _pid(*p))[:n_within]:
        add(ki[a], ki[b], "within_cluster")
    # T3 disputed sample
    t3 = sorted((tuple(p) for p in T["t3"]), key=lambda p: _pid(*p))
    got = 0
    for a, b in t3:
        if a in ki and b in ki and got < n_t3:
            add(ki[a], ki[b], "t3_disputed")
            got += 1
    # anchors (expected labels saved separately, payloads stay blind)
    anchors = {}
    pos = sorted(((tuple(p)) for p in (T["t1"] | T["t2"])), key=lambda p: _pid(*p))
    pv = {(_pid(*tuple(p))) for p in (T["t1"] | T["t2"])}
    n_pos = 0
    for r in verd:
        if n_pos >= n_anchor:
            break
        ka, kb = r.get("key_a"), r.get("key_b")
        if not (ka and kb) or ka not in ki or kb not in ki:
            continue
        if r.get("score") == 2 and (r.get("cos") or 0) >= 0.97 and _pid(ka, kb) in pv:
            add(ki[ka], ki[kb], "anchor")
            anchors[_pid(ka, kb)] = 2
            n_pos += 1
    negs = sorted((r for r in verd if r.get("score") == 0), key=lambda r: (r.get("cos") or 1))
    n_neg = 0
    for r in negs:
        if n_neg >= n_anchor:
            break
        ka, kb = r.get("key_a"), r.get("key_b")
        if ka in ki and kb in ki and _pid(ka, kb) not in pool:
            add(ki[ka], ki[kb], "anchor")
            anchors[_pid(ka, kb)] = 0
            n_neg += 1

    rows = sorted(pool.values(), key=lambda r: _h(r["pair_id"], "shuffle"))
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    json.dump(anchors, open(os.path.join(OUT, f"arbiter_anchors_{task}.json"), "w"))
    from collections import Counter
    print(f"[{task}] {len(rows)} eval pairs -> {path}  strata="
          f"{dict(Counter(r['stratum'] if not r['stratum'].startswith('spectrum') else 'spectrum' for r in rows))}"
          f"  anchors={len(anchors)}")
    return path


def payloads(task: str, per_agent: int = 100) -> List[str]:
    rows = [json.loads(l) for l in open(os.path.join(OUT, f"arbiter_eval_{task}.jsonl"))]
    pd = os.path.join(OUT, "arbiter_payloads")
    os.makedirs(pd, exist_ok=True)
    outs = []
    for a in range(0, len(rows), per_agent):
        i = a // per_agent
        p = os.path.join(pd, f"{task}_agent{i:02d}.jsonl")
        with open(p, "w") as f:
            for r in rows[a: a + per_agent]:
                f.write(json.dumps({"pair_id": r["pair_id"], "task": r["task"],
                                    "canonical_a": r["canonical_a"],
                                    "canonical_b": r["canonical_b"]}) + "\n")
        outs.append(p)
    print(f"[{task}] {len(outs)} payloads of <= {per_agent} -> {pd}")
    return outs


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--per-agent", type=int, default=100)
    a = ap.parse_args()
    for t in a.tasks.split(","):
        build_eval(t.strip())
        payloads(t.strip(), a.per_agent)
