#!/usr/bin/env python
"""Candidate-net RECALL CEILING diagnostic.

The candidate net caps repair recall: a true under-merge (adjudicated SAME but split across clusters
in the base partition) that is NOT in the net can never be recovered — downstream stages only prune.
So we measure, per width, what fraction of the repairable under-merges the net captures.

ceiling(net) = |SAME_cross ∩ net| / |SAME_cross|
  SAME_cross = adjudicated-SAME pairs that are CROSS-cluster in base (the repairable ones).
  max achievable recall = (SAME_already_co + |SAME_cross ∩ net|) / n_same.
Also reports the lexically-invisible residual: SAME_cross pairs with ~0 TF-IDF cos AND no shared
name — these need a semantic (BGE) net, not a lexical one.
"""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict

from .judge import canon_map
from .census import norm_term
from .sources import ROOT

OUT = os.path.join(ROOT, "outputs", "lexicon")


def _truth_pairs(task):
    truth = {k: bool(v) for k, v in json.load(open(os.path.join(OUT, f"adjudicated_truth_{task}.json"))).items()}
    ev = {r["pair_id"]: r for r in (json.loads(l) for l in open(os.path.join(OUT, f"arbiter_eval_{task}.jsonl")))}
    out = []
    for pid, same in truth.items():
        r = ev.get(pid)
        if r:
            out.append((r["key_a"], r["key_b"], same))
    return out


def sweep(task, cos_cuts=(0.5, 0.4, 0.3, 0.2, 0.1, 0.05), knn=30, name_df=12):
    from .repair import load_base_partition
    base = load_base_partition(task)
    cmap = canon_map(task)
    keys = sorted(k for k in cmap if k in base)
    texts = [cmap[k] for k in keys]
    ki = {k: i for i, k in enumerate(keys)}
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors
    X = TfidfVectorizer(min_df=2, max_features=40000, sublinear_tf=True).fit_transform(texts)
    nn = NearestNeighbors(n_neighbors=min(knn + 1, len(keys)), metric="cosine").fit(X)
    dist, idx = nn.kneighbors(X)

    # cos for every kNN cross-cluster pair
    knn_cos = {}
    for i in range(len(keys)):
        for jp in range(1, idx.shape[1]):
            j = int(idx[i][jp])
            if i < j and base[keys[i]] != base[keys[j]]:
                knn_cos[frozenset((keys[i], keys[j]))] = max(
                    knn_cos.get(frozenset((keys[i], keys[j])), 0.0), 1.0 - float(dist[i][jp]))
    # shared-rare-name cross-cluster pairs
    tok2 = defaultdict(list)
    for k in keys:
        for w in set(norm_term(cmap[k].split(".")[0][:80]).split()):
            if len(w) > 5:
                tok2[w].append(k)
    name_net = set()
    for w, ks in tok2.items():
        if 2 <= len(ks) <= name_df:
            for x in range(len(ks) - 1):
                a, b = ks[x], ks[x + 1]
                if base[a] != base[b]:
                    name_net.add(frozenset((a, b)))

    # v6 BGE candidate universe: every pair that got judged came from BGE kNN — this is the
    # semantic net (catches lexical paraphrases the TF-IDF/name nets miss). Cross-cluster subset.
    from .audit import load_verdicts
    bge_net = set()
    for r in load_verdicts(task):
        ka, kb = r.get("key_a"), r.get("key_b")
        if ka in base and kb in base and base[ka] != base[kb]:
            bge_net.add(frozenset((ka, kb)))

    tp = _truth_pairs(task)
    same_cross = [frozenset((a, b)) for a, b, s in tp if s and base.get(a) and base.get(b) and base[a] != base[b]]
    same_co = sum(1 for a, b, s in tp if s and base.get(a) == base.get(b))
    n_same = sum(1 for a, b, s in tp if s)
    sc = set(same_cross)

    def cov(net):
        return sum(1 for p in sc if p in net)

    rows = []
    for c in cos_cuts:
        knn_at = {p for p, v in knn_cos.items() if v >= c}
        net = knn_at | name_net
        cap = cov(net)
        rows.append({"min_cos": c, "knn_pairs": len(knn_at), "name_pairs": len(name_net),
                     "net_total": len(net), "same_cross_captured": cap,
                     "ceiling": round(cap / len(sc), 3) if sc else None,
                     "max_recall": round((same_co + cap) / n_same, 3) if n_same else None})
    # lexically-invisible residual (widest net misses)
    widest = {p for p in knn_cos} | name_net
    missed = [p for p in sc if p not in widest]
    return {"task": task, "n_same": n_same, "same_already_co": same_co,
            "n_same_cross_repairable": len(sc),
            "current_recall": round(same_co / n_same, 3) if n_same else None,
            "sweep": rows, "widest_net": len(widest), "widest_ceiling": round(cov(widest) / len(sc), 3) if sc else None,
            "lexically_invisible_missed": len(missed),
            "missed_examples": [[cmap[tuple(p)[0]][:70], cmap[tuple(p)[1]][:70]] for p in missed[:6]]}


def level_sweep(task, level="R1", caps=(9000, 18000, 30000, 50000), k=20, min_cos=.12):
    """Version-frozen R-level routing ceiling against LLM arbiter truth.

    Similarity generates/ranks candidates only. SAME labels come exclusively from complete arbiter
    vote files. The diagnostic includes eval pairs in its hypothetical candidate ranking while the
    actual build excludes them, matching ``build_level.emit_verify_net``'s separation.
    """
    from . import build_level as b
    import glob
    nodes, _ = b.nodes_from_level(task, level)
    ids = [n["node_id"] for n in nodes]
    reps = [b.rep_text(n) for n in nodes]
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors
    X = TfidfVectorizer(min_df=1, max_features=40000, sublinear_tf=True).fit_transform(reps)
    dist, idx = NearestNeighbors(n_neighbors=min(k + 1, len(ids)), metric="cosine").fit(X).kneighbors(X)
    cand = {}
    for i in range(len(ids)):
        for jp in range(1, idx.shape[1]):
            j = int(idx[i][jp])
            if j == i:
                continue
            cos = 1.0 - float(dist[i][jp])
            if cos >= min_cos:
                pair = frozenset((ids[i], ids[j]))
                cand[pair] = max(cand.get(pair, 0.0), cos)
    ranked = [pair for pair, _ in sorted(cand.items(), key=lambda x: -x[1])]
    votes = {}
    for path in glob.glob(os.path.join(OUT, "level_votes", f"arb_{task}_{level}_[0-9]*.jsonl")):
        for line in open(path):
            if not line.strip():
                continue
            row = json.loads(line); value = row.get("score")
            if type(value) is int and value in (0, 1, 2):
                votes[row["pair_id"]] = value
    eval_path = os.path.join(OUT, f"level_eval_{task}_{level}.jsonl")
    eval_rows = [json.loads(x) for x in open(eval_path) if x.strip()]
    same = {frozenset((row["node_a"], row["node_b"])) for row in eval_rows
            if votes.get(row["pair_id"]) == 2}
    rows = []
    for cap in list(caps) + [len(ranked)]:
        width = min(cap, len(ranked)); net = set(ranked[:width])
        rows.append({"cap": width, "same_captured": sum(x in net for x in same),
                     "n_truth_same": len(same),
                     "ceiling": round(sum(x in net for x in same)/len(same), 3) if same else None})
    # De-duplicate repeated full-width rows while retaining order.
    seen = set(); rows = [r for r in rows if not (r["cap"] in seen or seen.add(r["cap"]))]
    result = {"task": task, "level": level, "n_nodes": len(ids), "net": len(ranked),
              "k": k, "min_cos": min_cos, "truth_source": "LLM arbiter score==2",
              "semantic_measurement": False, "rows": rows}
    out = os.path.join(OUT, f"level_net_sweep_{task}_{level}.json")
    json.dump(result, open(out, "w"), indent=1)
    return result


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="humor,creative-writing")
    a = ap.parse_args()
    for t in a.tasks.split(","):
        print(json.dumps(sweep(t.strip()), indent=1))
        print("=" * 80)


if __name__ == "__main__":
    main()
