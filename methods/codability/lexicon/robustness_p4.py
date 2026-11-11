#!/usr/bin/env python3
"""PREREG-4 pre-publication robustness passes (descriptive, NOT new preregs — the
confirmatory result is the frozen 20260721_v2 R1 run; these check its two recorded caveats):

  A. same-class text-reuse dedup — if two docs share heavy verbatim text (boilerplate,
     quoted guidelines), identical names could be literal reuse, not independent convention.
     Pass: doc-level 5-gram shingle Jaccard over extracted quotes; greedily drop any doc
     with Jaccard >= .5 against a kept doc (same guard threshold as mirror-collapse), then
     re-run the same-class-vs-cross-class coincidence gap + doc-label permutation null.

  B. drop-top-K concept pooling — the pooled gap could ride on a few fat concepts.
     Pass: drop the K most-populous R1 concepts per field (K=1 and K=5), re-run.

Same machinery as prereg_tests (primary split, 1000 doc-level perms, Fisher combine).
Output: outputs/lexicon/prereg4_robustness_20260722.json
"""
import json
import random
import re
from collections import Counter, defaultdict

import numpy as np
from scipy import stats

from methods.codability.lexicon.codability_sampling_model import LEX
from methods.codability.lexicon.prereg_tests import (
    N_PERM, SPLITS, WIDENED, coincidence_gap, field_data)


def shingles(text, n=5):
    toks = re.findall(r"[a-z0-9]+", text.casefold())
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def dedup_docs(rows):
    """Greedy doc-level dedup by quote-shingle Jaccard >= .5 (deterministic order)."""
    doc_sh = defaultdict(set)
    for r in rows:
        doc_sh[r["doc"]] |= shingles(r.get("quote", ""))
    kept, dropped = [], set()
    for d in sorted(doc_sh):
        sh = doc_sh[d]
        dup = False
        for k in kept:
            ks = doc_sh[k]
            if sh and ks:
                inter = len(sh & ks)
                if inter and inter / len(sh | ks) >= 0.5:
                    dup = True
                    break
        if dup:
            dropped.add(d)
        else:
            kept.append(d)
    return [r for r in rows if r["doc"] not in dropped], len(dropped)


def drop_top_k(rows, k):
    top = {c for c, _ in Counter(r["con"] for r in rows).most_common(k)}
    return [r for r in rows if r["con"] not in top]


def run_pass(name, transform, cmap, rng):
    out = {}
    ps = []
    for task in WIDENED:
        rows, rungs = field_data(task, "R1")
        rows, meta = transform(rows)
        doc_cls = {d: cmap[r] for d, r in rungs.items()}
        obs, ds, dc = coincidence_gap(rows, doc_cls)
        if obs is None or ds < 50 or dc < 50:
            out[task] = {"usable": False, "same_pairs": ds, "cross_pairs": dc, "meta": meta}
            continue
        docs = sorted({r["doc"] for r in rows})
        labels = [doc_cls[d] for d in docs]
        ge = valid = 0
        for _ in range(N_PERM):
            rng.shuffle(labels)
            g, _, _ = coincidence_gap(rows, dict(zip(docs, labels)))
            if g is None:
                continue
            valid += 1
            if g >= obs:
                ge += 1
        p = (1 + ge) / (1 + valid)
        out[task] = {"usable": True, "gap": round(obs, 4), "same_pairs": ds,
                     "cross_pairs": dc, "p_perm": round(p, 4), "meta": meta}
        ps.append(p)
    if ps:
        X = -2 * sum(np.log(p) for p in ps)
        out["_fisher"] = {"chi2": round(float(X), 2), "df": 2 * len(ps),
                          "p": round(float(1 - stats.chi2.cdf(X, 2 * len(ps))), 6),
                          "fields_used": len(ps)}
    print(name, json.dumps(out.get("_fisher"), indent=None), flush=True)
    return out


def main():
    cmap = SPLITS["primary_12v345"]
    results = {}
    results["A_textreuse_dedup"] = run_pass(
        "A_textreuse_dedup", lambda rows: (lambda rr, nd: (rr, {"docs_dropped": nd}))(*dedup_docs(rows)),
        cmap, random.Random(41))
    for k in (1, 5):
        results[f"B_drop_top{k}"] = run_pass(
            f"B_drop_top{k}", lambda rows, k=k: (drop_top_k(rows, k), {"k": k}),
            cmap, random.Random(41 + k))
    path = f"{LEX}/prereg4_robustness_20260722.json"
    json.dump(results, open(path, "w"), indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
