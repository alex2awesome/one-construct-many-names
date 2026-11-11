"""Convergent validity of the dialect-battery instruments: do they agree on WHERE dialect
lives? Three levels (2026-07-09, follows the user's "do these results all correlate?"):

  construct  per-construct within-cross deltas: jaccard / chrf3 / semantic / community-LM
             advantage. Spearman per task + pooled with within-task [0,1] rank normalization
             (the CODA pooling lesson — never pool raw ranks across unequal groups).
  bucket     per-community signal: classifier per-class OVR AUC / LM per-bucket advantage /
             pairwise per-bucket jaccard delta / Fightin' Words significant-term count.
             Pooled across tasks, within-task rank-normalized.
  domain     n=4 — rank table only, no correlation claimed.

No permutation nulls here (point estimates only; the battery already carries the nulls).
  python -m methods.codability.lexicon.battery_convergence
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np

from .dialect import OUT, load_groups
from .dialect_battery import (MIRROR, classifier_probe, community_lm, embed_records,
                              fightin_words, flatten, mirror_components, model_records,
                              paired_contrast)

TASKS = ["humor", "creative-writing", "news-homepages", "math-stackexchange"]


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    # midranks for ties
    for v in np.unique(a):
        m = a == v
        ra[m] = ra[m].mean()
    for v in np.unique(b):
        m = b == v
        rb[m] = rb[m].mean()
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def nrank01(v):
    v = np.asarray(v, float)
    r = np.argsort(np.argsort(v)).astype(float)
    for u in np.unique(v):
        m = v == u
        r[m] = r[m].mean()
    return r / max(len(v) - 1, 1)


def per_bucket_jaccard(rows):
    """bucket -> mean over constructs of (within-b pair jac − cross-pairs-involving-b jac)."""
    by_c = defaultdict(list)
    for r in rows:
        by_c[r["construct"]].append(r)
    acc = defaultdict(list)
    for rs in by_c.values():
        n = len(rs)
        if n < 2:
            continue
        tsets = [set(r["terms"]) for r in rs]
        pairs = defaultdict(lambda: {"w": [], "c": []})
        for i in range(n):
            for j in range(i + 1, n):
                a, b = tsets[i], tsets[j]
                if not (a or b):
                    continue
                qa, qb = rs[i]["qtok"], rs[j]["qtok"]
                if qa and qb and len(qa & qb) / len(qa | qb) >= MIRROR:
                    continue
                jac = len(a & b) / len(a | b)
                bi, bj = rs[i]["bucket"], rs[j]["bucket"]
                if bi == bj:
                    pairs[bi]["w"].append(jac)
                else:
                    pairs[bi]["c"].append(jac)
                    pairs[bj]["c"].append(jac)
        for b, d in pairs.items():
            if d["w"] and d["c"]:
                acc[b].append(float(np.mean(d["w"]) - np.mean(d["c"])))
    return {b: float(np.mean(v)) for b, v in acc.items() if len(v) >= 3}


def corr_matrix(cols: dict, min_n: int = 10):
    names = list(cols)
    M = {}
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            keys = [k for k in cols[x] if k in cols[y]]
            if len(keys) < min_n:
                M[(x, y)] = (np.nan, len(keys))
                continue
            M[(x, y)] = (spearman([cols[x][k] for k in keys], [cols[y][k] for k in keys]),
                         len(keys))
    return M


def main():
    con_pool = defaultdict(dict)   # instrument -> (task, cid) -> rank01
    buck_pool = defaultdict(dict)  # instrument -> (task, bucket) -> rank01
    for task in TASKS:
        groups = load_groups(task, os.path.join(OUT, f"partition_{task}.json"))
        rows = flatten(groups, task)
        mc = mirror_components(rows)
        recs, classes = model_records(rows, mc)
        emb = embed_records(rows)
        jac = paired_contrast(rows, "jaccard", B=1)["per_construct"]
        chf = paired_contrast(rows, "chrf3", B=1)["per_construct"]
        sem = paired_contrast(rows, "cosine", emb=emb, B=1)["per_construct"]
        lm = community_lm(recs, classes, B=1)
        clf = classifier_probe(recs, classes, B=1)
        fw = fightin_words(recs, classes)
        con = {"jaccard": jac, "chrf3": chf, "semantic": sem, "lm_adv": lm["per_construct"]}
        buck = {"clf_auc": clf["per_class_auc"], "lm_adv": lm["per_bucket"],
                "jaccard": per_bucket_jaccard(rows), "fw_nsig": fw["n_sig_per_bucket"]}
        print(f"\n===== {task} =====")
        M = corr_matrix(con)
        print("  construct-level Spearman (within task):")
        for (x, y), (r, n) in M.items():
            print(f"    {x:9s} ~ {y:9s}  rho={r:+.3f}  (n={n})")
        Mb = corr_matrix(buck, min_n=5)
        print("  bucket-level Spearman (within task):")
        for (x, y), (r, n) in Mb.items():
            print(f"    {x:9s} ~ {y:9s}  rho={r:+.3f}  (n={n})")
        # pool with within-task rank normalization
        for inst, d in con.items():
            ks = sorted(d)
            for k, v in zip(ks, nrank01([d[k] for k in ks])):
                con_pool[inst][(task, k)] = v
        for inst, d in buck.items():
            ks = sorted(d)
            for k, v in zip(ks, nrank01([d[k] for k in ks])):
                buck_pool[inst][(task, k)] = v

    print("\n===== POOLED (within-task rank-normalized) =====")
    print("  construct level:")
    for (x, y), (r, n) in corr_matrix(con_pool).items():
        print(f"    {x:9s} ~ {y:9s}  rho={r:+.3f}  (n={n})")
    print("  bucket level:")
    for (x, y), (r, n) in corr_matrix(buck_pool, min_n=10).items():
        print(f"    {x:9s} ~ {y:9s}  rho={r:+.3f}  (n={n})")
    json.dump({"construct_pooled": {f"{x}~{y}": v for (x, y), v in corr_matrix(con_pool).items()},
               "bucket_pooled": {f"{x}~{y}": v for (x, y), v in
                                 corr_matrix(buck_pool, min_n=10).items()}},
              open(os.path.join(OUT, "dialect_battery_convergence.json"), "w"), indent=1,
              default=float)
    print(f"\n-> {os.path.join(OUT, 'dialect_battery_convergence.json')}")


if __name__ == "__main__":
    main()
