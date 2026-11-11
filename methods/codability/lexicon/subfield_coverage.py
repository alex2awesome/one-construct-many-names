#!/usr/bin/env python3
"""Per-SUBFIELD (dialect bucket) coverage: Heaps alpha + Good-Turing, within task.

Doc -> bucket via strata.subtask_short (contexts_<task>.jsonl) + dialect.BUCKETS keyword rules.
Cross-bucket saturation compared at COMMON RAREFIED DEPTH m* (GT missing mass is n-dependent;
raw full-sample GT also reported but never compared across buckets of different size).
Only the 4 tasks with BUCKETS entries. junk_doc excluded.
"""
import json, os, sys, random
from collections import Counter, defaultdict
import numpy as np

ROOT = "ROOT"
LEX = f"{ROOT}/outputs/lexicon"
sys.path.insert(0, ROOT)
from methods.codability.lexicon.dialect import bucket_of, BUCKETS

TASKS = ["humor", "creative-writing", "news-homepages", "math-stackexchange"]
MIN_TOK = 250   # bucket eligibility
REPS = 200

def jload(p):
    return {str(k): str(v) for k, v in json.load(open(p)).items()}

def gt_at_depth(keys_by_doc, lab, docs, mstar, reps, seed0=0):
    """mean f1/m over doc-subsamples truncated at m* tokens; also mean distinct species."""
    rates, rich = [], []
    for rep in range(reps):
        order = docs[:]; random.Random(seed0 + rep).shuffle(order)
        toks = []
        for d in order:
            toks.extend(lab[k] for k in keys_by_doc[d])
            if len(toks) >= mstar: break
        if len(toks) < mstar: return None, None
        c = Counter(toks[:mstar])
        rates.append(sum(1 for v in c.values() if v == 1) / mstar)
        rich.append(len(c))
    return float(np.mean(rates)), float(np.mean(rich))

def heaps_interior(m, S):
    m = np.asarray(m, float); S = np.asarray(S, float)
    sel = (m >= max(0.05 * m[-1], 2)) & (m <= 0.95 * m[-1]) & (S > 0)
    if sel.sum() < 5: return float("nan")
    return float(np.polyfit(np.log(m[sel]), np.log(S[sel]), 1)[0])

out = {}
for task in TASKS:
    l0 = jload(f"{LEX}/partition_{task}_L0v4.json")
    r1 = jload(f"{LEX}/partition_{task}_R1.json")
    lab = {"L0": l0, "R1": {k: r1[v] for k, v in l0.items()}}
    # doc -> bucket (subtask_short is doc-level; first record wins)
    doc_sub = {}
    for ln in open(f"{LEX}/contexts_{task}.jsonl"):
        r = json.loads(ln)
        d = r["doc"]
        if d not in doc_sub:
            doc_sub[d] = bucket_of(task, (r.get("strata") or {}).get("subtask_short") or "")
    keys_by_doc = defaultdict(list)
    for k in l0:
        keys_by_doc[k.split("::")[2]].append(k)
    by_bucket = defaultdict(list)
    for d in keys_by_doc:
        b = doc_sub.get(d, "other")
        if b != "junk_doc":
            by_bucket[b].append(d)
    sizes = {b: sum(len(keys_by_doc[d]) for d in ds) for b, ds in by_bucket.items()}
    elig = {b for b, n in sizes.items() if n >= MIN_TOK}
    if not elig: continue
    mstar = min(sizes[b] for b in elig)
    rows = []
    for b in sorted(elig, key=lambda x: -sizes[x]):
        docs = by_bucket[b]
        res = {"bucket": b, "n_docs": len(docs), "n_criteria": sizes[b]}
        for g in ["L0", "R1"]:
            cnt = Counter(lab[g][k] for d in docs for k in keys_by_doc[d])
            f1 = sum(1 for v in cnt.values() if v == 1)
            res[f"{g}_S"] = len(cnt)
            res[f"{g}_gt_raw"] = round(f1 / sizes[b], 3)
            gt_r, rich_r = gt_at_depth(keys_by_doc, lab[g], docs, mstar, REPS)
            res[f"{g}_gt_at_mstar"] = round(gt_r, 3) if gt_r is not None else None
            res[f"{g}_S_at_mstar"] = round(rich_r, 1) if rich_r is not None else None
            # heaps on doc-permuted mean curve (25 perms)
            curves = []
            for rep in range(25):
                order = docs[:]; random.Random(rep).shuffle(order)
                seen = set(); m = 0; xs, ys = [], []
                for d in order:
                    for k in keys_by_doc[d]:
                        seen.add(lab[g][k]); m += 1
                    xs.append(m); ys.append(len(seen))
                curves.append(ys)
            res[f"{g}_heaps"] = round(heaps_interior(xs, np.mean(curves, axis=0)), 2)
        rows.append(res)
    out[task] = {"mstar": mstar, "buckets": rows,
                 "note": "compare across buckets ONLY at mstar; junk_doc excluded"}
    print(f"\n=== {task} (m*={mstar} criteria; buckets >= {MIN_TOK})")
    print(f"{'bucket':22}{'docs':>5}{'crit':>6} | L0: {'gt@m*':>6}{'S@m*':>7}{'alpha':>6} | R1: {'gt@m*':>6}{'S@m*':>7}{'alpha':>6}")
    for r in sorted(rows, key=lambda r: r["R1_gt_at_mstar"]):
        print(f"{r['bucket']:22}{r['n_docs']:>5}{r['n_criteria']:>6} |     {r['L0_gt_at_mstar']:>6}{r['L0_S_at_mstar']:>7}{r['L0_heaps']:>6}"
              f" |     {r['R1_gt_at_mstar']:>6}{r['R1_S_at_mstar']:>7}{r['R1_heaps']:>6}")

json.dump(out, open(f"{LEX}/coverage_census_subfields_20260719.json", "w"), indent=1)
print("\nwrote", f"{LEX}/coverage_census_subfields_20260719.json")
