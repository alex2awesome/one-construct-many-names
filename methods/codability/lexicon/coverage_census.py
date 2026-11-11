#!/usr/bin/env python3
"""Species-accumulation / Good-Turing coverage census over the certified L0->R3 hierarchy.

Token = one raw author criterion (canon key task::layer::doc::item_idx).
Species = its cluster at grain L0 / R1 construct / R2 theme / R3 category.
Independence unit for rarefaction = source DOC (census design lock), permuted.

Outputs per task x grain:
  - Good-Turing missing mass f1/N  (P(next criterion is a new species))
  - GT coverage 1 - f1/N
  - Chao1 lower-bound richness + observed/Chao1 (labelled lower-bound, f2 guard)
  - Heaps alpha (interior 5-95% log-log OLS on doc-permuted mean curve)
  - pct of final species seen by 10% / 50% of docs
"""
import json, os, sys, random
from collections import Counter, defaultdict
import numpy as np

ROOT = "ROOT"
LEX = f"{ROOT}/outputs/lexicon"
TASKS = ["humor", "creative-writing", "code-review", "peer-review", "math-stackexchange",
         "news-homepages", "press-releases", "grant-funding", "legal-outcome-prediction",
         "notice-and-comment", "patents"]
REPS = 25
rng = random.Random(0)

def jload(p):
    return {str(k): str(v) for k, v in json.load(open(p)).items()}

def heaps_alpha_interior(m, S):
    """log-log OLS on interior 5-95% of tokens (endpoint-bend guard per 2026-07-18 audit)."""
    m = np.asarray(m, float); S = np.asarray(S, float)
    lo, hi = 0.05 * m[-1], 0.95 * m[-1]
    sel = (m >= max(lo, 2)) & (m <= hi) & (S > 0)
    if sel.sum() < 5: return float("nan")
    x, y = np.log(m[sel]), np.log(S[sel])
    return float(np.polyfit(x, y, 1)[0])

results = {}
for task in TASKS:
    l0 = jload(f"{LEX}/partition_{task}_L0v4.json")          # raw key -> L0 cluster
    r1 = jload(f"{LEX}/partition_{task}_R1.json")            # L0 -> construct
    r2 = jload(f"{LEX}/partition_{task}_R2.json")            # construct -> theme
    r3 = jload(f"{LEX}/partition_{task}_R3.json")            # theme -> category
    # species label per raw key at each grain
    grains = {}
    grains["L0"] = dict(l0)
    grains["R1"] = {k: r1.get(v) for k, v in l0.items()}
    grains["R2"] = {k: r2.get(g) for k, g in grains["R1"].items()}
    grains["R3"] = {k: r3.get(t) for k, t in grains["R2"].items()}
    # doc per key
    doc_of = {k: k.split("::")[2] for k in l0}
    docs = sorted(set(doc_of.values()))
    keys_by_doc = defaultdict(list)
    for k in l0: keys_by_doc[doc_of[k]].append(k)
    N = len(l0)
    tres = {"n_criteria": N, "n_docs": len(docs)}
    for g, lab in grains.items():
        miss = [k for k, v in lab.items() if v is None]
        cnt = Counter(v for v in lab.values() if v is not None)
        S_obs = len(cnt)
        f1 = sum(1 for c in cnt.values() if c == 1)
        f2 = sum(1 for c in cnt.values() if c == 2)
        n_tok = sum(cnt.values())
        gt_miss = f1 / n_tok
        chao1 = S_obs + (f1 * f1) / (2 * f2) if f2 > 0 else float("nan")
        # doc-permuted accumulation (mean over REPS)
        curves = []
        for rep in range(REPS):
            order = docs[:]; random.Random(rep).shuffle(order)
            seen = set(); m = 0; xs = []; ys = []
            for d in order:
                for k in keys_by_doc[d]:
                    v = lab.get(k)
                    m += 1
                    if v is not None: seen.add(v)
                xs.append(m); ys.append(len(seen))
            curves.append((xs, ys))
        xs0 = curves[0][0]
        Sm = np.mean([c[1] for c in curves], axis=0)
        alpha = heaps_alpha_interior(xs0, Sm)
        nd = len(docs)
        pct10 = float(Sm[max(0, int(0.10 * nd) - 1)] / S_obs)
        pct50 = float(Sm[max(0, int(0.50 * nd) - 1)] / S_obs)
        # empirical new-species rate in the last 10% of tokens
        tail_new = float((Sm[-1] - Sm[max(0, int(0.9 * nd) - 1)]) / (xs0[-1] - xs0[max(0, int(0.9 * nd) - 1)] + 1e-9))
        tres[g] = dict(S_obs=S_obs, f1=f1, f2=f2, unmapped=len(miss),
                       gt_missing_mass=round(gt_miss, 4), gt_coverage=round(1 - gt_miss, 4),
                       chao1=round(chao1, 1) if chao1 == chao1 else None,
                       obs_over_chao1=round(S_obs / chao1, 3) if chao1 == chao1 else None,
                       heaps_alpha=round(alpha, 3),
                       pct_species_at_10pct_docs=round(pct10, 3),
                       pct_species_at_50pct_docs=round(pct50, 3),
                       tail10_new_species_rate=round(tail_new, 4))
    results[task] = tres
    r = tres
    print(f"{task:26} N={N:5} docs={len(docs):4} | " + " | ".join(
        f"{g}: S={r[g]['S_obs']:5} GTmiss={r[g]['gt_missing_mass']:.3f} a={r[g]['heaps_alpha']:.2f}"
        for g in ["L0", "R1", "R2", "R3"]))

out = f"{LEX}/coverage_census_20260719.json"
json.dump(results, open(out, "w"), indent=1)
print("\nwrote", out)
