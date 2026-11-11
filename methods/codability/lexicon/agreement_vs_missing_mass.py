#!/usr/bin/env python3
"""Per-construct naming AGREEMENT vs NAME-SPACE MISSING MASS (Good-Turing f1/N over head_terms).

Probe for: is codability ≈ 1 - name-space missing mass (same distribution measured at head vs
tail), or do the two instruments carry different information? Both statistics are functionals of
the per-concept naming distribution: agreement = modal share p_max (Brown-Lenneberg), missing
mass = f1/N (Turing). They are NOT complements mechanically ({2,2,1}: p_max .4, f1/N .2), so the
correlation is empirical — but both are size-coupled, so the reported quantity is the PARTIAL
rank correlation given log(N) (size-confound lesson, 2026-07-07: raw agreement~size rho ~ -.85).

Concepts with n_named_sources >= MIN_N only (GT degenerate at n=2). Doc-level independence and
mirror caveats inherit from the census; this is a within-census consistency probe, not a new
population claim.
"""
import json
import sys

import numpy as np
from scipy import stats

LEX = "ROOT/outputs/lexicon"
TASKS = ["humor", "creative-writing", "news-homepages", "math-stackexchange"]
MIN_N = 3

def partial_spearman(x, y, z):
    """rank-based partial corr of x,y given z (residualize ranks on rank(z))."""
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    def resid(a, b):
        b1 = np.column_stack([np.ones_like(b), b])
        return a - b1 @ np.linalg.lstsq(b1, a, rcond=None)[0]
    ex, ey = resid(rx, rz), resid(ry, rz)
    r, p = stats.pearsonr(ex, ey)
    return r, p

print(f"{'task':22}{'n_concepts':>10} | {'rho(agr,mm)':>12}{'p':>8} | {'partial|logN':>13}{'p':>8} | note")
out = {}
for task in TASKS:
    d = json.load(open(f"{LEX}/census_{task}.json"))
    rows = []
    for cid, c in d["concept_level"]["concepts"].items():
        ht = c.get("head_terms") or {}
        N = sum(ht.values())
        if N < MIN_N:
            continue
        f1 = sum(1 for v in ht.values() if v == 1)
        rows.append((c["naming_agreement"], f1 / N, N))
    if len(rows) < 10:
        print(f"{task:22}{len(rows):>10} | too few multi-named concepts"); continue
    agr, mm, n = map(np.array, zip(*rows))
    rho, p = stats.spearmanr(agr, mm)
    pr, pp = partial_spearman(agr, mm, np.log(n))
    out[task] = {"n_concepts": len(rows), "spearman_raw": round(float(rho), 3),
                 "p_raw": round(float(p), 5), "partial_given_logN": round(float(pr), 3),
                 "p_partial": round(float(pp), 5),
                 "median_N": int(np.median(n)), "max_N": int(n.max())}
    print(f"{task:22}{len(rows):>10} | {rho:>12.3f}{p:>8.4f} | {pr:>13.3f}{pp:>8.4f} | medN={int(np.median(n))}")

json.dump(out, open(f"{LEX}/agreement_vs_missing_mass_20260720.json", "w"), indent=1)
print("\nwrote", f"{LEX}/agreement_vs_missing_mass_20260720.json")
