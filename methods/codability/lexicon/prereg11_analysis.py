#!/usr/bin/env python3
"""PREREG-11 frozen analysis (registry 2026-07-23): name-register scoring variance.
Per family (gpt56 / glm / claude), per construct:
  - PRIMARY: mean split-half Spearman of item ranking WITHIN name (run halves) minus
    Spearman BETWEEN names (run-mean item rankings, low vs high). One-sided Wilcoxon
    (within > between) over constructs.
  - SECONDARY: per-item |name gap| vs permutation null (shuffle name labels over the 10
    runs), per-construct p, Fisher over constructs.
  - TERTIARY (descriptive): mean shift high-vs-low; |shift| x judged height gap rho;
    ASYMMETRY: within-name run reliability low-name vs high-name (official-anchoring).
ONE run per family, reported separately, never pooled.
Output: outputs/lexicon/prereg11_results_20260723.json
"""
import glob
import itertools
import json
import os
import random
from collections import defaultdict

import numpy as np
from scipy import stats

SP = ("/private/tmp/claude-502/-Users-anon-Projects-stanford-research-norm-research/"
      "6eb8228d-7011-4800-87e2-61a172f6003c/scratchpad")
LANES = {"gpt56": "p11_out_p{pi:02d}_{arm}_r{r}.jsonl",
         "glm": "p11_glm_out_p{pi:02d}_{arm}_r{r}.jsonl",
         "claude": "p11_claude_out_p{pi:02d}_{arm}_r{r}.jsonl"}
N_PAIRS, N_RUNS, N_ITEMS = 14, 5, 30
N_PERM = 2000


def load(lane):
    data = {}
    for pi in range(N_PAIRS):
        for arm in ("low", "high"):
            runs = []
            for r in range(N_RUNS):
                p = f"{SP}/" + LANES[lane].format(pi=pi, arm=arm, r=r)
                sc = {}
                if os.path.exists(p):
                    for l in open(p):
                        try:
                            row = json.loads(l)
                            s = int(row["score"])
                            if 1 <= s <= 7:
                                sc[int(row["item"])] = s
                        except Exception:
                            pass
                runs.append(sc)
            data[(pi, arm)] = runs
    return data


def item_matrix(runs):
    """items x runs matrix (nan for missing)."""
    M = np.full((N_ITEMS, N_RUNS), np.nan)
    for r, sc in enumerate(runs):
        for i, s in sc.items():
            if i < N_ITEMS:
                M[i, r] = s
    return M


def sp(a, b):
    m = ~np.isnan(a) & ~np.isnan(b)
    if m.sum() < 10 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return np.nan
    return stats.spearmanr(a[m], b[m]).statistic


def analyze(lane, rng):
    data = load(lane)
    res = {"constructs": []}
    d_within_between, gaps_p, shifts, rel_low, rel_high = [], [], [], [], []
    for pi in range(N_PAIRS):
        Ml = item_matrix(data[(pi, "low")])
        Mh = item_matrix(data[(pi, "high")])
        # within-name split-half (all 10 half-splits of 5 runs -> use all (2,3) splits)
        def within(M):
            vals = []
            for combo in itertools.combinations(range(N_RUNS), 2):
                a = np.nanmean(M[:, list(combo)], axis=1)
                rest = [x for x in range(N_RUNS) if x not in combo]
                b = np.nanmean(M[:, rest], axis=1)
                v = sp(a, b)
                if not np.isnan(v):
                    vals.append(v)
            return float(np.mean(vals)) if vals else np.nan
        w = np.nanmean([within(Ml), within(Mh)])
        btw = sp(np.nanmean(Ml, axis=1), np.nanmean(Mh, axis=1))
        if not (np.isnan(w) or np.isnan(btw)):
            d_within_between.append(w - btw)
        # secondary: permutation on |mean gap| per construct (pooled over items)
        allruns = np.concatenate([Ml, Mh], axis=1)   # items x 10
        obs = np.nanmean(np.abs(np.nanmean(Ml, 1) - np.nanmean(Mh, 1)))
        ge = 0
        for _ in range(N_PERM):
            perm = rng.sample(range(10), 10)
            A, B = allruns[:, perm[:5]], allruns[:, perm[5:]]
            g = np.nanmean(np.abs(np.nanmean(A, 1) - np.nanmean(B, 1)))
            if g >= obs:
                ge += 1
        p = (1 + ge) / (1 + N_PERM)
        gaps_p.append(p)
        shifts.append(float(np.nanmean(Mh) - np.nanmean(Ml)))
        rel_low.append(within(Ml))
        rel_high.append(within(Mh))
        res["constructs"].append({"pair": pi, "within_rel": round(float(w), 3),
                                  "between": round(float(btw), 3),
                                  "gap_perm_p": round(p, 4),
                                  "shift_high_minus_low": round(shifts[-1], 3)})
    d = np.array(d_within_between)
    wilc = stats.wilcoxon(d, alternative="greater")
    X = -2 * sum(np.log(p) for p in gaps_p)
    fisher_p = float(1 - stats.chi2.cdf(X, 2 * len(gaps_p)))
    asym = stats.wilcoxon(np.array(rel_high) - np.array(rel_low))
    res["primary"] = {"mean_within_minus_between": round(float(d.mean()), 3),
                      "wilcoxon_p": round(float(wilc.pvalue), 6), "n_constructs": len(d)}
    res["secondary_fisher"] = {"chi2": round(float(X), 2), "p": round(fisher_p, 6),
                               "n_sig_constructs": sum(1 for p in gaps_p if p < .05)}
    res["tertiary"] = {"mean_shift_high_minus_low": round(float(np.mean(shifts)), 3),
                       "rel_high_mean": round(float(np.nanmean(rel_high)), 3),
                       "rel_low_mean": round(float(np.nanmean(rel_low)), 3),
                       "asym_wilcoxon_p": round(float(asym.pvalue), 4)}
    print(lane, "PRIMARY", res["primary"], "| SECONDARY", res["secondary_fisher"],
          "| TERTIARY", res["tertiary"])
    return res


def main():
    out = {}
    for lane in LANES:
        out[lane] = analyze(lane, random.Random(11))
    path = ("ROOT/outputs/lexicon/"
            "prereg11_results_20260723.json")
    json.dump(out, open(path, "w"), indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
