#!/usr/bin/env python3
"""PREREG-12 frozen analysis (registry 2026-07-23): name-swap vs natural-prompt-variation.
Per construct (14 PREREG-11 pairs, gpt56 lane):
  D_name = within − between_names   (scaffold A, from the PREREG-11 machinery)
  D_scaffold = within − between_scaffolds, between_scaffolds = Spearman(run-mean ranking
      scaffold A, run-mean ranking scaffold B) with the SAME name, averaged over arms.
PRIMARY: one-sided paired Wilcoxon over constructs, D_name > D_scaffold.
SECONDARY (separate): |level shift| name-swap (A: high vs low) vs scaffold-swap (same
name, B vs A, averaged over arms), one-sided paired Wilcoxon.
Descriptive: scaffold-B within-name split-half reliability (instrument check).
ONE analysis run. Output: outputs/lexicon/prereg12_results_20260723.json
"""
import itertools
import json
import os

import numpy as np
from scipy import stats

SP = ("/private/tmp/claude-502/-Users-anon-Projects-stanford-research-norm-research/"
      "6eb8228d-7011-4800-87e2-61a172f6003c/scratchpad")
N_PAIRS, N_RUNS, N_ITEMS = 14, 5, 30


def load(prefix):
    data = {}
    for pi in range(N_PAIRS):
        for arm in ("low", "high"):
            runs = []
            for r in range(N_RUNS):
                p = f"{SP}/{prefix}_out_p{pi:02d}_{arm}_r{r}.jsonl"
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


def main():
    A, B = load("p11"), load("p12")
    res = {"constructs": []}
    d_name, d_scaf, s_name, s_scaf, wB = [], [], [], [], []
    for pi in range(N_PAIRS):
        MlA, MhA = item_matrix(A[(pi, "low")]), item_matrix(A[(pi, "high")])
        MlB, MhB = item_matrix(B[(pi, "low")]), item_matrix(B[(pi, "high")])
        w = np.nanmean([within(MlA), within(MhA)])
        btw_name = sp(np.nanmean(MlA, 1), np.nanmean(MhA, 1))
        btw_scaf = np.nanmean([sp(np.nanmean(MlA, 1), np.nanmean(MlB, 1)),
                               sp(np.nanmean(MhA, 1), np.nanmean(MhB, 1))])
        if np.isnan(w) or np.isnan(btw_name) or np.isnan(btw_scaf):
            print(f"pair {pi}: INCOMPLETE (w={w} bn={btw_name} bs={btw_scaf}) — skipped")
            continue
        d_name.append(w - btw_name)
        d_scaf.append(w - btw_scaf)
        s_name.append(abs(float(np.nanmean(MhA) - np.nanmean(MlA))))
        s_scaf.append(float(np.mean([abs(np.nanmean(MlB) - np.nanmean(MlA)),
                                     abs(np.nanmean(MhB) - np.nanmean(MhA))])))
        wB.append(np.nanmean([within(MlB), within(MhB)]))
        res["constructs"].append({
            "pair": pi, "within_A": round(float(w), 3),
            "between_names": round(float(btw_name), 3),
            "between_scaffolds": round(float(btw_scaf), 3),
            "D_name": round(d_name[-1], 3), "D_scaffold": round(d_scaf[-1], 3),
            "shift_name": round(s_name[-1], 3), "shift_scaffold": round(s_scaf[-1], 3),
            "within_B": round(float(wB[-1]), 3)})
    dn, ds = np.array(d_name), np.array(d_scaf)
    w1 = stats.wilcoxon(dn - ds, alternative="greater")
    w2 = stats.wilcoxon(np.array(s_name) - np.array(s_scaf), alternative="greater")
    res["primary"] = {"mean_D_name": round(float(dn.mean()), 3),
                      "mean_D_scaffold": round(float(ds.mean()), 3),
                      "wilcoxon_p": float(w1.pvalue), "n_constructs": len(dn),
                      "n_name_gt_scaffold": int((dn > ds).sum())}
    res["secondary"] = {"mean_shift_name": round(float(np.mean(s_name)), 3),
                        "mean_shift_scaffold": round(float(np.mean(s_scaf)), 3),
                        "wilcoxon_p": float(w2.pvalue)}
    res["descriptive"] = {"within_B_mean": round(float(np.nanmean(wB)), 3)}
    print("PRIMARY", res["primary"])
    print("SECONDARY", res["secondary"])
    print("DESCRIPTIVE", res["descriptive"])
    out = ("ROOT/outputs/lexicon/"
           "prereg12_results_20260723.json")
    json.dump(res, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
