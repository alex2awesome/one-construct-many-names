#!/usr/bin/env python
"""PREREG-26 analysis (frozen design: notes/2026-10-12__prereg26-cw-name-swap.md, with Amendment 1).

Mirrors prereg25_analysis.py without the AUC secondaries. Per family (GPT-5.6, Claude; never pooled):
  within(A) = rho(A_S0_r0, A_S0_r1); within(B) likewise; within = mean of the two
  between_name = mean of rho(A_S0_ri, B_S0_rj), 4 (i,j);   Delta_name = within - between_name
  between_scaf = mean of rho(A_S0_ri, A_Sk_r0), i in {0,1}, k in {1,2};   Delta_scaf = within(A) - between_scaf
  PRIMARY: one-sided paired Wilcoxon over constructs, Delta_name > Delta_scaf, alpha .05
  TOST equivalence of Delta_scaf to 0, margin .10 (two one-sided one-sample t-tests)
  register (descriptive): Spearman of Delta_name with the formality gap
Analyses: PRIMARY = calibrated-standard selection (p26_pairs.json);
  SECONDARY (a) = primary restricted to constructs whose selected pair is also strict-same by both families;
  SECONDARY (b) = strict-standard selection (p26_pairs_strict.json; runs reused when the pair is identical,
  otherwise the s_-prefixed runs).
Runs: OUT/runs/<gpt56|claude>/<job>.json from prereg26_run.py (parsed = [{item_idx, score}] x 40).
"""
from __future__ import annotations

import json
import os
from itertools import product

import numpy as np
from scipy.stats import spearmanr, ttest_1samp, wilcoxon

ROOT = "ROOT"
OUT = f"{ROOT}/outputs/lexicon/prereg26_cw_20261012"
MARGIN = 0.10
FAMS = ("gpt56", "claude")


def load_runs(fam):
    d = f"{OUT}/{fam}/score"
    R = {}
    if not os.path.isdir(d):
        return R
    for f in os.listdir(d):
        if not f.endswith(".json"):
            continue
        r = json.load(open(f"{d}/{f}"))
        if r.get("status") != "pass":
            continue
        sc = {int(x["item_idx"]): float(x["score"]) for x in r["parsed"]}
        if len(sc) == 40:
            R[f[:-5]] = sc
    return R


def rho(x, y):
    if x is None or y is None:
        return None
    ks = sorted(x)
    a, b = [x[k] for k in ks], [y[k] for k in ks]
    if len(set(a)) < 2 or len(set(b)) < 2:
        return None
    return float(spearmanr(a, b).statistic)


def mean_defined(xs):
    xs = [x for x in xs if x is not None]
    return (float(np.mean(xs)) if xs else None), len(xs)


def analyse(fam, pairs, R, prefix_of):
    per, undefined = [], []
    for p in pairs:
        mid = p["mid"]
        pre = prefix_of(p)
        g = lambda arm, scaf, k: R.get(f"{pre}{mid}_{arm}_{scaf}_r{k}")
        rec = {"mid": mid, "name_A": p["name_A"], "name_B": p["name_B"], "gap": round(p["gap"], 3),
               "n_valid_runs": sum(g(a, s, k) is not None for a, s, k in
                                   [("A", "S0", 0), ("A", "S0", 1), ("B", "S0", 0), ("B", "S0", 1), ("A", "S1", 0), ("A", "S2", 0)])}
        wA, wB = rho(g("A", "S0", 0), g("A", "S0", 1)), rho(g("B", "S0", 0), g("B", "S0", 1))
        rec["within_A"], rec["within_B"] = wA, wB
        rec["within"], _ = mean_defined([wA, wB])
        bn = [rho(g("A", "S0", i), g("B", "S0", j)) for i, j in product((0, 1), (0, 1))]
        rec["between_name"], rec["n_between_name"] = mean_defined(bn)
        bs = [rho(g("A", "S0", i), g("A", sk, 0)) for i, sk in product((0, 1), ("S1", "S2"))]
        rec["between_scaf"], rec["n_between_scaf"] = mean_defined(bs)
        for nm, v in (("within_A", wA), ("within_B", wB)):
            if v is None and all(g(nm[-1], "S0", k) is not None for k in (0, 1)):
                undefined.append((mid, nm))
        rec["delta_name"] = (rec["within"] - rec["between_name"]
                             if rec["within"] is not None and rec["between_name"] is not None else None)
        rec["delta_scaf"] = (wA - rec["between_scaf"] if wA is not None and rec["between_scaf"] is not None else None)
        per.append(rec)
    out = {"family": fam, "n_selected": len(pairs)}
    both = [r for r in per if r["delta_name"] is not None and r["delta_scaf"] is not None]
    out["n_constructs_primary"] = len(both)
    if both:
        dn = np.array([r["delta_name"] for r in both])
        ds = np.array([r["delta_scaf"] for r in both])
        for k in ("within", "between_name", "between_scaf"):
            out[f"mean_{k}"] = round(float(np.mean([r[k] for r in both])), 4)
        out["mean_delta_name"] = round(float(dn.mean()), 4)
        out["mean_delta_scaf"] = round(float(ds.mean()), 4)
        out["median_delta_name"] = round(float(np.median(dn)), 4)
        out["median_delta_scaf"] = round(float(np.median(ds)), 4)
        out["n_delta_name_gt_delta_scaf"] = int((dn > ds).sum())
        if len(both) >= 5:
            w = wilcoxon(dn, ds, alternative="greater")
            out["PRIMARY_wilcoxon_delta_name_gt_delta_scaf"] = {"statistic": float(w.statistic),
                                                                "p_one_sided": float(f"{w.pvalue:.3g}"),
                                                                "reject_at_.05": bool(w.pvalue < 0.05)}
        if len(both) >= 4:
            s = spearmanr([r["delta_name"] for r in both], [r["gap"] for r in both])
            out["register_rho_delta_name_vs_gap"] = {"rho": round(float(s.statistic), 4), "p": round(float(s.pvalue), 4)}
    sc = [r["delta_scaf"] for r in per if r["delta_scaf"] is not None]
    if len(sc) >= 3:
        lo = ttest_1samp(sc, -MARGIN, alternative="greater").pvalue
        hi = ttest_1samp(sc, MARGIN, alternative="less").pvalue
        out["TOST_delta_scaf_margin_.10"] = {"n": len(sc), "mean": round(float(np.mean(sc)), 4),
                                             "p_lower": float(f"{lo:.3g}"), "p_upper": float(f"{hi:.3g}"),
                                             "p_tost": float(f"{max(lo, hi):.3g}"),
                                             "equivalent_at_.05": bool(max(lo, hi) < 0.05)}
    out["undefined_correlations"] = undefined
    out["per_construct"] = per
    return out


def main():
    prim = json.load(open(f"{OUT}/p26_pairs.json"))
    strict = json.load(open(f"{OUT}/p26_pairs_strict.json")) if os.path.exists(f"{OUT}/p26_pairs_strict.json") else []
    psel = {(s["mid"], s["form_a"], s["form_b"]) for s in prim}
    strict_same = lambda s: len(s.get("strict_verdicts", {})) == 2 and all(v == "same" for v in s["strict_verdicts"].values())
    res = {"design": "notes/2026-10-12__prereg26-cw-name-swap.md (+ Amendment 1)",
           "n_primary_selected": len(prim), "n_strict_selected": len(strict)}
    for fam in FAMS:
        R = load_runs(fam)
        res[fam] = {
            "n_valid_runs": len(R),
            "PRIMARY_calibrated": analyse(fam, prim, R, lambda p: ""),
            "SECONDARY_a_primary_restricted_to_strict_same": analyse(fam, [s for s in prim if strict_same(s)], R, lambda p: ""),
            "SECONDARY_b_strict_selection": analyse(fam, strict, R, lambda p: "" if (p["mid"], p["form_a"], p["form_b"]) in psel else "s_"),
        }
    json.dump(res, open(f"{OUT}/prereg26_results.json", "w"), indent=1, ensure_ascii=False)
    with open(f"{OUT}/prereg26_per_construct.tsv", "w") as f:
        f.write("analysis\tfamily\tmid\tgap\tn_valid_runs\twithin_A\twithin_B\tbetween_name\tbetween_scaf\t"
                "delta_name\tdelta_scaf\tname_A\tname_B\n")
        fmt = lambda v: "" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))
        for an in ("PRIMARY_calibrated", "SECONDARY_b_strict_selection"):
            for fam in FAMS:
                for r in res[fam][an]["per_construct"]:
                    f.write("\t".join([an, fam] + [fmt(r[k]) for k in ("mid", "gap", "n_valid_runs", "within_A", "within_B",
                            "between_name", "between_scaf", "delta_name", "delta_scaf", "name_A", "name_B")]) + "\n")
    for fam in FAMS:
        for an in ("PRIMARY_calibrated", "SECONDARY_a_primary_restricted_to_strict_same", "SECONDARY_b_strict_selection"):
            o = res[fam][an]
            print(fam, an, {k: v for k, v in o.items() if k not in ("per_construct",)})


if __name__ == "__main__":
    main()
