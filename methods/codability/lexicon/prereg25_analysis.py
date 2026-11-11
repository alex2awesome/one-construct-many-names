#!/usr/bin/env python
"""PREREG-25 analysis (frozen design: notes/2026-10-12__prereg25-matched-meaning-name-swap.md).

Per family (GPT-5.6, Claude; never pooled), over the metrics selected in p25_pairs.json:
  within(A) = rho(A_S0_r0, A_S0_r1); within(B) likewise; within = mean of the two
  between_name = mean of rho(A_S0_ri, B_S0_rj), 4 (i,j);   Delta_name = within - between_name
  between_scaf = mean of rho(A_S0_ri, A_Sk_r0), i in {0,1}, k in {1,2};   Delta_scaf = within(A) - between_scaf
  PRIMARY: one-sided paired Wilcoxon over metrics, Delta_name > Delta_scaf, alpha .05
  (a) TOST equivalence of Delta_scaf to 0, margin .10 (two one-sided one-sample t-tests)
  (b) AUC of each name (mean of its 2 S0 runs) vs community labels; |AUC_A - AUC_B| median/IQR/max;
      per-metric paired bootstrap CI for AUC_B - AUC_A (2,000 resamples within label strata); CIs excluding 0,
      raw and after Benjamini-Hochberg (q = .05)
  (c) register: win split (sign test) and Spearman of signed AUC_B - AUC_A with the formality gap
  (d) exploratory: rho(Delta_d, Delta_AUC), A = 2.75, one-sided < 0, as PREREG-15-v2
Spearman over the 40 items, average ranks for ties; a correlation with a constant run is undefined, reported and
dropped. A run is valid with 40 scores in 1..7.

Run files: lane B layout runs/<gpt56|claude>/<mid>_<A|B>_<S0|S1|S2>_r<k>.jsonl (family from the parent dir).
Name check: lane B runner logs ("queued <mid> A=... B=...") are compared with p25_pairs.json; metrics scored under
other names are reported, and every statistic is also given excluding them ("sensitivity_excluding").
Rerunnable: python methods/codability/lexicon/prereg25_analysis.py [--runs DIR] [--run-re REGEX]
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from itertools import product

import numpy as np
from scipy.stats import binomtest, rankdata, spearmanr, ttest_1samp, wilcoxon

ROOT = "ROOT"
V2 = f"{ROOT}/outputs/lexicon/prereg15v2_20260828"
OUT = f"{ROOT}/outputs/lexicon/prereg25_20261012"
# lane B layout: runs/<family dir: gpt56|claude>/<mid>_<A|B>_<S0|S1|S2>_r<k>.jsonl
RUN_RE = r"^(?P<mid>a\d+)_(?P<name>[AB])_(?P<scaf>S[012])_r(?P<k>\d)\.jsonl$"
FAM_DIR = {"gpt56": "gpt", "gpt": "gpt", "gpt-5.6": "gpt", "claude": "claude"}
# metrics where lane B kept names from a PARTIAL p25_pairs.json (see LANE_A_NOTICE_stale_pairs_a17_a118.md);
# overridden by parsing lane B runner logs ("queued <mid> A=... B=...") when they are on disk.
KNOWN_NAME_DEVIATIONS = ["a17", "a118"]
N_BOOT, SEED, MARGIN, A_REG = 2000, 20261012, 0.10, 2.75


def auc(s, y) -> float:
    s, y = np.asarray(s, float), np.asarray(y, int)
    r = rankdata(s)
    npos, nneg = int(y.sum()), int((1 - y).sum())
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def load_run(path: str):
    try:
        rows = [json.loads(l) for l in open(path) if l.strip()]
    except Exception:
        return None
    d = {}
    for r in rows:
        try:
            sc = float(r["score"])
            if 1 <= sc <= 7:
                d[int(r["item_idx"])] = sc
        except Exception:
            pass
    return d if len(d) == 40 else None


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


def bh(p, q=0.05):
    m = len(p)
    order = sorted(range(m), key=lambda i: p[i])
    k = max([r for r, i in enumerate(order, 1) if p[i] <= q * r / m], default=0)
    rej = [False] * m
    for r, i in enumerate(order, 1):
        rej[i] = r <= k
    return rej


def analyse_family(fam: str, pairs: list, runs: dict, rng) -> dict:
    per, undefined = [], []
    for p in pairs:
        mid = p["mid"]
        R = runs.get((fam, mid), {})
        lab = {it["item_idx"]: int(it["label"]) for it in json.load(open(f"{V2}/v2_items_{mid}.json"))["items"]}
        g = lambda name, scaf, k: R.get((name, scaf, k))
        rec = {"mid": mid, "n_valid_runs": len(R)}
        wA, wB = rho(g("A", "S0", 0), g("A", "S0", 1)), rho(g("B", "S0", 0), g("B", "S0", 1))
        rec["within_A"], rec["within_B"] = wA, wB
        rec["within"], _ = mean_defined([wA, wB])
        bn = [rho(g("A", "S0", i), g("B", "S0", j)) for i, j in product((0, 1), (0, 1))]
        rec["between_name"], rec["n_between_name"] = mean_defined(bn)
        bs = [rho(g("A", "S0", i), g("A", sk, 0)) for i, sk in product((0, 1), ("S1", "S2"))]
        rec["between_scaf"], rec["n_between_scaf"] = mean_defined(bs)
        for nm, v in (("within_A", wA), ("within_B", wB)):
            if v is None and all(g(n, "S0", k) is not None for n in nm[-1] for k in (0, 1)):
                undefined.append((mid, nm))
        rec["delta_name"] = (rec["within"] - rec["between_name"]
                             if rec["within"] is not None and rec["between_name"] is not None else None)
        rec["delta_scaf"] = (wA - rec["between_scaf"] if wA is not None and rec["between_scaf"] is not None else None)
        # AUC per name = mean of its two S0 runs
        ks = sorted(lab)
        y = np.array([lab[k] for k in ks])
        S = {}
        for name in ("A", "B"):
            r0, r1 = g(name, "S0", 0), g(name, "S0", 1)
            if r0 is not None and r1 is not None:
                S[name] = np.array([(r0[k] + r1[k]) / 2 for k in ks])
        if len(S) == 2:
            rec["auc_A"], rec["auc_B"] = auc(S["A"], y), auc(S["B"], y)
            rec["dauc_B_minus_A"] = rec["auc_B"] - rec["auc_A"]
            pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
            bsv = np.empty(N_BOOT)
            for b in range(N_BOOT):
                idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
                bsv[b] = auc(S["B"][idx], y[idx]) - auc(S["A"][idx], y[idx])
            rec["dauc_ci95"] = [float(np.percentile(bsv, 2.5)), float(np.percentile(bsv, 97.5))]
            rec["dauc_boot_p"] = float(min(1.0, 2 * min((bsv <= 0).mean(), (bsv >= 0).mean())))
        rec["formality_a"], rec["formality_b"], rec["gap"] = p["formality_a"], p["formality_b"], p["gap"]
        per.append(rec)

    out = {"family": fam, "n_selected": len(pairs)}
    both = [r for r in per if r["delta_name"] is not None and r["delta_scaf"] is not None]
    out["n_metrics_primary"] = len(both)
    if both:
        dn = np.array([r["delta_name"] for r in both])
        ds = np.array([r["delta_scaf"] for r in both])
        out["mean_within"] = round(float(np.mean([r["within"] for r in both])), 4)
        out["mean_between_name"] = round(float(np.mean([r["between_name"] for r in both])), 4)
        out["mean_between_scaf"] = round(float(np.mean([r["between_scaf"] for r in both])), 4)
        out["mean_delta_name"] = round(float(dn.mean()), 4)
        out["mean_delta_scaf"] = round(float(ds.mean()), 4)
        out["median_delta_name"] = round(float(np.median(dn)), 4)
        out["median_delta_scaf"] = round(float(np.median(ds)), 4)
        out["n_delta_name_gt_delta_scaf"] = int((dn > ds).sum())
    if len(both) >= 5:
        w = wilcoxon(dn, ds, alternative="greater")
        out["PRIMARY_wilcoxon_delta_name_gt_delta_scaf"] = {"statistic": float(w.statistic),
                                                            "p_one_sided": round(float(w.pvalue), 5),
                                                            "reject_at_.05": bool(w.pvalue < 0.05)}
    sc = [r["delta_scaf"] for r in per if r["delta_scaf"] is not None]
    if len(sc) >= 3:
        lo = ttest_1samp(sc, -MARGIN, alternative="greater").pvalue
        hi = ttest_1samp(sc, MARGIN, alternative="less").pvalue
        out["a_TOST_delta_scaf_margin_.10"] = {"n": len(sc), "mean": round(float(np.mean(sc)), 4),
                                               "p_lower": round(float(lo), 5), "p_upper": round(float(hi), 5),
                                               "p_tost": round(float(max(lo, hi)), 5),
                                               "equivalent_at_.05": bool(max(lo, hi) < 0.05)}
    au = [r for r in per if "auc_A" in r]
    if au:
        d = np.array([r["dauc_B_minus_A"] for r in au])
        a = np.abs(d)
        rej = bh([r["dauc_boot_p"] for r in au])
        out["b_auc"] = {"n": len(au), "mean_auc_A": round(float(np.mean([r["auc_A"] for r in au])), 4),
                        "mean_auc_B": round(float(np.mean([r["auc_B"] for r in au])), 4),
                        "median_abs_dauc": round(float(np.median(a)), 4),
                        "iqr_abs_dauc": [round(float(np.percentile(a, 25)), 4), round(float(np.percentile(a, 75)), 4)],
                        "max_abs_dauc": round(float(a.max()), 4),
                        "n_ci_excludes_0_raw": sum(r["dauc_ci95"][0] > 0 or r["dauc_ci95"][1] < 0 for r in au),
                        "n_bh_q05": int(sum(rej))}
        bw, aw = int((d > 0).sum()), int((d < 0).sum())
        out["c_register"] = {"B_formal_wins": bw, "A_informal_wins": aw, "ties": int((d == 0).sum()),
                             "sign_p_two_sided": round(binomtest(bw, bw + aw).pvalue, 4) if bw + aw else None}
        if len(au) >= 4:
            s = spearmanr(d, [r["gap"] for r in au])
            out["c_register"]["rho_signed_dauc_vs_gap"] = round(float(s.statistic), 4)
            out["c_register"]["p_two_sided"] = round(float(s.pvalue), 4)
            dd = [abs(r["formality_a"] - A_REG) - abs(r["formality_b"] - A_REG) for r in au]
            dA = [r["auc_A"] - r["auc_B"] for r in au]
            s2 = spearmanr(dd, dA, alternative="less")
            out["d_exploratory_register_matching_A2.75"] = {"rho": round(float(s2.statistic), 4),
                                                           "p_one_sided_less": round(float(s2.pvalue), 4)}
    out["undefined_correlations"] = undefined
    out["per_metric"] = per
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=f"{OUT}/runs")
    ap.add_argument("--run-re", default=RUN_RE)
    ap.add_argument("--pairs", default=f"{OUT}/p25_pairs.json")
    ap.add_argument("--laneb-logs", default=f"{OUT}/laneB_logs")
    args = ap.parse_args()
    P = json.load(open(args.pairs))
    pairs = P["pairs"]
    # which names did lane B actually score?
    import ast
    import glob
    used, logs = {}, sorted(glob.glob(f"{args.laneb_logs}/**/runner_*.log", recursive=True))
    qre = re.compile(r"queued (a\d+) A=(.+?) B=(.+)$")
    for lp in logs:
        for line in open(lp, errors="replace"):
            m = qre.search(line.strip())
            if m:  # the LAST queued names per metric are the ones in runs/ (superseded runs are moved out)
                try:
                    used[m[1]] = (ast.literal_eval(m[2]), ast.literal_eval(m[3]))
                except Exception:
                    pass
    if used:
        deviant = sorted({p["mid"] for p in pairs if p["mid"] in used and used[p["mid"]] != (p["name_a"], p["name_b"])})
        missing_names = sorted({p["mid"] for p in pairs} - set(used))
    else:
        deviant, missing_names = KNOWN_NAME_DEVIATIONS, None
    # sensitivity set: metrics flagged by the name check, else the two metrics rescored after the partial-pairs
    # episode (a17, a118), so that the coordinator always sees the primary without them
    sens = deviant if deviant else KNOWN_NAME_DEVIATIONS
    rx = re.compile(args.run_re)
    runs = defaultdict(dict)
    invalid, unmatched = [], 0
    for dp, _, fs in os.walk(args.runs):
        fam = FAM_DIR.get(os.path.basename(dp))
        for f in fs:
            if not f.endswith(".jsonl"):
                continue
            m = rx.search(f)
            if not m or fam is None:
                unmatched += 1
                continue
            r = load_run(os.path.join(dp, f))
            if r is None:
                invalid.append(f"{os.path.basename(dp)}/{f}")
                continue
            runs[(fam, m["mid"])][(m["name"], m["scaf"], int(m["k"]))] = r
    rng = np.random.default_rng(SEED)
    res = {"design": "notes/2026-10-12__prereg25-matched-meaning-name-swap.md", "pairs_file": args.pairs,
           "n_selected_metrics": len(pairs), "gap_floor": P.get("gap_floor"),
           "n_run_files_valid": sum(len(v) for v in runs.values()), "invalid_runs": invalid,
           "unmatched_jsonl_files": unmatched,
           "name_check": {"laneB_logs_found": logs, "metrics_scored_under_other_names": deviant,
                          "metrics_without_queued_names_in_logs": missing_names,
                          "names_used_by_laneB": {m: used[m] for m in deviant if m in used}},
           "families": {}, "sensitivity_excluding": {}}
    for fam in ("gpt", "claude"):
        res["families"][fam] = analyse_family(fam, pairs, runs, rng)
        sub = analyse_family(fam, [p for p in pairs if p["mid"] not in sens], runs, rng)
        res["sensitivity_excluding"][fam] = {"excluded": sens,
                                             **{k: v for k, v in sub.items() if k != "per_metric"}}
    tmp = f"{OUT}/prereg25_results.json.tmp"
    json.dump(res, open(tmp, "w"), indent=1)
    os.replace(tmp, f"{OUT}/prereg25_results.json")
    print(json.dumps({f: {k: v for k, v in d.items() if k != "per_metric"} for f, d in res["families"].items()},
                     indent=1)[:5000])
    print("valid runs:", res["n_run_files_valid"], "invalid:", len(invalid), "unmatched:", unmatched)


if __name__ == "__main__":
    main()
