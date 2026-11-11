#!/usr/bin/env python
"""PREREG-15-v2 analysis (frozen design: notes/2026-08-28__prereg15v2-community-matching.md)
plus clearly labelled EXPLORATORY re-analyses requested after [author]'s review (2026-10-12):
orientation correction, equivalence-verified subset, name-swap Delta from the k=2 runs,
per-metric paired bootstrap CIs with Benjamini-Hochberg, |Delta AUC| distribution, win split,
and the polarity (orientation) effect before vs after correction.

Inputs (read-only):
  outputs/lexicon/prereg15v2_20260828/{v2_arms.json, v2_items_<mid>.json, v2_score_<mid>_<arm>_r<k>.jsonl}
  outputs/lexicon/prereg25_20261012/{v2_arm_verdicts.json}   (PREREG-25 Step-1 adjudication; optional)
  outputs/lexicon/{prereg15_results_20260723.json, prereg17_results_20260723.json, paper1_figdata.json}
Outputs:
  outputs/lexicon/prereg25_20261012/prereg15v2_results.json  (+ a section in the report .md)
Rerunnable: python methods/codability/lexicon/prereg15v2_analysis.py
"""
from __future__ import annotations

import json
import os
import statistics
from itertools import product

import numpy as np
from scipy.stats import binomtest, rankdata, spearmanr, wilcoxon

ROOT = "ROOT"
V2 = f"{ROOT}/outputs/lexicon/prereg15v2_20260828"
OUT = f"{ROOT}/outputs/lexicon/prereg25_20261012"
A_PRIMARY, A_ALT = 2.75, {"humor_lay_critics": 3.08, "pooled_community": 3.20}
N_BOOT = 2000
SEED = 20261012


# ---------------------------------------------------------------- helpers
def auc(scores, labels) -> float:
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    r = rankdata(s)
    npos, nneg = int(y.sum()), int((1 - y).sum())
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def load_run(path: str):
    if not os.path.exists(path):
        return None
    try:
        rows = [json.loads(l) for l in open(path) if l.strip()]
    except Exception:
        return None
    d = {int(r["item_idx"]): float(r["score"]) for r in rows
         if isinstance(r.get("score"), (int, float)) and 1 <= r["score"] <= 7}
    return d if len(d) == 40 else None


def rho(x: dict, y: dict):
    ks = sorted(x)
    a, b = [x[k] for k in ks], [y[k] for k in ks]
    if len(set(a)) < 2 or len(set(b)) < 2:
        return None
    return float(spearmanr(a, b).statistic)


def bh(pvals: list, q: float = 0.05) -> list:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    thresh = -1
    for rank, i in enumerate(order, 1):
        if pvals[i] <= q * rank / m:
            thresh = rank
    rej = [False] * m
    for rank, i in enumerate(order, 1):
        if rank <= thresh:
            rej[i] = True
    return rej


def dist(xs: list) -> dict:
    if not xs:
        return {"n": 0}
    a = np.abs(np.asarray(xs, float))
    return {"n": len(xs), "median_abs": round(float(np.median(a)), 4),
            "iqr_abs": [round(float(np.percentile(a, 25)), 4), round(float(np.percentile(a, 75)), 4)],
            "max_abs": round(float(a.max()), 4), "mean_signed": round(float(np.mean(xs)), 4)}


def win_split(dauc: list) -> dict:
    lo = sum(d > 0 for d in dauc)
    hi = sum(d < 0 for d in dauc)
    out = {"n": len(dauc), "low_favoring": lo, "high_favoring": hi, "ties": len(dauc) - lo - hi}
    if lo + hi:
        out["sign_p_two_sided"] = round(binomtest(lo, lo + hi, 0.5).pvalue, 4)
    nz = [d for d in dauc if d != 0]
    if len(nz) >= 5:
        out["wilcoxon_p_two_sided"] = round(float(wilcoxon(nz).pvalue), 4)
    return out


def primary_stat(rows: list, A: float, auc_key=("auc_low", "auc_high")) -> dict:
    dd = [abs(r["f_low"] - A) - abs(r["f_high"] - A) for r in rows]
    da = [r[auc_key[0]] - r[auc_key[1]] for r in rows]
    if len(rows) < 4:
        return {"n": len(rows)}
    res = spearmanr(dd, da, alternative="less")
    return {"n": len(rows), "A": A, "rho": round(float(res.statistic), 4),
            "p_one_sided_less": round(float(res.pvalue), 4)}


# ---------------------------------------------------------------- main
def main() -> None:
    rng = np.random.default_rng(SEED)
    arms = json.load(open(f"{V2}/v2_arms.json"))
    verd_path = f"{OUT}/v2_arm_verdicts.json"
    verd = json.load(open(verd_path)) if os.path.exists(verd_path) else {}

    rows, incomplete = [], {}
    for mid, a in sorted(arms.items(), key=lambda x: int(x[0][1:])):
        ip = f"{V2}/v2_items_{mid}.json"
        if not os.path.exists(ip):
            incomplete[mid] = "no item set"
            continue
        lab = {it["item_idx"]: int(it["label"]) for it in json.load(open(ip))["items"]}
        runs = {(arm, k): load_run(f"{V2}/v2_score_{mid}_{arm}_r{k}.jsonl") for arm in ("low", "high") for k in (0, 1)}
        have = [key for key, v in runs.items() if v is not None]
        if len(have) < 4:
            incomplete[mid] = f"{len(have)}/4 valid runs"
            continue
        ks = sorted(lab)
        y = np.array([lab[k] for k in ks])
        S = {arm: np.array([(runs[(arm, 0)][k] + runs[(arm, 1)][k]) / 2 for k in ks]) for arm in ("low", "high")}
        r = {"mid": mid, "f_low": a["f_low"], "f_high": a["f_high"], "low_rubric": a["low_rubric"],
             "high_rubric": a["high_rubric"], "auc_low": auc(S["low"], y), "auc_high": auc(S["high"], y)}
        r["dauc"] = r["auc_low"] - r["auc_high"]
        # paired bootstrap within label strata (same resampled items for both arms)
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        bs = np.empty(N_BOOT)
        for b in range(N_BOOT):
            idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
            bs[b] = auc(S["low"][idx], y[idx]) - auc(S["high"][idx], y[idx])
        r["dauc_ci95"] = [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]
        r["dauc_boot_p_two_sided"] = float(min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean())))
        # name-swap Delta from the k=2 runs (exploratory)
        wl, wh = rho(runs[("low", 0)], runs[("low", 1)]), rho(runs[("high", 0)], runs[("high", 1)])
        btw = [rho(runs[("low", i)], runs[("high", j)]) for i, j in product((0, 1), (0, 1))]
        within = [w for w in (wl, wh) if w is not None]
        btw = [x for x in btw if x is not None]
        r["within_low"], r["within_high"] = wl, wh
        r["within"] = float(np.mean(within)) if within else None
        r["between"] = float(np.mean(btw)) if btw else None
        # orientation verdicts (PREREG-25 Step 1)
        v = verd.get(mid)
        if v:
            r["low_worse"] = bool(v["low"]["both_worse"])
            r["high_worse"] = bool(v["high"]["both_worse"])
            r["verified_equivalent"] = bool(v["verified_equivalent"])
            r["verified_equivalent_same_orientation"] = bool(v["verified_equivalent_same_orientation"])
            r["orientation_low"] = {"gpt": v["low"]["gpt"], "claude": v["low"]["claude"]}
            r["orientation_high"] = {"gpt": v["high"]["gpt"], "claude": v["high"]["claude"]}
            r["equivalence"] = {"gpt": v["equivalence"]["gpt"], "claude": v["equivalence"]["claude"]}
            r["auc_low_oc"] = 1 - r["auc_low"] if r["low_worse"] else r["auc_low"]
            r["auc_high_oc"] = 1 - r["auc_high"] if r["high_worse"] else r["auc_high"]
            r["dauc_oc"] = r["auc_low_oc"] - r["auc_high_oc"]
            flip = r["low_worse"] != r["high_worse"]
            r["between_oc"] = (-r["between"] if (flip and r["between"] is not None) else r["between"])
        for k in ("auc_low", "auc_high", "dauc", "within", "between"):
            if r.get(k) is not None:
                r[k] = round(r[k], 4)
        rows.append(r)

    # BH over the per-metric bootstrap p-values
    rej = bh([r["dauc_boot_p_two_sided"] for r in rows]) if rows else []
    for r, x in zip(rows, rej):
        r["dauc_bh_reject_q05"] = x
        r["dauc_ci_excludes_0"] = r["dauc_ci95"][0] > 0 or r["dauc_ci95"][1] < 0

    res = {"design": "notes/2026-08-28__prereg15v2-community-matching.md",
           "n_arm_pairs": len(arms), "n_complete_metrics": len(rows), "incomplete": incomplete}
    # ---------------- PRIMARY (frozen)
    res["PRIMARY_H1_rho_ddist_dauc_lt0_A2.75"] = primary_stat(rows, A_PRIMARY)
    res["primary_floor_check"] = {
        "mean_auc_low": round(float(np.mean([r["auc_low"] for r in rows])), 4) if rows else None,
        "mean_auc_high": round(float(np.mean([r["auc_high"] for r in rows])), 4) if rows else None,
        "n_metrics_both_arms_within_.40-.60": sum(0.4 <= r["auc_low"] <= 0.6 and 0.4 <= r["auc_high"] <= 0.6
                                                  for r in rows)}
    # ---------------- SECONDARY (frozen, descriptive)
    sec = {}
    for nm, A in A_ALT.items():
        sec[f"rho_ddist_dauc_A{A}_{nm}"] = primary_stat(rows, A)
    dists = [abs(r[f"f_{s}"] - A_PRIMARY) for r in rows for s in ("low", "high")]
    aucs = [r[f"auc_{s}"] for r in rows for s in ("low", "high")]
    if len(rows) >= 3:
        sr = spearmanr(dists, aucs)
        sec["pooled_rho_absdist_vs_auc_all_arms(clustered;descriptive)"] = {
            "n_arms": len(aucs), "rho": round(float(sr.statistic), 4), "naive_p": round(float(sr.pvalue), 4)}
    closer_high = [r for r in rows if abs(r["f_high"] - A_PRIMARY) < abs(r["f_low"] - A_PRIMARY)]
    wins = sum(r["auc_high"] > r["auc_low"] for r in closer_high)
    losses = sum(r["auc_high"] < r["auc_low"] for r in closer_high)
    sec["sign_test_closer_arm_is_high_arm"] = {
        "n": len(closer_high), "closer(high)_wins": wins, "closer(high)_loses": losses,
        "p_two_sided": round(binomtest(wins, wins + losses).pvalue, 4) if wins + losses else None}
    sec["v1_readout_win_split"] = win_split([r["dauc"] for r in rows])
    sec["v1_readout_dauc_distribution"] = dist([r["dauc"] for r in rows])
    sec["v1_readout_prohibition_contrast"] = (
        "NOT COMPUTABLE on v2 as frozen: PREREG-17 per-form polarity codes are not on disk "
        "(outputs/lexicon/prereg17_results_20260723.json holds aggregates and 16 per-metric code pairs without "
        "form strings). The exploratory orientation contrast below uses the PREREG-25 adjudicated orientation.")
    res["SECONDARY"] = sec

    # ---------------- EXPLORATORY
    ex = {"label": "EXPLORATORY (post hoc, after [author] review 2026-10-12); not part of the frozen PREREG-15-v2 "
                   "analysis"}
    ex["per_metric_bootstrap"] = {
        "n_metrics": len(rows), "n_ci_excludes_0_raw": sum(r["dauc_ci_excludes_0"] for r in rows),
        "n_bh_q05": sum(r["dauc_bh_reject_q05"] for r in rows),
        "n_boot": N_BOOT, "resampling": "items with replacement within label strata (20 pos, 20 neg), paired"}
    if verd and rows and "low_worse" in rows[0]:
        oc_rows = [r for r in rows if "dauc_oc" in r]
        n_flip_forms = sum(r["low_worse"] + r["high_worse"] for r in oc_rows)
        ex["orientation_correction"] = {
            "rule": "AUC -> 1 - AUC for an arm form that BOTH families call 'higher = worse'",
            "n_forms_flipped": n_flip_forms,
            "flipped": [(r["mid"], s, r[f"{s}_rubric"], r[f"auc_{s}"]) for r in oc_rows for s in ("low", "high")
                        if r[f"{s}_worse"]],
            "primary_stat_corrected_A2.75": primary_stat(oc_rows, A_PRIMARY, ("auc_low_oc", "auc_high_oc")),
            "win_split_corrected": win_split([r["dauc_oc"] for r in oc_rows]),
            "dauc_distribution_before": dist([r["dauc"] for r in oc_rows]),
            "dauc_distribution_corrected": dist([r["dauc_oc"] for r in oc_rows]),
        }
        # polarity effect = AUC(non-fault arm) - AUC(fault arm) on metrics with exactly one fault-oriented arm
        mixed = [r for r in oc_rows if r["low_worse"] != r["high_worse"]]

        def adv(r, suffix=""):
            good, bad = ("high", "low") if r["low_worse"] else ("low", "high")
            return r[f"auc_{good}{suffix}"] - r[f"auc_{bad}{suffix}"]

        def adv_stats(vals):
            out = {"n": len(vals), "mean_adv_nonfault": round(float(np.mean(vals)), 4) if vals else None}
            nz = [v for v in vals if v != 0]
            if len(nz) >= 3:
                out["wilcoxon_p_one_sided_greater"] = round(float(wilcoxon(nz, alternative="greater").pvalue), 4)
                out["wilcoxon_p_two_sided"] = round(float(wilcoxon(nz).pvalue), 4)
            return out
        ex["orientation_polarity_effect"] = {
            "definition": "metrics where exactly one arm is adjudicated 'higher = worse' by both families; "
                          "advantage = AUC(non-fault arm) - AUC(fault arm)",
            "before_correction": adv_stats([adv(r) for r in mixed]),
            "after_correction": adv_stats([adv(r, "_oc") for r in mixed]),
            "metrics": [r["mid"] for r in mixed]}
        eq = [r for r in oc_rows if r["verified_equivalent"]]
        eqs = [r for r in oc_rows if r["verified_equivalent_same_orientation"]]
        ex["equivalence_verified_subset"] = {
            "n_verified_equivalent": len(eq), "n_verified_equivalent_same_orientation": len(eqs),
            "mids_equivalent_same_orientation": [r["mid"] for r in eqs],
            "primary_stat_equivalent_A2.75": primary_stat(eq, A_PRIMARY),
            "primary_stat_equivalent_same_orientation_A2.75": primary_stat(eqs, A_PRIMARY),
            "win_split_equivalent": win_split([r["dauc"] for r in eq]),
            "dauc_distribution_equivalent": dist([r["dauc"] for r in eq]),
            "win_split_equivalent_same_orientation": win_split([r["dauc"] for r in eqs]),
            "dauc_distribution_equivalent_same_orientation": dist([r["dauc"] for r in eqs]),
            "n_ci_excludes_0_equivalent_same_orientation": sum(r["dauc_ci_excludes_0"] for r in eqs),
        }
    # name-swap Delta from v2 k=2 runs
    def ns(sub, between_key="between"):
        d = [r["within"] - r[between_key] for r in sub if r.get("within") is not None and r.get(between_key) is not None]
        if not d:
            return {"n": 0}
        out = {"n": len(d), "mean_within": round(float(np.mean([r["within"] for r in sub if r.get("within") is not None])), 4),
               "mean_between": round(float(np.mean([r[between_key] for r in sub if r.get(between_key) is not None])), 4),
               "mean_delta": round(float(np.mean(d)), 4), "median_delta": round(float(np.median(d)), 4)}
        nz = [x for x in d if x != 0]
        if len(nz) >= 5:
            out["wilcoxon_p_one_sided_delta_gt0"] = round(float(wilcoxon(nz, alternative="greater").pvalue), 4)
        return out
    ex["name_swap_delta_v2_k2"] = {
        "definition": "within = mean of rho(r0,r1) over the two arms; between = mean of the 4 cross-arm run rhos; "
                      "Delta = within - between (GPT-5.6 only; v2 scored with gpt-5.6-sol)",
        "all_complete_metrics": ns(rows)}
    if verd and rows and "between_oc" in rows[0]:
        ex["name_swap_delta_v2_k2"]["orientation_corrected(between->-between when exactly one arm is fault)"] = \
            ns(rows, "between_oc")
        ex["name_swap_delta_v2_k2"]["equivalent_subset"] = ns([r for r in rows if r["verified_equivalent"]])
        ex["name_swap_delta_v2_k2"]["equivalent_same_orientation_subset"] = \
            ns([r for r in rows if r["verified_equivalent_same_orientation"]])
    # name-swap Delta and |Delta AUC| by the adjudicated equivalence verdict of the arm pair
    if verd and rows and "equivalence" in rows[0]:
        def cat(r):
            e = r["equivalence"]
            if e["gpt"] == "same" and e["claude"] == "same":
                return "both_same"
            if "different" in (e["gpt"], e["claude"]):
                return "at_least_one_different"
            return "overlap_or_mixed_same_overlap"
        by = {}
        for c in ("both_same", "overlap_or_mixed_same_overlap", "at_least_one_different"):
            sub = [r for r in rows if cat(r) == c]
            by[c] = {"name_swap": ns(sub), "dauc": dist([r["dauc"] for r in sub]), "mids": [r["mid"] for r in sub]}
        ex["by_equivalence_verdict"] = by
    res["EXPLORATORY"] = ex

    # ---------------- v1 (published) prohibition test, orientation sensitivity (exploratory)
    res["EXPLORATORY_v1"] = v1_reanalysis()
    res["per_metric"] = rows
    tmp = f"{OUT}/prereg15v2_results.json.tmp"
    json.dump(res, open(tmp, "w"), indent=1, ensure_ascii=False)
    os.replace(tmp, f"{OUT}/prereg15v2_results.json")
    print(json.dumps({k: v for k, v in res.items() if k != "per_metric"}, indent=1, ensure_ascii=False)[:6000])


def v1_reanalysis() -> dict:
    """Published v1 numbers (PREREG-15/17, 2026-07-23) under orientation correction where names exist."""
    p17 = json.load(open(f"{ROOT}/outputs/lexicon/prereg17_results_20260723.json"))
    v1 = {r["mid"]: r for r in json.load(open(f"{ROOT}/outputs/lexicon/prereg15_results_20260723.json"))["per_metric"]}
    t3p = f"{OUT}/v1_table3_verdicts.json"
    t3 = {r["mid"]: r for r in json.load(open(t3p))["rows"]} if os.path.exists(t3p) else {}
    out = {"label": "EXPLORATORY: published v1 numbers re-read under orientation correction"}

    def prohib(flip_rule: str) -> dict:
        adv = []
        for mid, lo, hi, dauc in p17["mismatched_detail"]:
            r = v1[mid]
            al, ah = r["auc_low"], r["auc_high"]
            pa = "low" if lo == "prohibition_antipattern" else ("high" if hi == "prohibition_antipattern" else None)
            if pa is None:
                continue
            if flip_rule == "adjudicated" and mid in t3:
                if t3[mid]["low_flipped"]:
                    al = 1 - al
                if t3[mid]["high_flipped"]:
                    ah = 1 - ah
            adv.append((ah - al) if pa == "low" else (al - ah))
        nz = [a for a in adv if a != 0]
        return {"n": len(adv), "mean_adv_nonprohibition": round(float(np.mean(adv)), 4),
                "wilcoxon_p_one_sided_greater": round(float(wilcoxon(nz, alternative="greater").pvalue), 4),
                "wilcoxon_p_two_sided": round(float(wilcoxon(nz).pvalue), 4)}
    out["prohibition_contrast_published_reproduced"] = prohib("none")
    out["prohibition_contrast_flip_adjudicated_table3_forms_only"] = prohib("adjudicated")
    out["note"] = ("v1 arm names exist only for the 10 Table-3 metrics (paper1_figdata.json); of the 16 "
                   "prohibition-mismatched metrics, names (hence adjudicated orientation) exist for "
                   + ", ".join(sorted(set(t3) & {m[0] for m in p17["mismatched_detail"]})) +
                   ". The other 12 prohibition-coded arms have no recoverable names and are left as published (a prohibition "
                   "code is not the same as fault orientation: e.g. 'Punch up, not down' is a prohibition whose top "
                   "score means better comedy).")
    if t3:
        all_d = {m: r["dauc"] for m, r in v1.items()}
        corr = dict(all_d)
        for m, r in t3.items():
            corr[m] = r["dauc_corrected"]
        out["v1_dauc_distribution_published"] = dist(list(all_d.values()))
        out["v1_dauc_distribution_table3_corrected"] = dist(list(corr.values()))
        out["v1_win_split_published"] = win_split(list(all_d.values()))
        out["v1_win_split_table3_corrected"] = win_split(list(corr.values()))
        out["table3_rows"] = [{k: r[k] for k in ("mid", "low", "high", "auc_low_published", "auc_high_published",
                                                 "dauc_published", "low_flipped", "high_flipped",
                                                 "auc_low_corrected", "auc_high_corrected", "dauc_corrected",
                                                 "verified_equivalent", "equivalence", "orientation_low",
                                                 "orientation_high")} for r in t3.values()]
    return out


if __name__ == "__main__":
    main()
