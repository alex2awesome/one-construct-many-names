#!/usr/bin/env python
"""PREREG-30 analysis (notes/2026-10-12__prereg30-humor-construct-level-name-swap.md).
= PREREG-25 analysis (prereg25_analysis.analyse_family, unchanged statistics) with paths changed, plus the PREREG-26
register statistic (Spearman of Delta_name with the formality gap). Per family, constructs with fewer than 6 valid
runs are dropped and listed. Rerunnable: python methods/codability/lexicon/prereg30_analysis.py
"""
import json, os, re, sys
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prereg25_analysis as P25

ROOT = "ROOT"
OUT = f"{ROOT}/outputs/lexicon/prereg30_humor_20261012"


def main():
    P = json.load(open(f"{OUT}/p30_pairs.json"))
    pairs = P["pairs"]
    rx = re.compile(P25.RUN_RE)
    runs, invalid = defaultdict(dict), []
    for fd in ("gpt56", "claude"):
        fam = P25.FAM_DIR[fd]
        for f in sorted(os.listdir(f"{OUT}/runs/{fd}")):
            m = rx.search(f)
            if not m:
                continue
            r = P25.load_run(f"{OUT}/runs/{fd}/{f}")
            if r is None:
                invalid.append(f"{fd}/{f}"); continue
            runs[(fam, m["mid"])][(m["name"], m["scaf"], int(m["k"]))] = r
    rng = np.random.default_rng(P25.SEED)
    res = {"design": "notes/2026-10-12__prereg30-humor-construct-level-name-swap.md",
           "rule_applied": P["rule_applied"], "n_selected": len(pairs), "invalid_runs": invalid,
           "reused_from_prereg25": [p["mid"] for p in pairs if p["same_as_prereg25"]], "families": {}}
    for fam in ("gpt", "claude"):
        comp = [p for p in pairs if len(runs.get((fam, p["mid"]), {})) == 6]
        out = P25.analyse_family(fam, comp, runs, rng)
        out["incomplete_constructs_dropped"] = {p["mid"]: len(runs.get((fam, p["mid"]), {}))
                                                for p in pairs if p not in comp}
        both = [r for r in out["per_metric"] if r["delta_name"] is not None and r["delta_scaf"] is not None]
        if len(both) >= 4:
            s = spearmanr([r["delta_name"] for r in both], [r["gap"] for r in both])
            out["register_rho_delta_name_vs_gap"] = {"n": len(both), "rho": round(float(s.statistic), 4),
                                                     "p_two_sided": round(float(s.pvalue), 4)}
        res["families"][fam] = out
    json.dump(res, open(f"{OUT}/prereg30_results.json", "w"), indent=1)
    for fam, d in res["families"].items():
        print(fam, json.dumps({k: d.get(k) for k in ("n_selected", "n_metrics_primary", "mean_delta_name",
              "mean_delta_scaf", "n_delta_name_gt_delta_scaf", "PRIMARY_wilcoxon_delta_name_gt_delta_scaf",
              "a_TOST_delta_scaf_margin_.10", "register_rho_delta_name_vs_gap", "incomplete_constructs_dropped")}))
        print("  b_auc", d.get("b_auc"), "c_register", d.get("c_register"))
    print("invalid:", invalid)


if __name__ == "__main__":
    main()
