#!/usr/bin/env python
"""Write outputs/lexicon/prereg25_20261012/prereg25_report.md from the result JSONs.

Reads adjudication_summary.json, p25_pairs.json, v1_table3_verdicts.json, prereg15v2_results.json and
prereg25_results.json (each optional). Rerun after prereg25_adjudication_aggregate.py, prereg15v2_analysis.py and
prereg25_analysis.py.
"""
from __future__ import annotations

import json
import os
import time

OUT = "ROOT/outputs/lexicon/prereg25_20261012"


def j(name):
    p = f"{OUT}/{name}"
    return json.load(open(p)) if os.path.exists(p) else None


def fmt(x, nd=3):
    return "NA" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def main() -> None:
    S, P, T3, V2R, P25 = (j("adjudication_summary.json"), j("p25_pairs.json"), j("v1_table3_verdicts.json"),
                          j("prereg15v2_results.json"), j("prereg25_results.json"))
    L = [f"# PREREG-25 adjudication and re-analyses (lane A)", "",
         f"Generated {time.strftime('%Y-%m-%d %H:%M %Z')} by `methods/codability/lexicon/prereg25_report.py`. "
         "Designs: `notes/2026-10-12__prereg25-matched-meaning-name-swap.md` (Steps 1-2, statistics) and "
         "`notes/2026-08-28__prereg15v2-community-matching.md` (v2 primary). Everything marked EXPLORATORY is post hoc.",
         ""]
    if S:
        ke, ko = S["kappa_equivalence"], S["kappa_orientation"]
        L += ["## Step 1: adjudication", "",
              f"Judges: GPT-5.6 ({S['judges']['gpt']}) and Claude ({S['judges']['claude']}), blind to every score "
              "and AUC; names shown without definitions or metric context.",
              f"Scope: {S['n_pairs_total']} form pairs (all within-metric pairs with formality gap >= 1 pooled SD in "
              f"the 86 item metrics, plus the 86 v2 arm pairs and the 10 v1 Table-3 pairs), {S['n_forms_total']} forms.",
              f"Judged: pairs {S['n_pairs_judged']}, forms {S['n_forms_judged']}.",
              f"Anchor accuracy in accepted batches: {json.dumps(S['anchor_accuracy_passed_batches'])}.", "",
              "| question | n | Cohen's kappa | % agree |", "|---|---|---|---|",
              f"| equivalence (3-way) | {ke.get('n')} | {fmt(ke.get('kappa'))} | {fmt(ke.get('pct_agree'))} |",
              f"| equivalence (same vs not) | {S['kappa_equivalence_same_vs_not'].get('n')} | "
              f"{fmt(S['kappa_equivalence_same_vs_not'].get('kappa'))} | {fmt(S['kappa_equivalence_same_vs_not'].get('pct_agree'))} |",
              f"| orientation (3-way) | {ko.get('n')} | {fmt(ko.get('kappa'))} | {fmt(ko.get('pct_agree'))} |",
              f"| orientation (worse vs not) | {S['kappa_orientation_worse_vs_not'].get('n')} | "
              f"{fmt(S['kappa_orientation_worse_vs_not'].get('kappa'))} | {fmt(S['kappa_orientation_worse_vs_not'].get('pct_agree'))} |",
              "", f"Equivalence confusion (gpt|claude): `{json.dumps(ke.get('confusion_gpt|claude'))}`",
              f"Orientation confusion (gpt|claude): `{json.dumps(ko.get('confusion_gpt|claude'))}`", "",
              f"- Pairs both families call 'same requirement': {S['pairs_both_same_total']} of {S['n_pairs_total']}.",
              f"- Forms both families call 'higher = worse': {S['forms_both_worse_total']} of {S['n_forms_total']} "
              f"(either family: {S['forms_either_worse_total']}).",
              f"- v2 arm pairs (with items, n={S['v2_arms_with_items']}): verified equivalent "
              f"{S['v2_arms_verified_equivalent']}; equivalent and both forms higher = better "
              f"{S['v2_arms_verified_equivalent_same_orientation']}.",
              f"- v2 arm forms both families call 'higher = worse': {S['v2_arm_forms_both_worse']}",
              f"- v1 Table-3 pairs verified equivalent: {S['v1_table3_verified_equivalent']} of 10; "
              f"fault-oriented forms: {S['v1_table3_flipped_forms']}", ""]
    if P:
        L += ["## Step 2: PREREG-25 selection", "",
              f"Gap floor {fmt(P['gap_floor'])} ({P['gap_floor_note']}); selected {P['n_selected']} of "
              f"{P['n_metrics_universe']} metrics. File: `p25_pairs.json`.", "",
              "| mid | name A (less formal) | name B | f_A | f_B |", "|---|---|---|---|---|"]
        for r in P["pairs"]:
            L.append(f"| {r['mid']} | {r['name_a']} | {r['name_b']} | {r['formality_a']} | {r['formality_b']} |")
        L.append("")
    if T3:
        L += ["## v1 Table 3 (published pairs), orientation-corrected AUCs (EXPLORATORY)", "",
              "| mid | low | high | AUC low | AUC high | dAUC pub | equiv (gpt/claude) | orient low | orient high "
              "| AUC low corr | AUC high corr | dAUC corr |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in T3["rows"]:
            L.append(f"| {r['mid']} | {r['low']} | {r['high']} | {r['auc_low_published']} | {r['auc_high_published']} | "
                     f"{r['dauc_published']} | {r['equivalence']['gpt']}/{r['equivalence']['claude']} | "
                     f"{r['orientation_low']['gpt']}/{r['orientation_low']['claude']} | "
                     f"{r['orientation_high']['gpt']}/{r['orientation_high']['claude']} | {r['auc_low_corrected']} | "
                     f"{r['auc_high_corrected']} | {r['dauc_corrected']} |")
        L.append("")
    if V2R:
        L += ["## PREREG-15-v2 (frozen primary + secondaries; then EXPLORATORY)", "",
              f"Complete metrics (all 4 runs): {V2R['n_complete_metrics']} of {V2R['n_arm_pairs']} arm pairs.", "",
              "```json", json.dumps({k: V2R[k] for k in ("PRIMARY_H1_rho_ddist_dauc_lt0_A2.75", "primary_floor_check",
                                                          "SECONDARY")}, indent=1, ensure_ascii=False), "```", "",
              "### EXPLORATORY", "```json", json.dumps(V2R["EXPLORATORY"], indent=1, ensure_ascii=False), "```", "",
              "### EXPLORATORY: published v1 numbers under orientation correction", "```json",
              json.dumps({k: v for k, v in V2R["EXPLORATORY_v1"].items() if k != "table3_rows"}, indent=1,
                         ensure_ascii=False), "```", ""]
    if P25:
        L += ["## PREREG-25 statistics (per family)", "",
              f"Valid run files: {P25['n_run_files_valid']}; invalid: {len(P25['invalid_runs'])}.", "", "```json",
              json.dumps({f: {k: v for k, v in d.items() if k != "per_metric"} for f, d in P25["families"].items()},
                         indent=1), "```", ""]
    open(f"{OUT}/prereg25_report.md", "w").write("\n".join(L) + "\n")
    print(f"wrote {OUT}/prereg25_report.md ({len(L)} lines)")


if __name__ == "__main__":
    main()
