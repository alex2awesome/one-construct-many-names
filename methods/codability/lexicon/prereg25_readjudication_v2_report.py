#!/usr/bin/env python
"""Write readjudication_v2/report.md from results.json (EXPLORATORY)."""
from __future__ import annotations

import json

ROOT = "ROOT"
RV = f"{ROOT}/outputs/lexicon/prereg25_20261012/readjudication_v2"
FAMS = (("gpt", "GPT-5.6"), ("claude", "Claude Sonnet 5"))


def f3(x) -> str:
    if x is None:
        return "n/a"
    s = f"{x:.3f}"
    return s.replace("0.", ".", 1) if abs(x) < 1 else s


def conf(c: dict, rows=("same", "overlap", "different"), cols=("same", "overlap", "different")) -> str:
    out = "| author \\ judge | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols) + "\n"
    for r in rows:
        if not any(k.startswith(r + "|") for k in c):
            continue
        out += f"| {r} | " + " | ".join(str(c.get(f"{r}|{k}", 0)) for k in cols) + " |\n"
    return out


def main() -> None:
    R = json.load(open(f"{RV}/results.json"))
    sp = json.load(open(f"{RV}/fewshot_split.json"))
    L = []
    A = L.append
    A("# PREREG-25 re-adjudication v2: author-calibrated standard (EXPLORATORY)\n")
    A("Post hoc, after the first author's blind annotation of 2026-10-12. Not part of any frozen design. "
      "Same harness, anchors, gate (>= 8/10 anchors) and batch contents as lane A; prompt v2 = lane-A prompt + "
      "calibration paragraph + 32 author-labeled examples after the PART 1 question (`prompt_v2_head.txt`). "
      "Orientation question unchanged.\n")
    A(f"- Author labels: `{R['author_snapshot']}` ({R['n_author_labels']} pairs). Few-shot split: `fewshot_split.json` "
      f"(examples {sp['counts']['fewshot']}, by set {sp['counts']['fewshot_by_set']}; held out {sp['counts']['heldout']}, "
      f"by set {sp['counts']['heldout_by_set']}).")
    A(f"- Pairs judged: {R['n_pairs_judged']}; forms judged: {R['n_forms_judged']}; PREREG-11 pairs judged: {R['n_p11_judged']}.")
    A(f"- Batch status: GPT {sorted(set(v.split(' ')[0] for v in R['batch_status']['gpt'].values()))}, "
      f"Claude {sorted(set(v.split(' ')[0] for v in R['batch_status']['claude'].values()))}; anchors {R['anchor_accuracy']}.")
    b = R["between_family_calibrated"]
    A(f"- Between families (calibrated): equivalence kappa {f3(b['kappa_3way']['kappa'])} (3-way, n = {b['kappa_3way']['n']}), "
      f"{f3(b['kappa_same_vs_not']['kappa'])} (same vs not); orientation kappa {f3(b['kappa_orientation']['kappa'])}. "
      f"Strict run for comparison: .622 / .666; orientation {f3(R['strict_orientation_kappa'])}.")
    A(f"- Marginals over all pairs: calibrated {R['marginals']['calibrated']}; strict {R['marginals']['strict']}.\n")

    a = R["a_heldout_agreement"]
    A("## (a) Agreement with the author on held-out pairs\n")
    A(f"Held out: {a['n_heldout']} pairs {a['heldout_author_labels']}, by set {a['heldout_by_set']}. "
      "The author used 'different' only 3 times, all in the few-shot set, so no held-out pair is author-'different'.\n")
    A("| Subset | Standard | n | GPT kappa 3-way | GPT same-vs-not | Claude kappa 3-way | Claude same-vs-not | both-same vs author-same |")
    A("|---|---|---|---|---|---|---|---|")
    for key, lab in (("calibrated_all_heldout", "all held-out|calibrated"),
                     ("calibrated_heldout_excl_p11", "held-out excl. PREREG-11|calibrated"),
                     ("strict_heldout_excl_p11", "held-out excl. PREREG-11|strict"),
                     ("calibrated_heldout_excl_p25", "held-out excl. PREREG-25|calibrated"),
                     ("calibrated_heldout_excl_p11_p25", "held-out excl. PREREG-11 and PREREG-25|calibrated"),
                     ("strict_heldout_excl_p11_p25", "held-out excl. PREREG-11 and PREREG-25|strict")):
        x = a[key]
        s, st = lab.split("|")
        A(f"| {s} | {st} | {x['gpt']['kappa_3way']['n']} | {f3(x['gpt']['kappa_3way'].get('kappa'))} | "
          f"{f3(x['gpt']['kappa_same_vs_not'].get('kappa'))} | {f3(x['claude']['kappa_3way'].get('kappa'))} | "
          f"{f3(x['claude']['kappa_same_vs_not'].get('kappa'))} | {f3(x['both_same_vs_author_same'].get('kappa'))} |")
    A("")
    d = a["disagreement_cell_heldout"]
    A(f"Held-out pairs on which at least one strict family disagreed with the author (n = {d['n']}): judge verdict equals the "
      f"author's label for calibrated {d['calibrated_matches_author']} vs strict {d['strict_matches_author']}. "
      f"Few-shot examples reproduced by the judges: {a['fewshot_reproduced']}.\n")
    for key, lab in (("calibrated_all_heldout", "calibrated, all held-out"), ("strict_heldout_excl_p11", "strict, held-out excl. PREREG-11")):
        for fam, fn in FAMS:
            A(f"Confusion ({lab}, {fn}):\n")
            A(conf(a[key][fam]["kappa_3way"]["confusion_rowA|colB"]))

    c = R["b_counts"]
    A("## (b) Counts under the calibrated standard (both families 'same')\n")
    A("| Set | n | strict both-same | calibrated both-same | GPT same | Claude same | few-shot examples among calibrated-same |")
    A("|---|---|---|---|---|---|---|")
    for key, lab in (("v2_arms_86", "PREREG-15-v2 arm pairs"), ("v2_arms_with_items_85", "  of which with items"),
                     ("table3_10", "Table 3 pairs"), ("p25_35", "PREREG-25 pairs")):
        x = c[key]
        A(f"| {lab} | {x['n']} | {x['strict_both_same']} | {x['calibrated_both_same']} | {x['per_family_same']['gpt']} | "
          f"{x['per_family_same']['claude']} | {len(x['calibrated_both_same_fewshot_examples'])} |")
    p = c["p11_14"]
    A(f"| PREREG-11 pairs | 14 | not judged | {p['n_calibrated_both_same']} | {p['per_family_same']['gpt']} | "
      f"{p['per_family_same']['claude']} | {sum(x in p['fewshot_examples'] for x in p['calibrated_both_same'])} |")
    A("")
    al = c["all_pairs"]
    A(f"All {al['n']} pairs: calibrated both-same {al['calibrated_both_same']} vs strict {al['strict_both_same']}; "
      f"intersection {al['intersection']}; strict-only {al['strict_only']}; calibrated-only {al['calibrated_only']}. "
      f"({al['fewshot_keys_in_pool']} few-shot pairs are in the pool; their verdicts are not independent.)\n")
    A(f"- v2 arms calibrated both-same: {c['v2_arms_86']['calibrated_both_same_mids']}; of these both-better orientation "
      f"(calibrated run): {c['v2_arms_86']['calibrated_both_same_and_both_better']}.")
    A(f"- Table 3 calibrated both-same: {c['table3_10']['calibrated_both_same_mids']}.")
    A(f"- PREREG-11 calibrated both-same: {p['calibrated_both_same']}; consensus {p['calibrated_consensus']}; "
      f"few-shot examples {p['fewshot_examples']}; author {p['author']}.")
    A(f"- PREREG-25 calibrated consensus: {c['p25_35']['calibrated_consensus']}.\n")

    A("## (c) Name-swap disruption by equivalence standard (existing runs, no new scoring)\n")
    v = R["c_name_swap_delta_v2"]
    A(f"PREREG-15-v2 arm pairs, GPT-5.6 Delta_name (n = {v['n']}; mean {f3(v['all']['mean'])}, median {f3(v['all']['median'])}). {v['def']}.\n")
    A("| Grouping | label | n | mean Delta | median Delta |\n|---|---|---|---|---|")
    for key, lab in (("by_strict_consensus", "strict (both-family consensus)"),
                     ("by_calibrated_consensus", "calibrated (both-family consensus)"),
                     ("by_calibrated_consensus_excl_fewshot", "calibrated, few-shot pairs removed"), ("by_author", "author")):
        for g, x in v[key].items():
            A(f"| {lab} | {g} | {x['n']} | {f3(x['mean'])} | {f3(x['median'])} |")
    A("")
    for key, lab in (("strict_both_same_vs_not", "strict both-same vs not"),
                     ("calibrated_both_same_vs_not", "calibrated both-same vs not"),
                     ("calibrated_both_same_vs_not_excl_fewshot", "calibrated both-same vs not, few-shot pairs removed"),
                     ("author_same_vs_not", "author same vs not")):
        x = v[key]
        A(f"- {lab}: n {x['n_same']} vs {x['n_not']}; mean {f3(x.get('mean_same'))} vs {f3(x.get('mean_not'))}; "
          f"median {f3(x.get('median_same'))} vs {f3(x.get('median_not'))}; Mann-Whitney p = {x.get('mannwhitney_p_two_sided')}")
    q = R["c_name_swap_delta_p11"]
    A("\nPREREG-11 pairs, stored GPT-5.6 D_n:\n")
    A("| Grouping | label | n | mean D_n | median D_n |\n|---|---|---|---|---|")
    for key, lab in (("by_author", "author"), ("by_calibrated_consensus", "calibrated (both-family consensus)")):
        for g, x in q[key].items():
            A(f"| {lab} | {g} | {x['n']} | {f3(x['mean'])} | {f3(x['median'])} |")
    A("\n| Pair | task | D_n | author | calibrated GPT/Claude | few-shot example |\n|---|---|---|---|---|---|")
    p11v = json.load(open(f"{RV}/verdicts_p11.json"))
    for r in q["rows"]:
        pv = p11v[r["doc"]]
        A(f"| {r['doc']} {pv['a']} / {pv['b']} | {r['task']} | {f3(r['D_n'])} | {r['author']} | "
          f"{pv['calibrated']['gpt']}/{pv['calibrated']['claude']} | {'yes' if r['fewshot_example'] else ''} |")

    s = R["d_sensitivity_prereg25"]
    A("\n## (d) Sensitivity\n")
    A(f"PREREG-25 primary (Delta_name > Delta_scaf, one-sided Wilcoxon), restricted to the {s['n_subset']} of 35 metrics whose "
      f"pair is also both-same under the calibrated standard (dropped: {s['dropped_mids']}); "
      f"{s['n_subset_also_both_better_calibrated']} of these also both-better in the calibrated orientation run.\n")
    A("| Family | subset | n | mean Delta_name | mean Delta_scaf | n name > scaf | Wilcoxon p |\n|---|---|---|---|---|---|---|")
    for fam, fn in FAMS:
        for key, lab in (("full_35", "all 35"), ("calibrated_same_subset", "calibrated-same"),
                         ("calibrated_same_and_orientation_subset", "calibrated-same + both-better")):
            x = s[key][fam]
            if not x.get("n"):
                A(f"| {fn} | {lab} | 0 | | | | |")
                continue
            A(f"| {fn} | {lab} | {x['n']} | {f3(x['mean_delta_name'])} | {f3(x['mean_delta_scaf'])} | "
              f"{x['n_delta_name_gt_delta_scaf']} | {x.get('wilcoxon_one_sided', {}).get('p', 'n/a')} |")
    t = R["d_sensitivity_prereg15v2"]
    A("\nPREREG-15-v2 win split (low-favoring / high-favoring / ties; Delta AUC = AUC_low - AUC_high) and median |Delta AUC|:\n")
    A("| Subset | n | low / high / tie | sign p | median abs dAUC | IQR |\n|---|---|---|---|---|---|")
    for key, lab in (("all_85", "all"), ("calibrated_same", "calibrated both-same"),
                     ("calibrated_same_excl_fewshot", "calibrated both-same, few-shot pairs removed"),
                     ("calibrated_same_and_both_better", "calibrated both-same + both-better"),
                     ("calibrated_not_same", "calibrated not both-same")):
        x = t[key]
        w, dd = x["win_split"], x["dauc"]
        if not w.get("n"):
            A(f"| {lab} | 0 | | | | |")
            continue
        A(f"| {lab} | {w['n']} | {w['low_favoring']} / {w['high_favoring']} / {w['ties']} | {w.get('sign_p_two_sided')} | "
          f"{f3(dd['median_abs'])} | {f3(dd['iqr_abs'][0])} to {f3(dd['iqr_abs'][1])} |")
    if t["calibrated_same"]["n"]:
        w = t["calibrated_same"]["win_split_orientation_corrected"]
        A(f"\nCalibrated both-same, orientation-corrected (strict-run orientation, as in prereg15v2_results EXPLORATORY): "
          f"{w['low_favoring']} / {w['high_favoring']} / {w['ties']}, median |dAUC| "
          f"{f3(t['calibrated_same']['dauc_orientation_corrected']['median_abs'])}. Mids: {t['calibrated_same']['mids']}.")
    A("\n## Files\n")
    A("`prompt_v2_head.txt`, `prompt_v2_head_creative_writing.txt` (batch b27 only), `fewshot_split.json`, `batches.json`, "
      "`key_private.json`, `gpt/`, `claude/` (every attempt kept), `run.log`, `verdicts_pairs.json`, `verdicts_forms.json`, "
      "`verdicts_p11.json`, `heldout_rows.json`, `results.json`. Scripts: methods/codability/lexicon/"
      "prereg25_readjudication_v2_{build,run,analysis,report}.py.")
    A("\nDesign notes: the 7 humor PREREG-11 pairs were appended one each to batches b00 to b06; the 7 creative-writing "
      "PREREG-11 pairs form batch b27, whose head substitutes creative-writing wording for the comedy wording (including "
      "'a writing community' in the calibration paragraph); anchors there are the same comedy anchor lists. One few-shot "
      "example (v2 a43, 'Blend of writing and drawing' / 'Cover art', author 'different') is also a cross-metric "
      "'different' anchor string pair, so that anchor is answered by an example in the batches that carry it.")
    open(f"{RV}/report.md", "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
