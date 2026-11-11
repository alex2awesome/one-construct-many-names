#!/usr/bin/env python
"""PREREG-25 Steps 1-2: aggregate the two judge families, Cohen's kappa, mechanical selection.

Inputs : outputs/lexicon/prereg25_20261012/adjudication/{key_private.json, gpt/bNN.json, claude/bNN.json}
Outputs: outputs/lexicon/prereg25_20261012/{p25_pairs.json, v2_arm_verdicts.json, v1_table3_verdicts.json,
         adjudication_summary.json, form_verdicts.json, pair_verdicts.json}
Selection rule (frozen): eligible pair = both families "same" AND both families "better" for both forms;
per metric the eligible pair with the largest formality gap (ties: lowest sha256(form_id_a + form_id_b),
A = less formal form). Gap floor 1 pooled SD; drop to 0.5 SD (reported) only if < 15 metrics qualify.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict

ROOT = "ROOT"
V2 = f"{ROOT}/outputs/lexicon/prereg15v2_20260828"
OUT = f"{ROOT}/outputs/lexicon/prereg25_20261012"
ADJ = f"{OUT}/adjudication"
FAMS = ("gpt", "claude")


def kappa(a: list, b: list) -> dict:
    n = len(a)
    if n == 0:
        return {"n": 0}
    cats = sorted(set(a) | set(b))
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[c] * cb[c] for c in cats) / n ** 2
    k = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    conf = {f"{x}|{y}": v for (x, y), v in sorted(Counter(zip(a, b)).items())}
    return {"n": n, "kappa": round(k, 4), "pct_agree": round(po, 4), "confusion_gpt|claude": conf,
            "marginals": {"gpt": dict(ca), "claude": dict(cb)}}


def atomic_dump(obj, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def main() -> None:
    K = json.load(open(f"{ADJ}/key_private.json"))
    pool = json.load(open(f"{V2}/v2_form_pool.json"))
    rub = {r["form_id"]: r["rubric"] for r in pool}
    by_rub = {r["rubric"]: r["form_id"] for r in pool}
    fmj = json.load(open(f"{V2}/v2_form_formality.json"))
    fm, sd = fmj["form_means"], fmj["pooled_sd"]

    pv = {f: {} for f in FAMS}      # pair key "fa|fb" -> verdict
    fv = {f: {} for f in FAMS}      # form id -> verdict
    batch_status = {f: {} for f in FAMS}
    anchor_detail = {f: Counter() for f in FAMS}
    for fam in FAMS:
        for bid, kb in K["key"].items():
            p = f"{ADJ}/{fam}/{bid}.json"
            if not os.path.exists(p):
                batch_status[fam][bid] = "missing"
                continue
            rec = json.load(open(p))
            batch_status[fam][bid] = f"{rec['status']}" + (f" a{rec.get('attempt')} acc={rec.get('anchor_acc'):.2f}"
                                                         if rec["status"] == "pass" else "")
            if rec["status"] != "pass":
                continue
            P = {k: str(v).strip().lower() for k, v in rec["reply"]["pairs"].items()}
            F = {k: str(v).strip().lower() for k, v in rec["reply"]["forms"].items()}
            for lid, e in kb["pairs"].items():
                if e["kind"] == "real":
                    pv[fam]["|".join(e["forms"])] = P[lid]
                else:
                    anchor_detail[fam][f"{e['kind']}_n"] += 1
                    anchor_detail[fam][f"{e['kind']}_ok"] += P[lid] == e["answer"]
            for lid, e in kb["forms"].items():
                if e["kind"] == "real":
                    fv[fam][e["form"]] = F[lid]
                else:
                    anchor_detail[fam][f"{e['kind']}_n"] += 1
                    anchor_detail[fam][f"{e['kind']}_ok"] += F[lid] == e["answer"]

    common_p = sorted(set(pv["gpt"]) & set(pv["claude"]))
    common_f = sorted(set(fv["gpt"]) & set(fv["claude"]))
    summary = {
        "design": "notes/2026-10-12__prereg25-matched-meaning-name-swap.md (Step 1-2)",
        "judges": {"gpt": "gpt-5.6-sol via codex exec 0.154 on sk2 (web search disabled, read-only sandbox, ephemeral "
                          "session per batch; box-default reasoning effort = none, as served in all 27 logs)",
                   "claude": "claude-sonnet-5 via Claude Code CLI -p on sk2 (all 27 batches served on the "
                             "anon token: the account was at its session limit until 23:20 PDT)"},
        "batch_design": "27 batches x (58-59 real pairs + 3 paraphrase-same + 3 cross-metric-different anchors; "
                        "29-30 real forms + 4 explicit-fault anchors); gate >= 8/10 anchors; names only, no metric "
                        "context, no definitions, no scores",
        "n_pairs_total": len(K["pairs"]), "n_forms_total": len(K["forms"]),
        "n_pairs_judged": {f: len(pv[f]) for f in FAMS}, "n_forms_judged": {f: len(fv[f]) for f in FAMS},
        "batch_status": batch_status,
        "anchor_accuracy_passed_batches": {f: dict(anchor_detail[f]) for f in FAMS},
        "kappa_equivalence": kappa([pv["gpt"][k] for k in common_p], [pv["claude"][k] for k in common_p]),
        "kappa_orientation": kappa([fv["gpt"][k] for k in common_f], [fv["claude"][k] for k in common_f]),
    }
    # kappa on the candidate-pair subset only and on equivalence collapsed to same vs not-same
    summary["kappa_equivalence_same_vs_not"] = kappa(
        ["same" if pv["gpt"][k] == "same" else "not" for k in common_p],
        ["same" if pv["claude"][k] == "same" else "not" for k in common_p])
    summary["kappa_orientation_worse_vs_not"] = kappa(
        ["worse" if fv["gpt"][k] == "worse" else "not" for k in common_f],
        ["worse" if fv["claude"][k] == "worse" else "not" for k in common_f])

    def form_rec(f: str) -> dict:
        g, c = fv["gpt"].get(f), fv["claude"].get(f)
        return {"form_id": f, "name": rub[f], "formality": fm[f], "gpt": g, "claude": c,
                "both_better": g == "better" and c == "better", "both_worse": g == "worse" and c == "worse"}

    def pair_rec(fa: str, fb: str) -> dict:
        k = "|".join(sorted((fa, fb)))
        g, c = pv["gpt"].get(k), pv["claude"].get(k)
        return {"gpt": g, "claude": c, "both_same": g == "same" and c == "same",
                "judged_by_both": g is not None and c is not None}

    atomic_dump({f: form_rec(f) for f in K["forms"]}, f"{OUT}/form_verdicts.json")
    atomic_dump({k: {**pair_rec(*k.split("|")), **v} for k, v in K["pairs"].items()}, f"{OUT}/pair_verdicts.json")

    # ---- selection
    item_mids = sorted((f[len("v2_items_"):-5] for f in os.listdir(V2)
                        if f.startswith("v2_items_") and f.endswith(".json")), key=lambda m: int(m[1:]))
    bym = defaultdict(set)
    for r in pool:
        for m in r["metric_ids"]:
            bym[m].add(r["form_id"])

    def select(floor: float) -> tuple[list, dict]:
        chosen, excluded = [], {}
        for mid in item_mids:
            elig, n_cand, n_judged = [], 0, 0
            fs = sorted(bym[mid])
            for i, x in enumerate(fs):
                for y in fs[i + 1:]:
                    gap = abs(fm[x] - fm[y])
                    if gap < floor:
                        continue
                    n_cand += 1
                    a, b = (x, y) if fm[x] < fm[y] else (y, x)
                    pr = pair_rec(a, b)
                    ra, rb = form_rec(a), form_rec(b)
                    if pr["judged_by_both"] and None not in (ra["gpt"], ra["claude"], rb["gpt"], rb["claude"]):
                        n_judged += 1
                    if pr["both_same"] and ra["both_better"] and rb["both_better"]:
                        elig.append((-gap, hashlib.sha256((a + b).encode()).hexdigest(), a, b, gap, pr, ra, rb))
            if not elig:
                excluded[mid] = {"reason": "no eligible pair", "n_candidate_pairs": n_cand,
                                 "n_fully_judged": n_judged}
                continue
            elig.sort()
            _, _, a, b, gap, pr, ra, rb = elig[0]
            chosen.append({"mid": mid, "form_id_a": a, "form_id_b": b, "name_a": rub[a], "name_b": rub[b],
                           "formality_a": fm[a], "formality_b": fm[b], "gap": round(gap, 4),
                           "gap_sd": round(gap / sd, 3), "n_eligible_pairs": len(elig),
                           "n_candidate_pairs": n_cand,
                           "verdicts": {"equivalence": {"gpt": pr["gpt"], "claude": pr["claude"]},
                                        "orientation_a": {"gpt": ra["gpt"], "claude": ra["claude"]},
                                        "orientation_b": {"gpt": rb["gpt"], "claude": rb["claude"]}}})
        return chosen, excluded

    chosen, excluded = select(sd)
    floor_used, floor_note = sd, "1.0 pooled SD (frozen default)"
    if len(chosen) < 15:
        floor_note = (f"only {len(chosen)} metrics qualified at 1.0 SD; frozen rule drops the floor to 0.5 SD "
                      "(pairs between 0.5 and 1.0 SD need adjudication first)")
    atomic_dump({"design": "notes/2026-10-12__prereg25-matched-meaning-name-swap.md",
                 "rule": "both families 'same requirement' AND both families 'higher = better' for both forms; "
                         "largest formality gap per metric; ties lowest sha256(form_id_a+form_id_b); "
                         "A = less formal",
                 "gap_floor": floor_used, "gap_floor_note": floor_note, "pooled_sd": sd,
                 "n_metrics_universe": len(item_mids), "n_selected": len(chosen),
                 "pairs": chosen, "excluded": excluded,
                 "batch_status": batch_status},
                f"{OUT}/p25_pairs.json")

    # ---- v2 arm verdicts
    arms = json.load(open(f"{V2}/v2_arms.json"))
    arm_out = {}
    for mid, a in sorted(arms.items(), key=lambda x: int(x[0][1:])):
        lo, hi = a["low_form_id"], a["high_form_id"]
        pr = pair_rec(lo, hi)
        arm_out[mid] = {"low": form_rec(lo), "high": form_rec(hi), "equivalence": pr,
                        "verified_equivalent": pr["both_same"],
                        "verified_equivalent_same_orientation": pr["both_same"] and form_rec(lo)["both_better"]
                        and form_rec(hi)["both_better"],
                        "has_items": mid in item_mids}
    atomic_dump(arm_out, f"{OUT}/v2_arm_verdicts.json")

    # ---- v1 table 3 verdicts with orientation-corrected AUCs (exploratory)
    t3 = json.load(open(f"{ROOT}/outputs/lexicon/paper1_figdata.json"))["table3"]
    rows = []
    for grp in ("low_wins", "high_wins"):
        for r in t3[grp]:
            lo, hi = by_rub[r["low"]], by_rub[r["high"]]
            rl, rh, pr = form_rec(lo), form_rec(hi), pair_rec(lo, hi)
            al = 1 - r["auc_low"] if rl["both_worse"] else r["auc_low"]
            ah = 1 - r["auc_high"] if rh["both_worse"] else r["auc_high"]
            rows.append({"group": grp, "mid": r["mid"], "low": r["low"], "high": r["high"],
                         "auc_low_published": r["auc_low"], "auc_high_published": r["auc_high"],
                         "dauc_published": r["dauc"],
                         "orientation_low": {"gpt": rl["gpt"], "claude": rl["claude"]},
                         "orientation_high": {"gpt": rh["gpt"], "claude": rh["claude"]},
                         "equivalence": {"gpt": pr["gpt"], "claude": pr["claude"]},
                         "verified_equivalent": pr["both_same"],
                         "low_flipped": rl["both_worse"], "high_flipped": rh["both_worse"],
                         "auc_low_corrected": round(al, 4), "auc_high_corrected": round(ah, 4),
                         "dauc_corrected": round(al - ah, 4),
                         "winner_published": "low" if r["dauc"] > 0 else "high",
                         "winner_corrected": "low" if al - ah > 0 else ("high" if al - ah < 0 else "tie")})
    atomic_dump({"note": "EXPLORATORY: orientation-corrected AUC = 1 - AUC for a form both families call "
                         "'higher = worse'. Published values from outputs/lexicon/paper1_figdata.json table3.",
                 "rows": rows}, f"{OUT}/v1_table3_verdicts.json")

    summary["n_selected_p25"] = len(chosen)
    summary["v2_arms_verified_equivalent"] = sum(v["verified_equivalent"] for v in arm_out.values() if v["has_items"])
    summary["v2_arms_verified_equivalent_same_orientation"] = sum(
        v["verified_equivalent_same_orientation"] for v in arm_out.values() if v["has_items"])
    summary["v2_arms_with_items"] = sum(v["has_items"] for v in arm_out.values())
    summary["v2_arm_forms_both_worse"] = sorted(
        {(m, s, v[s]["name"]) for m, v in arm_out.items() if v["has_items"] for s in ("low", "high")
         if v[s]["both_worse"]})
    summary["v1_table3_verified_equivalent"] = sum(r["verified_equivalent"] for r in rows)
    summary["v1_table3_flipped_forms"] = [(r["mid"], side, r[side]) for r in rows for side in ("low", "high")
                                          if r[f"{side}_flipped"]]
    summary["forms_both_worse_total"] = sum(form_rec(f)["both_worse"] for f in K["forms"])
    summary["forms_either_worse_total"] = sum(fv["gpt"].get(f) == "worse" or fv["claude"].get(f) == "worse"
                                              for f in K["forms"])
    summary["pairs_both_same_total"] = sum(pair_rec(*k.split("|"))["both_same"] for k in K["pairs"])
    atomic_dump(summary, f"{OUT}/adjudication_summary.json")
    print(json.dumps({k: summary[k] for k in ("n_pairs_judged", "n_forms_judged", "kappa_equivalence",
                                                "kappa_orientation", "n_selected_p25",
                                                "v2_arms_verified_equivalent",
                                                "v2_arms_verified_equivalent_same_orientation",
                                                "v1_table3_verified_equivalent", "v1_table3_flipped_forms",
                                                "forms_both_worse_total", "pairs_both_same_total")},
                     indent=1, ensure_ascii=False)[:4000])


if __name__ == "__main__":
    main()
