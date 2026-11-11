#!/usr/bin/env python
"""PREREG-25 re-adjudication v2 (author-calibrated standard): aggregation and analysis.

Inputs : outputs/lexicon/prereg25_20261012/readjudication_v2/{key_private.json, fewshot_split.json,
         gpt/bNN.json, claude/bNN.json}; lane-A strict verdicts (pair_verdicts.json, form_verdicts.json);
         author annotations (newest scratchpad snapshot); PREREG-15-v2 score runs (v2_score_*);
         prereg25_results.json, prereg15v2_results.json; PREREG-11 per-pair D_n
         (review_repro_20261012/b_prereg11_12_spec.md, section 3).
Outputs: readjudication_v2/{verdicts_pairs.json, verdicts_forms.json, verdicts_p11.json, results.json}.
All analyses here are EXPLORATORY (post hoc, after the author's annotation of 2026-10-12).
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from collections import Counter

import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon

ROOT = "ROOT"
sys.path.insert(0, f"{ROOT}/methods/codability/lexicon")
import prereg15v2_analysis as A15  # noqa: E402  (load_run, rho, dist, win_split)

P25 = f"{ROOT}/outputs/lexicon/prereg25_20261012"
RV = f"{P25}/readjudication_v2"
V2 = f"{ROOT}/outputs/lexicon/prereg15v2_20260828"
SCR = "/private/tmp/claude-502/-Users-anon-Projects-stanford-research-norm-research/6eb8228d-7011-4800-87e2-61a172f6003c/scratchpad"
FAMS = ("gpt", "claude")


def newest_snapshot() -> str:
    snaps = sorted(glob.glob(f"{SCR}/author_annotations_snapshot*.json"),
                   key=lambda p: int(re.search(r"snapshot(\d+)", p).group(1)))
    return snaps[-1]


def kappa(a: list, b: list) -> dict:
    n = len(a)
    if n == 0:
        return {"n": 0}
    cats = sorted(set(a) | set(b))
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[c] * cb[c] for c in cats) / n ** 2
    k = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    return {"n": n, "kappa": round(k, 4), "pct_agree": round(po, 4),
            "confusion_rowA|colB": {f"{x}|{y}": v for (x, y), v in sorted(Counter(zip(a, b)).items())}}


def sv(x):
    return None if x is None else ("same" if x == "same" else "not")


def consensus(g, c) -> str | None:
    if g is None or c is None:
        return None
    return g if g == c else "split"


def main() -> None:
    K = json.load(open(f"{RV}/key_private.json"))
    split = json.load(open(f"{RV}/fewshot_split.json"))
    few_ids = {c["doc"] for c in split["fewshot"]}
    snap_path = newest_snapshot()
    author = json.load(open(snap_path))
    annot = {p["id"]: p for s in json.load(open(f"{SCR}/annot_data.json")) for p in s["pairs"]}
    orig_pv = json.load(open(f"{P25}/pair_verdicts.json"))
    orig_fv = json.load(open(f"{P25}/form_verdicts.json"))
    pool = json.load(open(f"{V2}/v2_form_pool.json"))
    by_rub = {r["rubric"]: r["form_id"] for r in pool}
    arms = json.load(open(f"{V2}/v2_arms.json"))
    p25sel = json.load(open(f"{P25}/p25_pairs.json"))["pairs"]

    # ---------------- aggregate calibrated verdicts
    pv = {f: {} for f in FAMS}
    fv = {f: {} for f in FAMS}
    status, anchors = {f: {} for f in FAMS}, {f: Counter() for f in FAMS}
    for fam in FAMS:
        for bid, kb in K["key"].items():
            p = f"{RV}/{fam}/{bid}.json"
            if not os.path.exists(p):
                status[fam][bid] = "missing"
                continue
            rec = json.load(open(p))
            status[fam][bid] = rec["status"] + (f" a{rec['attempt']} acc={rec['anchor_acc']:.2f}" if rec["status"] == "pass" else "")
            if rec["status"] != "pass":
                continue
            P = {k: str(v).strip().lower() for k, v in rec["reply"]["pairs"].items()}
            F = {k: str(v).strip().lower() for k, v in rec["reply"]["forms"].items()}
            for lid, e in kb["pairs"].items():
                if e["kind"] == "real":
                    pv[fam]["|".join(e["forms"])] = P[lid]
                elif e["kind"] == "p11":
                    pv[fam][e["doc"]] = P[lid]
                else:
                    anchors[fam][e["kind"] + "_n"] += 1
                    anchors[fam][e["kind"] + "_ok"] += P[lid] == e["answer"]
            for lid, e in kb["forms"].items():
                if e["kind"] == "real":
                    fv[fam][e["form"]] = F[lid]
                elif e["kind"] == "p11":
                    fv[fam][f"{e['doc']}:{e['side']}"] = F[lid]
                else:
                    anchors[fam][e["kind"] + "_n"] += 1
                    anchors[fam][e["kind"] + "_ok"] += F[lid] == e["answer"]

    def cal(key: str) -> dict:
        g, c = pv["gpt"].get(key), pv["claude"].get(key)
        return {"gpt": g, "claude": c, "both_same": g == "same" and c == "same", "consensus": consensus(g, c)}

    def org(key: str) -> dict:
        r = orig_pv.get(key)
        if r is None:
            return {"gpt": None, "claude": None, "both_same": None, "consensus": None}
        return {"gpt": r["gpt"], "claude": r["claude"], "both_same": r["both_same"],
                "consensus": consensus(r["gpt"], r["claude"])}

    def fcal(f: str) -> dict:
        g, c = fv["gpt"].get(f), fv["claude"].get(f)
        return {"gpt": g, "claude": c, "both_better": g == "better" and c == "better",
                "both_worse": g == "worse" and c == "worse"}

    real_keys = sorted(K["pairs"])
    common = [k for k in real_keys if k in pv["gpt"] and k in pv["claude"]]
    forms_common = [f for f in K["forms"] if f in fv["gpt"] and f in fv["claude"]]
    vp = {k: {"calibrated": cal(k), "strict": org(k), **K["pairs"][k]} for k in real_keys}
    json.dump(vp, open(f"{RV}/verdicts_pairs.json", "w"), indent=1, ensure_ascii=False)
    json.dump({f: {"calibrated": fcal(f), "strict": {"gpt": orig_fv[f]["gpt"], "claude": orig_fv[f]["claude"]}}
               for f in K["forms"]}, open(f"{RV}/verdicts_forms.json", "w"), indent=1, ensure_ascii=False)
    p11v = {}
    for i in range(14):
        d = f"p11:{i}"
        p11v[d] = {**K["p11"][d], "calibrated": cal(d),
                   "orientation_a": fcal(f"{d}:a"), "orientation_b": fcal(f"{d}:b"),
                   "author": author.get(d, {}).get("label"), "fewshot_example": d in few_ids}
    json.dump(p11v, open(f"{RV}/verdicts_p11.json", "w"), indent=1, ensure_ascii=False)

    R: dict = {"label": "EXPLORATORY (post hoc): author-calibrated re-adjudication of PREREG-25 Step 1",
               "author_snapshot": snap_path, "n_author_labels": len(author),
               "batch_status": status, "anchor_accuracy": {f: dict(anchors[f]) for f in FAMS},
               "n_pairs_judged": {f: sum(k in pv[f] for k in real_keys) for f in FAMS},
               "n_forms_judged": {f: sum(x in fv[f] for x in K["forms"]) for f in FAMS},
               "n_p11_judged": {f: sum(f"p11:{i}" in pv[f] for i in range(14)) for f in FAMS}}
    R["between_family_calibrated"] = {
        "kappa_3way": kappa([pv["gpt"][k] for k in common], [pv["claude"][k] for k in common]),
        "kappa_same_vs_not": kappa([sv(pv["gpt"][k]) for k in common], [sv(pv["claude"][k]) for k in common]),
        "kappa_orientation": kappa([fv["gpt"][f] for f in forms_common], [fv["claude"][f] for f in forms_common])}
    R["strict_orientation_kappa"] = json.load(open(f"{P25}/adjudication_summary.json"))["kappa_orientation"]["kappa"]
    R["marginals"] = {"calibrated": {f: dict(Counter(pv[f][k] for k in common)) for f in FAMS},
                      "strict": {f: dict(Counter(orig_pv[k][f] for k in common)) for f in FAMS}}

    # ---------------- (a) agreement with the author on held-out pairs
    def doc_key(doc: str) -> str:
        st, mid = doc.split(":", 1)
        if st == "p11":
            return doc
        if st == "v2":
            return "|".join(sorted((arms[mid]["low_form_id"], arms[mid]["high_form_id"])))
        if st == "p25":
            p = next(x for x in p25sel if x["mid"] == mid)
            return "|".join(sorted((p["form_id_a"], p["form_id_b"])))
        a = annot[doc]
        return "|".join(sorted((by_rub[a["a"]], by_rub[a["b"]])))

    held = sorted(d for d in author if d not in few_ids)
    rows = []
    for d in held:
        k = doc_key(d)
        rows.append({"doc": d, "set": author[d]["set"], "author": author[d]["label"], "key": k,
                     "cal": cal(k), "org": org(k) if not d.startswith("p11") else None})
    json.dump(rows, open(f"{RV}/heldout_rows.json", "w"), indent=1, ensure_ascii=False)

    def agree(sub: list, which: str) -> dict:
        out = {}
        for fam in FAMS:
            r2 = [r for r in sub if r[which] and r[which][fam] is not None]
            out[fam] = {"kappa_3way": kappa([r["author"] for r in r2], [r[which][fam] for r in r2]),
                        "kappa_same_vs_not": kappa([sv(r["author"]) for r in r2], [sv(r[which][fam]) for r in r2])}
        r2 = [r for r in sub if r[which] and r[which]["gpt"] is not None and r[which]["claude"] is not None]
        out["both_same_vs_author_same"] = kappa([sv(r["author"]) for r in r2],
                                                ["same" if r[which]["both_same"] else "not" for r in r2])
        return out

    nonp11 = [r for r in rows if r["set"] != "p11"]
    dis_docs = {c["doc"] for c in split["heldout"] if c.get("cell") == "dis"}
    R["a_heldout_agreement"] = {
        "note": "rows = author, cols = judge; held-out = author-labeled pairs not used as few-shot examples. "
                "PREREG-11 pairs have no strict verdicts, so strict-vs-calibrated comparisons use the non-PREREG-11 subset.",
        "n_heldout": len(rows), "heldout_author_labels": dict(Counter(r["author"] for r in rows)),
        "heldout_by_set": dict(Counter(r["set"] for r in rows)),
        "calibrated_all_heldout": agree(rows, "cal"),
        "calibrated_heldout_excl_p11": agree(nonp11, "cal"),
        "strict_heldout_excl_p11": agree(nonp11, "org"),
        "calibrated_heldout_excl_p25": agree([r for r in rows if r["set"] != "p25"], "cal"),
        "strict_heldout_excl_p11_p25": agree([r for r in nonp11 if r["set"] != "p25"], "org"),
        "calibrated_heldout_excl_p11_p25": agree([r for r in nonp11 if r["set"] != "p25"], "cal"),
        "disagreement_cell_heldout": {
            "def": "held-out pairs where >= 1 strict family disagreed with the author",
            "n": len([r for r in rows if r["doc"] in dis_docs]),
            "calibrated_matches_author": {f: sum(r["cal"][f] == r["author"] for r in rows if r["doc"] in dis_docs)
                                          for f in FAMS},
            "strict_matches_author": {f: sum(r["org"][f] == r["author"] for r in rows if r["doc"] in dis_docs)
                                      for f in FAMS}},
        "fewshot_reproduced": {f: f"{sum(cal(doc_key(c['doc']))[f] == c['label'] for c in split['fewshot'])}/"
                                  f"{len(split['fewshot'])}" for f in FAMS}}

    # ---------------- (b) counts
    orig_same = {k for k in real_keys if orig_pv[k]["both_same"]}
    cal_same = {k for k in common if cal(k)["both_same"]}
    few_keys = {doc_key(c["doc"]) for c in split["fewshot"]}
    v2keys = {mid: "|".join(sorted((a["low_form_id"], a["high_form_id"]))) for mid, a in arms.items()}
    items = {m for m in arms if os.path.exists(f"{V2}/v2_items_{m}.json")}
    t3 = json.load(open(f"{P25}/v1_table3_verdicts.json"))["rows"]
    t3keys = {r["mid"]: "|".join(sorted((by_rub[r["low"]], by_rub[r["high"]]))) for r in t3}

    def setcount(keys: dict, name: str) -> dict:
        ks = list(keys.items())
        cs = [m for m, k in ks if cal(k)["both_same"]]
        lo_hi = {}
        return {"n": len(ks), "calibrated_both_same": len(cs), "calibrated_both_same_mids": cs,
                "calibrated_both_same_fewshot_examples": [m for m in cs if keys[m] in few_keys],
                "strict_both_same": sum(bool(org(k)["both_same"]) for _, k in ks),
                "calibrated_consensus": dict(Counter(cal(k)["consensus"] for _, k in ks)),
                "strict_consensus": dict(Counter(org(k)["consensus"] for _, k in ks)),
                "per_family_same": {f: sum(cal(k)[f] == "same" for _, k in ks) for f in FAMS}}

    v2c = setcount(v2keys, "v2")
    v2c_items = setcount({m: k for m, k in v2keys.items() if m in items}, "v2_items")
    for blk, keys in ((v2c, v2keys), (v2c_items, {m: k for m, k in v2keys.items() if m in items})):
        blk["calibrated_both_same_and_both_better"] = [
            m for m in blk["calibrated_both_same_mids"]
            if fcal(arms[m]["low_form_id"])["both_better"] and fcal(arms[m]["high_form_id"])["both_better"]]
    p11c = {"n": 14, "calibrated_both_same": [d for d in p11v if p11v[d]["calibrated"]["both_same"]],
            "calibrated_consensus": dict(Counter(p11v[d]["calibrated"]["consensus"] for d in p11v)),
            "per_family_same": {f: sum(p11v[d]["calibrated"][f] == "same" for d in p11v) for f in FAMS},
            "fewshot_examples": sorted(d for d in p11v if p11v[d]["fewshot_example"]),
            "author": dict(Counter(p11v[d]["author"] for d in p11v))}
    p11c["n_calibrated_both_same"] = len(p11c["calibrated_both_same"])
    p25keys = {p["mid"]: "|".join(sorted((p["form_id_a"], p["form_id_b"]))) for p in p25sel}
    R["b_counts"] = {
        "all_pairs": {"n": len(common), "calibrated_both_same": len(cal_same), "strict_both_same": len(orig_same),
                      "intersection": len(cal_same & orig_same), "strict_only": len(orig_same - cal_same),
                      "calibrated_only": len(cal_same - orig_same),
                      "fewshot_keys_in_pool": len(few_keys & set(real_keys))},
        "v2_arms_86": v2c, "v2_arms_with_items_85": v2c_items, "table3_10": setcount(t3keys, "t3"),
        "p11_14": p11c, "p25_35": setcount(p25keys, "p25")}

    # ---------------- (c) name-swap Delta by standard
    p15 = {r["mid"]: r for r in json.load(open(f"{P25}/prereg15v2_results.json"))["per_metric"]}
    drows, maxdiff = [], 0.0
    for mid in sorted(items, key=lambda m: int(m[1:])):
        runs = {(arm, k): A15.load_run(f"{V2}/v2_score_{mid}_{arm}_r{k}.jsonl") for arm in ("low", "high") for k in (0, 1)}
        if any(v is None for v in runs.values()):
            continue
        w = [A15.rho(runs[(a, 0)], runs[(a, 1)]) for a in ("low", "high")]
        b = [A15.rho(runs[("low", i)], runs[("high", j)]) for i in (0, 1) for j in (0, 1)]
        w, b = [x for x in w if x is not None], [x for x in b if x is not None]
        if not w or not b:
            continue
        within, between = float(np.mean(w)), float(np.mean(b))
        if mid in p15 and p15[mid].get("within") is not None:
            maxdiff = max(maxdiff, abs(within - p15[mid]["within"]), abs(between - p15[mid]["between"]))
        k = v2keys[mid]
        drows.append({"mid": mid, "within": within, "between": between, "delta": within - between,
                      "strict": org(k)["consensus"], "calibrated": cal(k)["consensus"],
                      "strict_both_same": bool(org(k)["both_same"]), "calibrated_both_same": cal(k)["both_same"],
                      "author": author.get(f"v2:{mid}", {}).get("label"), "fewshot_example": k in few_keys,
                      "dauc": p15.get(mid, {}).get("dauc"), "dauc_oc": p15.get(mid, {}).get("dauc_oc")})

    def grp(rs: list, key: str, val: str = "delta") -> dict:
        out = {}
        for g in sorted({str(r[key]) for r in rs}):
            x = [r[val] for r in rs if str(r[key]) == g]
            out[g] = {"n": len(x), "mean": round(float(np.mean(x)), 4), "median": round(float(np.median(x)), 4)}
        return out

    def same_vs_not(rs: list, key: str, val: str = "delta") -> dict:
        a = [r[val] for r in rs if r[key] in ("same", True)]
        b = [r[val] for r in rs if r[key] is not None and r[key] not in ("same", True)]
        out = {"n_same": len(a), "n_not": len(b)}
        if a and b:
            out.update({"mean_same": round(float(np.mean(a)), 4), "mean_not": round(float(np.mean(b)), 4),
                        "median_same": round(float(np.median(a)), 4), "median_not": round(float(np.median(b)), 4),
                        "mannwhitney_p_two_sided": round(float(mannwhitneyu(a, b).pvalue), 4)})
        return out

    auth_rows = [r for r in drows if r["author"]]
    R["c_name_swap_delta_v2"] = {
        "def": "GPT-5.6 2x2 runs (prereg15v2_20260828/v2_score_*): within = mean of rho(r0,r1) over the two arms; "
               "between = mean of the 4 cross-arm rhos; Delta = within - between. Recomputed from the runs; "
               f"max |diff| vs prereg15v2_results.json = {maxdiff:.2e}",
        "n": len(drows), "all": {"mean": round(float(np.mean([r['delta'] for r in drows])), 4),
                                  "median": round(float(np.median([r['delta'] for r in drows])), 4)},
        "by_strict_consensus": grp(drows, "strict"), "by_calibrated_consensus": grp(drows, "calibrated"),
        "by_author": grp(auth_rows, "author"),
        "by_calibrated_consensus_excl_fewshot": grp([r for r in drows if not r["fewshot_example"]], "calibrated"),
        "calibrated_both_same_vs_not_excl_fewshot": same_vs_not([r for r in drows if not r["fewshot_example"]],
                                                                "calibrated_both_same"),
        "strict_both_same_vs_not": same_vs_not(drows, "strict_both_same"),
        "calibrated_both_same_vs_not": same_vs_not(drows, "calibrated_both_same"),
        "author_same_vs_not": same_vs_not(auth_rows, "author"),
        "rows": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in drows]}

    md = open(f"{ROOT}/outputs/lexicon/review_repro_20261012/b_prereg11_12_spec.md").read()
    p11d = {}
    for m in re.finditer(r"^\s*(\d+) & (humor|creative writing) & (.+?) & ([-.\d]+) & ([-.\d]+) & ([-.\d]+) & ([-.\d]+) & ([-.\d]+)", md, re.M):
        p11d[f"p11:{m.group(1)}"] = {"D_n": float(m.group(7)), "D_s": float(m.group(8)), "task": m.group(2)}
    assert len(p11d) == 14
    prow = [{"doc": d, "D_n": p11d[d]["D_n"], "task": p11d[d]["task"], "author": p11v[d]["author"],
             "calibrated": p11v[d]["calibrated"]["consensus"], "calibrated_both_same": p11v[d]["calibrated"]["both_same"],
             "fewshot_example": p11v[d]["fewshot_example"]} for d in sorted(p11d, key=lambda x: int(x[4:]))]
    R["c_name_swap_delta_p11"] = {
        "def": "stored per-pair GPT-5.6 D_n = W - B_n (b_prereg11_12_spec.md section 3)",
        "by_author": grp(prow, "author", "D_n"), "by_calibrated_consensus": grp(prow, "calibrated", "D_n"),
        "author_same_vs_not": same_vs_not(prow, "author", "D_n"),
        "calibrated_both_same_vs_not": same_vs_not(prow, "calibrated_both_same", "D_n"), "rows": prow}

    # ---------------- (d) sensitivity
    p25r = json.load(open(f"{P25}/prereg25_results.json"))
    sub = [m for m, k in p25keys.items() if cal(k)["both_same"]]
    sub_or = [m for m in sub if fcal(next(p for p in p25sel if p["mid"] == m)["form_id_a"])["both_better"]
              and fcal(next(p for p in p25sel if p["mid"] == m)["form_id_b"])["both_better"]]

    def prim(fam: str, mids: list) -> dict:
        pm = {r["mid"]: r for r in p25r["families"][fam]["per_metric"]}
        rs = [pm[m] for m in mids if m in pm and pm[m].get("delta_name") is not None and pm[m].get("delta_scaf") is not None]
        if not rs:
            return {"n": 0}
        dn, ds = [r["delta_name"] for r in rs], [r["delta_scaf"] for r in rs]
        out = {"n": len(rs), "mean_delta_name": round(float(np.mean(dn)), 4), "mean_delta_scaf": round(float(np.mean(ds)), 4),
               "median_delta_name": round(float(np.median(dn)), 4), "median_delta_scaf": round(float(np.median(ds)), 4),
               "n_delta_name_gt_delta_scaf": sum(a > b for a, b in zip(dn, ds))}
        if len(rs) >= 5:
            w = wilcoxon(dn, ds, alternative="greater")
            out["wilcoxon_one_sided"] = {"statistic": float(w.statistic), "p": round(float(w.pvalue), 5)}
        return out

    allm = [p["mid"] for p in p25sel]
    R["d_sensitivity_prereg25"] = {
        "subset_def": "PREREG-25 metrics whose selected pair is ALSO 'same' for both families under the calibrated standard",
        "n_subset": len(sub), "subset_mids": sub, "dropped_mids": [m for m in allm if m not in sub],
        "n_subset_also_both_better_calibrated": len(sub_or),
        "full_35": {f: prim(f, allm) for f in FAMS},
        "calibrated_same_subset": {f: prim(f, sub) for f in FAMS},
        "calibrated_same_and_orientation_subset": {f: prim(f, sub_or) for f in FAMS}}
    vs = [r for r in drows if r["calibrated_both_same"] and r["dauc"] is not None]
    vs_or = [r for r in vs if fcal(arms[r["mid"]]["low_form_id"])["both_better"]
             and fcal(arms[r["mid"]]["high_form_id"])["both_better"]]
    full = [r for r in drows if r["dauc"] is not None]
    R["d_sensitivity_prereg15v2"] = {
        "all_85": {"win_split": A15.win_split([r["dauc"] for r in full]), "dauc": A15.dist([r["dauc"] for r in full])},
        "calibrated_same": {"n": len(vs), "mids": [r["mid"] for r in vs],
                            "win_split": A15.win_split([r["dauc"] for r in vs]), "dauc": A15.dist([r["dauc"] for r in vs]),
                            "win_split_orientation_corrected": A15.win_split([r["dauc_oc"] for r in vs]),
                            "dauc_orientation_corrected": A15.dist([r["dauc_oc"] for r in vs])},
        "calibrated_same_excl_fewshot": {"n": len([r for r in vs if not r["fewshot_example"]]),
                                         "mids": [r["mid"] for r in vs if not r["fewshot_example"]],
                                         "win_split": A15.win_split([r["dauc"] for r in vs if not r["fewshot_example"]]),
                                         "dauc": A15.dist([r["dauc"] for r in vs if not r["fewshot_example"]])},
        "calibrated_same_and_both_better": {"n": len(vs_or), "mids": [r["mid"] for r in vs_or],
                                            "win_split": A15.win_split([r["dauc"] for r in vs_or]),
                                            "dauc": A15.dist([r["dauc"] for r in vs_or])},
        "calibrated_not_same": {"n": len(full) - len(vs),
                                "win_split": A15.win_split([r["dauc"] for r in full if not r["calibrated_both_same"]]),
                                "dauc": A15.dist([r["dauc"] for r in full if not r["calibrated_both_same"]])}}
    json.dump(R, open(f"{RV}/results.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: R[k] for k in ("n_pairs_judged", "n_p11_judged", "anchor_accuracy")}, indent=0))


if __name__ == "__main__":
    main()
