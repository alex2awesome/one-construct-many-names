#!/usr/bin/env python
"""PREREG-26 Steps 0-3 aggregation: formality means, adjudication verdicts (strict + calibrated), kappas,
mechanical selection (PRIMARY = calibrated standard, per Amendment 1; SECONDARY = strict), item sets, scoring jobs.

Inputs (rsynced from sk2/sk3): OUT/{claude,gpt56}/{formality,adj}/<job>.json, OUT/private/keys.json.
Outputs: OUT/formality.json, OUT/adjudication_summary.json, OUT/pair_verdicts.json, OUT/form_verdicts.json,
OUT/p26_pairs.json (primary, calibrated), OUT/p26_pairs_strict.json (secondary), OUT/items/*, OUT/jobs/score.json.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import defaultdict
from itertools import combinations

import numpy as np
from scipy.stats import spearmanr

ROOT = "ROOT"
SJ = f"{ROOT}/outputs/lexicon/silver_join"
OUT = f"{ROOT}/outputs/lexicon/prereg26_cw_20261012"
sys.path.insert(0, f"{ROOT}/methods/codability/lexicon")
from prereg26_build import S  # noqa: E402

ITEMS_HDR = "ITEMS (JSON: items = list of {item_idx, text}):\n"
OUT_FMT = ('Reply with exactly one JSON line per item: {"item_idx": <idx>, "score": <1-7>}, in the order given. '
           "40 lines total. Output nothing else.")
SLOTS = [("A", "S0", 0), ("B", "S0", 0), ("A", "S0", 1), ("B", "S0", 1), ("A", "S1", 0), ("A", "S2", 0)]


def h(s):
    return hashlib.sha256(s.encode()).hexdigest()


def kappa(a, b):
    cats = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else None


def load(fam, kind):
    d = f"{OUT}/{fam}/{kind}"
    res = {}
    if not os.path.isdir(d):
        return res
    for f in os.listdir(d):
        if f.endswith(".json"):
            res[f[:-5]] = json.load(open(f"{d}/{f}"))
    return res


def main():
    K = json.load(open(f"{OUT}/private/keys.json"))
    name_of = K["name_of"]
    summ = {}
    # ---- formality
    fr = load("claude", "formality")
    passes = defaultdict(dict)
    anchor_f = defaultdict(list)
    st = {"pass": 0, "failed": 0, "attempts": {}}
    for jid, r in fr.items():
        if r["status"] != "pass":
            st["failed"] += 1
            continue
        st["pass"] += 1
        st["attempts"][jid] = r["attempt"]
        k = int(jid[2])
        for iid, row in r["parsed"]["rows"].items():
            e = K["formality"][jid].get(iid)
            if e is None:
                continue
            if e["kind"] == "real":
                passes[k][e["form"]] = float(row["formality"])
            else:
                anchor_f[e["stratum"]].append(float(row["formality"]))
    forms = K["forms"]
    fmean = {f: float(np.mean([passes[k][f] for k in (1, 2) if f in passes[k]]))
             for f in forms if any(f in passes[k] for k in (1, 2))}
    both = [f for f in forms if f in passes[1] and f in passes[2]]
    sd_within = float(np.sqrt(np.mean([np.var([passes[1][f], passes[2][f]], ddof=1) for f in both]))) if both else None
    summ["formality"] = {"chunks": st, "n_forms_with_mean": len(fmean), "n_forms_both_passes": len(both),
                         "pooled_within_form_sd": sd_within,
                         "sd_of_form_means": float(np.std(list(fmean.values()), ddof=1)) if fmean else None,
                         "pass1_vs_pass2_spearman": float(spearmanr([passes[1][f] for f in both],
                                                                    [passes[2][f] for f in both]).statistic) if len(both) > 3 else None,
                         "anchor_mean_formality_by_stratum": {k: round(float(np.mean(v)), 3) for k, v in anchor_f.items()}}
    json.dump({"form_means": fmean, "passes": {str(k): v for k, v in passes.items()}, "summary": summ["formality"]},
              open(f"{OUT}/formality.json", "w"), indent=1)

    # ---- adjudication
    PV = defaultdict(dict)   # (std) -> pair -> fam -> verdict
    FV = defaultdict(dict)
    for fam in ("gpt56", "claude"):
        ar = load(fam, "adj")
        for jid, r in ar.items():
            std = "calibrated" if jid.startswith("v") else "strict"
            kb = K["adj_v2" if std == "calibrated" else "adj"][jid]
            summ.setdefault("adj_batches", {}).setdefault(f"{fam}/{std}", {"pass": 0, "failed": 0, "anchor_acc": []})
            s = summ["adj_batches"][f"{fam}/{std}"]
            if r["status"] != "pass":
                s["failed"] += 1
                continue
            s["pass"] += 1
            s["anchor_acc"].append(r["parsed"]["anchor_acc"])
            for pid, e in kb["pairs"].items():
                if e["kind"] == "real":
                    PV[std].setdefault("|".join(e["forms"]), {})[fam] = r["parsed"]["pairs"][pid]
            for fid, e in kb["forms"].items():
                if e["kind"] == "real":
                    FV[std].setdefault(e["form"], {})[fam] = r["parsed"]["forms"][fid]
    for s in summ.get("adj_batches", {}).values():
        s["mean_anchor_acc"] = round(float(np.mean(s["anchor_acc"])), 4) if s["anchor_acc"] else None
        del s["anchor_acc"]
    summ["adjudication"] = {}
    for std in ("calibrated", "strict"):
        pv, fv = PV[std], FV[std]
        bothp = [p for p in pv if len(pv[p]) == 2]
        bothf = [f for f in fv if len(fv[f]) == 2]
        dist = {fam: {c: sum(pv[p].get(fam) == c for p in pv) for c in ("same", "overlap", "different")}
                for fam in ("gpt56", "claude")}
        summ["adjudication"][std] = {
            "n_pairs_judged_both": len(bothp), "n_forms_judged_both": len(bothf), "pair_dist": dist,
            "n_both_same": sum(all(v == "same" for v in pv[p].values()) for p in bothp),
            "kappa_equivalence": kappa([pv[p]["gpt56"] for p in bothp], [pv[p]["claude"] for p in bothp]) if bothp else None,
            "kappa_orientation": kappa([fv[f]["gpt56"] for f in bothf], [fv[f]["claude"] for f in bothf]) if bothf else None,
            "orientation_dist": {fam: {c: sum(fv[f].get(fam) == c for f in fv) for c in ("better", "worse", "unclear")}
                                 for fam in ("gpt56", "claude")}}
    json.dump({std: PV[std] for std in PV}, open(f"{OUT}/pair_verdicts.json", "w"), indent=1)
    json.dump({std: FV[std] for std in FV}, open(f"{OUT}/form_verdicts.json", "w"), indent=1)

    # ---- selection
    bym = defaultdict(list)
    for p in K["pairs"]:
        bym[p.split(":")[0]].append(p)

    def select(std):
        pv, fv = PV[std], FV[std]
        sel, out_ = [], []
        for mid in sorted(bym, key=lambda m: int(m[1:])):
            elig = []
            for p in bym[mid]:
                fa, fb = p.split("|")
                v = pv.get(p, {})
                if not (len(v) == 2 and all(x == "same" for x in v.values())):
                    continue
                if not all(len(fv.get(f, {})) == 2 and all(x == "better" for x in fv[f].values()) for f in (fa, fb)):
                    continue
                if fa not in fmean or fb not in fmean:
                    continue
                elig.append((abs(fmean[fa] - fmean[fb]), h(fa + fb), fa, fb))
            if not elig:
                out_.append(mid)
                continue
            elig.sort(key=lambda t: (-t[0], t[1]))
            gap, _, fa, fb = elig[0]
            if fmean[fa] > fmean[fb] or (fmean[fa] == fmean[fb] and h(fa) > h(fb)):
                fa, fb = fb, fa
            sel.append({"mid": mid, "form_a": fa, "form_b": fb, "name_A": name_of[fa], "name_B": name_of[fb],
                        "formality_a": fmean[fa], "formality_b": fmean[fb], "gap": fmean[fb] - fmean[fa],
                        "n_eligible": len(elig),
                        "strict_verdicts": PV["strict"].get("|".join(sorted((fa, fb))), {}),
                        "calibrated_verdicts": PV["calibrated"].get("|".join(sorted((fa, fb))), {})})
        return sel, out_

    selp, outp = select("calibrated")
    sels, outs = select("strict")
    summ["selection"] = {"calibrated_primary": {"n_selected": len(selp), "constructs_out": outp,
                                                "gaps": [round(s["gap"], 3) for s in selp]},
                         "strict_secondary": {"n_selected": len(sels), "constructs_out": outs,
                                              "gaps": [round(s["gap"], 3) for s in sels]},
                         "primary_pairs_also_strict_same_both": sum(
                             len(s["strict_verdicts"]) == 2 and all(v == "same" for v in s["strict_verdicts"].values())
                             for s in selp)}
    json.dump(selp, open(f"{OUT}/p26_pairs.json", "w"), indent=1, ensure_ascii=False)
    json.dump(sels, open(f"{OUT}/p26_pairs_strict.json", "w"), indent=1, ensure_ascii=False)

    # ---- items
    texts = [json.loads(l) for l in open(f"{SJ}/cw_candidate_texts.jsonl")]
    ypos = json.load(open(f"{SJ}/variant_ypos_cw.json"))
    os.makedirs(f"{OUT}/items", exist_ok=True)
    keyf = lambda t: (h(t["post_id"] + "p26-cw"), h(t["unit_id"] + "p26-cw"))
    mids = sorted({s["mid"] for s in selp + sels}, key=lambda m: int(m[1:]))
    for mid in mids:
        pos = set(ypos.get(mid, []))
        P = sorted([t for t in texts if t["post_id"] in pos], key=keyf)[:20]
        N = sorted([t for t in texts if t["post_id"] not in pos], key=keyf)[:40 - len(P)]
        chosen = P + N
        chosen.sort(key=lambda t: h(t["unit_id"] + mid + "p26-order"))
        lab = [{"item_idx": i, "unit_id": t["unit_id"], "post_id": t["post_id"], "positive": int(t["post_id"] in pos)}
               for i, t in enumerate(chosen)]
        sc = [{"item_idx": i, "text": t["text"][:1000]} for i, t in enumerate(chosen)]
        json.dump({"mid": mid, "items": lab}, open(f"{OUT}/items/p26_items_{mid}.json", "w"), indent=1)
        json.dump({"items": sc}, open(f"{OUT}/items/p26_scoring_items_{mid}.json", "w"), indent=1, ensure_ascii=False)
    summ["items"] = {mid: sum(x["positive"] for x in json.load(open(f"{OUT}/items/p26_items_{mid}.json"))["items"])
                     for mid in mids}

    # ---- scoring jobs: primary (calibrated) metric-major, then strict-only extras (prefix s_)
    jobs = []

    def add(sel, prefix):
        for s in sel:
            items_txt = open(f"{OUT}/items/p26_scoring_items_{s['mid']}.json").read()
            for arm, scaf, k in SLOTS:
                name = s["name_A"] if arm == "A" else s["name_B"]
                prompt = S[scaf].format(name=name) + "\n\n" + ITEMS_HDR + items_txt + "\n\n" + OUT_FMT
                jobs.append({"job": f"{prefix}{s['mid']}_{arm}_{scaf}_r{k}", "prompt": prompt, "idxs": list(range(40)),
                             "name": name})
    add(selp, "")
    psel = {(s["mid"], s["form_a"], s["form_b"]) for s in selp}
    extra = [s for s in sels if (s["mid"], s["form_a"], s["form_b"]) not in psel]
    if os.environ.get("P26_PRIMARY_ONLY"):  # strict adjudication still running: queue primary runs only
        extra = []
    json.dump(jobs, open(f"{OUT}/jobs/score.json", "w"), indent=1, ensure_ascii=False)
    jobs = []
    add(extra, "s_")
    if extra:
        json.dump(jobs, open(f"{OUT}/jobs/score_strict.json", "w"), indent=1, ensure_ascii=False)
    summ["scoring_jobs"] = {"primary": 6 * len(selp), "strict_extra": 6 * len(extra),
                            "strict_reusing_primary_runs": len(sels) - len(extra)}
    json.dump(summ, open(f"{OUT}/adjudication_summary.json", "w"), indent=1)
    print(json.dumps(summ, indent=1)[:4000])


if __name__ == "__main__":
    main()
