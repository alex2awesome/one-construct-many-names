#!/usr/bin/env python
"""PREREG-27 Steps 0-3 aggregation for domain argv[1] (math | news): formality means, verdicts under both standards,
kappas, calibrated selection (cap 20 qualifying constructs in construct-hash order), strict counts only, items,
scoring jobs (calibrated selection only). Mirrors prereg26_select.py. Writes the same files under OUT.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

ROOT = "ROOT"
sys.path.insert(0, f"{ROOT}/methods/codability/lexicon")
from prereg26_select import kappa, load as _load, ITEMS_HDR, OUT_FMT, SLOTS  # noqa: E402
import prereg26_select  # noqa: E402

DOM = sys.argv[1]
OUT = f"{ROOT}/outputs/lexicon/prereg27_{DOM}_20261012"
prereg26_select.OUT = OUT
CAP = 20


def h(s):
    return hashlib.sha256(s.encode()).hexdigest()


def text_pool():
    """-> list of {item_id, text, label, source}"""
    if os.path.exists(f"{OUT}/text_pool.jsonl"):
        return [json.loads(l) for l in open(f"{OUT}/text_pool.jsonl")]
    if DOM == "math":
        import pandas as pd
        src = "datasets/math/stackexchange/math_se_v3_position_matched.csv.gz"
        d = pd.read_csv(f"{ROOT}/{src}", usecols=["text", "judgement", "answer_id"])
        d = d[d.text.str.len().between(400, 1000)].drop_duplicates("answer_id")
        pool = []
        for lab in (1, 0):
            sub = d[d.judgement == lab].copy()
            sub["k"] = sub.answer_id.astype(str).map(lambda a: h(a + "p27-math"))
            for _, r in sub.sort_values("k").head(100).iterrows():
                pool.append({"item_id": str(r.answer_id), "text": r.text, "label": int(lab), "source": src})
    else:
        import pandas as pd
        src = "datasets/news-homepages/homepage_newsworthiness_clean_v9.csv.gz"
        d = pd.read_csv(f"{ROOT}/{src}")
        d = d[d.text.str.len().between(400, 1000)].copy()
        d["item_id"] = d.snapshot_id.astype(str) + "#" + d.text.map(lambda t: h(t)[:12])
        d = d.drop_duplicates("item_id")
        pool = []
        for lab in (1, 0):
            sub = d[d.judgement == lab].copy()
            sub["k"] = sub.item_id.map(lambda a: h(a + "p27-news"))
            for _, r in sub.sort_values("k").head(100).iterrows():
                pool.append({"item_id": r.item_id, "text": r.text, "label": int(lab), "source": src})
    with open(f"{OUT}/text_pool.jsonl", "w") as f:
        for p in pool:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    return pool


def main():
    K = json.load(open(f"{OUT}/private/keys.json"))
    name_of = K["name_of"]
    S = json.load(open(f"{OUT}/scaffolds.json"))
    summ = {}
    fr = _load("claude", "formality")
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
    summ["formality"] = {"chunks": st, "n_forms": len(forms), "n_forms_with_mean": len(fmean), "n_forms_both_passes": len(both),
                         "pooled_within_form_sd": float(np.sqrt(np.mean([np.var([passes[1][f], passes[2][f]], ddof=1) for f in both]))) if both else None,
                         "sd_of_form_means": float(np.std(list(fmean.values()), ddof=1)) if fmean else None,
                         "pass1_vs_pass2_spearman": float(spearmanr([passes[1][f] for f in both], [passes[2][f] for f in both]).statistic) if len(both) > 3 else None,
                         "anchor_mean_formality_by_stratum": {k: round(float(np.mean(v)), 3) for k, v in anchor_f.items()}}
    json.dump({"form_means": fmean, "summary": summ["formality"]}, open(f"{OUT}/formality.json", "w"), indent=1)

    PV, FV = defaultdict(dict), defaultdict(dict)
    summ["adj_batches"] = {}
    for fam in ("gpt56", "claude"):
        for jid, r in _load(fam, "adj").items():
            std = "calibrated" if jid.startswith("v") else "strict"
            kb = K["adj_v2" if std == "calibrated" else "adj"][jid]
            s = summ["adj_batches"].setdefault(f"{fam}/{std}", {"pass": 0, "failed": 0, "acc": []})
            if r["status"] != "pass":
                s["failed"] += 1
                continue
            s["pass"] += 1
            s["acc"].append(r["parsed"]["anchor_acc"])
            for pid, e in kb["pairs"].items():
                if e["kind"] == "real":
                    PV[std].setdefault("|".join(e["forms"]), {})[fam] = r["parsed"]["pairs"][pid]
            for fid, e in kb["forms"].items():
                if e["kind"] == "real":
                    FV[std].setdefault(e["form"], {})[fam] = r["parsed"]["forms"][fid]
    for s in summ["adj_batches"].values():
        s["mean_anchor_acc"] = round(float(np.mean(s.pop("acc") or [np.nan])), 4)
    summ["n_pairs"] = len(K["pairs"])
    summ["adjudication"] = {}
    for std in ("calibrated", "strict"):
        pv, fv = PV[std], FV[std]
        bp = [p for p in pv if len(pv[p]) == 2]
        bf = [f for f in fv if len(fv[f]) == 2]
        summ["adjudication"][std] = {
            "n_pairs_judged_both": len(bp), "n_forms_judged_both": len(bf),
            "same_by_family": {fam: sum(pv[p].get(fam) == "same" for p in pv) for fam in ("gpt56", "claude")},
            "n_both_same": sum(all(v == "same" for v in pv[p].values()) for p in bp),
            "kappa_equivalence": kappa([pv[p]["gpt56"] for p in bp], [pv[p]["claude"] for p in bp]) if bp else None,
            "kappa_orientation": kappa([fv[f]["gpt56"] for f in bf], [fv[f]["claude"] for f in bf]) if bf else None}
    json.dump(PV, open(f"{OUT}/pair_verdicts.json", "w"), indent=1)
    json.dump(FV, open(f"{OUT}/form_verdicts.json", "w"), indent=1)

    bym = defaultdict(list)
    for p in K["pairs"]:
        bym[p.split(":")[0]].append(p)
    sel, out_ = [], []
    pv, fv = PV["calibrated"], FV["calibrated"]
    for mid in K["construct_order"]:
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
        if len(sel) >= CAP:
            continue
        elig.sort(key=lambda t: (-t[0], t[1]))
        gap, _, fa, fb = elig[0]
        if fmean[fa] > fmean[fb] or (fmean[fa] == fmean[fb] and h(fa) > h(fb)):
            fa, fb = fb, fa
        sv = PV["strict"].get("|".join(sorted((fa, fb))), {})
        sel.append({"mid": mid, "form_a": fa, "form_b": fb, "name_A": name_of[fa], "name_B": name_of[fb],
                    "formality_a": fmean[fa], "formality_b": fmean[fb], "gap": fmean[fb] - fmean[fa],
                    "n_eligible": len(elig), "strict_verdicts": sv,
                    "calibrated_verdicts": pv.get("|".join(sorted((fa, fb))), {})})
    n_qual = sum(1 for m in K["construct_order"] if m not in out_)
    summ["selection"] = {"n_constructs_adjudicated": len(K["construct_order"]), "n_qualifying": n_qual,
                         "n_selected_cap20": len(sel), "constructs_without_eligible_pair": out_,
                         "gaps": [round(s["gap"], 3) for s in sel],
                         "selected_also_strict_same_both": sum(len(s["strict_verdicts"]) == 2 and all(
                             v == "same" for v in s["strict_verdicts"].values()) for s in sel)}
    json.dump(sel, open(f"{OUT}/p27_pairs.json", "w"), indent=1, ensure_ascii=False)

    pool = text_pool()
    os.makedirs(f"{OUT}/items", exist_ok=True)
    jobs = []
    for s in sel:
        mid = s["mid"]
        P = sorted([t for t in pool if t["label"] == 1], key=lambda t: h(t["item_id"] + mid + "p27"))[:20]
        N = sorted([t for t in pool if t["label"] == 0], key=lambda t: h(t["item_id"] + mid + "p27"))[:20]
        ch = sorted(P + N, key=lambda t: h(t["item_id"] + mid + "p27-order"))
        json.dump({"mid": mid, "items": [{"item_idx": i, "item_id": t["item_id"], "label": t["label"]} for i, t in enumerate(ch)]},
                  open(f"{OUT}/items/p27_items_{mid}.json", "w"), indent=1)
        json.dump({"items": [{"item_idx": i, "text": t["text"][:1000]} for i, t in enumerate(ch)]},
                  open(f"{OUT}/items/p27_scoring_items_{mid}.json", "w"), indent=1, ensure_ascii=False)
        items_txt = open(f"{OUT}/items/p27_scoring_items_{mid}.json").read()
        for arm, scaf, k in SLOTS:
            name = s["name_A"] if arm == "A" else s["name_B"]
            jobs.append({"job": f"{mid}_{arm}_{scaf}_r{k}", "name": name, "idxs": list(range(40)),
                         "prompt": S[scaf].format(name=name) + "\n\n" + ITEMS_HDR + items_txt + "\n\n" + OUT_FMT})
    summ["pool"] = {"n": len(pool), "source": pool[0]["source"] if pool else None,
                    "len_min_max": [min(len(t["text"]) for t in pool), max(len(t["text"]) for t in pool)] if pool else None}
    json.dump(jobs, open(f"{OUT}/jobs/score.json", "w"), indent=1, ensure_ascii=False)
    summ["scoring_jobs"] = len(jobs)
    json.dump(summ, open(f"{OUT}/adjudication_summary.json", "w"), indent=1)
    print(json.dumps(summ, indent=1)[:3500])


if __name__ == "__main__":
    main()
