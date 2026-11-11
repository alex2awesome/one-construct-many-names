#!/usr/bin/env python3
"""PREREG-30 selection (notes/2026-10-12__prereg30-humor-construct-level-name-swap.md). Mechanical."""
import hashlib, json, os
R = "ROOT/outputs/lexicon"
V2 = f"{R}/prereg15v2_20260828"
RA = f"{R}/prereg25_20261012/readjudication_v2"
OUT = f"{R}/prereg30_humor_20261012"
h = lambda s: hashlib.sha256(s.encode()).hexdigest()

pool = json.load(open(f"{V2}/v2_form_pool.json"))
rub = {p["form_id"]: p["rubric"] for p in pool}
form_mids = {p["form_id"]: set(p["metric_ids"]) for p in pool}
fm = json.load(open(f"{V2}/v2_form_formality.json"))["form_means"]
vp = json.load(open(f"{RA}/verdicts_pairs.json"))
vf = json.load(open(f"{RA}/verdicts_forms.json"))
fs = json.load(open(f"{RA}/fewshot_split.json"))
fewshot = {x["pair_key"] for x in fs["fewshot"] if x.get("pair_key")}
mids = sorted([f[len("v2_scoring_items_"):-5] for f in os.listdir(V2) if f.startswith("v2_scoring_items_")],
              key=lambda m: int(m[1:]))
p25 = {p["mid"]: p for p in json.load(open(f"{R}/prereg25_20261012/p25_pairs.json"))["pairs"]}


def better(fid, run):
    v = vf.get(fid, {}).get(run, {})
    return v.get("gpt") == "better" and v.get("claude") == "better"


def eligible_by_mid(drop_fewshot):
    out = {}
    for key, v in vp.items():
        if not v["calibrated"]["both_same"]:
            continue
        if drop_fewshot and key in fewshot:
            continue
        a, b = key.split("|")
        if a not in fm or b not in fm:
            continue
        cal_ok = better(a, "calibrated") and better(b, "calibrated")
        str_ok = better(a, "strict") and better(b, "strict")
        if not (cal_ok or str_ok):
            continue
        gap = abs(fm[a] - fm[b])
        if gap <= 0:
            continue
        for mid in v["mids"]:
            if mid not in mids or not (mid in form_mids.get(a, ()) and mid in form_mids.get(b, ())):
                continue
            lo, hi = (a, b) if fm[a] < fm[b] else (b, a)
            out.setdefault(mid, []).append(dict(mid=mid, form_id_a=lo, form_id_b=hi, name_a=rub[lo], name_b=rub[hi],
                                                formality_a=fm[lo], formality_b=fm[hi], gap=round(gap, 4),
                                                gap_sd=round(gap / 1.0425, 3), pair_key=key, fewshot=key in fewshot,
                                                orient_ok_calibrated=cal_ok, orient_ok_strict=str_ok,
                                                calibrated=v["calibrated"], strict=v["strict"]))
    sel = {}
    for mid, ps in out.items():
        ps.sort(key=lambda p: (-p["gap"], h(p["form_id_a"] + p["form_id_b"])))
        sel[mid] = dict(ps[0], n_eligible_pairs=len(ps))
    return sel


sel_ex = eligible_by_mid(True)
if len(sel_ex) >= 15:
    rule, sel = "few-shot excluded", sel_ex
else:
    rule, sel = "few-shot kept", eligible_by_mid(False)
sel_kept = eligible_by_mid(False)
qual = sorted(sel, key=lambda m: h(m + "p30"))
capped = qual[:20]
pairs = []
for m in capped:
    p = sel[m]
    q = p25.get(m)
    p["same_as_prereg25"] = bool(q and q["form_id_a"] == p["form_id_a"] and q["form_id_b"] == p["form_id_b"])
    p["prereg25_pair"] = ({k: q[k] for k in ("form_id_a", "form_id_b", "name_a", "name_b")} if q else None)
    p["sha_p30"] = h(m + "p30")
    pairs.append(p)
res = dict(design="notes/2026-10-12__prereg30-humor-construct-level-name-swap.md",
           rule_applied=rule, n_universe=len(mids), n_qualify_fewshot_excluded=len(sel_ex),
           n_qualify_fewshot_kept=len(sel_kept), n_qualify_under_rule=len(sel), cap=20, n_selected=len(pairs),
           qualifying_mids=sorted(sel, key=lambda m: int(m[1:])),
           dropped_by_cap=sorted(qual[20:], key=lambda m: int(m[1:])),
           n_same_as_prereg25=sum(p["same_as_prereg25"] for p in pairs), pairs=pairs)
os.makedirs(OUT, exist_ok=True)
json.dump(res, open(f"{OUT}/p30_pairs.json", "w"), indent=1)
print(rule, "excl:", len(sel_ex), "kept:", len(sel_kept), "selected:", len(pairs),
      "same as P25:", res["n_same_as_prereg25"])
for p in pairs:
    print(p["mid"], p["gap"], p["same_as_prereg25"], p["fewshot"], "|", p["name_a"][:50], "||", p["name_b"][:50])
