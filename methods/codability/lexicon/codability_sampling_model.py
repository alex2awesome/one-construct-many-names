#!/usr/bin/env python3
"""Codability as a SAMPLING problem: missing-mass decomposition + CRP/Pitman-Yor naming model
+ held-out prediction test. (The 'sampling angle' suite, user-approved 2026-07-20.)

CONFIRMATORY SUITE = E2 + E3 + E5 (E4 TASTE/CRAFT contrast intentionally EXCLUDED pending a
preregistered hypothesis and a validity review of the type codes — user challenge 2026-07-20).

Data: per-record author namings from extract_<task>_glm-4.7.jsonl (status=ok, head_term
present = the author's own head lexicalization), concept = partition_<task>.json (Jul-6
repaired concept grain; same grain as census_<task>.json).

E2  Three-way novelty decomposition (pooled Turing estimators over named records):
      P(next articulated+named criterion is a NOVEL CONCEPT)   = f1_concept / N
      P(... KNOWN concept, NOVEL name)                         = (f1_joint - f1_concept) / N
      P(... KNOWN concept, KNOWN name)                         = 1 - f1_joint / N
    (every concept-singleton token is a joint singleton, so the middle term is >= 0)
    Plus concept-level novelty over ALL ok records (the stream comparable to CRP-wave items).

E3  Generative naming model: per-concept naming partition ~ CRP (DP, concentration th) or
    Pitman-Yor (d, th), parameters shared per task, ML over the EPPF on concepts with >=2
    named records. Size-free codability = asymptotic P(two authors coincide):
      DP: 1/(1+th)     PY: (1-d)/(1+th)
    Bootstrap CI over concepts; posterior-predictive check on mean K and singleton fraction.
    Mirror-guard arm: within-concept records with quote token-Jaccard >= .5 collapsed first.

E5  Held-out prediction (fair in-population point test): stable-hash split by DOC (md5, 20%
    test), prequential scoring of each test record given its concept's train+seen-test state:
      model   P(new name) = (th + K d)/(th + N)   [PY predictive]
      plug-in P(new name) = concept's current singleton fraction f1/N
    Brier + log-loss, model vs plug-in; concept-level: realized new-concept rate in test vs
    Turing f1/N computed on train only.

Output: outputs/lexicon/codability_sampling_20260720.json. Seeded; conservative name
normalization; census caveats (junk sources, conditional-on-pool) inherit.
"""
import hashlib
import json
import re
from collections import Counter, defaultdict

import numpy as np
from scipy import optimize, special

ROOT = "ROOT"
LEX = f"{ROOT}/outputs/lexicon"
TASKS = ["humor", "creative-writing", "news-homepages", "math-stackexchange"]
SEED = 0
B_BOOT = 300

def norm_name(t):
    t = re.sub(r"[^a-z0-9 ]+", " ", (t or "").lower())
    return re.sub(r"\s+", " ", t).strip()

def toks(q):
    return set(re.findall(r"[a-z0-9]+", (q or "").lower()))

def load_records(task):
    part = {str(k): str(v) for k, v in json.load(open(f"{LEX}/partition_{task}.json")).items()}
    rows = []
    for ln in open(f"{LEX}/extract_{task}_glm-4.7.jsonl"):
        r = json.loads(ln)
        if r.get("status") != "ok" or str(r.get("key", "")).startswith("ANCHOR"):
            continue
        key = str(r["key"])
        con = part.get(key)
        if con is None:
            continue
        doc = key.split("::")[2] if key.count("::") >= 3 else key
        rows.append({"key": key, "con": con, "doc": doc,
                     "name": norm_name(r.get("head_term")) or None,
                     "quote": r.get("quote") or ""})
    return rows

def mirror_collapse(named):
    out = []
    by_con = defaultdict(list)
    for r in named:
        ts = toks(r["quote"])
        if any(len(ts & kept) / max(1, len(ts | kept)) >= .5 for kept in by_con[r["con"]]):
            continue
        by_con[r["con"]].append(ts)
        out.append(r)
    return out

# ---- E2 ---------------------------------------------------------------------------------------
def decompose(named):
    N = len(named)
    cc = Counter(r["con"] for r in named)
    jc = Counter((r["con"], r["name"]) for r in named)
    f1c = sum(1 for v in cc.values() if v == 1)
    f1j = sum(1 for v in jc.values() if v == 1)
    return {"N": N, "p_new_concept": round(f1c / N, 4),
            "p_known_concept_new_name": round((f1j - f1c) / N, 4),
            "p_known_concept_known_name": round(1 - f1j / N, 4)}

# ---- E3: histogram sufficient statistics for the EPPF -----------------------------------------
def partitions(named, min_n=2):
    g = defaultdict(list)
    for r in named:
        g[r["con"]].append(r["name"])
    return [tuple(sorted(Counter(v).values(), reverse=True)) for v in g.values() if len(v) >= min_n]

def hists(parts):
    hN, hK, hI, hS = Counter(), 0, Counter(), Counter()
    for p in parts:
        n, k = sum(p), len(p)
        hN[n] += 1
        hK += k
        for i in range(1, k):
            hI[i] += 1
        for s in p:
            hS[s] += 1
    Ns = np.array(sorted(hN)); cN = np.array([hN[x] for x in Ns], float)
    Is = np.array(sorted(hI)); cI = np.array([hI[x] for x in Is], float)
    Ss = np.array(sorted(hS)); cS = np.array([hS[x] for x in Ss], float)
    return Ns, cN, float(hK), Is, cI, Ss, cS

def dp_nll(log_th, H):
    Ns, cN, K, _, cI, _, _ = H
    th = np.exp(log_th)
    return -(K * log_th + (cN * (special.gammaln(th) - special.gammaln(th + Ns))).sum())

def py_nll(x, H):
    Ns, cN, _, Is, cI, Ss, cS = H
    d = 1 / (1 + np.exp(-x[0])) * 0.99
    th = np.exp(x[1]) - d + 1e-9
    ll = (cN * (special.gammaln(th + 1) - special.gammaln(th + Ns))).sum()
    ll += (cI * np.log(th + Is * d)).sum()
    ll += (cS * (special.gammaln(Ss - d) - special.gammaln(1 - d))).sum()
    return -ll

def fit_dp(H):
    r = optimize.minimize_scalar(dp_nll, bounds=(-4, 5), args=(H,), method="bounded")
    return float(np.exp(r.x)), float(-r.fun)

def fit_py(H):
    best = None
    for x0 in ([-1.0, 0.0], [0.0, 1.0], [-2.0, -0.5]):
        r = optimize.minimize(py_nll, x0, args=(H,), method="Nelder-Mead",
                              options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 400})
        if best is None or r.fun < best.fun:
            best = r
    d = 1 / (1 + np.exp(-best.x[0])) * 0.99
    th = float(np.exp(best.x[1]) - d + 1e-9)
    return float(d), th, float(-best.fun)

def crp_sim(n, d, th, rng):
    counts = []
    for _ in range(n):
        tot = sum(counts)
        if not counts or rng.random() < (th + len(counts) * d) / (th + tot):
            counts.append(1)
        else:
            w = np.array([c - d for c in counts], float)
            counts[rng.choice(len(counts), p=w / w.sum())] += 1
    return counts

def fit_group(parts, rng, boot=B_BOOT):
    H = hists(parts)
    th_dp, ll_dp = fit_dp(H)
    d, th, ll_py = fit_py(H)
    cois = []
    idx = np.arange(len(parts))
    for _ in range(boot):
        bs = [parts[i] for i in rng.choice(idx, size=len(idx), replace=True)]
        try:
            db, thb, _ = fit_py(hists(bs))
            cois.append((1 - db) / (1 + thb))
        except Exception:
            pass
    lo, hi = (np.percentile(cois, [2.5, 97.5]) if cois else (np.nan, np.nan))
    simK, simf1, obsK, obsf1 = [], [], [], []
    for p in parts:
        obsK.append(len(p)); obsf1.append(sum(1 for v in p if v == 1) / len(p))
        sp = crp_sim(sum(p), d, th, rng)
        simK.append(len(sp)); simf1.append(sum(1 for v in sp if v == 1) / len(sp))
    return {"n_concepts": len(parts), "dp_theta": round(th_dp, 3),
            "py_d": round(d, 3), "py_theta": round(th, 3),
            "lrt_py_vs_dp": round(2 * (ll_py - ll_dp), 1),
            "coincidence": round((1 - d) / (1 + th), 4),
            "coincidence_ci": [round(float(lo), 4), round(float(hi), 4)],
            "ppc_meanK_obs": round(float(np.mean(obsK)), 2), "ppc_meanK_sim": round(float(np.mean(simK)), 2),
            "ppc_f1frac_obs": round(float(np.mean(obsf1)), 3), "ppc_f1frac_sim": round(float(np.mean(simf1)), 3)}

# ---- E5 ---------------------------------------------------------------------------------------
def heldout_name(named, d, th):
    tr, te = [], []
    for r in named:
        (te if int(hashlib.md5(r["doc"].encode()).hexdigest(), 16) % 5 == 0 else tr).append(r)
    state = defaultdict(Counter)
    for r in tr:
        state[r["con"]][r["name"]] += 1
    bm, bp, lm, lp, ys, pms = [], [], [], [], [], []
    for r in te:
        c = state[r["con"]]
        N, K = sum(c.values()), len(c)
        if N == 0:
            continue
        y = 1 if c.get(r["name"], 0) == 0 else 0
        pm = float(np.clip((th + K * d) / (th + N), 1e-6, 1 - 1e-6))
        f1 = sum(1 for v in c.values() if v == 1)
        pp = float(np.clip(f1 / N, 1e-6, 1 - 1e-6))
        ys.append(y); pms.append(pm)
        bm.append((pm - y) ** 2); bp.append((pp - y) ** 2)
        lm.append(-(y * np.log(pm) + (1 - y) * np.log(1 - pm)))
        lp.append(-(y * np.log(pp) + (1 - y) * np.log(1 - pp)))
        state[r["con"]][r["name"]] += 1
    n = len(ys)
    if not n:
        return {"n_test_scored": 0}
    return {"n_test_scored": n, "actual_new_rate": round(float(np.mean(ys)), 3),
            "model_pred_rate": round(float(np.mean(pms)), 3),
            "brier_model": round(float(np.mean(bm)), 4), "brier_plugin": round(float(np.mean(bp)), 4),
            "logloss_model": round(float(np.mean(lm)), 4), "logloss_plugin": round(float(np.mean(lp)), 4)}

def heldout_concept(rows):
    tr = [r for r in rows if int(hashlib.md5(r["doc"].encode()).hexdigest(), 16) % 5 != 0]
    te = [r for r in rows if int(hashlib.md5(r["doc"].encode()).hexdigest(), 16) % 5 == 0]
    seen = set(r["con"] for r in tr)
    y = [1 if r["con"] not in seen else 0 for r in te]
    cc = Counter(r["con"] for r in tr)
    return {"n_test": len(te), "actual_new_concept_rate": round(float(np.mean(y)), 4),
            "turing_pred_from_train": round(sum(1 for v in cc.values() if v == 1) / len(tr), 4)}

def main():
    rng = np.random.default_rng(SEED)
    out = {}
    for task in TASKS:
        rows = load_records(task)
        named = [r for r in rows if r["name"]]
        named_mc = mirror_collapse(named)
        res = {"n_records": len(rows), "n_named": len(named), "n_named_mirrorfree": len(named_mc),
               "decomp_named": decompose(named), "decomp_named_mirrorfree": decompose(named_mc)}
        cc = Counter(r["con"] for r in rows)
        res["p_new_concept_allrecords"] = round(sum(1 for v in cc.values() if v == 1) / len(rows), 4)
        res["fit"] = fit_group(partitions(named), rng)
        res["fit_mirrorfree"] = fit_group(partitions(named_mc), rng, boot=150)
        f = res["fit"]
        res["heldout_name"] = heldout_name(named, f["py_d"], f["py_theta"])
        res["heldout_concept"] = heldout_concept(rows)
        out[task] = res
        dn, h, hc = res["decomp_named"], res["heldout_name"], res["heldout_concept"]
        print(f"{task:22} named={len(named):5} | new-con {dn['p_new_concept']:.3f} "
              f"new-name {dn['p_known_concept_new_name']:.3f} reuse {dn['p_known_concept_known_name']:.3f} | "
              f"PY d={f['py_d']:.2f} th={f['py_theta']:.2f} coincide={f['coincidence']:.3f} "
              f"[{f['coincidence_ci'][0]:.3f},{f['coincidence_ci'][1]:.3f}] LRT={f['lrt_py_vs_dp']:.0f} | "
              f"PPC K {f['ppc_meanK_obs']}/{f['ppc_meanK_sim']} f1 {f['ppc_f1frac_obs']}/{f['ppc_f1frac_sim']}",
              flush=True)
        print(f"{'':22} heldout-name: actual {h.get('actual_new_rate')} pred {h.get('model_pred_rate')} "
              f"brier M/P {h.get('brier_model')}/{h.get('brier_plugin')} "
              f"logloss M/P {h.get('logloss_model')}/{h.get('logloss_plugin')} (n={h.get('n_test_scored')}) | "
              f"heldout-concept: actual {hc['actual_new_concept_rate']} turing {hc['turing_pred_from_train']}",
              flush=True)
    json.dump(out, open(f"{LEX}/codability_sampling_20260720.json", "w"), indent=1)
    print("\nwrote", f"{LEX}/codability_sampling_20260720.json", flush=True)

if __name__ == "__main__":
    main()
