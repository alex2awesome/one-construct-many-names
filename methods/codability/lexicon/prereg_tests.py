#!/usr/bin/env python3
"""SINGLE-RUN confirmatory tests PREREG-1 and PREREG-2 (registry: paper-plan note, incl.
2026-07-20 pre-data amendment). Run ONCE each; results append to
outputs/lexicon/prereg_results_20260721.json.

Frozen choices (written before execution):
- Records: load_records() named (head_term), norm_name'd, mirror_collapse'd; docs joined to
  provenance_rungs (last non-null rung per doc); docs w/o rung excluded.
- Classes: PRIMARY split I={1,2} vs N={3,4,5}; SENSITIVITY I={1,2,3} vs N={4,5}.
- PREREG-1 statistic (per field): pooled-ratio coincidence gap. Numerators/denominators
  pooled over concepts: same-class pairs P(name match) minus cross-class pairs P(name match).
  Null: permute doc->class within field (1,000 perms, class counts preserved); one-sided
  p=(1+#{perm>=obs})/1001. Combine fields: Fisher chi2(2k) over usable fields. Field usable
  iff >=50 same-class AND >=50 cross-class pairs.
- PREREG-2 statistic (per field): Spearman(inst_share, height) over variants with >=3 uses
  in field (post-filters) and judged height. Height = mean(z(formality), z(lat)) with
  lat={germanic:0, mixed:.5, latinate:1, greek:1}; z over all judged variants. Null: same
  doc->class permutation -> recompute inst_share -> Spearman; one-sided positive. Field
  usable iff >=20 eligible variants and nonconstant inst_share.
"""
import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict

import numpy as np
from scipy import stats

from methods.codability.lexicon.codability_sampling_model import (
    LEX, TASKS, load_records, mirror_collapse, norm_name)
from methods.codability.lexicon.subfield_richness import load_chain

WIDENED = ["notice-and-comment", "grant-funding", "peer-review",
           "legal-outcome-prediction", "patents", "press-releases", "code-review"]

N_PERM = 1000
SPLITS = {"primary_12v345": {1: "I", 2: "I", 3: "N", 4: "N", 5: "N"},
          "sensitivity_123v45": {1: "I", 2: "I", 3: "I", 4: "N", 5: "N"}}
LAT = {"germanic": 0.0, "mixed": 0.5, "latinate": 1.0, "greek": 1.0}


def doc_rungs(task):
    r = {}
    for line in open(f"{LEX}/provenance_rungs_{task}.jsonl"):
        d = json.loads(line)
        if d["rung"] is not None:
            r[d["id"]] = d["rung"]
    return r


def field_data(task, grain="census"):
    rungs = doc_rungs(task)
    if grain == "census":
        named = [r for r in load_records(task) if r["name"] and r["doc"] in rungs]
    else:                     # R1 construct grain (PREREG-4): key -> L0v4 -> R1 chain
        chain, _ = load_chain(task)
        named = []
        for line in open(f"{LEX}/extract_{task}_glm-4.7.jsonl"):
            r = json.loads(line)
            if r.get("status") != "ok" or str(r.get("key", "")).startswith("ANCHOR"):
                continue
            key = str(r["key"])
            if key not in chain:
                continue
            doc = key.split("::")[2] if key.count("::") >= 3 else key
            nm = norm_name(r.get("head_term"))
            if nm and doc in rungs:
                named.append({"key": key, "con": chain[key]["R1"], "doc": doc,
                              "name": nm, "quote": r.get("quote") or ""})
    # v2 correction (Codex audit): filter uncoded docs BEFORE mirror collapse so an
    # excluded doc cannot suppress a coded mirror.
    return mirror_collapse(named), rungs


def coincidence_gap(rows, doc_cls):
    """v2 correction (Codex audit): same-document pairs are excluded from the same-class
    numerator and denominator (a doc paired with itself is not an author pair)."""
    ns = ds = nc = dc = 0
    bycon = defaultdict(list)
    for r in rows:
        bycon[r["con"]].append((doc_cls[r["doc"]], r["name"], r["doc"]))
    for recs in bycon.values():
        cc = Counter(c for c, _, _ in recs)
        cn = Counter((c, n) for c, n, _ in recs)
        cd = Counter((c, d) for c, _, d in recs)
        cnd = Counter((c, n, d) for c, n, d in recs)
        nI, nN = cc.get("I", 0), cc.get("N", 0)
        ds += nI * (nI - 1) // 2 + nN * (nN - 1) // 2
        ds -= sum(k * (k - 1) // 2 for k in cd.values())
        dc += nI * nN
        ns += sum(k * (k - 1) // 2 for k in cn.values())
        ns -= sum(k * (k - 1) // 2 for k in cnd.values())
        for (c, n), k in cn.items():
            if c == "I":
                nc += k * cn.get(("N", n), 0)
    if ds <= 0 or dc <= 0:
        return None, ds, dc
    return ns / ds - nc / dc, ds, dc


def prereg5(cmap, rng, tasks, grain):
    """PREREG-5 adoption asymmetry: for concepts with >=2 uses in EACH class, dominant code
    per class = modal name WITHIN that class (cross-estimated); statistic = pooled
    P(informal record uses inst-dominant) - P(inst record uses informal-dominant)."""
    def stat(rows, doc_cls):
        num_i = den_i = num_n = den_n = 0
        bycon = defaultdict(list)
        for r in rows:
            bycon[r["con"]].append((doc_cls[r["doc"]], r["name"]))
        q = 0
        for recs in bycon.values():
            I = [n for c, n in recs if c == "I"]
            N = [n for c, n in recs if c == "N"]
            if len(I) < 2 or len(N) < 2:
                continue
            q += 1
            di = min(Counter(I).items(), key=lambda kv: (-kv[1], kv[0]))[0]
            dn = min(Counter(N).items(), key=lambda kv: (-kv[1], kv[0]))[0]
            num_n += sum(1 for n in N if n == di)
            den_n += len(N)
            num_i += sum(1 for n in I if n == dn)
            den_i += len(I)
        if den_i == 0 or den_n == 0:
            return None, q
        return num_n / den_n - num_i / den_i, q
    out = {}
    ps = []
    for task in tasks:
        rows, rungs = field_data(task, grain)
        doc_cls = {d: cmap[r] for d, r in rungs.items()}
        obs, q = stat(rows, doc_cls)
        if obs is None or q < 15:
            out[task] = {"usable": False, "qual_concepts": q}
            continue
        docs = sorted({r["doc"] for r in rows})
        labels = [doc_cls[d] for d in docs]
        ge = valid = 0
        for _ in range(N_PERM):
            rng.shuffle(labels)
            g, _ = stat(rows, dict(zip(docs, labels)))
            if g is None:
                continue
            valid += 1
            if g >= obs:
                ge += 1
        p = (1 + ge) / (1 + valid)
        out[task] = {"usable": True, "asymmetry": round(obs, 4), "qual_concepts": q,
                     "p_perm": round(p, 4)}
        ps.append(p)
    if ps:
        X = -2 * sum(np.log(p) for p in ps)
        out["_fisher"] = {"chi2": round(float(X), 2), "df": 2 * len(ps),
                          "p": round(float(1 - stats.chi2.cdf(X, 2 * len(ps))), 5),
                          "fields_used": len(ps)}
    return out


def prereg1(split_name, cmap, rng, tasks=TASKS, grain="census"):
    out = {}
    ps = []
    for task in tasks:
        rows, rungs = field_data(task, grain)
        doc_cls = {d: cmap[r] for d, r in rungs.items()}
        obs, ds, dc = coincidence_gap(rows, doc_cls)
        if obs is None or ds < 50 or dc < 50:
            out[task] = {"usable": False, "same_pairs": ds, "cross_pairs": dc}
            continue
        docs = sorted({r["doc"] for r in rows})
        labels = [doc_cls[d] for d in docs]
        ge = valid = 0
        for _ in range(N_PERM):
            rng.shuffle(labels)
            perm = dict(zip(docs, labels))
            g, _, _ = coincidence_gap(rows, perm)
            if g is None:
                continue
            valid += 1
            if g >= obs:
                ge += 1
        p = (1 + ge) / (1 + valid)
        out[task] = {"usable": True, "gap": round(obs, 4), "same_pairs": ds,
                     "cross_pairs": dc, "p_perm": round(p, 4)}
        ps.append(p)
    if ps:
        X = -2 * sum(np.log(p) for p in ps)
        out["_fisher"] = {"chi2": round(float(X), 2), "df": 2 * len(ps),
                          "p": round(float(1 - stats.chi2.cdf(X, 2 * len(ps))), 5),
                          "fields_used": len(ps)}
    return out


def heights():
    rows = [json.loads(l) for l in open(f"{LEX}/register_height_judgments.jsonl")]
    seen = {}
    for r in rows:
        if r["stratum"] in LAT and r.get("formality"):
            seen[r["variant"]] = (float(r["formality"]), LAT[r["stratum"]])
    f = np.array([v[0] for v in seen.values()])
    l = np.array([v[1] for v in seen.values()])
    zf = (f - f.mean()) / f.std()
    zl = (l - l.mean()) / l.std()
    return {v: float((a + b) / 2) for v, (a, b) in
            zip(seen, zip(zf, zl))}


def prereg2(split_name, cmap, rng):
    H = heights()
    out = {}
    ps = []
    for task in TASKS:
        rows, rungs = field_data(task)
        doc_cls = {d: cmap[r] for d, r in rungs.items()}
        uses = defaultdict(set)
        for r in rows:
            if r["name"] in H:
                uses[r["name"]].add(r["doc"])   # v2: UNIQUE docs, not record repeats
        eli = {v: sorted(ds) for v, ds in uses.items() if len(ds) >= 3}
        share = {v: sum(1 for d in ds if doc_cls[d] == "I") / len(ds) for v, ds in eli.items()}
        vs = sorted(eli)
        if len(vs) < 20 or len(set(share.values())) < 2:
            out[task] = {"usable": False, "n_variants": len(vs)}
            continue
        x = np.array([share[v] for v in vs])
        y = np.array([H[v] for v in vs])
        obs = stats.spearmanr(x, y).statistic
        # v2: permutation universe = docs contributing eligible uses (not the whole field)
        docs = sorted({d for ds in eli.values() for d in ds})
        labels = [doc_cls[d] for d in docs]
        ge = valid = 0
        for _ in range(N_PERM):
            rng.shuffle(labels)
            perm = dict(zip(docs, labels))
            xp = np.array([sum(1 for d in eli[v] if perm[d] == "I") / len(eli[v]) for v in vs])
            if len(set(xp)) < 2:
                continue
            valid += 1
            if stats.spearmanr(xp, y).statistic >= obs:
                ge += 1
        # v2: valid-permutation denominator (constant permutations no longer occupy slots)
        p = (1 + ge) / (1 + valid)
        out[task] = {"usable": True, "rho": round(float(obs), 4), "n_variants": len(vs),
                     "p_perm": round(p, 4)}
        ps.append(p)
    if ps:
        X = -2 * sum(np.log(p) for p in ps)
        out["_fisher"] = {"chi2": round(float(X), 2), "df": 2 * len(ps),
                          "p": round(float(1 - stats.chi2.cdf(X, 2 * len(ps))), 5),
                          "fields_used": len(ps)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grain", default="census", choices=["census", "R1"])
    ap.add_argument("--tasks", default="")
    ap.add_argument("--tests", default="p1,p2")
    a = ap.parse_args()
    tasks = a.tasks.split(",") if a.tasks else TASKS
    tests = set(a.tests.split(","))
    rng = random.Random(2026)
    res = {"_run": {"grain": a.grain, "tasks": tasks, "tests": sorted(tests)}}
    for split_name, cmap in SPLITS.items():
        res[split_name] = {}
        if "p1" in tests:
            print(f"=== PREREG-1/4 [{split_name}] grain={a.grain} ===")
            r1 = prereg1(split_name, cmap, rng, tasks, a.grain)
            for k, v in r1.items():
                print(" ", k, v)
            res[split_name]["prereg1"] = r1
        if "p2" in tests:
            print(f"=== PREREG-2 [{split_name}] ===")
            r2 = prereg2(split_name, cmap, rng)
            for k, v in r2.items():
                print(" ", k, v)
            res[split_name]["prereg2"] = r2
        if "p5" in tests:
            print(f"=== PREREG-5 [{split_name}] grain={a.grain} ===")
            r5 = prereg5(cmap, rng, tasks, a.grain)
            for k, v in r5.items():
                print(" ", k, v)
            res[split_name]["prereg5"] = r5
    # never overwrite an existing execution: version the output file
    tag = f"{a.grain}_{'-'.join(sorted(tests))}"
    base = f"{LEX}/prereg_results_20260721_v2_{tag}"
    path = f"{base}.json"
    k = 2
    while __import__("os").path.exists(path):
        k += 1
        path = f"{base}_run{k}.json"
    res["_meta"] = {"label": "v2_deviation_corrected (Codex audit 2026-07-21): same-doc pairs "
                             "excluded, mirror-after-filter, unique-doc inst_share, eligible-doc "
                             "perm universe, valid-perm denominators. Grain unchanged (census) — "
                             "grain change would be a new prereg, not a correction.",
                    "original": "prereg_results_20260721_ORIGINAL_frozen.json"}
    json.dump(res, open(path, "w"), indent=1)
    print("\nwrote", path)


if __name__ == "__main__":
    main()
