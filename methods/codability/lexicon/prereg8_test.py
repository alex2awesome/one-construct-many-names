#!/usr/bin/env python3
"""PREREG-8 (frozen 2026-07-22, registry): author-level class contrast.

Class per doc = Codex-coded AUTHOR type (author_institutionality_20260722.json, gates
passed .941/.968): INST = {institution, credentialed_individual}, LAY = {lay_individual},
unknown EXCLUDED. H (primary): within-class naming coincidence > cross-class at R1
construct grain, 7 widened fields; v2 machinery (same-doc pairs excluded, mirror-guarded),
doc-level author-class permutation x1000, Fisher over fields passing >=50/>=50 pair gates.
Census fields secondary, reported separately.

Secondary family (P7 retest, separate): variant-level author-INST share (>=3 docs) x
metaphoricity NEGATIVE, same 5-field universe as PREREG-7.
ONE run. Output: outputs/lexicon/prereg8_results_20260722.json
"""
import json
import random
from collections import defaultdict

import numpy as np
from scipy import stats

from methods.codability.lexicon.codability_sampling_model import LEX, norm_name
from methods.codability.lexicon.prereg_tests import (
    N_PERM, TASKS, WIDENED, coincidence_gap, field_data)
from methods.codability.lexicon.subfield_richness import load_chain

AUTH = json.load(open(f"{LEX}/author_institutionality_20260722.json"))


def doc_auth(task):
    out = {}
    for k, v in AUTH.items():
        t, d = k.split("/", 1)
        if t == task and v in ("institution", "credentialed_individual", "lay_individual"):
            out[d] = "I" if v != "lay_individual" else "N"
    return out


def run_block(tasks, rng, label):
    out = {}
    ps = []
    for task in tasks:
        rows, _ = field_data(task, "R1")
        cls = doc_auth(task)
        rows = [r for r in rows if r["doc"] in cls]
        obs, ds, dc = coincidence_gap(rows, cls)
        if obs is None or ds < 50 or dc < 50:
            out[task] = {"usable": False, "same_pairs": ds, "cross_pairs": dc}
            continue
        docs = sorted({r["doc"] for r in rows})
        labels = [cls[d] for d in docs]
        ge = valid = 0
        for _ in range(N_PERM):
            rng.shuffle(labels)
            g, _, _ = coincidence_gap(rows, dict(zip(docs, labels)))
            if g is None:
                continue
            valid += 1
            if g >= obs:
                ge += 1
        p = (1 + ge) / (1 + valid)
        n_lay = sum(1 for d in docs if cls[d] == "N")
        out[task] = {"usable": True, "gap": round(obs, 4), "same_pairs": ds,
                     "cross_pairs": dc, "n_docs": len(docs), "n_lay_docs": n_lay,
                     "p_perm": round(p, 4)}
        ps.append(p)
        print(label, task, out[task], flush=True)
    if ps:
        X = -2 * sum(np.log(p) for p in ps)
        out["_fisher"] = {"chi2": round(float(X), 2), "df": 2 * len(ps),
                          "p": round(float(1 - stats.chi2.cdf(X, 2 * len(ps))), 6),
                          "fields_used": len(ps)}
        print(label, "FISHER", out["_fisher"])
    return out


def literalness_retest():
    axes = {}
    for l in open(f"{LEX}/axis_metaphoricity_widened_20260722.jsonl"):
        r = json.loads(l)
        axes[r["variant"]] = r["score"]
    out = {}
    ps = []
    for task in WIDENED:
        cls = doc_auth(task)
        chain, _ = load_chain(task)
        uses = defaultdict(set)
        for l in open(f"{LEX}/extract_{task}_glm-4.7.jsonl"):
            r = json.loads(l)
            if r.get("status") != "ok" or str(r.get("key", "")).startswith("ANCHOR"):
                continue
            key = str(r["key"])
            if key not in chain:
                continue
            doc = key.split("::")[2] if key.count("::") >= 3 else key
            nm = norm_name(r.get("head_term"))
            if nm and doc in cls:
                uses[nm].add((doc, cls[doc]))
        pairs = []
        for v, ds in uses.items():
            if len(ds) >= 3 and v in axes:
                share = sum(1 for _, c in ds if c == "I") / len(ds)
                pairs.append((share, axes[v]))
        if len(pairs) < 20:
            out[task] = {"usable": False, "eligible": len(pairs)}
            continue
        r = stats.spearmanr([a for a, _ in pairs], [b for _, b in pairs])
        p1 = r.pvalue / 2 if r.statistic < 0 else 1 - r.pvalue / 2
        out[task] = {"usable": True, "n": len(pairs),
                     "rho_metaphoricity": round(float(r.statistic), 3),
                     "p1": round(float(p1), 5)}
        ps.append(p1)
        print("P8-literalness", task, out[task])
    if ps:
        X = -2 * sum(np.log(p) for p in ps)
        out["_fisher"] = {"chi2": round(float(X), 2), "df": 2 * len(ps),
                          "p": round(float(1 - stats.chi2.cdf(X, 2 * len(ps))), 6),
                          "fields_used": len(ps)}
        print("P8-literalness FISHER", out["_fisher"])
    return out


def main():
    res = {"primary_widened": run_block(WIDENED, random.Random(8), "P8-primary"),
           "secondary_census": run_block(TASKS, random.Random(88), "P8-census"),
           "secondary_literalness": literalness_retest()}
    path = f"{LEX}/prereg8_results_20260722.json"
    json.dump(res, open(path, "w"), indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
