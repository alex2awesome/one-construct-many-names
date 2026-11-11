#!/usr/bin/env python3
"""PREREG-6 (frozen 2026-07-22, plan note): subfield-conditioned within-class coincidence.

H: within-class naming coincidence > cross-class among SAME-SUBFIELD author pairs, R1
construct grain, binary split {1,2}v{3,4,5}, 7 widened fields.

Implementation notes (all frozen):
- doc -> subfield: normalized `subtask_short` from contexts, mapped through the W2b judged
  union-find (SAME edges only, subfield_merges_20260720.jsonl final state).
- pair restriction = group records by (R1 construct, subfield cluster): pairs across
  subfields never enter numerator or denominator. Same-doc pairs excluded, mirror-guarded
  (inherited from the v2 coincidence machinery).
- docs with no subfield label, or whose cluster has <2 docs in the field's record set, are
  EXCLUDED from the pair universe.
- null: 1,000 permutations of class labels shuffled WITHIN subfield-cluster strata.
- Fisher over usable fields (>=50 same-class and >=50 cross-class eligible pairs).
- secondary descriptive: share of unconditioned PREREG-4 gap explained by composition.
ONE run. Output: outputs/lexicon/prereg6_results_20260722.json
"""
import ast
import json
import random
from collections import defaultdict

import numpy as np
from scipy import stats

from methods.codability.lexicon.codability_sampling_model import LEX
from methods.codability.lexicon.prereg_tests import (
    N_PERM, SPLITS, WIDENED, coincidence_gap, field_data)
from methods.codability.lexicon.subfield_cluster_candidates import norm_label


def subfield_uf():
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for line in open(f"{LEX}/subfield_merges_20260720.jsonl"):
        r = json.loads(line)
        if r["same"] == 1:
            a, b = (r["task"], r["a"]), (r["task"], r["b"])
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
    return find


def doc_subfields(task, find):
    out = {}
    for line in open(f"{LEX}/contexts_{task}.jsonl"):
        c = json.loads(line)
        s = c.get("strata")
        if isinstance(s, str):
            try:
                s = ast.literal_eval(s)
            except Exception:
                s = {}
        lab = norm_label((s or {}).get("subtask_short", ""))
        if lab and c["doc"] not in out:
            out[c["doc"]] = find((task, lab))
    return out


def main():
    cmap = SPLITS["primary_12v345"]
    find = subfield_uf()
    rng = random.Random(6)
    out = {}
    ps = []
    for task in WIDENED:
        rows, rungs = field_data(task, "R1")
        sf = doc_subfields(task, find)
        rows = [r for r in rows if r["doc"] in sf]
        # drop docs in singleton clusters (no possible same-subfield partner)
        cl_docs = defaultdict(set)
        for r in rows:
            cl_docs[sf[r["doc"]]].add(r["doc"])
        rows = [r for r in rows if len(cl_docs[sf[r["doc"]]]) >= 2]
        doc_cls = {d: cmap[rungs[d]] for d in {r["doc"] for r in rows}}
        # conditioning: concept x subfield as the grouping key
        crows = [{"con": (r["con"], sf[r["doc"]]), "doc": r["doc"], "name": r["name"]}
                 for r in rows]
        obs, ds, dc = coincidence_gap(crows, doc_cls)
        # unconditioned gap on the same doc universe (secondary descriptive)
        u_obs, _, _ = coincidence_gap(
            [{"con": r["con"], "doc": r["doc"], "name": r["name"]} for r in rows], doc_cls)
        if obs is None or ds < 50 or dc < 50:
            out[task] = {"usable": False, "same_pairs": ds, "cross_pairs": dc}
            continue
        strata = defaultdict(list)
        for d in doc_cls:
            strata[sf[d]].append(d)
        ge = valid = 0
        for _ in range(N_PERM):
            perm = {}
            for docs in strata.values():
                labs = [doc_cls[d] for d in docs]
                rng.shuffle(labs)
                perm.update(dict(zip(docs, labs)))
            g, _, _ = coincidence_gap(crows, perm)
            if g is None:
                continue
            valid += 1
            if g >= obs:
                ge += 1
        p = (1 + ge) / (1 + valid)
        out[task] = {"usable": True, "gap_conditioned": round(obs, 4),
                     "gap_unconditioned_same_universe": round(u_obs, 4) if u_obs is not None else None,
                     "same_pairs": ds, "cross_pairs": dc,
                     "n_docs": len(doc_cls), "n_strata": len(strata),
                     "p_perm": round(p, 4)}
        ps.append(p)
        print(task, out[task], flush=True)
    if ps:
        X = -2 * sum(np.log(p) for p in ps)
        out["_fisher"] = {"chi2": round(float(X), 2), "df": 2 * len(ps),
                          "p": round(float(1 - stats.chi2.cdf(X, 2 * len(ps))), 6),
                          "fields_used": len(ps)}
        print("FISHER", out["_fisher"])
    path = f"{LEX}/prereg6_results_20260722.json"
    json.dump(out, open(path, "w"), indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
