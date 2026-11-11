#!/usr/bin/env python
"""Evaluate the frozen 70B preregistrations against the landed 70B reader pass.

The two freeze artifacts (2026-07-05) predate any code that reads them back — this is that
code path. Inputs are the per-domain grid auc_report.json files after the Llama-3.3-70B-FP8
reader pass (reader tag = its snapshot-dir basename). Everything on the AUC scale, floor .55,
mirroring name_sufficiency.py's definitions exactly (measurable over ARTIC_RUNGS; name-sufficient
= AUC_name >= floor AND deficit <= eps; prereg uses eps = 0).

  prediction 1 (persistence): every metric name-sufficient at 8B stays so at 70B.
  prediction 2 (ordinal):     70B flips among the pending set form a PREFIX of the frozen
                              ascending-8B-deficit ranking; readout = rank-AUC + prefix
                              violations + permutation p.
  scaling-law prereg:         per class-x-tag cell, observed mean 70B AUC vs pred_70B_AUC /
                              CI_G (G = 2*(AUC-.5)); REJECTION RULE: > half of cells outside
                              their 95% CI => parametric law rejected, ordinal claims only.

CONTAMINATION SPLIT: cw + humor 70B grids were scored 2026-07-02, BEFORE the 2026-07-05 freeze
(the freeze text "before any 70B reader pass" is wrong for those two domains). All headline
numbers are therefore reported for the 7 clean domains; cw/humor shown separately.

Usage: python -m methods.codability.eval_prereg_70b [--tag70 fde04ee76a27704c88f569542ef023b57d4d0362]
Writes <DATA>/prereg_70b_evaluation.json.
"""
import argparse
import hashlib
import json
import os

import numpy as np

from methods.codability.name_sufficiency import (DATA, DOMAINS, ARTIC_RUNGS, FLOOR,
                                                 load_tags)

CONTAMINATED = {"cw", "humor"}      # 70B pass predates the freeze in these domains
READER_8B = "Llama-3.1-8B-Instruct"


def canonical_payload_sha256(payload):
    body = {k: v for k, v in payload.items() if k != "sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def verify_frozen_hash(payload, label):
    expected = payload.get("sha256")
    observed = canonical_payload_sha256(payload)
    if not expected or observed != expected:
        raise ValueError(f"{label} hash mismatch: embedded={expected!r}, recomputed={observed}")
    return {"embedded": expected, "recomputed": observed, "verified": True}


def load_auc_rows(tag70):
    """(domain, gi) -> {'8B': {rung: auc}, '70B': {rung: auc}} from auc_report.json files."""
    rows = {}
    for short, (gdir, _aliases, dclass) in DOMAINS.items():
        p = os.path.join(DATA, gdir, "auc_report.json")
        if not os.path.exists(p):
            continue
        rep = json.load(open(p))
        for size, tag in [("8B", READER_8B), ("70B", tag70)]:
            for gi_s, rungs in rep.get(tag, {}).items():
                r = rows.setdefault((short, int(gi_s)), {"domain_class": dclass})
                r[size] = {rung: v.get("auc") for rung, v in rungs.items()}
    return rows


def measurable(sizes, size, floor=FLOOR):
    vals = [v for r, v in sizes.get(size, {}).items() if r in ARTIC_RUNGS and v is not None]
    return bool(vals) and max(vals) >= floor


def name_sufficient(sizes, size, eps=0.0, floor=FLOOR):
    if not measurable(sizes, size, floor):
        return None
    a_name = sizes.get(size, {}).get("name")
    a_def = sizes.get(size, {}).get("definition")
    if a_name is None or a_def is None:
        return None
    return bool(a_name >= floor and (a_def - a_name) <= eps)


def rank_auc(ranks, labels):
    """AUC that LOW rank (small 8B deficit) predicts flip=True."""
    pos = [rank for rank, label in zip(ranks, labels) if label]
    neg = [rank for rank, label in zip(ranks, labels) if not label]
    if not pos or not neg:
        return None
    wins = sum((p < n) + 0.5 * (p == n) for p in pos for n in neg)
    return float(wins / (len(pos) * len(neg)))


def prefix_violations(labels):
    """# of (non-flip before flip) pairs — 0 iff flips are a perfect prefix."""
    seen_nonflip, v = 0, 0
    for label in labels:
        if label:
            v += seen_nonflip
        else:
            seen_nonflip += 1
    return v


def eval_ordinal(prereg, rows):
    out = {}
    for scope, domsel in [("clean7", lambda d: d not in CONTAMINATED),
                          ("cw_humor_pre_freeze", lambda d: d in CONTAMINATED)]:
        keep = [e for e in prereg["name_sufficient_8B"] if domsel(e["domain"])]
        res = {"persist": 0, "lost": 0, "unmeasurable_70B": 0, "no_70B_data": 0, "lost_list": []}
        for e in keep:
            sizes = rows.get((e["domain"], e["gi"]))
            if not sizes or "70B" not in sizes:
                res["no_70B_data"] += 1
                continue
            ns = name_sufficient(sizes, "70B")
            if ns is None:
                res["unmeasurable_70B"] += 1
            elif ns:
                res["persist"] += 1
            else:
                res["lost"] += 1
                res["lost_list"].append({**e, "auc_name_70B": sizes["70B"].get("name"),
                                         "deficit_70B": (None if sizes["70B"].get("definition") is None
                                                         or sizes["70B"].get("name") is None else
                                                         round(sizes["70B"]["definition"]
                                                               - sizes["70B"]["name"], 4))})
        n_eval = res["persist"] + res["lost"]
        res["persistence_rate"] = round(res["persist"] / n_eval, 4) if n_eval else None
        res["persistence_prediction_verdict"] = (
            "FALSIFIED" if res["lost"] else
            "not yet fully evaluated" if res["unmeasurable_70B"] or res["no_70B_data"] else
            "supported")

        pend = [e for e in prereg["ranking_pending"] if domsel(e["domain"])]
        ranks, labels = [], []
        for i, e in enumerate(pend):
            sizes = rows.get((e["domain"], e["gi"]))
            if not sizes or "70B" not in sizes:
                continue
            ns = name_sufficient(sizes, "70B")
            if ns is None:
                continue
            ranks.append(i)
            labels.append(bool(ns))
        res["pending_evaluated"] = len(labels)
        res["pending_flipped_at_70B"] = int(sum(labels))
        res["rank_auc"] = (lambda v: round(v, 4) if v is not None else None)(
            rank_auc(ranks, labels))
        res["prefix_violations"] = prefix_violations(labels)
        if labels and any(labels) and not all(labels):
            rng = np.random.default_rng(0)
            arr = np.array(labels)
            obs = rank_auc(ranks, labels)
            null = [rank_auc(ranks, list(rng.permutation(arr))) for _ in range(2000)]
            exceed = sum(n >= obs for n in null)
            res["rank_auc_perm_p"] = round(float((exceed + 1) / (len(null) + 1)), 4)
        res["prefix_prediction_verdict"] = (
            "FALSIFIED" if res["prefix_violations"] else
            "not yet fully evaluated" if len(labels) < len(pend) else "supported")
        out[scope] = res
    return out


def eval_law(law, rows, tags, *, clean_only=True):
    cells = {}
    for (short, gi), sizes in rows.items():
        if "70B" not in sizes:
            continue
        tag = None
        for al in DOMAINS[short][1]:
            tag = tag or tags.get((al, gi))
        if not tag:
            continue
        for rung in ("name", "definition"):
            a = sizes["70B"].get(rung)
            if a is None:
                continue
            key = f"{sizes['domain_class']}|{tag}::{rung}"
            cells.setdefault(key, {"aucs": [], "clean": []})
            cells[key]["aucs"].append(a)
            cells[key]["clean"].append(short not in CONTAMINATED)
    res, outside, n_eval, contaminated_only = {}, 0, 0, 0
    for key, pred in law["predictions"].get("class_x_tag", {}).items():
        got = cells.get(key)
        if not got:
            res[key] = {"observed": None}
            continue
        clean_aucs = [a for a, c in zip(got["aucs"], got["clean"]) if c]
        if clean_only and not clean_aucs:
            contaminated_only += 1
            res[key] = {"observed": None, "reason": "only pre-freeze-contaminated domains"}
            continue
        use = clean_aucs if clean_only else got["aucs"]
        obs_auc = float(np.mean(use))
        obs_g = 2 * (obs_auc - 0.5)
        lo, hi = pred["CI_G"]
        n_eval += 1
        out_ci = not (lo <= obs_g <= hi)
        outside += out_ci
        res[key] = {"observed_auc_70B": round(obs_auc, 4), "observed_G": round(obs_g, 4),
                    "pred_70B_AUC": pred["pred_70B_AUC"], "CI_G": pred["CI_G"],
                    "n_metric_rungs": len(use), "clean_only": clean_only,
                    "outside_CI": bool(out_ci)}
    verdict = ("REJECTED (ordinal claims only)" if n_eval and outside > n_eval / 2
               else "not rejected" if n_eval else "no evaluable cells")
    return {"cells": res, "n_evaluated": n_eval, "n_outside_CI": outside,
            "n_contaminated_only_excluded": contaminated_only,
            "scope": "clean7" if clean_only else "all9_descriptive",
            "rejection_rule": law.get("evaluation_protocol"), "verdict": verdict}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag70", default="fde04ee76a27704c88f569542ef023b57d4d0362")
    ap.add_argument("--out", default=os.path.join(DATA, "prereg_70b_evaluation.json"))
    a = ap.parse_args()

    prereg = json.load(open(os.path.join(DATA, "prereg_70b_name_sufficiency.json")))
    law = json.load(open(os.path.join(DATA, "prereg_scaling_law_70b.json")))
    hash_verification = {"name_sufficiency": verify_frozen_hash(prereg, "name preregistration"),
                         "scaling_law": verify_frozen_hash(law, "law preregistration")}
    rows = load_auc_rows(a.tag70)
    n70 = sum(1 for v in rows.values() if "70B" in v)
    print(f"rows: {len(rows)} (with 70B data: {n70})")

    ordinal = eval_ordinal(prereg, rows)
    law_clean = eval_law(law, rows, load_tags(), clean_only=True)
    out = {"prereg_sha256": prereg["sha256"], "law_sha256": law["sha256"],
           "hash_verification": hash_verification,
           "tag70": a.tag70, "floor": FLOOR, "scale": "auc",
           "contamination_note": "cw+humor 70B grids scored 2026-07-02, before the "
                                 "2026-07-05 freeze; excluded from clean7 headline",
           "ordinal": ordinal,
           "frozen_claim_status": {
               "persistence": ordinal["clean7"]["persistence_prediction_verdict"],
               "prefix_order": ordinal["clean7"]["prefix_prediction_verdict"],
               "parametric_law": law_clean["verdict"],
           },
           "parametric_law": law_clean,
           "parametric_law_all9_descriptive": eval_law(
               law, rows, load_tags(), clean_only=False)}
    path = a.out
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    json.dump(out, open(path, "w"), indent=1)
    o = out["ordinal"]["clean7"]
    print(f"clean7 persistence: {o['persist']}/{o['persist'] + o['lost']} "
          f"(unmeasurable {o['unmeasurable_70B']}, missing {o['no_70B_data']})")
    print(f"clean7 pending flips: {o['pending_flipped_at_70B']}/{o['pending_evaluated']}, "
          f"rank AUC {o['rank_auc']}, prefix violations {o['prefix_violations']}")
    print(f"parametric law: {out['parametric_law']['verdict']} "
          f"({out['parametric_law']['n_outside_CI']}/{out['parametric_law']['n_evaluated']} "
          f"cells outside CI)")
    print(f"-> {path}")


if __name__ == "__main__":
    main()
