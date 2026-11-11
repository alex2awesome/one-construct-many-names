#!/usr/bin/env python
"""Sonnet-4.5 vs Sonnet-5 arbiter validation on the FROZEN adjudicated eval.

Both panels judged the IDENTICAL blind payloads (arbiter_payloads/<task>_agent*.jsonl) under the
IDENTICAL protocol (scripts/judge_prompt.py). Only the model differs -> clean single covariate.

Rulers (never mixed):
  1. vs ADJUDICATED TRUTH (adjudicated_truth_<task>.json = Sonnet4.5-panel & GLM agree + Opus on
     disagreements). NOTE: on agreement pairs, S4.5 defined the truth -> S4.5 is favored there.
  2. HONEST HEAD-TO-HEAD = the Opus-decided DISAGREEMENT subset (S4.5-binary != GLM-binary). Truth
     there = Opus, independent of both S4.5 and S5. S4.5 acc on this subset == "Opus sided with
     Sonnet" rate; if S5 beats it, S5 is the better arbiter where it actually matters.
  3. Blinded anchors (arbiter_anchors_<task>.json, known 0/2 labels) — sanity, both must pass.

SAME <=> score==2 (the census/merge convention). Threshold-free AUC uses the raw 0/1/2 score as an
ordinal ranker vs the truth bool (per feedback_threshold_free_readouts — don't conflate calibration
with signal). Judgments are for VALIDATION only; they never become a census score.
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict
from typing import Dict, Optional

from .sources import ROOT

OUT = os.path.join(ROOT, "outputs", "lexicon")


def _load_votes(pattern: str) -> Dict[str, int]:
    """pair_id -> 0/1/2 score, concatenated over agent shards (last write wins)."""
    out: Dict[str, int] = {}
    for f in sorted(glob.glob(os.path.join(OUT, pattern))):
        for line in open(f):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            s = r.get("score")
            if isinstance(s, int) and s in (0, 1, 2) and r.get("pair_id"):
                out[r["pair_id"]] = s
    return out


def _load_glm(task: str) -> Dict[str, bool]:
    """pair_id -> GLM fresh_same bool, over all arbiter_glm52*.jsonl (task-tagged rows only)."""
    out: Dict[str, bool] = {}
    for f in sorted(glob.glob(os.path.join(OUT, "arbiter_glm52*.jsonl"))):
        for line in open(f):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("task") != task or r.get("fresh_same") is None:
                continue
            out[r["pair_id"]] = bool(r["fresh_same"])
    return out


def _auc(scores, labels) -> Optional[float]:
    try:
        from sklearn.metrics import roc_auc_score
        if len(set(labels)) < 2:
            return None
        return round(float(roc_auc_score(labels, scores)), 3)
    except Exception:
        return None


def _binstats(pred_same: Dict[str, bool], truth: Dict[str, bool], ids) -> dict:
    tp = fp = tn = fn = 0
    for pid in ids:
        p, t = pred_same[pid], truth[pid]
        if t and p:
            tp += 1
        elif t and not p:
            fn += 1
        elif (not t) and p:
            fp += 1
        else:
            tn += 1
    n = tp + fp + tn + fn
    acc = (tp + tn) / n if n else None
    prec = tp / (tp + fp) if (tp + fp) else None
    rec = tp / (tp + fn) if (tp + fn) else None
    tnr = tn / (tn + fp) if (tn + fp) else None
    f1 = (2 * prec * rec / (prec + rec)) if (prec and rec) else None
    bal = ((rec + tnr) / 2) if (rec is not None and tnr is not None) else None
    return {"n": n, "acc": round(acc, 3) if acc is not None else None,
            "precision": round(prec, 3) if prec is not None else None,
            "recall": round(rec, 3) if rec is not None else None,
            "bal_acc": round(bal, 3) if bal is not None else None,
            "f1": round(f1, 3) if f1 is not None else None,
            "n_same_truth": tp + fn, "n_same_pred": tp + fp}


def validate(task: str) -> dict:
    eval_rows = {r["pair_id"]: r for r in
                 (json.loads(l) for l in open(os.path.join(OUT, f"arbiter_eval_{task}.jsonl")))}
    truth_path = os.path.join(OUT, f"adjudicated_truth_{task}.json")
    if not os.path.exists(truth_path):
        return {"task": task, "error": "no adjudicated truth"}
    truth = {k: bool(v) for k, v in json.load(open(truth_path)).items()}
    anchors = {}
    ap = os.path.join(OUT, f"arbiter_anchors_{task}.json")
    if os.path.exists(ap):
        anchors = {k: int(v) for k, v in json.load(open(ap)).items()}

    s45 = _load_votes(f"arbiter_votes/sonnet_{task}_agent*.jsonl")
    s5 = _load_votes(f"arbiter_votes/sonnet5_{task}_agent*.jsonl")
    glm = _load_glm(task)

    res = {"task": task, "n_eval": len(eval_rows), "n_truth": len(truth),
           "coverage": {"s45": len(s45), "s5": len(s5), "glm": len(glm)}}

    # ---- ruler 1: vs adjudicated truth (both models judged) ----
    common = [pid for pid in truth if pid in s45 and pid in s5 and pid not in anchors]
    same45 = {pid: (s45[pid] == 2) for pid in s45}
    same5 = {pid: (s5[pid] == 2) for pid in s5}
    res["vs_adjudicated"] = {
        "n": len(common),
        "sonnet4.5": {**_binstats(same45, truth, common),
                      "auc": _auc([s45[p] for p in common], [truth[p] for p in common])},
        "sonnet5": {**_binstats(same5, truth, common),
                    "auc": _auc([s5[p] for p in common], [truth[p] for p in common])},
    }

    # ---- ruler 2: HONEST head-to-head — Opus-decided disagreement subset ----
    dis = [pid for pid in common if pid in glm and same45[pid] != glm[pid]]
    res["opus_disagreement_subset"] = {
        "n": len(dis),
        "note": "truth = Opus, independent of both Sonnets; S4.5 acc == Opus-sided-with-Sonnet rate",
        "sonnet4.5_acc": _binstats(same45, truth, dis)["acc"],
        "sonnet5_acc": _binstats(same5, truth, dis)["acc"],
        "sonnet5_agrees_glm_here": round(
            sum(same5[p] == glm[p] for p in dis) / len(dis), 3) if dis else None,
    }
    # agreement subset (truth = S4.5&GLM consensus; S4.5 correct by construction) — S5 replication
    agr = [pid for pid in common if pid in glm and same45[pid] == glm[pid]]
    res["consensus_subset"] = {"n": len(agr),
                               "sonnet5_acc": _binstats(same5, truth, agr)["acc"],
                               "note": "S4.5 == truth by construction; measures S5 replication of easy consensus"}

    # ---- ruler 3: blinded anchors ----
    def anchor_pass(votes):
        pos = [k for k, v in anchors.items() if v == 2 and k in votes]
        neg = [k for k, v in anchors.items() if v == 0 and k in votes]
        return {"n_pos": len(pos), "pos_pass": round(sum(votes[k] == 2 for k in pos) / len(pos), 3) if pos else None,
                "n_neg": len(neg), "neg_pass": round(sum(votes[k] != 2 for k in neg) / len(neg), 3) if neg else None}
    res["anchors"] = {"sonnet4.5": anchor_pass(s45), "sonnet5": anchor_pass(s5)}

    # ---- inter-model agreement ----
    both = [pid for pid in eval_rows if pid in s45 and pid in s5]
    agree_bin = sum(same45[p] == same5[p] for p in both) / len(both) if both else None
    # Cohen's kappa on binary SAME
    if both:
        po = agree_bin
        p45 = sum(same45[p] for p in both) / len(both)
        p5 = sum(same5[p] for p in both) / len(both)
        pe = p45 * p5 + (1 - p45) * (1 - p5)
        kappa = round((po - pe) / (1 - pe), 3) if pe < 1 else None
    else:
        kappa = None
    res["inter_model"] = {"n": len(both), "binary_agreement": round(agree_bin, 3) if agree_bin else None,
                          "kappa": kappa,
                          "s45_same_rate": round(sum(same45[p] for p in both) / len(both), 3) if both else None,
                          "s5_same_rate": round(sum(same5[p] for p in both) / len(both), 3) if both else None}

    # ---- by-stratum acc (both models) ----
    by = defaultdict(lambda: {"n": 0, "s45": 0, "s5": 0})
    for pid in common:
        st = eval_rows[pid]["stratum"]
        st = "spectrum" if st.startswith("spectrum") else st
        by[st]["n"] += 1
        by[st]["s45"] += int(same45[pid] == truth[pid])
        by[st]["s5"] += int(same5[pid] == truth[pid])
    res["by_stratum"] = {k: {"n": v["n"], "s45_acc": round(v["s45"] / v["n"], 3),
                             "s5_acc": round(v["s5"] / v["n"], 3)} for k, v in sorted(by.items())}
    return res


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="humor,creative-writing")
    a = ap.parse_args()
    allres = {}
    for t in a.tasks.split(","):
        t = t.strip()
        r = validate(t)
        allres[t] = r
        print(json.dumps(r, indent=1))
        print("=" * 80)
    json.dump(allres, open(os.path.join(OUT, "sonnet5_validation.json"), "w"), indent=1)
    print("wrote", os.path.join(OUT, "sonnet5_validation.json"))


if __name__ == "__main__":
    main()
