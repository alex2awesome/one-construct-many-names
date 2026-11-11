#!/usr/bin/env python
"""PREREG-23 scorer: our frozen level prompts vs external human-coded gold links.

Headline readout is AUC (threshold-free, per the standing rule) of the 0/1/2 prompt score
against the gold binary. P/R at the prompt's own cut (score==2 -> "same") is reported
alongside but is cut-dependent and is never the headline.

Reported SEPARATELY for hard and easy negatives. Pooled AUC is composition-dependent: it
moves when the hard share moves, so the hard-negative AUC is the number that describes the
instrument. A big easy-vs-hard gap means the prompt is a topic detector, not a level detector.

Anchor gate: >=5/6 anchors correct (known-SAME scored 2, known-DIFFERENT scored 0 or 1),
else the batch is DISCARDED and reported as discarded, never silently dropped.
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

import numpy as np

ROOT = "ROOT"
OUT = f"{ROOT}/outputs/lexicon/prereg23"
B = f"{OUT}/batches"
GATE = 5


def auc(scores, gold):
    """Rank AUC with ties handled by mid-rank (Mann-Whitney U / (n_pos*n_neg))."""
    s, g = np.asarray(scores, float), np.asarray(gold, int)
    pos, neg = s[g == 1], s[g == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), float)
    cat = np.concatenate([pos, neg])[order]
    i = 0
    while i < len(cat):
        j = i
        while j + 1 < len(cat) and cat[j + 1] == cat[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def boot_ci(scores, gold, n=2000, seed=7):
    rng = np.random.default_rng(seed)
    s, g = np.asarray(scores, float), np.asarray(gold, int)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(s), len(s))
        a = auc(s[i], g[i])
        if a == a:
            vals.append(a)
    return (float(np.quantile(vals, .025)), float(np.quantile(vals, .975))) if vals else (float("nan"),) * 2


def score_all():
    manifest = json.load(open(f"{OUT}/batch_manifest.json"))
    pairs = {}
    for fn in glob.glob(f"{OUT}/pairs_*.jsonl"):
        for l in open(fn):
            r = json.loads(l)
            pairs[r["pair_id"]] = r

    # LEGACY ID SHIM. The dedupe fix started sorting the two node ids before hashing, which
    # changed pair_id for every pair that had been drawn in b->a order. Votes collected before
    # that fix are keyed by the old id but are judgements of the SAME two texts, so they stay
    # valid. Map each pair's pre-sort id onto its current row rather than re-judging.
    import hashlib
    for r in list(pairs.values()):
        legacy = hashlib.sha1(
            f"{r['corpus']}|{r['rung']}|{r['a_id']}|{r['b_id']}".encode()).hexdigest()[:16]
        pairs.setdefault(legacy, r)

    cells, gates = defaultdict(list), []
    seen_pid = defaultdict(set)
    for m in manifest:
        vp = m["votes"]
        if not os.path.exists(vp):
            continue
        votes = {}
        for l in open(vp):
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                continue
            if isinstance(r.get("score"), (int, float)):
                votes[r["pair_id"]] = float(r["score"])
        anchors = json.load(open(f"{B}/anchors_{m['tag']}.json"))
        ok = sum(1 for pid, exp in anchors.items()
                 if pid in votes and ((exp == 2 and votes[pid] == 2) or (exp == 0 and votes[pid] < 2)))
        passed = ok >= GATE
        gates.append({"tag": m["tag"], "anchors_ok": ok, "n_anchors": len(anchors),
                      "passed": passed, "n_votes": len(votes),
                      "coverage": round(len(votes) / m["n"], 3)})
        if not passed:
            continue
        # one vote per pair per cell: batches were re-emitted after the dedupe fix, so a
        # pair_id can appear in an old and a new batch file. First vote wins; never both.
        for pid, sc in votes.items():
            if pid in pairs and pid not in seen_pid[(m["corpus"], m["rung"])]:
                seen_pid[(m["corpus"], m["rung"])].add(pid)
                cells[(m["corpus"], m["rung"])].append((sc, pairs[pid]["gold"], pairs[pid]["stratum"]))

    rows = []
    for (corpus, rung), rec in sorted(cells.items()):
        s = [x[0] for x in rec]
        g = [x[1] for x in rec]
        st = [x[2] for x in rec]
        lo, hi = boot_ci(s, g)
        r = {"corpus": corpus, "rung": rung, "n": len(rec), "auc": auc(s, g), "ci": [lo, hi]}
        for tag, keep in (("hard", "neg_hard"), ("easy", "neg_easy")):
            sub = [(a, b) for a, b, c in rec if c in ("pos", keep)]
            r[f"auc_{tag}"] = auc([x[0] for x in sub], [x[1] for x in sub]) if sub else float("nan")
            r[f"n_{tag}"] = sum(1 for c in st if c == keep)
        yhat = [1 if x >= 2 else 0 for x in s]
        tp = sum(1 for a, b in zip(yhat, g) if a and b)
        r["precision"] = tp / max(sum(yhat), 1)
        r["recall"] = tp / max(sum(g), 1)
        r["mean_pos"] = float(np.mean([a for a, b in zip(s, g) if b]))
        r["mean_neg"] = float(np.mean([a for a, b in zip(s, g) if not b]))
        rows.append(r)

    json.dump({"cells": rows, "gates": gates}, open(f"{OUT}/arm_a_results.json", "w"), indent=1)
    print(f"{'corpus':<12}{'rung':<5}{'n':>5}{'AUC':>7}{'95% CI':>16}{'AUChard':>9}{'AUCeasy':>9}"
          f"{'prec':>7}{'rec':>7}")
    for r in rows:
        ci = f"[{r['ci'][0]:.2f},{r['ci'][1]:.2f}]"
        print(f"{r['corpus']:<12}{r['rung']:<5}{r['n']:>5}{r['auc']:>7.3f}{ci:>16}"
              f"{r['auc_hard']:>9.3f}{r['auc_easy']:>9.3f}{r['precision']:>7.2f}{r['recall']:>7.2f}")
    bad = [g for g in gates if not g["passed"]]
    if bad:
        print("\nDISCARDED (anchor gate):", ", ".join(f"{g['tag']} {g['anchors_ok']}/{g['n_anchors']}" for g in bad))
    thin = [g for g in gates if g["passed"] and g["coverage"] < .95]
    if thin:
        print("LOW COVERAGE:", ", ".join(f"{g['tag']} {g['coverage']:.2f}" for g in thin))
    return rows, gates


if __name__ == "__main__":
    score_all()
