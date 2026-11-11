#!/usr/bin/env python
"""Step-1 finisher for the 9 remaining tasks' L0 repair (after the screen workflow wuujje94h).

Flow: screen (Sonnet, done by workflow) -> [confirm-build] -> Opus confirm (workflow) -> [apply].
  confirm-build : per task, filter Sonnet-SAME (screen==2), emit Opus confirm payloads (canonicals
                  pulled from the CANDIDATES, not the eval map — the 2026-07-07 bug), + ride anchors.
                  Also reports screen coverage so missing/failed shards can be re-run.
  apply         : per task, two-family verified (screen==2 AND Opus confirm==2) -> >=2-edge star ->
                  partition_<task>_L0v2.json + score vs adjudicated truth (before/after).

Idempotent. Usage:
  python -m methods.codability.lexicon.harvest_screen9 confirm-build
  python -m methods.codability.lexicon.harvest_screen9 apply
"""
import glob
import json
import os
import random
import sys

from methods.codability.lexicon import repair

OUT = repair.OUT
PD = os.path.join(OUT, "repair_payloads")
VD = os.path.join(OUT, "repair_votes")
NINE = ["code-review", "peer-review", "math-stackexchange", "press-releases", "news-homepages",
        "grant-funding", "legal-outcome-prediction", "notice-and-comment", "patents"]


def _cands(task):
    return json.load(open(os.path.join(OUT, f"repair_candidates_{task}.json")))[:2500]


def _anchor_rows(task):
    ev = {r["pair_id"]: r for r in (json.loads(l) for l in open(os.path.join(OUT, f"arbiter_eval_{task}.jsonl")))}
    ap = os.path.join(OUT, f"arbiter_anchors_{task}.json")
    anch = json.load(open(ap)) if os.path.exists(ap) else {}
    return [{"pair_id": p, "task": task, "canonical_a": ev[p]["canonical_a"],
             "canonical_b": ev[p]["canonical_b"]} for p in anch if p in ev]


def confirm_build():
    for t in NINE:
        cand = _cands(t)
        by_pid = {c["pair_id"]: c for c in cand}
        screen = repair._load_scored(f"repair_votes/screen_{t}_*.jsonl")
        cov = sum(1 for p in by_pid if p in screen)
        same = [p for p, s in screen.items() if s == 2 and p in by_pid]
        # emit Opus confirm payloads (canonicals from candidates)
        for f in glob.glob(os.path.join(PD, f"confirm_{t}_*.jsonl")):
            os.remove(f)
        anch = _anchor_rows(t)
        rng = random.Random(0)
        n = 0
        for a in range(0, len(same), 130):
            rows = [{"pair_id": by_pid[p]["pair_id"], "task": t,
                     "canonical_a": by_pid[p]["canonical_a"], "canonical_b": by_pid[p]["canonical_b"]}
                    for p in same[a:a + 130]]
            rows = rows + rng.sample(anch, min(6, len(anch)))
            rng.shuffle(rows)
            with open(os.path.join(PD, f"confirm_{t}_{n:03d}.jsonl"), "w") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
            n += 1
        miss = 2500 - cov
        print(f"{t:26s} screen_cov={cov}/2500{' MISSING '+str(miss) if miss>50 else ''} "
              f"Sonnet-SAME={len(same)} -> {n} Opus confirm shards")


def apply():
    print(f"{'task':26s} {'edges':>5} {'merges':>6}  recall(before->after)  precision(before->after)")
    for t in NINE:
        cand = _cands(t)
        base = repair.load_base_partition(t)
        edges = repair.ingest_verified(t, cand, f"repair_votes/screen_{t}_*.jsonl",
                                       confirm_glob=f"repair_votes/confirm_{t}_*.jsonl", screen_min=2)
        before = repair.score_vs_truth(t, base)
        res = repair.apply_merges(base, edges, min_edges=2, task=t)
        after = repair.score_vs_truth(t, res["partition"])
        json.dump(res["partition"], open(os.path.join(OUT, f"partition_{t}_L0v2.json"), "w"))
        print(f"{t:26s} {len(edges):5d} {res['n_merges']:6d}  "
              f"{before['recall']}->{after['recall']}    {before['precision']}->{after['precision']}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "confirm-build"
    {"confirm-build": confirm_build, "apply": apply}[cmd]()
