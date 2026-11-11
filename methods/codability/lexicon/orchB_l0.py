#!/usr/bin/env python
"""Orchestrator-B L0 real-repair finisher — scoped ONLY to press-releases / grant-funding /
legal-outcome-prediction / patents. Additive sibling of harvest_screen9.py (which hardcodes the
other 9-task NINE list and would touch Orchestrator A's confirm/apply files if called directly).
repair.py / harvest_screen9.py are NOT modified; this module only calls their pure/read functions
and writes exclusively to <this task>-namespaced files.

Differences from harvest_screen9.py (deliberate, per the runbook naming convention):
  (a) task-scoped to MY_TASKS, never the global NINE.
  (b) apply() writes partition_<task>_L0v3.json (L0v3 = "the frozen real-repaired L0" naming used
      by the runbook), NOT L0v2 — partition_<task>_L0v2.json is left untouched as the GLM-placeholder
      historical record. build_level.nodes_from_level already prefers L0v3 when present.

Usage:
  PYTHONPATH=. python -m methods.codability.lexicon.orchB_l0 screen-status
  PYTHONPATH=. python -m methods.codability.lexicon.orchB_l0 confirm-build
  PYTHONPATH=. python -m methods.codability.lexicon.orchB_l0 apply
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
MY_TASKS = ["press-releases", "grant-funding", "legal-outcome-prediction", "patents"]


def _cands(task):
    return json.load(open(os.path.join(OUT, f"repair_candidates_{task}.json")))[:2500]


def _anchor_rows(task):
    ev = {r["pair_id"]: r for r in (json.loads(l) for l in open(os.path.join(OUT, f"arbiter_eval_{task}.jsonl")))}
    ap = os.path.join(OUT, f"arbiter_anchors_{task}.json")
    anch = json.load(open(ap)) if os.path.exists(ap) else {}
    return [{"pair_id": p, "task": task, "canonical_a": ev[p]["canonical_a"],
             "canonical_b": ev[p]["canonical_b"]} for p in anch if p in ev]


def screen_status(tasks=MY_TASKS):
    for t in tasks:
        cand = _cands(t)
        by_pid = {c["pair_id"]: c for c in cand}
        screen = repair._load_scored(f"repair_votes/screen_{t}_*.jsonl")
        cov = sum(1 for p in by_pid if p in screen)
        same = sum(1 for p, s in screen.items() if s == 2 and p in by_pid)
        related = sum(1 for p, s in screen.items() if s >= 1 and p in by_pid)
        # per-shard line-count check against the payload manifest
        shard_report = []
        for f in sorted(glob.glob(os.path.join(PD, f"{t}_screen*.jsonl"))):
            base = os.path.basename(f)
            vf = os.path.join(VD, "screen_" + t + "_" + base.split("screen")[-1])
            n_in = sum(1 for _ in open(f))
            n_out = sum(1 for _ in open(vf)) if os.path.exists(vf) else 0
            shard_report.append(f"{base}:{n_in}->{n_out}")
        print(f"{t:26s} screen_cov={cov}/{len(by_pid)}  score>=1={related}  score==2={same}")
        print("   " + " ".join(shard_report))


def confirm_build(tasks=MY_TASKS):
    for t in tasks:
        cand = _cands(t)
        by_pid = {c["pair_id"]: c for c in cand}
        screen = repair._load_scored(f"repair_votes/screen_{t}_*.jsonl")
        cov = sum(1 for p in by_pid if p in screen)
        same = [p for p, s in screen.items() if s == 2 and p in by_pid]
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


def apply(tasks=MY_TASKS):
    print(f"{'task':26s} {'edges':>5} {'merges':>6}  recall(before->after)  precision(before->after)  collapse")
    rows = []
    for t in tasks:
        cand = _cands(t)
        base = repair.load_base_partition(t)
        edges = repair.ingest_verified(t, cand, f"repair_votes/screen_{t}_*.jsonl",
                                       confirm_glob=f"repair_votes/confirm_{t}_*.jsonl", screen_min=2)
        before = repair.score_vs_truth(t, base)
        res = repair.apply_merges(base, edges, min_edges=2, task=t)
        after = repair.score_vs_truth(t, res["partition"])
        json.dump(res["partition"], open(os.path.join(OUT, f"partition_{t}_L0v3.json"), "w"))
        collapse = f"{res['n_clusters_before']}->{res['n_clusters_after']}"
        print(f"{t:26s} {len(edges):5d} {res['n_merges']:6d}  "
              f"{before['recall']}->{after['recall']}    {before['precision']}->{after['precision']}  {collapse}")
        rows.append({"task": t, "n_edges": len(edges), "n_merges": res["n_merges"],
                     "before": before, "after": after, "collapse": collapse})
    json.dump(rows, open(os.path.join(OUT, "orchB_l0_apply_report.json"), "w"), indent=1)
    return rows


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "screen-status"
    {"screen-status": screen_status, "confirm-build": confirm_build, "apply": apply}[cmd]()
