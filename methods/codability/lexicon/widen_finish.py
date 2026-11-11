#!/usr/bin/env python
"""Widen-band L0 finisher — ADDITIVE ONLY, does not modify repair.py or harvest_screen9.py
(coordinator guardrail: those two are live in the humor/CW L0v3 lane concurrently).

Extends the top-2500 harvest_screen9 flow to the FULL repair_candidates_<task>.json
(widen1/widen2 bands, ranks 2500-15000), reusing repair.py's pure functions
(ingest_verified/apply_merges/score_vs_truth) UNCHANGED with a wider candidate slice.
Applies strictly on top of the REAL partition_<task>_L0v2.json base (never mutated)
into a NEW partition_<task>_L0v3.json. Confirm delta is computed by pair_id so
already-confirmed pairs (confirm_<task>_NNN.jsonl, the original top-2500 pass) are
never re-judged; new confirm payloads land at confirm_<task>_widenNNN.jsonl (a
distinct filename that still matches repair.ingest_verified's confirm_{t}_*.jsonl glob).

Usage:
  python -m methods.codability.lexicon.widen_finish screen-resume --task code-review --cap 2500
  python -m methods.codability.lexicon.widen_finish confirm-build --task math-stackexchange
  python -m methods.codability.lexicon.widen_finish apply --task math-stackexchange
"""
import glob
import hashlib
import json
import os
import random
import re
import sys

from methods.codability.lexicon import repair

OUT = repair.OUT
PD = os.path.join(OUT, "repair_payloads")
L0_CONFIRM_PROTOCOL = os.path.join(OUT, "CONFIRM_PROTOCOL_L0_V2.txt")


def _full_cands(task, cap=None):
    """Ranked candidate list — NOT the [:2500] slice harvest_screen9._cands() uses.
    repair_candidates_<task>.json already contains the widen1 (rank 2500-8000) and
    widen2 (rank 8000-15000) bands; no rebuild needed. cap=None -> full list (default,
    unchanged behavior); cap=8000 -> top2500+widen1 only, EXCLUDES widen2 (coordinator
    2026-07-10: widen2's deep band has low measured payoff, don't block freezes on it)."""
    cand = json.load(open(os.path.join(OUT, f"repair_candidates_{task}.json")))
    # Frozen eval/QC anchors may have entered old candidate files before build/eval separation was
    # enforced.  They can ride along to test a judge, but must never become build edges.  Filter
    # first, then apply the cap so the requested band still contains `cap` genuine candidates.
    anchor_path = os.path.join(OUT, f"arbiter_anchors_{task}.json")
    anchor_ids = set(json.load(open(anchor_path))) if os.path.exists(anchor_path) else set()
    cand = [row for row in cand if row.get("pair_id") not in anchor_ids]
    return cand[:cap] if cap else cand


def _anchor_rows(task):
    ev = {r["pair_id"]: r for r in (json.loads(l) for l in open(os.path.join(OUT, f"arbiter_eval_{task}.jsonl")))}
    ap = os.path.join(OUT, f"arbiter_anchors_{task}.json")
    anch = json.load(open(ap)) if os.path.exists(ap) else {}
    return [{"pair_id": p, "task": task, "canonical_a": ev[p]["canonical_a"],
            "canonical_b": ev[p]["canonical_b"]} for p in anch if p in ev]


def screen_resume(task, cap=2500, per_agent=250):
    """Emit only candidate pairs that still lack a valid LLM screen vote.

    This is bookkeeping: it neither scores pairs nor infers similarity.  Ride-along anchors are
    included for judge QC but do not count toward candidate coverage.
    """
    cap = 2500 if cap is None else cap
    cand = _full_cands(task, cap)
    existing = repair._load_scored(f"repair_votes/screen_{task}_*.jsonl")
    missing = [r for r in cand if r["pair_id"] not in existing]
    # Never reuse an earlier resume filename: replacing its vote file would silently erase the
    # valid judgments that supplied earlier candidate coverage.  Generation 1 retains the legacy
    # ``resume000`` name; later generations use ``resume2_000``, ``resume3_000``, ...
    existing_votes = glob.glob(os.path.join(OUT, "repair_votes",
                                            f"screen_{task}_resume*.jsonl"))
    generations = [1]
    for path in existing_votes:
        m = re.search(r"_resume(\d+)_", os.path.basename(path))
        if m:
            generations.append(int(m.group(1)))
    generation = max(generations) + 1 if existing_votes else 1
    prefix = f"screen_{task}_resume" if generation == 1 else f"screen_{task}_resume{generation}_"
    for f in glob.glob(os.path.join(PD, f"{prefix}*.jsonl")):
        os.remove(f)
    anchors = _anchor_rows(task)
    rng = random.Random(0)
    n = 0
    for lo in range(0, len(missing), per_agent):
        rows = [{"pair_id": r["pair_id"], "task": task,
                 "canonical_a": r["canonical_a"], "canonical_b": r["canonical_b"]}
                for r in missing[lo:lo + per_agent]]
        rows += rng.sample(anchors, min(8, len(anchors)))
        rng.shuffle(rows)
        path = os.path.join(PD, f"{prefix}{n:03d}.jsonl")
        with open(path, "w") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        n += 1
    print(f"{task}: screen coverage={len(cand) - len(missing)}/{len(cand)}; "
          f"missing={len(missing)} -> {n} resume shards")
    return n


def confirm_build(task, cap=None):
    """Delta-only strict L0 confirm payloads: screen-advanced (score>=1, ANY screen shard incl.
    widen1/widen2) pairs that do NOT already have a confirm vote (by pair_id, dedup
    against the original top-2500 confirm_<task>_NNN.jsonl AND any prior widen round).
    cap restricts the candidate band (see _full_cands)."""
    cand = _full_cands(task, cap)
    by_pid = {c["pair_id"]: c for c in cand}
    screen = repair._load_scored(f"repair_votes/screen_{task}_*.jsonl")
    existing_confirm = repair._load_scored(f"repair_votes/confirm_{task}_*.jsonl")
    advanced = [p for p, s in screen.items() if s >= 1 and p in by_pid]
    delta = [p for p in advanced if p not in existing_confirm]
    for f in glob.glob(os.path.join(PD, f"confirm_{task}_widen*.jsonl")):
        os.remove(f)
    anch = _anchor_rows(task)
    rng = random.Random(0)
    n = 0
    for a in range(0, len(delta), 130):
        rows = [{"pair_id": by_pid[p]["pair_id"], "task": task,
                 "canonical_a": by_pid[p]["canonical_a"], "canonical_b": by_pid[p]["canonical_b"]}
                for p in delta[a:a + 130]]
        rows = rows + rng.sample(anch, min(6, len(anch)))
        rng.shuffle(rows)
        with open(os.path.join(PD, f"confirm_{task}_widen{n:03d}.jsonl"), "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        n += 1
    # Freeze the exact semantic boundary. Historical L0 runs predate this manifest and used the
    # broader CONFIRM_PROTOCOL_R1.txt; never silently reuse that R1 same-construct prompt for new L0
    # confirmation work.
    with open(L0_CONFIRM_PROTOCOL, "rb") as fh:
        protocol_sha = hashlib.sha256(fh.read()).hexdigest()
    with open(os.path.join(PD, f"confirm_{task}_widen_manifest.json"), "w") as fh:
        json.dump({"task": task, "relation": "L0 same criterion",
                   "protocol_path": os.path.relpath(L0_CONFIRM_PROTOCOL, repair.ROOT),
                   "protocol_sha256": protocol_sha, "screen_min": 1,
                   "confirm_accept_score": 2, "n_delta": len(delta), "n_shards": n}, fh, indent=1)
    print(f"{task}: screen-advanced>=1(all bands)={len(advanced)} "
          f"already-confirmed={len(advanced) - len(delta)} "
          f"DELTA={len(delta)} -> {n} new strict-L0 confirm shards "
          f"(protocol=CONFIRM_PROTOCOL_L0_V2.txt)")
    return n


def apply(task, cap=None):
    """Apply verified edges (top-2500 + widen1 [+ widen2 if cap allows]) onto the REAL
    L0v2 base. Writes partition_<task>_L0v3.json (new file). partition_<task>_L0v2.json
    is only READ here, never opened for writing. cap=8000 excludes widen2 entirely
    (candidate ids beyond the cap are absent from `cand`, so ingest_verified's
    `pid in by_pid` filter drops them even if a confirm vote exists for one)."""
    cand = _full_cands(task, cap)
    base_path = os.path.join(OUT, f"partition_{task}_L0v2.json")
    base = {k: str(v) for k, v in json.load(open(base_path)).items()}
    edges = repair.ingest_verified(task, cand, f"repair_votes/screen_{task}_*.jsonl",
                                   confirm_glob=f"repair_votes/confirm_{task}_*.jsonl", screen_min=1)
    before = repair.score_vs_truth(task, base)
    res = repair.apply_merges(base, edges, min_edges=2, task=task)
    after = repair.score_vs_truth(task, res["partition"])
    out_path = os.path.join(OUT, f"partition_{task}_L0v3.json")
    json.dump(res["partition"], open(out_path, "w"))
    print(f"{task}: verified_edges={len(edges)} merges={res['n_merges']} "
          f"clusters {res['n_clusters_before']}->{res['n_clusters_after']}")
    print(f"  recall {before['recall']}->{after['recall']}   precision {before['precision']}->{after['precision']}")
    print(f"  wrote {out_path}")
    return {"before": before, "after": after,
            **{k: v for k, v in res.items() if k != "partition"}}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["screen-resume", "confirm-build", "apply"])
    ap.add_argument("--task", required=True)
    ap.add_argument("--cap", type=int, default=None)
    a = ap.parse_args()
    {"screen-resume": screen_resume, "confirm-build": confirm_build,
     "apply": apply}[a.cmd](a.task, a.cap)
