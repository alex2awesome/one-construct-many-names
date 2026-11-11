#!/usr/bin/env python
"""Resumable GLM-5.2 arbiter over the frozen eval sets, all tasks.

Role (user directive 2026-07-06): GLM is a NON-BLOCKING cross-family tie-break / validation
layer. Sonnet-screen + Opus-confirm is the bulk arbiter and never waits on GLM; GLM contributes
an independent second-family vote where available, is RESTARTED ON REBOOT, and is used to
validate adjudicated truth and break uncertain (Sonnet<->Opus disagreement / borderline) cases.
Never on the critical path.

Idempotent + resumable: seeds already-voted pair_ids from ANY outputs/lexicon/arbiter_glm52*.jsonl
(tranche files t1/t1b/t2/t3 + per-task files), then judges only the still-unlabeled eval pairs
per task, appending to per-task arbiter_glm52_<task>.jsonl (itself resume-safe). Safe to run
repeatedly; each run only does what's left.

Restart command (laptop or sk3; z.ai key auto-resolved, 0-GPU HTTP):
    python -m methods.codability.lexicon.glm_arbiter_resume
"""
import glob
import json
import os

from methods.codability.lexicon.judge import FreshJudge
from methods.codability.lexicon.sources import ROOT

OUT = os.path.join(ROOT, "outputs", "lexicon")
TASKS = ["humor", "creative-writing", "code-review", "peer-review", "press-releases",
         "news-homepages", "grant-funding", "legal-outcome-prediction", "notice-and-comment",
         "patents", "math-stackexchange"]


def prior_pairids():
    done = set()
    for f in glob.glob(os.path.join(OUT, "arbiter_glm52*.jsonl")):
        for line in open(f):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if [x for x in r.get("votes", []) if x is not None]:
                done.add(r.get("pair_id"))
    return done


def main():
    done = prior_pairids()
    print(f"seeded {len(done)} already-voted pairs from existing arbiter_glm52*.jsonl", flush=True)
    j = FreshJudge(model="glm-5", votes=1, temperature=0.0)
    for task in TASKS:
        ep = os.path.join(OUT, f"arbiter_eval_{task}.jsonl")
        if not os.path.exists(ep):
            print(f"skip {task}: no frozen eval present", flush=True)
            continue
        rows = []
        for line in open(ep):
            r = json.loads(line)
            if r["pair_id"] in done:
                continue
            rows.append({"pair_id": r["pair_id"], "task": r["task"], "kind": r["stratum"],
                         "key_a": r["key_a"], "key_b": r["key_b"],
                         "canonical_a": r["canonical_a"], "canonical_b": r["canonical_b"]})
        if not rows:
            print(f"{task}: complete (all eval pairs already GLM-voted)", flush=True)
            continue
        op = os.path.join(OUT, f"arbiter_glm52_{task}.jsonl")
        print(f"{task}: {len(rows)} pairs still to judge -> {op}", flush=True)
        st = j.run(rows, op, flush_every=200)
        print(f"{task}: {st}", flush=True)
    print("GLM-ARBITER-RESUME-DONE", flush=True)


if __name__ == "__main__":
    main()
