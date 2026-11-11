#!/usr/bin/env python3
"""Parse-error reject sweep (ledger 2026-07-21e). Under API degradation, parse_error /
no_reply rejects are INSTRUMENT noise, not content verdicts — but the extractor's resume
logic treats any written key as done, so they would be lost forever. This tool re-runs
exactly those keys under (presumably) healthier API conditions.

Discipline: append-only. Sweep rows go to extract_<task>_glm-4.7_sweep<N>.jsonl; ok rows
are then APPENDED to the main extract file (the original reject rows remain in place as
provenance; loaders filter status!=ok, and the extractor's done-set sees the key as ok).
Content-driven rejects (quote_not_in_source, quote_too_long, term_not_in_source,
notfound_nonempty, ...) are FINAL and never re-run.

Usage: python3 -m methods.codability.lexicon.reject_sweep --task notice-and-comment
"""
import argparse
import json
import os

from methods.codability.lexicon.extract import GLMExtractor

LEX = "ROOT/outputs/lexicon"
SP = ("/private/tmp/claude-502/-Users-anon-Projects-stanford-research-norm-research/"
      "6eb8228d-7011-4800-87e2-61a172f6003c/scratchpad")
NOISE = {"parse_error", "no_reply"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    a = ap.parse_args()
    main_path = f"{LEX}/extract_{a.task}_glm-4.7.jsonl"
    ok_keys, noise_keys = set(), set()
    for line in open(main_path):
        r = json.loads(line)
        (ok_keys if r["status"] == "ok" else
         noise_keys if r["status"] in NOISE else set()).add(r["key"])
    todo_keys = noise_keys - ok_keys
    print(f"{a.task}: {len(todo_keys)} noise-rejected keys to re-run "
          f"({len(ok_keys)} ok already)")
    if not todo_keys:
        return
    ctxs = []
    for line in open(f"{SP}/census_payload_{a.task}.jsonl"):
        c = json.loads(line)
        if c["key"] in todo_keys:
            ctxs.append(c)
    n = 1
    while os.path.exists(f"{LEX}/extract_{a.task}_glm-4.7_sweep{n}.jsonl"):
        n += 1
    sweep_path = f"{LEX}/extract_{a.task}_glm-4.7_sweep{n}.jsonl"
    ex = GLMExtractor()
    st = ex.run(ctxs, sweep_path, flush_every=50)
    print("sweep stats:", st)
    rescued = 0
    with open(main_path, "a") as fo:
        for line in open(sweep_path):
            r = json.loads(line)
            if r["status"] == "ok":
                fo.write(line)
                rescued += 1
    print(f"rescued {rescued}/{len(ctxs)} -> appended to {os.path.basename(main_path)} "
          f"(reject provenance rows retained)")


if __name__ == "__main__":
    main()
