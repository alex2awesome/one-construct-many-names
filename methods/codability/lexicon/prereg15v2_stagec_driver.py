#!/usr/bin/env python
"""PREREG-15-v2 Stage C driver: dispatch Codex gpt-5.6-sol scoring sessions.

Per (metric, arm, replicate): one fresh Codex session reads the metric's BLIND item
file (label-free, shuffled) and writes v2_score_<mid>_<arm>_r<k>.jsonl with one line
per item: {"item_idx": <idx>, "score": 1-7}. Scaffold frozen verbatim per the prereg.

Dispatch uses the companion's --background mode (the launch call returns immediately
and the session runs in the companion's own runtime): long-lived foreground children
were SIGKILLed by the sandbox, which is what killed the first two driver attempts.
Completion is detected by the OUTPUT FILE becoming valid, with a per-cell timeout.
Resume-safe; capped retries; progress in v2_stagec.log.
"""
from __future__ import annotations

import json
import os
import subprocess
import time

D = "ROOT/outputs/lexicon/prereg15v2_20260828"
MJS = os.path.expanduser("~/.claude/plugins/cache/openai-codex/codex/1.0.5/scripts/codex-companion.mjs")
IN_FLIGHT = 3
CELL_TIMEOUT = 15 * 60
MAX_ATTEMPTS = 3
SCAFFOLD = ("You are evaluating stand-up comedy material. The criterion named below is one "
            "used by the comedy community to judge such material. No definition is provided "
            "--- apply the criterion as you understand it from its name.")


def cell_done(path: str) -> bool:
    if not os.path.exists(path):
        return False
    try:
        rows = [json.loads(l) for l in open(path)]
    except Exception:
        return False
    return (len(rows) == 40 and
            all(isinstance(r.get("score"), (int, float)) and 1 <= r["score"] <= 7 for r in rows))


def prompt_for(mid: str, arm: str, rubric: str, k: int) -> str:
    return (
        f"Scoring task, no investigation. {SCAFFOLD}\n\n"
        f"CRITERION NAME: {rubric}\n\n"
        f"Read the file {D}/v2_scoring_items_{mid}.json (JSON: items = list of "
        f"{{item_idx, text}}). For EVERY item, judge how well the item satisfies the "
        f"criterion named above on a 1-7 scale (1 = clearly fails it, 7 = clearly "
        f"satisfies it). Judge each item independently from its text alone. Write "
        f"{D}/v2_score_{mid}_{arm}_r{k}.jsonl with exactly one JSON line per item: "
        f'{{"item_idx": <idx>, "score": <1-7>}} in the file order. 40 lines total; '
        f"verify the count. Do not read or write any other file. Final output: the "
        f"line count only.")


def main() -> None:
    arms = json.load(open(f"{D}/v2_arms.json"))
    todo = []
    for mid, a in sorted(arms.items()):
        if not os.path.exists(f"{D}/v2_scoring_items_{mid}.json"):
            continue
        for arm in ("low", "high"):
            for k in (0, 1):
                if not cell_done(f"{D}/v2_score_{mid}_{arm}_r{k}.jsonl"):
                    todo.append((mid, arm, k))
    log = open(f"{D}/v2_stagec.log", "a")
    log.write(f"=== bg-driver start {time.ctime()}: {len(todo)} cells ===\n"); log.flush()

    attempts: dict[tuple, int] = {}
    inflight: dict[tuple, float] = {}
    gave_up = 0
    while todo or inflight:
        for cell in list(inflight):
            mid, arm, k = cell
            out = f"{D}/v2_score_{mid}_{arm}_r{k}.jsonl"
            if cell_done(out):
                del inflight[cell]
                log.write(f"OK {mid} {arm} r{k}\n"); log.flush()
            elif time.time() - inflight[cell] > CELL_TIMEOUT:
                del inflight[cell]
                attempts[cell] = attempts.get(cell, 0) + 1
                if attempts[cell] < MAX_ATTEMPTS:
                    todo.append(cell)
                    log.write(f"TIMEOUT {mid} {arm} r{k} (attempt {attempts[cell]}), requeued\n")
                else:
                    gave_up += 1
                    log.write(f"GIVING UP {mid} {arm} r{k}\n")
                log.flush()
        while todo and len(inflight) < IN_FLIGHT:
            mid, arm, k = todo.pop(0)
            rubric = arms[mid][f"{arm}_rubric"]
            r = subprocess.run(
                ["node", MJS, "task", "--background", "--fresh", "--write",
                 "--model", "gpt-5.6-sol", prompt_for(mid, arm, rubric, k)],
                capture_output=True, text=True, timeout=120)
            log.write(f"launched {mid} {arm} r{k}: {(r.stdout or r.stderr).strip()[:90]}\n")
            log.flush()
            inflight[(mid, arm, k)] = time.time()
            time.sleep(5)
        time.sleep(30)
    log.write(f"=== bg-driver done {time.ctime()}; gave_up={gave_up} ===\n")


if __name__ == "__main__":
    main()
