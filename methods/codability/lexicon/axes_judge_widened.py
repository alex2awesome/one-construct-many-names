#!/usr/bin/env python3
"""PREREG-7 instrument run: metaphoricity + transparency GLM axis judging for the widened-
field variant queue (widened_axis_queue_20260722.txt), using the exact 2026-07-21 protocol
from axes_judge_glm (same prompts, same anchors, 6-anchor camouflage gate per batch,
abort <5/6). Output: axis_<axis>_widened_20260722.jsonl. Resume-safe."""
import json
import os
import random

from methods.codability.lexicon.axes_judge_glm import AXES, LEX, _backend, _judge

QUEUE = f"{LEX}/widened_axis_queue_20260722.txt"


def run(axis):
    rng = random.Random(11)
    variants = [v for v in open(QUEUE).read().splitlines() if v]
    out_path = f"{LEX}/axis_{axis}_widened_20260722.jsonl"
    done = set()
    if os.path.exists(out_path):
        done = {json.loads(l)["variant"] for l in open(out_path)}
    todo = [v for v in variants if v not in done]
    anchors = AXES[axis]["anchors"]
    be = _backend()
    print(f"{axis}: {len(todo)} to judge")
    with open(out_path, "a") as fo:
        for lo in range(0, len(todo), 180):
            chunk = list(todo[lo:lo + 180])
            camo = [(t, y) for t, y in rng.sample(anchors, 6)]
            items = chunk + [t for t, _ in camo]
            rng.shuffle(items)
            got = dict(zip(items, _judge(be, axis, items)))
            a_ok = sum(1 for t, y in camo if got.get(t) == y)
            if a_ok < 5:
                print(f"  ANCHOR GATE FAIL ({a_ok}/6) — chunk NOT ingested; aborting")
                return
            for v in chunk:
                if got.get(v) is not None:
                    fo.write(json.dumps({"variant": v, "axis": axis, "score": got[v],
                                         "judge": "glm-4.7_20260722"}) + "\n")
            fo.flush()
            print(f"  {min(lo + 180, len(todo))}/{len(todo)} (batch anchors {a_ok}/6)",
                  flush=True)


if __name__ == "__main__":
    for axis in ("metaphoricity", "transparency"):
        run(axis)
