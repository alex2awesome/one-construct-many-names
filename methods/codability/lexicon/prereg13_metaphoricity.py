#!/usr/bin/env python3
"""PREREG-13 collection (registry 2026-07-23): metaphoricity judging of community-rule
terms + lay heads with the EXACT PREREG-8/bank instrument (axes_judge_glm.py metaphoricity
axis: GLM-4.7 zai, binary, 6 camouflaged anchors/180-term batch, gate >=5/6, temp 0).
Resume-safe append. Output: outputs/lexicon/axis_metaphoricity_prereg13_20260723.jsonl
rows {"variant","cls","score","judge"}.
"""
import glob
import json
import os
import random

import time

from methods.codability.lexicon.axes_judge_glm import AXES, _backend, _judge


def patient_judge(be, axis, terms, rounds=8):
    """z.ai drops ~50% of responses some days (empty-content 200s). Re-query unparsed
    items with backoff until parsed or rounds exhausted. Instrument (prompt/anchors/gate)
    unchanged — transport robustness only, per the GLM patient-retry standing rule."""
    got = {t: None for t in terms}
    todo = list(terms)
    for rd in range(rounds):
        vals = _judge(be, axis, todo)
        for t, v in zip(todo, vals):
            if v is not None:
                got[t] = v
        todo = [t for t in todo if got[t] is None]
        if not todo:
            break
        time.sleep(min(120, 10 * (rd + 1)))
    return [got[t] for t in terms]
from methods.codability.lexicon.codability_sampling_model import norm_name

ROOT = "ROOT"
LEX = f"{ROOT}/outputs/lexicon"
SP = ("/private/tmp/claude-502/-Users-anon-Projects-stanford-research-norm-research/"
      "6eb8228d-7011-4800-87e2-61a172f6003c/scratchpad")
OUT = f"{LEX}/axis_metaphoricity_prereg13_20260723.jsonl"


def pool():
    items = []
    com = set()
    for l in open(f"{LEX}/community_rule_criteria_20260723.jsonl"):
        r = json.loads(l)
        for t in r.get("criterion_terms", []):
            t = (t or "").strip().lower()
            if t:
                com.add(t)
    items += [("community_rule", t) for t in sorted(com)]
    lay = set()
    for p in glob.glob(f"{LEX}/register_corpus_20260723/lay_extract_*.jsonl"):
        for l in open(p):
            try:
                r = json.loads(l)
            except Exception:
                continue
            if r.get("doc_summary_row") or not r.get("head_term"):
                continue
            nm = norm_name(r["head_term"])
            if nm:
                lay.add(nm)
    items += [("individual_lay", t) for t in sorted(lay)]
    return items


def main():
    rng = random.Random(13)
    items = pool()
    rng.shuffle(items)
    done = set()
    if os.path.exists(OUT):
        done = {(json.loads(l)["cls"], json.loads(l)["variant"]) for l in open(OUT)}
    todo = [(c, t) for c, t in items if (c, t) not in done]
    print(f"pool {len(items)} | done {len(done)} | todo {len(todo)}")
    anchors = AXES["metaphoricity"]["anchors"]
    be = _backend()
    n_fail = 0
    with open(OUT, "a") as fo:
        for lo in range(0, len(todo), 180):
            chunk = todo[lo:lo + 180]
            camo = rng.sample(anchors, 6)
            terms = [t for _, t in chunk] + [t for t, _ in camo]
            rng.shuffle(terms)
            got = dict(zip(terms, patient_judge(be, "metaphoricity", terms)))
            a_ok = sum(1 for t, y in camo if got.get(t) == y)
            if a_ok < 5:
                n_fail += 1
                print(f"  ANCHOR GATE FAIL ({a_ok}/6) at {lo} — chunk NOT ingested "
                      f"({n_fail} fails so far)", flush=True)
                if n_fail >= 6:
                    print("  3 gate fails — aborting for inspection")
                    return
                continue
            for c, t in chunk:
                if got.get(t) is not None:
                    fo.write(json.dumps({"variant": t, "cls": c, "score": got[t],
                                         "judge": "glm-4.7_prereg13_20260723"}) + "\n")
            fo.flush()
            print(f"  {min(lo + 180, len(todo))}/{len(todo)} (anchors {a_ok}/6)", flush=True)
    print("COLLECTION DONE")


if __name__ == "__main__":
    main()
