#!/usr/bin/env python
"""PREREG-23 arm A: emit judging batches that put our FROZEN level prompts, verbatim, in
front of external human-coded gold links.

The prompt text is imported from the modules that actually built our corpus, never retyped,
so arm A measures the instrument we really used:
  L0  l0_precision_audit.PROTOCOL          (SAME CRITERION, 0/1/2)
  R1  build_level.RELATIONS['R1']          (SAME CONSTRUCT)
  R2  build_level.RELATIONS['R2']          (SAME THEME)
  R3  build_level.RELATIONS['R3']          (SAME CATEGORY)

Every batch carries 6 camouflaged anchors (3 known-SAME, 3 known-DIFFERENT) with ordinary
pair_ids. Gate: >=5/6 correct, else the batch is discarded, per the standing anchor rule.
"""
from __future__ import annotations

import hashlib
import json
import os
import random

from .build_level import RELATIONS
from .l0_precision_audit import PROTOCOL as L0_PROTOCOL

ROOT = "ROOT"
OUT = f"{ROOT}/outputs/lexicon/prereg23"
BATCH = 140

SCALE = """Score each pair 0, 1 or 2:
  2 = SAME at this level
  1 = RELATED BUT DISTINCT (same topic/area, different thing at this level)
  0 = DIFFERENT
When torn between 1 and 2, answer 1. Do not infer sameness from shared vocabulary alone."""


def header(rung: str) -> str:
    if rung == "L0":
        return L0_PROTOCOL
    rel, guidance = RELATIONS[rung]
    return (f"# {rung} {rel.upper()} judgement\n\nYou see two nodes from an evaluative "
            f"hierarchy. Decide whether they are the {rel.upper()}.\n\n{guidance}\n\n{SCALE}\n")


# level-appropriate anchors: (a, b, expected) with expected in {2, 0}
ANCHORS = {
    "L0": [("Prose must be free of spelling errors.", "Text should contain no misspelled words.", 2),
           ("Cite every claim to a primary source.", "Each assertion needs a primary-source citation.", 2),
           ("Respond to reviewers within two weeks.", "Reply to referee comments inside a fortnight.", 2),
           ("Prose must be free of spelling errors.", "The argument should be original.", 0),
           ("Cite every claim to a primary source.", "Figures should use colorblind-safe palettes.", 0),
           ("Respond to reviewers within two weeks.", "Code must compile without warnings.", 0)],
    "R1": [("Spelling accuracy", "Orthographic correctness", 2),
           ("Citation completeness", "Thoroughness of referencing", 2),
           ("Response timeliness", "Promptness of reply", 2),
           ("Spelling accuracy", "Novelty of contribution", 0),
           ("Citation completeness", "Colour accessibility of figures", 0),
           ("Response timeliness", "Numerical stability of the solver", 0)],
    "R2": [("Surface language correctness", "Mechanical quality of the writing", 2),
           ("Evidential grounding", "Support and sourcing of claims", 2),
           ("Process responsiveness", "Timeliness of engagement", 2),
           ("Surface language correctness", "Originality and contribution", 0),
           ("Evidential grounding", "Visual accessibility", 0),
           ("Process responsiveness", "Computational correctness", 0)],
    "R3": [("Craft and presentation", "Execution quality", 2),
           ("Substantive contribution", "Intellectual merit", 2),
           ("Craft and presentation", "Ethical conduct", 0),
           ("Substantive contribution", "Turnaround speed", 0),
           ("Process and conduct", "Professional behaviour", 2),
           ("Process and conduct", "Aesthetic appeal", 0)],
}


def emit(cells=None):
    os.makedirs(f"{OUT}/batches", exist_ok=True)
    manifest = []
    files = sorted(f for f in os.listdir(OUT) if f.startswith("pairs_"))
    for fn in files:
        corpus, rung = fn[len("pairs_"):-len(".jsonl")].rsplit("_", 1)
        if cells and (corpus, rung) not in cells:
            continue
        rows = [json.loads(l) for l in open(f"{OUT}/{fn}")]
        rng = random.Random(f"anchor|{corpus}|{rung}")
        for bi in range(0, len(rows), BATCH):
            chunk = list(rows[bi:bi + BATCH])
            truth = {}
            for j, (a, b, exp) in enumerate(ANCHORS[rung]):
                pid = hashlib.sha1(f"anch|{corpus}|{rung}|{bi}|{j}".encode()).hexdigest()[:16]
                chunk.append({"pair_id": pid, "a": a, "b": b})
                truth[pid] = exp
            rng.shuffle(chunk)
            tag = f"{corpus}_{rung}_{bi // BATCH:02d}"
            with open(f"{OUT}/batches/payload_{tag}.jsonl", "w") as fh:
                for r in chunk:
                    fh.write(json.dumps({"pair_id": r["pair_id"], "a": r["a"], "b": r["b"]}) + "\n")
            json.dump(truth, open(f"{OUT}/batches/anchors_{tag}.json", "w"))
            manifest.append({"tag": tag, "corpus": corpus, "rung": rung, "n": len(chunk),
                             "payload": f"{OUT}/batches/payload_{tag}.jsonl",
                             "votes": f"{OUT}/batches/votes_{tag}.jsonl"})
    for m in manifest:
        m["prompt"] = header(m["rung"])
    json.dump(manifest, open(f"{OUT}/batch_manifest.json", "w"), indent=1)
    for m in manifest:
        print(f"{m['tag']:<22}{m['n']:>5} pairs")
    print(f"\n{len(manifest)} batches, {sum(m['n'] for m in manifest)} judgements")
    return manifest


if __name__ == "__main__":
    emit()
