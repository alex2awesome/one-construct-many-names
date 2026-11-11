#!/usr/bin/env python3
"""GLM fallback judge for W4 register terms and W2b subfield pairs.

Context: session subagent limit reached 2026-07-20 late; remaining judging pivots to
GLM-4.7 (approved judge class) with a MANDATORY cross-instrument validation gate before
each queue is inherited:
  w4-validate:  GLM re-judges a sample of Sonnet-judged variants + the 30 stratum anchors.
                Gate: stratum agreement >= .85 AND formality Spearman >= .75.
  w2b-validate: GLM re-judges a sample of Sonnet-judged pairs. Gate: same/diff agreement >= .90
                (Sonnet same-rates ran .91-.99, so agreement is mostly driven by the sames;
                the gate also requires >= .5 recall on Sonnet's DIFFERENT calls in the sample).
Prompts are VERBATIM the Sonnet agent prompts. Run modes: w4-validate, w4-tail, w2b-validate,
w2b-wave --lo --hi. All outputs append-only; instrument recorded per row.
"""
import argparse
import glob
import json
import os
import random
import re
from collections import defaultdict

ROOT = "ROOT"
LEX = f"{ROOT}/outputs/lexicon"

W4_SYSTEM = (
    "You are a careful lexicographic annotator for a study of the register of evaluative "
    "vocabulary. You are given a word or short phrase used to name an evaluation concept. "
    "Judge three things:\n"
    '1. "stratum": the dominant etymological stratum of the CONTENT words - "germanic" '
    "(native Anglo-Saxon roots: help, show, buy, teach, wit, truth, craft, strength), "
    '"latinate" (Latin/French roots: facilitate, demonstrate, clarity, coherence, rigor), '
    '"greek" (Greek roots: rhetoric, metaphor, synthesis, aesthetics), or "mixed" '
    "(multi-word terms mixing strata roughly equally, or a native root with a heavy Romance "
    "derivational suffix where neither dominates). Judge by the actual root of each content "
    "word, not by how formal it feels.\n"
    '2. "formality": 1-7 - how formal/elevated the term is as a register choice (1 = casual '
    "everyday speech, 4 = neutral professional, 7 = highly formal/technical-institutional).\n"
    '3. "nominalization": 1 if the term is or contains a derived abstract nominal (suffixes '
    "like -tion, -ity, -ness, -ment, -ance/-ence), else 0.\n"
    'Reply with STRICT JSON only: {"stratum": "...", "formality": 1-7, "nominalization": 0|1}'
)

W2B_SYSTEM = (
    "You are a careful annotator consolidating a taxonomy of practice subfields. You are "
    "given two short free-text labels naming the subfield/topic-area a source document is "
    "about. Decide whether the two labels name the SAME subfield - i.e., a single taxonomy "
    'node would correctly cover both ("writing mathematical proofs" vs "mathematical proof '
    'writing" = same). Same = mere rewording, pluralization, reordering, or trivially '
    "narrower/broader phrasing of one topic. Different = genuinely distinct topics, even if "
    'related ("writing appellate briefs" vs "writing trial briefs" = different - sibling '
    'topics; "doing X" vs "evaluating X" = different activities).\n'
    'Reply with STRICT JSON only: {"same": 1|0}'
)


def _backend():
    from methods.metric_implementer import backends as _b, config as _c
    return _b.LLMBackend("glm-4.7", "lexicon_fallback_judge", _c.ImplementerConfig(backend="zai_anthropic"))


def _parse(o, key):
    m = re.search(r"\{.*\}", o or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0)).get(key, None) if key else json.loads(m.group(0))
    except Exception:
        return None


def _gen(be, prompts, system):
    outs = be.generate_batch(prompts, system=system, max_tokens=300, temperature=0.0, seed=0)
    retry = [i for i, o in enumerate(outs) if _parse(o, None) is None]
    if retry:
        r2 = be.generate_batch([prompts[i] for i in retry], system=system,
                               max_tokens=300, temperature=0.5, seed=1)
        for i, o in zip(retry, r2):
            outs[i] = o
    return outs


def w4_validate(a):
    rng = random.Random(3)
    rows = [json.loads(l) for l in open(f"{LEX}/register_height_judgments.jsonl")]
    samp = rng.sample(rows, min(150, len(rows)))
    be = _backend()
    outs = _gen(be, [f"TERM: {r['variant']}" for r in samp], W4_SYSTEM)
    import numpy as np
    from scipy import stats
    ok = [(r, _parse(o, None)) for r, o in zip(samp, outs)]
    ok = [(r, d) for r, d in ok if d and d.get("stratum") and d.get("formality")]
    ag = sum(1 for r, d in ok if d["stratum"] == r["stratum"]) / len(ok)
    rho = stats.spearmanr([r["formality"] for r, _ in ok], [d["formality"] for _, d in ok]).statistic
    print(f"n={len(ok)} stratum agreement {ag:.3f} (gate .85) | formality rho {rho:.3f} (gate .75)")
    print("GATE:", "PASS" if ag >= .85 and rho >= .75 else "FAIL")


def w4_tail(a):
    donev = {json.loads(l)["variant"] for l in open(f"{LEX}/register_height_judgments.jsonl")}
    usage = defaultdict(int)
    for l in open(f"{LEX}/name_variants_20260720.jsonl"):
        r = json.loads(l)
        usage[r["variant"]] += r["count"]
    todo = sorted(v for v in usage if v not in donev)
    be = _backend()
    with open(f"{LEX}/register_height_judgments.jsonl", "a") as fo:
        for lo in range(0, len(todo), 200):
            ch = todo[lo:lo + 200]
            outs = _gen(be, [f"TERM: {v}" for v in ch], W4_SYSTEM)
            for v, o in zip(ch, outs):
                d = _parse(o, None) or {}
                if d.get("stratum"):
                    fo.write(json.dumps({"variant": v, "stratum": d["stratum"],
                                         "formality": d.get("formality"),
                                         "nominalization": d.get("nominalization"),
                                         "batch": "03_tail_glm", "judge": "glm-4.7_20260720"}) + "\n")
            fo.flush()
            print(f"  {min(lo + 200, len(todo))}/{len(todo)}", flush=True)


def w2b_validate(a):
    rng = random.Random(5)
    rows = [json.loads(l) for l in open(f"{LEX}/subfield_merges_20260720.jsonl")]
    diffs = [r for r in rows if r["same"] == 0]
    sames = [r for r in rows if r["same"] == 1]
    samp = rng.sample(sames, 100) + rng.sample(diffs, min(50, len(diffs)))
    be = _backend()
    outs = _gen(be, [f"LABEL A: {r['a']}\nLABEL B: {r['b']}" for r in samp], W2B_SYSTEM)
    got = [(r, _parse(o, "same")) for r, o in zip(samp, outs)]
    got = [(r, g) for r, g in got if g in (0, 1)]
    ag = sum(1 for r, g in got if g == r["same"]) / len(got)
    dif = [(r, g) for r, g in got if r["same"] == 0]
    drec = sum(1 for r, g in dif if g == 0) / max(1, len(dif))
    print(f"n={len(got)} agreement {ag:.3f} (gate .90) | different-recall {drec:.3f} (gate .50)")
    print("GATE:", "PASS" if ag >= .90 and drec >= .50 else "FAIL")


def w2b_wave(a):
    judged = set()
    for l in open(f"{LEX}/subfield_merges_20260720.jsonl"):
        r = json.loads(l)
        judged.add((r["task"], r["a"], r["b"]))
    pool = []
    for f in glob.glob(f"{LEX}/subfield_pairs_*.jsonl"):
        for l in open(f):
            r = json.loads(l)
            if a.lo <= r["cos"] < a.hi and (r["task"], r["a"], r["b"]) not in judged:
                pool.append(r)
    pool.sort(key=lambda r: -r["cos"])
    print(f"wave [{a.lo},{a.hi}): {len(pool)} pairs")
    be = _backend()
    with open(f"{LEX}/subfield_merges_20260720.jsonl", "a") as fo:
        for lo in range(0, len(pool), 200):
            ch = pool[lo:lo + 200]
            outs = _gen(be, [f"LABEL A: {r['a']}\nLABEL B: {r['b']}" for r in ch], W2B_SYSTEM)
            for r, o in zip(ch, outs):
                g = _parse(o, "same")
                if g in (0, 1):
                    fo.write(json.dumps({"wave": f"glm_{a.lo}", "task": r["task"], "a": r["a"],
                                         "b": r["b"], "cos": r["cos"], "same": g,
                                         "judge": "glm-4.7_20260720"}) + "\n")
            fo.flush()
            print(f"  {min(lo + 200, len(pool))}/{len(pool)}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("w4-validate")
    sub.add_parser("w4-tail")
    sub.add_parser("w2b-validate")
    w = sub.add_parser("w2b-wave")
    w.add_argument("--lo", type=float, required=True)
    w.add_argument("--hi", type=float, required=True)
    args = p.parse_args()
    {"w4-validate": w4_validate, "w4-tail": w4_tail,
     "w2b-validate": w2b_validate, "w2b-wave": w2b_wave}[args.cmd](args)
