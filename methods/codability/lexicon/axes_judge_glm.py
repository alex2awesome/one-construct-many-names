#!/usr/bin/env python3
"""GLM-judged lens axes: metaphoricity, semantic transparency, thick-vs-thin.

No Sonnet lane this session, so GLM-4.7 is the judge — GATED PER AXIS on hand-built
unambiguous anchors (>= .85 required to run the full inventory; anchors camouflaged among
real items in full runs too). Binary judgments for gateability.

Modes: pilot (anchors only, prints gate) | full --axis <name> (all code_axes variants,
resume-safe, writes outputs/lexicon/axis_<name>_20260721.jsonl).
"""
import argparse
import json
import os
import random
import re

ROOT = "ROOT"
LEX = f"{ROOT}/outputs/lexicon"

AXES = {
    "metaphoricity": {
        "system": (
            "You judge whether an evaluation-criterion term is METAPHORICAL — a figurative "
            "borrowing from another domain (physical action, body, space, music...) used to "
            "name a quality of work — or LITERAL. Examples: metaphorical = 'punch', 'flow', "
            "'hook', 'landing'; literal = 'grammatical correctness', 'citation accuracy'.\n"
            'Reply STRICT JSON only: {"score": 1} if metaphorical, {"score": 0} if literal.'),
        "anchors": [("landing the punchline", 1), ("punch", 1), ("flow", 1), ("hook", 1),
                    ("kill your darlings", 1), ("skeleton of the proof", 1),
                    ("signposting", 1), ("burying the lede", 1), ("purple prose", 1),
                    ("grammatical correctness", 0), ("citation accuracy", 0),
                    ("word count limit", 0), ("proof correctness", 0),
                    ("statistical significance", 0), ("spelling errors", 0),
                    ("sample size", 0), ("response time", 0), ("factual accuracy", 0)]},
    "transparency": {
        "system": (
            "You judge whether an evaluation-criterion term is SEMANTICALLY TRANSPARENT — "
            "its meaning is predictable by composing its words ('excessive setup', 'accurate "
            "citations') — or OPAQUE/IDIOMATIC — the meaning cannot be derived from the parts "
            "without cultural knowledge ('kill your darlings', 'deus ex machina').\n"
            'Reply STRICT JSON only: {"score": 1} if transparent, {"score": 0} if opaque.'),
        "anchors": [("excessive setup", 1), ("clear structure", 1), ("accurate citations", 1),
                    ("logical organization", 1), ("concise wording", 1), ("correct grammar", 1),
                    ("relevant examples", 1), ("consistent notation", 1),
                    ("kill your darlings", 0), ("deus ex machina", 0), ("purple prose", 0),
                    ("dad joke", 0), ("burying the lede", 0), ("red herring", 0),
                    ("cherry picking", 0), ("straw man", 0), ("show dont tell", 0),
                    ("rule of three", 0)]},
    "thickthin": {
        "system": (
            "You judge whether an evaluative term is THIN — pure evaluation with little "
            "descriptive content ('good', 'excellent', 'effective') — or THICK — it carries "
            "substantial descriptive content along with its evaluation ('hacky', 'derivative', "
            "'clickbait': you learn WHAT the work is like, not just that it is good/bad).\n"
            'Reply STRICT JSON only: {"score": 1} if thick, {"score": 0} if thin.'),
        "anchors": [("hacky", 1), ("derivative", 1), ("pedantic", 1), ("clickbait", 1),
                    ("rigorous", 1), ("sloppy", 1), ("contrived", 1), ("preachy", 1),
                    ("formulaic", 1), ("good", 0), ("excellent quality", 0),
                    ("well written", 0), ("effective", 0), ("strong work", 0),
                    ("high quality", 0), ("great", 0), ("solid", 0), ("outstanding", 0)]},
}


def _backend():
    from methods.metric_implementer import backends as _b, config as _c
    return _b.LLMBackend("glm-4.7", "lexicon_axis_judge", _c.ImplementerConfig(backend="zai_anthropic"))


def _parse(o):
    m = re.search(r"\{.*\}", o or "", re.S)
    try:
        s = json.loads(m.group(0)).get("score")
        return s if s in (0, 1) else None
    except Exception:
        return None


def _judge(be, axis, terms):
    prompts = [f"TERM: {t}" for t in terms]
    outs = be.generate_batch(prompts, system=AXES[axis]["system"], max_tokens=300,
                             temperature=0.0, seed=0)
    bad = [i for i, o in enumerate(outs) if _parse(o) is None]
    if bad:
        r2 = be.generate_batch([prompts[i] for i in bad], system=AXES[axis]["system"],
                               max_tokens=300, temperature=0.5, seed=1)
        for i, o in zip(bad, r2):
            outs[i] = o
    return [_parse(o) for o in outs]


def pilot(_a):
    be = _backend()
    for axis, spec in AXES.items():
        terms = [t for t, _ in spec["anchors"]]
        truth = [y for _, y in spec["anchors"]]
        got = _judge(be, axis, terms)
        ok = sum(1 for g, y in zip(got, truth) if g == y)
        n = sum(1 for g in got if g is not None)
        misses = [(t, y, g) for (t, y), g in zip(spec["anchors"], got) if g != y]
        print(f"{axis:14} gate {ok}/{len(truth)} (parsed {n}) -> "
              f"{'PASS' if ok / len(truth) >= .85 else 'FAIL'}  misses={misses[:4]}")


def full(a):
    axis = a.axis
    rng = random.Random(9)
    variants = [json.loads(l)["variant"] for l in open(f"{LEX}/code_axes_20260721.jsonl")]
    out_path = f"{LEX}/axis_{axis}_20260721.jsonl"
    done = set()
    if os.path.exists(out_path):
        done = {json.loads(l)["variant"] for l in open(out_path)}
    todo = [v for v in variants if v not in done]
    anchors = AXES[axis]["anchors"]
    be = _backend()
    with open(out_path, "a") as fo:
        for lo in range(0, len(todo), 180):
            chunk = list(todo[lo:lo + 180])
            camo = [(f"__A{j}__", t, y) for j, (t, y) in enumerate(rng.sample(anchors, 6))]
            items = chunk + [t for _, t, _ in camo]
            rng.shuffle(items)
            got = dict(zip(items, _judge(be, axis, items)))
            a_ok = sum(1 for _, t, y in camo if got.get(t) == y)
            if a_ok < 5:
                print(f"  ANCHOR GATE FAIL in batch ({a_ok}/6) — chunk NOT ingested; aborting")
                return
            for v in chunk:
                if got.get(v) is not None:
                    fo.write(json.dumps({"variant": v, "axis": axis, "score": got[v],
                                         "judge": "glm-4.7_20260721"}) + "\n")
            fo.flush()
            print(f"  {min(lo + 180, len(todo))}/{len(todo)} (batch anchors {a_ok}/6)", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("pilot")
    fp = sub.add_parser("full")
    fp.add_argument("--axis", required=True, choices=sorted(AXES))
    args = p.parse_args()
    {"pilot": pilot, "full": full}[args.cmd](args)
