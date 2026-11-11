#!/usr/bin/env python3
"""Disambiguate the selection-audit finding (heads +.05 more latinate than key_terms,
5/5 fields): is the extractor CHOOSING the most latinate of the author's candidate names
(selection bias), or are NAMES-as-a-class more latinate than surrounding evaluative text
(naming is a nominal register act)?

Design: for a sample of named records, re-prompt GLM to list ALL names/terms-of-art the
author uses for the concept (each validated verbatim-in-source, same discipline as
extract.py). Readouts:
  A. P(original head == max-latinate candidate) vs chance among candidates  -> selection bias
  B. mean latinate of candidate-name CLASS vs same records' key_terms       -> nominal-act
Both can be true; the split apportions the +.056 delta.

Subcommands: build (sample+payload) | run (GLM, resume-safe) | analyze
"""
import argparse
import json
import os
import random
import re

ROOT = "ROOT"
LEX = f"{ROOT}/outputs/lexicon"
SP = ("/private/tmp/claude-502/-Users-anon-Projects-stanford-research-norm-research/"
      "6eb8228d-7011-4800-87e2-61a172f6003c/scratchpad")
FIELDS = ["humor", "creative-writing", "news-homepages", "math-stackexchange",
          "notice-and-comment"]
PER_FIELD = 60

SYSTEM = (
    "You are a careful corpus annotator. You are given a SOURCE DOCUMENT and one CRITERION "
    "previously extracted from it. List ALL the names or terms of art the ORIGINAL AUTHOR "
    "uses to label this concept in this document — every distinct name, not just the best "
    "one. Each name must appear VERBATIM in the document. If the author never names the "
    "concept, return an empty list.\n"
    'Reply STRICT JSON only: {"names": ["...", ...]} (at most 6).')


def _contains(needle, hay):
    n = re.findall(r"[a-z0-9]+", needle.casefold())
    h = re.findall(r"[a-z0-9]+", hay.casefold())
    return bool(n) and any(h[i:i + len(n)] == n for i in range(len(h) - len(n) + 1))


def build(_a):
    rng = random.Random(77)
    rows = []
    for f in FIELDS:
        ctx = {}
        for line in open(f"{LEX}/contexts_{f}.jsonl"):
            c = json.loads(line)
            ctx[c["key"]] = c
        cand = []
        for line in open(f"{LEX}/extract_{f}_glm-4.7.jsonl"):
            r = json.loads(line)
            if r.get("status") == "ok" and r.get("head_term") and r["key"] in ctx:
                cand.append(r)
        for r in rng.sample(cand, min(PER_FIELD, len(cand))):
            c = ctx[r["key"]]
            rows.append({"key": r["key"], "field": f, "head": r["head_term"],
                         "key_terms": r.get("key_terms") or [],
                         "name": c.get("name"), "description": c.get("description"),
                         "doc_text": c.get("doc_text", "")[:12000]})
    with open(f"{SP}/disambig_payload.jsonl", "w") as fo:
        for r in rows:
            fo.write(json.dumps(r) + "\n")
    print(f"built {len(rows)} sample records -> disambig_payload.jsonl")


def run(_a):
    from methods.metric_implementer import backends as _b, config as _c
    be = _b.LLMBackend("glm-4.7", "lexicon_disambig", _c.ImplementerConfig(backend="zai_anthropic"))
    rows = [json.loads(l) for l in open(f"{SP}/disambig_payload.jsonl")]
    out_path = f"{LEX}/head_selection_disambig_20260721.jsonl"
    done = set()
    if os.path.exists(out_path):
        done = {json.loads(l)["key"] for l in open(out_path)}
    todo = [r for r in rows if r["key"] not in done]
    with open(out_path, "a") as fo:
        for lo in range(0, len(todo), 60):
            ch = todo[lo:lo + 60]
            ps = [f"CRITERION:\nname: {r['name']}\ndesc: {r['description']}\n\n"
                  f"SOURCE DOCUMENT:\n{r['doc_text']}" for r in ch]
            outs = be.generate_batch(ps, system=SYSTEM, max_tokens=300, temperature=0.0, seed=0)
            for r, o in zip(ch, outs):
                m = re.search(r"\{.*\}", o or "", re.S)
                names = []
                if m:
                    try:
                        names = [n for n in json.loads(m.group(0)).get("names", [])
                                 if isinstance(n, str) and _contains(n, r["doc_text"])][:6]
                    except Exception:
                        pass
                fo.write(json.dumps({"key": r["key"], "field": r["field"],
                                     "head": r["head"], "key_terms": r["key_terms"],
                                     "candidates": names}) + "\n")
            fo.flush()
            print(f"  {min(lo + 60, len(todo))}/{len(todo)}", flush=True)


def analyze(_a):
    import numpy as np
    from methods.codability.lexicon.latinate_detector import V2, latinate_score
    v2 = V2()
    picked_max = n_multi = 0
    chance = []
    cand_lat, keys_lat = [], []
    for line in open(f"{LEX}/head_selection_disambig_20260721.jsonl"):
        r = json.loads(line)
        cands = r["candidates"]
        scores = [(c, latinate_score(c, v2.word)) for c in cands]
        scores = [(c, s) for c, s in scores if s is not None]
        if scores:
            cand_lat.append(float(np.mean([s for _, s in scores])))
            kk = [latinate_score(k, v2.word) for k in r["key_terms"]]
            kk = [x for x in kk if x is not None]
            if kk:
                keys_lat.append(float(np.mean(kk)))
        if len(scores) >= 2:
            n_multi += 1
            mx = max(s for _, s in scores)
            hs = latinate_score(r["head"], v2.word)
            if hs is not None and hs >= mx - 1e-9:
                picked_max += 1
            chance.append(sum(1 for _, s in scores if s >= mx - 1e-9) / len(scores))
    print(f"A. multi-candidate records: {n_multi}; original head == max-latinate candidate "
          f"{picked_max}/{n_multi} = {picked_max/max(1,n_multi):.3f} vs chance {np.mean(chance):.3f}")
    print(f"B. candidate-name CLASS latinate {np.mean(cand_lat):.3f} (n={len(cand_lat)}) vs "
          f"same-record key_terms {np.mean(keys_lat):.3f} (n={len(keys_lat)})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for c in ("build", "run", "analyze"):
        sub.add_parser(c)
    a = p.parse_args()
    {"build": build, "run": run, "analyze": analyze}[a.cmd](a)
