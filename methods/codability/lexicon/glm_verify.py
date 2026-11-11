#!/usr/bin/env python
"""GLM (z.ai) fallback verify for level pairs — direct HTTP, NO Claude subagents, so it runs when
the Claude Max weekly limit is exhausted. Judges the level-verify payloads at the level relation and
appends vrf_<task>_<level>_glm.jsonl (which apply_pairwise reads via its vrf_<task>_<level>_* glob).

Only CLEAN use: finish a cell whose independent (Claude) arbiter truth already exists — e.g. humor R1
(Opus eval done). GLM-build + GLM-measure would be circular, so this is a build-only fallback.

Sparing: max_pairs cap (GLM monthly quota is binding). Resumable (skips already-judged pair_ids).
    python -m methods.codability.lexicon.glm_verify humor R1 8000
"""
import glob
import json
import os
import re
import sys

from methods.codability.lexicon.sources import ROOT

OUT = os.path.join(ROOT, "outputs", "lexicon")


def _score(o):
    if not o:
        return None
    m = re.search(r'"?score"?\s*[:=]\s*([012])', o)
    if not m:
        m = re.search(r'\b([012])\b', o[::-1])  # last digit if no key
        if m:
            return int(m.group(1))
        return None
    return int(m.group(1))


def run(task, level, max_pairs=8000, batch=80):
    proto = open(os.path.join(OUT, f"ARBITER_PROTOCOL_{level}.txt")).read()
    pairs = {}
    for f in sorted(glob.glob(os.path.join(OUT, "level_arbiter", f"{task}_{level}_verify_*.jsonl"))):
        for l in open(f):
            if l.strip():
                r = json.loads(l); pairs[r["pair_id"]] = r
    done = set()
    for f in glob.glob(os.path.join(OUT, "level_votes", f"vrf_{task}_{level}_*.jsonl")):
        for l in open(f):
            if l.strip():
                try:
                    done.add(json.loads(l)["pair_id"])
                except json.JSONDecodeError:
                    pass
    todo = [r for pid, r in pairs.items() if pid not in done][:max_pairs]
    print(f"{task}/{level}: {len(todo)} pairs to GLM-verify (of {len(pairs)-len(done)} undone; {len(done)} done)", flush=True)
    if not todo:
        print("GLM-VERIFY-DONE (nothing to do)", flush=True); return
    from methods.metric_implementer import backends as B, config as C
    be = B.LLMBackend("glm-5", "glm_verify", C.ImplementerConfig(backend="zai_anthropic"))
    outp = os.path.join(OUT, "level_votes", f"vrf_{task}_{level}_glm.jsonl")
    n = 0
    with open(outp, "a") as out:
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            prompts = [f'Concept A: {r["canonical_a"]}\nConcept B: {r["canonical_b"]}' for r in chunk]
            try:
                outs = be.generate_batch(prompts, system=proto, max_tokens=150, temperature=0.0)
            except Exception as e:
                print(f"  GLM error at {i}: {str(e)[:120]} — stopping", flush=True); break
            for r, o in zip(chunk, outs):
                s = _score(o or "")
                out.write(json.dumps({"pair_id": r["pair_id"], "score": s if s in (0, 1, 2) else 1}) + "\n")
            out.flush(); n += len(chunk)
            print(f"  {min(i + batch, len(todo))}/{len(todo)}", flush=True)
    print(f"GLM-VERIFY-DONE ({n} judged)", flush=True)


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "humor",
        sys.argv[2] if len(sys.argv) > 2 else "R1",
        int(sys.argv[3]) if len(sys.argv) > 3 else 8000)
