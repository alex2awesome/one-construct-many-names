#!/usr/bin/env python
"""GLM (z.ai) L0 repair for a task — direct HTTP, NO Claude subagents (runs under the Claude weekly
limit). GLM-judges the task's candidate band at the SAME-CRITERION relation, keeps score==2 edges,
applies >=2-edge STAR (chain-proof; the >=2 requirement blunts GLM's leniency by demanding two
independent SAME member-edges per cluster-pair), scores vs the EXISTING Opus adjudicated truth
(locked earlier -> GLM-build + Opus-measure is NOT circular). Precision is the gate: if GLM+>=2-edge
comes in well below the Sonnet-built L0 bar (~.63-.85), report and do NOT trust it.

    python -m methods.codability.lexicon.glm_l0_repair code-review
"""
import glob
import json
import os
import re
import sys

from methods.codability.lexicon import repair

OUT = repair.OUT


def _score(o):
    if not o:
        return None
    m = re.search(r'"?score"?\s*[:=]\s*([012])', o)
    if m:
        return int(m.group(1))
    m = re.search(r'[012]', o.lstrip("*# \n\t"))  # GLM writes the score FIRST ("2 — reasoning")
    return int(m.group(0)) if m else None


def run(task, max_pairs=2500, batch=80):
    proto = open(os.path.join(OUT, "JUDGE_PROTOCOL.txt")).read()
    cand = json.load(open(os.path.join(OUT, f"repair_candidates_{task}.json")))[:max_pairs]
    outp = os.path.join(OUT, "repair_votes", f"glm_{task}.jsonl")
    done = set()
    if os.path.exists(outp):
        for l in open(outp):
            if l.strip():
                try:
                    done.add(json.loads(l)["pair_id"])
                except json.JSONDecodeError:
                    pass
    todo = [c for c in cand if c["pair_id"] not in done]
    print(f"{task}: {len(todo)} candidates to GLM-judge ({len(done)} done)", flush=True)
    if todo:
        from methods.metric_implementer import backends as B, config as C
        be = B.LLMBackend("glm-5", "glm_l0", C.ImplementerConfig(backend="zai_anthropic"))
        dom = task.replace("-", " ")
        with open(outp, "a") as out:
            for i in range(0, len(todo), batch):
                ch = todo[i:i + batch]
                pr = [f'Domain: {dom}\n\nRUBRIC A: {c["canonical_a"]}\nRUBRIC B: {c["canonical_b"]}' for c in ch]
                try:
                    outs = be.generate_batch(pr, system=proto, max_tokens=150, temperature=0.0)
                except Exception as e:
                    print(f"  GLM error at {i}: {str(e)[:120]} — stopping", flush=True); break
                for c, o in zip(ch, outs):
                    s = _score(o or "")
                    out.write(json.dumps({"pair_id": c["pair_id"], "key_a": c["key_a"],
                                          "key_b": c["key_b"], "score": s if s in (0, 1, 2) else 1}) + "\n")
                out.flush()
                print(f"  {min(i + batch, len(todo))}/{len(todo)}", flush=True)
    # apply >=2-edge star + score vs existing Opus truth
    edges = []
    for l in open(outp):
        if l.strip():
            r = json.loads(l)
            if r.get("score") == 2:
                edges.append((r["key_a"], r["key_b"]))
    base = repair.load_base_partition(task)
    res = repair.apply_merges(base, edges, min_edges=2, task=task)
    before = repair.score_vs_truth(task, base)
    after = repair.score_vs_truth(task, res["partition"])
    json.dump(res["partition"], open(os.path.join(OUT, f"partition_{task}_L0v2_glm.json"), "w"))
    out = {"task": task, "glm_same_edges": len(edges), "merges": res["n_merges"],
           "before": {"r": before["recall"], "p": before["precision"]},
           "after": {"r": after["recall"], "p": after["precision"]}}
    json.dump(out, open(os.path.join(OUT, f"glm_l0_{task}.json"), "w"))
    print(f"GLM-L0-DONE {json.dumps(out)}", flush=True)


if __name__ == "__main__":
    run(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 2500)
