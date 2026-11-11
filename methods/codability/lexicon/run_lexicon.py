"""Driver for the metric-lexicon codability census (notes/2026-07-06__metric-lexicon-codability-census.md).

Phases:
  build        materialize extraction contexts for tasks (CPU, local banks)
  sample       stable-hash equivalence sample + blinded anchors -> payload jsonl(s)
  glm          run GLM extraction over a payload/context file (HTTP; quota-aware caller)
  compare      GLM-vs-Sonnet equivalence report
  partition    trusted repaired partition + fresh-judge candidates (audit.py)
  census       conventionalization census over ok extractions + trusted partition

Sonnet extraction runs as Claude subagents orchestrated OUTSIDE this script (payloads from
`sample --split-payloads`); their JSONL output is validated here in `compare`/`ingest` via the
same mechanical checks (quotes re-validated against stored doc_text).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random

from . import anchors as anch
from . import audit, census, sources
from .extract import GLMExtractor, finalize, load_extractions, parse_reply, validate

OUT = os.path.join(sources.ROOT, "outputs", "lexicon")


def _hash_rank(key: str) -> str:
    return hashlib.sha1(key.encode()).hexdigest()


def phase_build(args):
    os.makedirs(OUT, exist_ok=True)
    for task in args.tasks.split(","):
        task = task.strip()
        st = sources.build_contexts(task, os.path.join(OUT, f"contexts_{task}.jsonl"),
                                    limit=args.limit)
        print(st)


def phase_sample(args):
    per = args.n // max(1, len(args.tasks.split(",")))
    rows = []
    for task in args.tasks.split(","):
        task = task.strip()
        with open(os.path.join(OUT, f"contexts_{task}.jsonl")) as f:
            ctxs = [json.loads(l) for l in f if l.strip()]
        ctxs.sort(key=lambda c: _hash_rank(c["key"]))
        rows += ctxs[:per]
    rows += anch.anchor_contexts()
    rng = random.Random(0)
    rng.shuffle(rows)
    out = os.path.join(OUT, args.out_name)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} items ({len(anch.anchor_keys())} anchors, blinded) -> {out}")
    if args.split_payloads:
        n = args.split_payloads
        chunks = [rows[i::n] for i in range(n)]
        for i, ch in enumerate(chunks):
            p = os.path.join(OUT, args.out_name.replace(".jsonl", f"_payload{i}.json"))
            json.dump(ch, open(p, "w"))
            print(f"  payload {i}: {len(ch)} items -> {p}")


def phase_glm(args):
    with open(args.contexts) as f:
        ctxs = [json.loads(l) for l in f if l.strip()]
    if args.limit:
        ctxs = ctxs[:args.limit]
    ex = GLMExtractor(model=args.model)
    st = ex.run(ctxs, args.out, flush_every=args.flush_every)
    print(st)
    _anchor_report(args.out)


def _anchor_report(path):
    recs = load_extractions(path, only_ok=False)
    ak = anch.anchor_keys() & set(recs)
    if ak:
        rep = anch.score_anchor_batch({k: recs[k] for k in ak})
        print(f"anchors: pass_rate={rep['pass_rate']:.2f} "
              f"fails={[r['key'] for r in rep['results'] if not r['pass']]}")


def phase_ingest(args):
    """Validate subagent-produced extraction rows against stored doc_text and write a clean
    JSONL (same mechanical checks as the GLM path — no trust in any annotator)."""
    ctx = {}
    for p in glob.glob(os.path.join(OUT, "contexts_*.jsonl")):
        with open(p) as f:
            for l in f:
                if l.strip():
                    c = json.loads(l)
                    ctx[c["key"]] = c
    for a in anch.anchor_contexts():
        ctx[a["key"]] = a
    n_ok = n_rej = 0
    with open(args.out, "w") as out:
        for p in args.inputs.split(","):
            with open(p.strip()) as f:
                for line in f:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    c = ctx.get(r.get("key"))
                    if c is None:
                        continue
                    d = {k: r.get(k) for k in
                         ("found", "quote", "key_terms", "head_term", "named_in_source")}
                    reason = validate(d, c["doc_text"])
                    rec = finalize(c, d, reason, args.model_name)
                    n_ok += rec["status"] == "ok"
                    n_rej += rec["status"] != "ok"
                    out.write(json.dumps(rec) + "\n")
    print(f"ingested ok={n_ok} rejected={n_rej} -> {args.out}")
    _anchor_report(args.out)


def phase_partition(args):
    for task in args.tasks.split(","):
        task = task.strip()
        rep = audit.repaired_partition(task)
        p = os.path.join(OUT, f"partition_{task}.json")
        json.dump(rep["partition"], open(p, "w"))
        meta = {k: v for k, v in rep.items() if k != "partition"}
        meta["fresh_judge_candidates"] = meta["fresh_judge_candidates"][:20000]
        json.dump(meta, open(os.path.join(OUT, f"partition_{task}_meta.json"), "w"))
        print(f"{task}: trusted={rep['n_trusted_edges']} t3(untrusted)={rep['n_untrusted_t3']} "
              f"adopt_v2={rep['adopt_v2_applied']} tail_adopt={rep['tail_adoptions']} "
              f"merge_cands={len(rep['merge_candidates'])} "
              f"clusters {rep['n_clusters_base']}->{rep['n_clusters_repaired']} -> {p}")


def phase_census(args):
    for task in args.tasks.split(","):
        task = task.strip()
        ex = load_extractions(os.path.join(OUT, f"extract_{task}_{args.model_name}.jsonl"),
                              only_ok=False)
        part = json.load(open(os.path.join(OUT, f"partition_{task}.json")))
        rep = {"concept_level": census.task_census(ex, part, args.min_sources),
               "strata": census.strata_decomposition(ex, part)}
        up = os.path.join(OUT, f"upmap_{task}.json")
        if os.path.exists(up):
            rep["r3_subtree"] = census.subtree_census(ex, part, json.load(open(up)),
                                                      args.min_sources)
        o = os.path.join(OUT, f"census_{task}.json")
        json.dump(rep, open(o, "w"), indent=1)
        c = rep["concept_level"]
        print(f"{task}: concepts={c['n_concepts']} multi={c['n_concepts_multi_source']} "
              f"synonymy={c['synonymy_rate']} unnamed={c['mean_unnamed_rate']} -> {o}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="phase", required=True)

    b = sub.add_parser("build")
    b.add_argument("--tasks", required=True)
    b.add_argument("--limit", type=int, default=0)

    s = sub.add_parser("sample")
    s.add_argument("--tasks", required=True)
    s.add_argument("--n", type=int, default=150)
    s.add_argument("--out-name", default="equiv_sample.jsonl")
    s.add_argument("--split-payloads", type=int, default=0)

    g = sub.add_parser("glm")
    g.add_argument("--contexts", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--model", default="glm-4.7")
    g.add_argument("--limit", type=int, default=0)
    g.add_argument("--flush-every", type=int, default=50)

    i = sub.add_parser("ingest")
    i.add_argument("--inputs", required=True, help="comma-sep subagent jsonl paths")
    i.add_argument("--out", required=True)
    i.add_argument("--model-name", default="sonnet")

    pa = sub.add_parser("partition")
    pa.add_argument("--tasks", required=True)

    c = sub.add_parser("census")
    c.add_argument("--tasks", required=True)
    c.add_argument("--model-name", default="glm-4.7")
    c.add_argument("--min-sources", type=int, default=2)

    args = p.parse_args()
    {"build": phase_build, "sample": phase_sample, "glm": phase_glm, "ingest": phase_ingest,
     "partition": phase_partition, "census": phase_census}[args.phase](args)


if __name__ == "__main__":
    main()
