#!/usr/bin/env python3
"""W3: codification-rung coding of SOURCES from text (GLM-4.7 batch, zai backend).

Why judge-coded and not orientation-mapped: the 2026-07-20 validation audit (311 stratified
docs, 2 Sonnet judges, blinded anchors 6/6) showed scrape-time `orientation` tags encode
GENRE not AUTHORITY — judged-vs-mapped agreement .60 exact / .69 3-class, failures
concentrated in formal_guideline/stylebook/course_syllabus. So every source gets its rung
from its own text. Prompt is IDENTICAL to the Sonnet validation judges for comparability.

Calibration gate (run mode=calibrate FIRST): GLM re-codes the same 311 docs; ship full-pool
coding only if GLM-vs-Sonnet 3-class agreement >= .80. Full pool: one row per doc per task,
resume-safe, excerpt 1600 chars.

Outputs: outputs/lexicon/provenance_rungs_<task>.jsonl ; calibration report to stdout.
"""
import argparse
import ast
import json
import os
import re

ROOT = "ROOT"
LEX = f"{ROOT}/outputs/lexicon"
SP = ("/private/tmp/claude-502/-Users-anon-Projects-stanford-research-norm-research/"
      "6eb8228d-7011-4800-87e2-61a172f6003c/scratchpad")
TASKS = ["code-review", "creative-writing", "grant-funding", "humor",
         "legal-outcome-prediction", "math-stackexchange", "news-homepages",
         "notice-and-comment", "patents", "peer-review", "press-releases"]

SYSTEM = (
    "You are a careful annotator for a study of how communities codify evaluation criteria. "
    "You are given the opening text of a web source document. Classify the source into ONE "
    "codification rung from its excerpt alone:\n"
    "1 = CODIFIED/OFFICIAL - issued by a governing authority and binding: statutes, "
    "regulations, federal notices, official agency guidelines, patent-office manuals, formal "
    "contest rules.\n"
    "2 = PROFESSIONAL STANDARD - authoritative within a profession but not legally binding: "
    "style guides/stylebooks, professional-body standards, organizational engineering "
    "guidelines, journal reviewer guidelines issued by the journal.\n"
    "3 = ACADEMIC - research articles, preprint/abstract pages, academic course pages, "
    "textbooks, scholarly dataset descriptions.\n"
    "4 = PRACTITIONER/SECONDARY - how-to guides, tutorials, trade-press/news articles about "
    "the practice, encyclopedia or wiki articles.\n"
    "5 = FOLK/INFORMAL - personal blogs, forum threads (Reddit/HN etc.), social-media "
    "aggregations, informal community advice.\n"
    "Judge from the TEXT content and voice, not from any URL fragments that happen to appear. "
    "If an excerpt is degenerate (pure navigation junk, raw JSON), still pick the most "
    "probable rung.\n"
    'Reply with STRICT JSON only: {"rung": 1|2|3|4|5}'
)

C3 = {1: "inst", 2: "inst", 3: "acad", 4: "informal", 5: "informal"}


def _backend():
    from methods.metric_implementer import backends as _b, config as _c
    return _b.LLMBackend("glm-4.7", "lexicon_rung_coder", _c.ImplementerConfig(backend="zai_anthropic"))


def _excerpt(txt):
    return re.sub(r"\s+", " ", (txt or ""))[:1600]


def _parse(o):
    m = re.search(r"\{.*\}", o or "", re.S)
    if not m:
        return None
    try:
        r = int(json.loads(m.group(0)).get("rung"))
        return r if 1 <= r <= 5 else None
    except Exception:
        return None


def _run_batch(be, items, out_path, flush=200):
    done = set()
    if os.path.exists(out_path):
        # null-rung rows are retried on resume (hybrid-thinking empty-content trap)
        done = {r["id"] for r in map(json.loads, open(out_path)) if r["rung"] is not None}
    todo = [it for it in items if it["id"] not in done]
    out = open(out_path, "a")
    for lo in range(0, len(todo), flush):
        chunk = todo[lo:lo + flush]
        prompts = [f"SOURCE DOCUMENT (opening text):\n{it['excerpt']}" for it in chunk]
        replies = be.generate_batch(prompts, system=SYSTEM, max_tokens=300, temperature=0.0, seed=0)
        retry = [i for i, o in enumerate(replies) if _parse(o) is None]
        if retry:
            r2 = be.generate_batch([prompts[i] for i in retry], system=SYSTEM,
                                   max_tokens=300, temperature=0.5, seed=1)
            for i, o in zip(retry, r2):
                replies[i] = o
        for it, o in zip(chunk, replies):
            r = _parse(o)
            out.write(json.dumps({"id": it["id"], "rung": r, **{k: it[k] for k in it if k not in ("id", "excerpt")}}) + "\n")
        out.flush()
        print(f"  {min(lo + flush, len(todo))}/{len(todo)}", flush=True)
    out.close()


def cmd_calibrate(_a):
    key = json.load(open(f"{SP}/w3_val_key.json"))
    idmap = key["idmap"]
    docs = json.load(open(f"{SP}/w3_val_sample.json"))
    bydoc = {d["doc"]: d for d in docs}
    sonnet = {}
    for sh in ["00", "01"]:
        for l in open(f"{SP}/w3_val_out_{sh}.jsonl"):
            r = json.loads(l)
            sonnet[r["id"]] = r["rung"]
    items = [{"id": fid, "excerpt": bydoc[m["doc"]]["excerpt"][:1600]}
             for fid, m in idmap.items() if m["doc"] in bydoc]
    _run_batch(_backend(), items, f"{SP}/w3_glm_calib.jsonl")
    glm = {json.loads(l)["id"]: json.loads(l)["rung"] for l in open(f"{SP}/w3_glm_calib.jsonl")}
    n = ex = c3 = 0
    for fid, sr in sonnet.items():
        gr = glm.get(fid)
        if fid not in idmap or gr is None:
            continue
        n += 1
        ex += gr == sr
        c3 += C3[gr] == C3[sr]
    print(f"GLM-vs-Sonnet on validation sample: n={n} exact {ex/n:.3f} 3-class {c3/n:.3f} "
          f"(GATE: 3-class >= .80 to ship full pool)")


def cmd_pool(a):
    be = _backend()
    for task in (a.tasks.split(",") if a.tasks else TASKS):
        seen = {}
        for line in open(f"{LEX}/contexts_{task}.jsonl"):
            r = json.loads(line)
            if r["doc"] in seen:
                continue
            s = r.get("strata")
            s = ast.literal_eval(s) if isinstance(s, str) else (s or {})
            seen[r["doc"]] = {"id": r["doc"], "task": task, "source_id": r.get("source_id", ""),
                              "orientation": s.get("orientation", "?"),
                              "excerpt": _excerpt(r.get("doc_text"))}
        items = sorted(seen.values(), key=lambda x: x["id"])
        print(f"{task}: {len(items)} docs")
        _run_batch(be, items, f"{LEX}/provenance_rungs_{task}.jsonl")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("calibrate")
    pp = sub.add_parser("pool")
    pp.add_argument("--tasks", default="")
    args = p.parse_args()
    {"calibrate": cmd_calibrate, "pool": cmd_pool}[args.cmd](args)
