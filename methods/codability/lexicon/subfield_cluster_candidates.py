#!/usr/bin/env python3
"""W2b step 1: candidate pairs for subfield-label clustering. The raw `subtask_short`
annotation layer is 91-96% singletons (audited 2026-07-20) — an unbounded string space that
must be clustered before "how many subfields per task" is answerable. Retrieval here is a
SHORTLIST ONLY (repo rule): every merge decision is made by a Sonnet judge with blinded
anchors in a later batch, never by similarity thresholds.

Candidates = BGE cosine >= CUT within task (union exact-normalized duplicates, which merge
mechanically without a judge). Output: per-task candidate-pair file + a judge payload with
blinded anchors (known-same = exact-dup label pairs held out of mechanical merge;
known-different = cross-task donor pairs).

Outputs: outputs/lexicon/subfield_pairs_<task>.jsonl, scratchpad payloads for judging.
"""
import ast
import hashlib
import json
import random
import re
from collections import defaultdict

import numpy as np

ROOT = "ROOT"
LEX = f"{ROOT}/outputs/lexicon"
TASKS = ["code-review", "creative-writing", "grant-funding", "humor",
         "legal-outcome-prediction", "math-stackexchange", "news-homepages",
         "notice-and-comment", "patents", "peer-review", "press-releases"]
CUT = 0.80
MAX_PAIRS_PER_LABEL = 6


def norm_label(s):
    return re.sub(r"[_\s]+", " ", str(s or "").lower()).strip()


def load_labels(task):
    bydoc = {}
    for line in open(f"{LEX}/contexts_{task}.jsonl"):
        r = json.loads(line)
        s = r.get("strata")
        if isinstance(s, str):
            try:
                s = ast.literal_eval(s)
            except Exception:
                s = {}
        lab = norm_label((s or {}).get("subtask_short", ""))
        if lab:
            bydoc[r["doc"]] = lab
    counts = defaultdict(int)
    for lab in bydoc.values():
        counts[lab] += 1
    return dict(counts)


def main():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    rng = random.Random(0)
    all_labels = {t: load_labels(t) for t in TASKS}
    summary = {}
    for task in TASKS:
        labels = sorted(all_labels[task])
        emb = model.encode(labels, normalize_embeddings=True, batch_size=256,
                           show_progress_bar=False)
        sims = emb @ emb.T
        pairs = []
        for i in range(len(labels)):
            js = np.argsort(-sims[i])
            kept = 0
            for j in js[1:]:
                if sims[i][j] < CUT or kept >= MAX_PAIRS_PER_LABEL:
                    break
                if i < j:
                    pairs.append((labels[i], labels[int(j)], float(sims[i][int(j)])))
                    kept += 1
        with open(f"{LEX}/subfield_pairs_{task}.jsonl", "w") as f:
            for a, b, s in sorted(pairs, key=lambda p: -p[2]):
                f.write(json.dumps({"task": task, "a": a, "b": b, "cos": round(s, 4)}) + "\n")
        summary[task] = {"n_labels": len(labels), "n_pairs": len(pairs)}
        print(f"{task:26} labels={len(labels):5} candidate_pairs={len(pairs):6}")
    json.dump(summary, open(f"{LEX}/subfield_pairs_summary.json", "w"), indent=1)
    print("wrote per-task subfield_pairs_*.jsonl")


if __name__ == "__main__":
    main()
