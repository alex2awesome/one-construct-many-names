#!/usr/bin/env python
"""PREREG-24 stage 1: generate CANDIDATE formal/casual rule pairs from AIRules.

Candidate generation only. Every measurement in PREREG-24 -- same-requirement
confirmation (G3), formality rating (G2), non-overlap (G4), recall matching (G5) -- is
done by an anchor-gated LLM judge downstream. Nothing here is a measurement; embeddings
and the register heuristic exist solely to hand the judge a pool worth judging.

What we want is a pair of rules from DIFFERENT subreddits that state the SAME requirement
in DIFFERENT words at DIFFERENT register. So the candidate signal is deliberately
two-sided: high semantic similarity AND low lexical overlap. Ranking by similarity alone
would return paraphrase-free near-duplicates, which are the same register by construction
-- exactly the wrong pool.

Pairs are then deduplicated by construct, because an undeduplicated pool is ~40% variants
of "no reposts" and the 140 "independent" pairs of the sign test would not be independent.

Out: outputs/lexicon/extraction_validity_20260727/p24_candidate_pairs.json
"""
from __future__ import annotations

import gzip
import json
import os
import re

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

ROOT = "ROOT"
OUT = f"{ROOT}/outputs/lexicon/extraction_validity_20260727"
AIRULES = f"{ROOT}/datasets/prior_norms/airules_frame.jsonl.gz"

MIN_W, MAX_W = 8, 40          # plantable as one sentence
COS_LO = .82                  # same requirement, probably
JAC_HI = .35                  # but not the same wording
N_CANDIDATES = 700            # judge pool; PREREG-24 needs 140 survivors
SIM_DEDUP = .88               # construct-level dedup threshold

# Quality-stratum restriction (PREREG-24 G6). The production extractor targets QUALITY
# criteria and recovers only .05 of governance gold, so planting governance rules would
# park both arms on the floor -- the mirror image of the 2026-07-24 ceiling. Unrestricted
# mining returns ~16% governance against ~6% quality (flair, NSFW tags, reposts, spam
# dominate), so the pool has to be cut to the quality stratum before pairing. This is a
# heuristic PRE-FILTER on what the judge sees; G3/G2 still decide what survives.
QUAL = re.compile(
    r"\b(quality|clear|clearly|clarity|detail|detailed|specific|explain|explanation|"
    r"effort|low.effort|constructive|helpful|accurate|accuracy|evidence|source|cite|"
    r"citation|proofread|grammar|spelling|formatting|readable|legible|concise|thorough|"
    r"context|justify|reasoning|critique|feedback|well.written|substantive|descriptive|"
    r"informative)\b", re.I)
GOV = re.compile(
    r"\b(flair|tag|nsfw|nsfl|spoiler|repost|spam|self.promo|promotion|advertis|ban|banned|"
    r"karma|account age|sidebar|modmail|report button|downvote|upvote|crosspost|"
    r"mega.?thread|discord|survey)\b", re.I)


def is_quality_stratum(t: str) -> bool:
    return bool(QUAL.search(t)) and not GOV.search(t)


# Crude register proxy. RANKING ONLY -- it decides which pairs the judge looks at, never
# which pairs survive. G2 (judged formality, gap >= 1.5) is the actual gate.
CASUAL = re.compile(r"\b(you|your|don't|doesn't|isn't|won't|can't|we're|it's|stuff|kinda|"
                    r"gonna|please|hey|yeah|ok|okay|guys|folks|just|pretty|really|"
                    r"basically|plz|pls)\b|[!?]{1,}", re.I)
FORMAL = re.compile(r"\b(must|shall|submissions?|prohibited|permitted|required|"
                    r"applicable|pertaining|constitute|adhere|comply|compliance|"
                    r"appropriate|content|material|participants?|contributors?|"
                    r"violation|subject to|thereof|herein)\b", re.I)


def register_proxy(t: str) -> float:
    """Positive = looks formal, negative = looks casual. Ranking heuristic only."""
    n = max(len(t.split()), 1)
    return (len(FORMAL.findall(t)) - len(CASUAL.findall(t))) / n * 10


def load_rules() -> list[dict]:
    """One plantable sentence per rule: prefer the short name, fall back to first desc line."""
    rows, seen = [], set()
    with gzip.open(AIRULES, "rt") as f:
        for line in f:
            r = json.loads(line)
            sub, topic = r.get("name", ""), r.get("topic", "")
            for rule in r.get("rules") or []:
                for field in ("sn", "desc"):
                    t = (rule.get(field) or "").strip()
                    t = re.sub(r"\[Read more\]\(\S+\)", "", t)
                    t = re.sub(r"https?://\S+", "", t)
                    t = re.sub(r"\s+", " ", t).strip()
                    if not t:
                        continue
                    t = re.split(r"(?<=[.!?]) ", t)[0].strip()   # first sentence
                    if not (MIN_W <= len(t.split()) <= MAX_W):
                        continue
                    if not is_quality_stratum(t):
                        continue
                    k = t.lower()
                    if k in seen:
                        continue
                    seen.add(k)
                    rows.append({"sub": sub, "topic": topic, "text": t,
                                 "reg": register_proxy(t)})
                    break
    return rows


def jaccard(a: str, b: str) -> float:
    A = set(re.findall(r"[a-z']+", a.lower()))
    B = set(re.findall(r"[a-z']+", b.lower()))
    return len(A & B) / max(len(A | B), 1)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    rules = load_rules()
    print(f"plantable rule sentences: {len(rules):,}")

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer("BAAI/bge-base-en-v1.5", device=dev)
    texts = [r["text"] for r in rules]
    emb = model.encode(texts, batch_size=256, convert_to_numpy=True,
                       normalize_embeddings=True, show_progress_bar=True)
    E = torch.tensor(emb, device=dev)

    # Chunked top-k. We keep several neighbours per rule because the first neighbour is
    # usually a near-verbatim copy from a sister subreddit -- high cosine, high Jaccard,
    # useless here.
    cands, CH = [], 2048
    for s in range(0, len(rules), CH):
        sims = (E[s:s + CH] @ E.T)
        sims[torch.arange(sims.shape[0]), torch.arange(s, min(s + CH, len(rules)))] = -1
        top = torch.topk(sims, k=12, dim=1)
        for i, (vals, idxs) in enumerate(zip(top.values.tolist(), top.indices.tolist())):
            a = rules[s + i]
            for v, j in zip(vals, idxs):
                if v < COS_LO or j <= s + i:
                    continue
                b = rules[j]
                if a["sub"] == b["sub"]:
                    continue
                if jaccard(a["text"], b["text"]) > JAC_HI:
                    continue
                hi, lo = (a, b) if a["reg"] >= b["reg"] else (b, a)
                gap = hi["reg"] - lo["reg"]
                if gap <= 0:
                    continue
                cands.append({"cos": v, "reg_gap": gap, "formal_cand": hi,
                              "casual_cand": lo, "idx_formal": (s + i) if hi is a else j})
        print(f"  scanned {min(s + CH, len(rules)):,}/{len(rules):,}  cands={len(cands):,}",
              end="\r")
    print()
    cands.sort(key=lambda c: -c["reg_gap"])
    print(f"raw candidates: {len(cands):,}")

    # Construct-level dedup: greedily keep a pair only if its formal member is not close
    # to an already-kept formal member. Without this the pool is dominated by one or two
    # very common rules and the pairs are not independent units.
    kept, kept_vecs = [], []
    for c in cands:
        v = E[c["idx_formal"]]
        if kept_vecs and float(torch.max(torch.stack(kept_vecs) @ v)) > SIM_DEDUP:
            continue
        kept.append(c)
        kept_vecs.append(v)
        if len(kept) >= N_CANDIDATES:
            break
    print(f"after construct dedup: {len(kept):,}")

    out = [{"pair_id": f"p24c{i:04d}",
            "cos": round(c["cos"], 4), "reg_proxy_gap": round(c["reg_gap"], 3),
            "formal_candidate": {k: c["formal_cand"][k] for k in ("sub", "topic", "text")},
            "casual_candidate": {k: c["casual_cand"][k] for k in ("sub", "topic", "text")}}
           for i, c in enumerate(kept)]
    p = f"{OUT}/p24_candidate_pairs.json"
    json.dump({"note": "CANDIDATES ONLY -- not measured. G2/G3 judging decides survival. "
                       "reg_proxy_gap is a ranking heuristic, never a gate.",
               "params": {"cos_lo": COS_LO, "jac_hi": JAC_HI, "sim_dedup": SIM_DEDUP,
                          "min_words": MIN_W, "max_words": MAX_W},
               "n_source_rules": len(rules), "n_candidates": len(out),
               "pairs": out}, open(p, "w"), indent=1)
    print(f"wrote {p}")
    for c in out[:4]:
        print(f"\n  cos={c['cos']}  F[{c['formal_candidate']['sub']}] "
              f"{c['formal_candidate']['text'][:95]}"
              f"\n            C[{c['casual_candidate']['sub']}] "
              f"{c['casual_candidate']['text'][:95]}")


if __name__ == "__main__":
    main()
