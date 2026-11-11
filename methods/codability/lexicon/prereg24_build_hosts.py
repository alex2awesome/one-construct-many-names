#!/usr/bin/env python
"""PREREG-24 stage 2: assemble the real host-page pool.

Hosts are untouched census pages, drawn from the same `contexts_*.jsonl` doc_text the
census itself was built over -- so the host population IS the census population, and
planted-rule recall is directly comparable to the .28 panel-union benchmark.

Hosts keep their pre-existing criteria. That density is the entire point: the 2026-07-24
and 2026-07-26 designs planted 2 targets in 309- and 516-word cleaned pages, where recall
saturated at 1.00; real census pages run ~1,283 words with many competing criteria, where
recall is .28.

**G7 (added 2026-07-27, before any judging).** Host text must be real parsed prose. An
audit of the Leg-3 harness found 2 of its 15 adjudicated pages were served as unparsed PDF
byte streams; both scored 100% recall on 2- and 3-unit denominators and inflated the
published figure from .276 to .293. Any page failing the cleanliness screen here is
dropped before it can do that again.

Note on formatting: this cache stores page text without paragraph breaks. That is fine and
is NOT a realism problem, because the G1 realism gate compares spliced pages against
untouched controls drawn from this same pool -- any format artifact is shared by both arms
and cannot discriminate them. Splices go in at sentence boundaries.

Out: outputs/lexicon/extraction_validity_20260727/p24_host_pool.json
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import statistics as st

ROOT = "ROOT"
OUT = f"{ROOT}/outputs/lexicon/extraction_validity_20260727"
LEX = f"{ROOT}/outputs/lexicon"

TASKS = ["code-review", "humor", "creative-writing", "peer-review", "grant-funding"]
MIN_WORDS, MAX_WORDS = 800, 4000
MIN_SENTENCES = 12          # need interior boundaries to splice at
MAX_NONASCII = .02          # G7 cleanliness
MIN_ALPHA = .70             # G7: real prose is mostly letters and spaces
N_HOSTS, N_CONTROLS = 140, 140
SEED = 20260727

SENT = re.compile(r"(?<=[.!?])\s+")


def clean_enough(t: str) -> tuple[bool, str]:
    """G7. Returns (ok, reason-if-not)."""
    head = t.lstrip()[:8]
    if head.startswith("%PDF") or head.startswith("\x89PNG") or head.startswith("PK\x03"):
        return False, "binary_container"
    s = t[:6000]
    if sum(1 for c in s if ord(c) > 127) / max(len(s), 1) > MAX_NONASCII:
        return False, "nonascii"
    if sum(1 for c in s if c.isalpha() or c.isspace()) / max(len(s), 1) < MIN_ALPHA:
        return False, "low_alpha"
    if len(SENT.split(t)) < MIN_SENTENCES:
        return False, "too_few_sentences"
    return True, ""


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    rng = random.Random(SEED)

    seen, pool, rejected = set(), [], {}
    for task in TASKS:
        path = f"{LEX}/contexts_{task}.jsonl"
        if not os.path.exists(path):
            continue
        for line in open(path):
            r = json.loads(line)
            doc, txt = r.get("doc"), r.get("doc_text") or ""
            if not doc or (task, doc) in seen:
                continue
            seen.add((task, doc))
            w = len(txt.split())
            if not (MIN_WORDS <= w <= MAX_WORDS):
                rejected["length"] = rejected.get("length", 0) + 1
                continue
            ok, why = clean_enough(txt)
            if not ok:
                rejected[why] = rejected.get(why, 0) + 1
                continue
            pool.append({"task": task, "doc": doc, "n_words": w,
                         "n_sentences": len(SENT.split(txt)),
                         "sha256": hashlib.sha256(txt.encode()).hexdigest()[:16]})
    print(f"unique docs seen: {len(seen):,}")
    print(f"rejected: {rejected}")
    print(f"eligible hosts: {len(pool):,}")
    by: dict[str, list] = {}
    for r in pool:
        by.setdefault(r["task"], []).append(r)
    for t in sorted(by):
        print(f"  {t:18s} {len(by[t]):,}")

    need = N_HOSTS + N_CONTROLS
    per = need // len(by)
    picked, used = [], set()
    for t in sorted(by):
        c = by[t][:]
        rng.shuffle(c)
        for r in c[:per]:
            picked.append(r)
            used.add((r["task"], r["doc"]))
    rng.shuffle(pool)
    for r in pool:
        if len(picked) >= need:
            break
        if (r["task"], r["doc"]) not in used:
            picked.append(r)
            used.add((r["task"], r["doc"]))
    rng.shuffle(picked)
    hosts, controls = picked[:N_HOSTS], picked[N_HOSTS:need]

    for nm, g in [("hosts", hosts), ("controls", controls)]:
        W = [r["n_words"] for r in g]
        print(f"{nm}: n={len(g)} median_words={st.median(W):.0f} min={min(W)} max={max(W)}")

    p = f"{OUT}/p24_host_pool.json"
    json.dump({"note": "Untouched real census pages (contexts_*.jsonl doc_text). "
                       "Pre-existing criteria LEFT IN PLACE -- they are the haystack. "
                       "G7 cleanliness screen applied; see rejected counts.",
               "params": {"tasks": TASKS, "min_words": MIN_WORDS, "max_words": MAX_WORDS,
                          "min_sentences": MIN_SENTENCES, "max_nonascii": MAX_NONASCII,
                          "min_alpha": MIN_ALPHA, "seed": SEED},
               "g7_rejected": rejected,
               "hosts": [dict(r, host_id=f"p24h{i:03d}") for i, r in enumerate(hosts)],
               "controls": [dict(r, control_id=f"p24k{i:03d}")
                            for i, r in enumerate(controls)]},
              open(p, "w"), indent=1)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
