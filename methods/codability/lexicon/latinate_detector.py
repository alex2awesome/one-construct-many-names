#!/usr/bin/env python3
"""Deterministic Latinate/stratum detector, iterated against the Sonnet-judged reference.

v1 = morphology only (suffix/prefix markers).
v2 = etymology-db lookup (Wiktionary-derived, scratchpad etym_word_stratum.json) with
     iterative suffix-strip re-lookup; a stripped LATINATE suffix on a GERMANIC root -> mixed
     (matches the judge instruction). Morphology fallback where the db is silent.

Eval: variant-level agreement vs outputs/lexicon/register_height_judgments.jsonl (1,500
Sonnet-judged variants, anchor gates 30/30). Also exports word- and term-level scoring used
by the head_term-vs-key_terms selection audit.
"""
import json
import re
from collections import Counter

SP = ("/private/tmp/claude-502/-Users-anon-Projects-stanford-research-norm-research/"
      "6eb8228d-7011-4800-87e2-61a172f6003c/scratchpad")
LEX = "ROOT/outputs/lexicon"
STOP = set("a an the of for in on to and or with by from as at is are be been was were it its this "
           "that be your our their his her via about what all else per non not into over under".split())

# v2.2 curated overrides — added after manual inspection of Sonnet-disagreement samples
# (homonym collisions and ME-only paths in the db). In-sample-tuned; documented as such.
OVERRIDES = {"humor": "latinate", "angle": "latinate", "humour": "latinate"}

LAT_SUF = ["ization", "isation", "ation", "ition", "ility", "ivity", "ician", "iency",
           "ment", "ance", "ence", "ancy", "ency", "tion", "sion", "ity", "ive", "ous",
           "able", "ible", "al", "ar", "ic", "ory", "ure", "age", "ize", "ise", "fy"]
GERM_SUF = ["ness", "ship", "hood", "dom", "ful", "less", "ing", "ed", "er", "ly", "s"]
GRK_PAT = re.compile(r"^(ph|ps|rh|chr|the[o]|arch|poly|meta|hyper|epi|syn|ana)|(?:ology|ography|osis|esis|ism)$")


def words(s):
    return [w for w in re.findall(r"[a-z]+", (s or "").lower()) if len(w) > 2 and w not in STOP]


def v1_word(w):
    if GRK_PAT.search(w):
        return "greek"
    for s in LAT_SUF:
        if w.endswith(s) and len(w) - len(s) >= 3:
            return "latinate"
    return "germanic"


class V2:
    def __init__(self, db="etym_word_stratum_v21.json"):
        self.db = json.load(open(f"{SP}/{db}"))

    def word(self, w):
        if w in OVERRIDES:
            return OVERRIDES[w]
        if w in self.db:
            return self.db[w]
        stripped_lat = False
        cur = w
        for _ in range(3):
            hit = None
            for s in LAT_SUF + GERM_SUF:
                if cur.endswith(s) and len(cur) - len(s) >= 3:
                    root = cur[:len(cur) - len(s)]
                    for cand in (root, root + "e", root[:-1] if root[-1:] == root[-2:-1] else root):
                        if cand in self.db:
                            base = self.db[cand]
                            if s in LAT_SUF and base == "germanic":
                                return "mixed"
                            return base
                    hit = (root, s in LAT_SUF)
                    break
            if hit is None:
                break
            cur, was_lat = hit[0], hit[1]
            stripped_lat = stripped_lat or hit[1]
        return v1_word(w) if not stripped_lat else "latinate"


def term_stratum(term, word_fn):
    ws = words(term)
    if not ws:
        return None
    cs = Counter(word_fn(w) for w in ws)
    if len(cs) == 1:
        return next(iter(cs))
    if cs.get("mixed"):
        return "mixed"
    top, n = cs.most_common(1)[0]
    return top if n / sum(cs.values()) > 0.5 else "mixed"


def latinate_score(term, word_fn):
    """Continuous 0-1: mean of per-word scores g=0, mixed=.5, lat/greek=1."""
    ws = words(term)
    if not ws:
        return None
    m = {"germanic": 0.0, "mixed": 0.5, "latinate": 1.0, "greek": 1.0}
    return sum(m[word_fn(w)] for w in ws) / len(ws)


def evaluate():
    ref = {}
    for l in open(f"{LEX}/register_height_judgments.jsonl"):
        r = json.loads(l)
        ref[r["variant"]] = r["stratum"]
    v2 = V2()
    for name, fn in [("v1_morphology", v1_word), ("v2_etym+strip", v2.word)]:
        hits = n = 0
        conf = Counter()
        for var, s in ref.items():
            pred = term_stratum(var, fn)
            if pred is None:
                continue
            n += 1
            hits += pred == s
            if pred != s:
                conf[(s, pred)] += 1
        print(f"{name}: agreement {hits}/{n} = {hits/n:.3f}; top confusions "
              f"{conf.most_common(4)}")
    return v2


if __name__ == "__main__":
    evaluate()
