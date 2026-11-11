"""Sub-community (dialect) contrast for the author-lexicon census — partition-agnostic.

Statistic (construct-matched, from outputs/lexicon/codability_audit/census_strata_buckets.py,
consolidated here 2026-07-09): unit = SOURCE (best-per-source record per construct); pair
statistic = Jaccard of canonicalized author key_terms; classes = same coarse sub-community
bucket vs different; paired within-construct delta; source->bucket permutation null.

★ MIRROR CONFOUND (2026-07-09 verification — always run the guard): sub-communities share
canonical texts mirrored across distinct URLs (SPJ/Reuters codes in journalism, Aristotle's
Poetics in CW drama, Lean/mathlib style guides in math). Mirrors are 62-88% same-bucket and
inflate within-bucket Jaccard 4-8x. Guard: drop within-construct pairs whose verbatim QUOTES
share >= `mirror` token-Jaccard (.5 = mirror guard; .3 = strict, aggressive — stopwords).
Post-guard verdicts (concept grain): humor/CW REAL, journalism/math DEAD (= shared-canonical-
text adoption, a different phenomenon than lexical dialect). URL-level independence is NOT
sufficient for any lexical-overlap statistic.

Rerun after any L0->R3 rebuild with the new partition:
  python -m methods.codability.lexicon.dialect <task> --partition <key->construct json> --arms
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT = os.path.join(ROOT, "outputs", "lexicon")

BUCKETS = {
    "humor": [
        ("junk_doc", ["wikipedia", "wiki ", "blocked", "catalog", "landing page", "metadata", "factbook"]),
        ("standup", ["stand-up", "standup", "open mic"]),
        ("improv", ["improv"]),
        ("sketch", ["sketch"]),
        ("sitcom_tv", ["sitcom", "tv comedy", "television", "late night", "snl"]),
        ("anekdoty", ["russian joke", "anekdot"]),
        ("satire_news", ["satir", "parody", "onion", "news headline", "editorial cartoon"]),
        ("cartoon_visual", ["cartoon", "comic strip", "meme", "visual humor", "caricature"]),
        ("screen_film", ["screenplay", "screenwriting", "film", "movie"]),
        ("speech_rhetoric", ["speech", "orator", "rhetoric", "court", "presentation", "toast", "debate"]),
        ("theory_academic", ["theory", "linguistic", "academic", "philosoph", "psycholog", "anthropolog",
                             "cognitive", "research"]),
        ("memoir_books", ["memoir", "book", "essay", "literary"]),
        ("joke_writing", ["joke", "one-liner", "pun", "wordplay", "comedy writing", "humor writing",
                          "writing humor", "writing comedy", "comic timing"]),
    ],
    "creative-writing": [
        ("junk_doc", ["wikipedia", "wiki ", "blocked", "catalog", "landing page", "metadata", "factbook",
                      "country profile"]),
        ("horror", ["horror", "gothic"]),
        ("mystery_thriller", ["myster", "thriller", "crime", "detective", "suspense"]),
        ("fantasy_sf", ["fantasy", "science fiction", "sci-fi", " sf", "speculative", "magic", "worldbuild"]),
        ("romance", ["romance", "romantic"]),
        ("historical", ["historical"]),
        ("poetry", ["poetry", "poem", "verse"]),
        ("drama_stage", ["play", "drama", "theatre", "theater", "tragedy", "tragic", "heroic"]),
        ("screenwriting", ["screenplay", "screenwriting", "film", "script"]),
        ("publishing_query", ["query", "agent", "submit", "publish", "pitch", "market"]),
        ("children_ya", ["children", "middle grade", "young adult", "picture book"]),
        ("fanfic_web", ["fanfic", "fan fiction", "web serial", "webnovel"]),
        ("memoir_essay", ["memoir", "essay", "personal narrative", "nonfiction"]),
        ("reading_review", ["review", "reading", "book club", "criticism"]),
        ("short_fiction", ["short stor", "flash fiction", "microfiction"]),
        ("novel", ["novel"]),
        ("craft_general", ["dialogue", "character", "plot", "pacing", "prose", "ensemble", "foreshadow",
                           "ending", "opening", "revision", "editing", "scene", "point of view", "structure",
                           "description", "voice", "writing fiction", "storytelling", "narrative"]),
    ],
    "news-homepages": [
        ("junk_doc", ["wikipedia", "wiki ", "blocked", "catalog", "landing page", "metadata",
                      "factbook", "esrb", "video game", "country profile"]),
        ("ethics_standards", ["ethic", "standard", "code of", "integrity", "accuracy", "corrections",
                              "editorial polic", "conduct", "impartial", "objectivity", "trust"]),
        ("identity_style", ["trans ", "trans people", "lgbtq", "diversity", "disability", "race",
                            "ethnic", "indigenous", "religion", "mental health", "suicide",
                            "immigrant", "gender", "inclusive", "bias-free", "style guide"]),
        ("solutions_constructive", ["solutions journalism", "solutions-focused", "constructive"]),
        ("newsworthiness", ["newsworth", "news value", "story selection", "news quality",
                            "pitch", "news judgment", "selecting"]),
        ("writing_craft", ["writing", "feature stor", "op-ed", "lede", "lead", "headline",
                           "explainer", "narrative", "longform", "structure", "interview"]),
        ("photo_visual", ["photo", "visual", "caption", "image"]),
        ("data_journalism", ["data journalism", "data_journalism", "data-driven"]),
        ("beats", ["science", "environment", "climate", "travel", "sports", "health", "business",
                   "crime", "politic", "education", "arts", "food", "obituar"]),
        ("tabloid_broadcast", ["tabloid", "broadcast", "radio", "podcast", "tv news"]),
    ],
    # v2 buckets (2026-07-09): v1 keywords missed the corpus's real communities — top label
    # "writing mathematical proofs" (n=98) fell to 'other'; 'other' was 61% of sources vs
    # 33-36% elsewhere (v1 delta +.0153 p=.049 -> v2 +.0354 p=.0025; disclose both).
    # First match wins: distinctive communities before broad ones.
    "math-stackexchange": [
        ("junk_doc", ["blocked", "catalog", "landing page", "metadata", "factbook"]),
        ("formalization", ["lean", "coq", "metamath", "mathlib", "macaulay", "formaliz", "agda",
                           "isabelle", "set.mm", "proof assistant"]),
        ("wiki_encyclopedic", ["wikipedia", "wiki", "nlab", "encyclopedi"]),
        ("latex_viz", ["latex", "tikz", "mathml", "typograph", "visualization", "tufte",
                       "typeset", "font", "diagram"]),
        ("research_pubs", ["research", "paper", "journal", "publish", "referee", "peer review",
                           "mathematical sciences", "thesis"]),
        ("exposition_popular", ["exposition", "expository", "popular", "beauty", "good mathematics",
                                "essay", "book review", "communicat"]),
        ("answering_qa", ["answer", "question", "stack exchange", "stackexchange", "mathoverflow",
                          "forum", "asking"]),
        ("problem_solving", ["problem", "olympiad", "competition", "contest", "puzzle"]),
        ("proof_writing", ["proof", "rigor"]),
        ("pedagogy", ["teach", "pedagog", "lecture", "course", "student", "homework", "tutor",
                      "textbook", "errata", "curriculum", "education", "undergraduate", "classroom"]),
        ("subfields", ["algebra", "analysis", "topology", "geometry", "number theory", "combinator",
                       "probability", "statistic", "calculus", "logic"]),
    ],
}

_W = re.compile(r"[a-z0-9']+")


def bucket_of(task: str, label: str) -> str:
    lab = " ".join((label or "").strip().lower().replace("_", " ").split())
    for name, kws in BUCKETS[task]:
        if any(kw in lab for kw in kws):
            return name
    return "other"


def canon(t: str) -> str:
    t = re.sub(r"[^a-z0-9 ]+", " ", (t or "").lower())
    t = re.sub(r"\s+", " ", t).strip()
    return " ".join(w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w
                    for w in t.split())


def qtok(q: str) -> set:
    return set(_W.findall((q or "").lower()))


def load_groups(task: str, partition_path: str, extract_path: str | None = None) -> dict:
    """construct -> source -> record (best per source, first wins; ok+found only)."""
    part = json.load(open(partition_path))
    groups: dict = defaultdict(dict)
    for l in open(extract_path or os.path.join(OUT, f"extract_{task}_glm-4.7.jsonl")):
        if not l.strip():
            continue
        r = json.loads(l)
        if r.get("status") != "ok" or not r.get("found"):
            continue
        cid = part.get(r["key"])
        if cid is None:
            continue
        groups[cid].setdefault(r.get("source_id") or r["key"], r)
    return groups


def dialect_contrast(task: str, groups: dict, mirror: float | None = None,
                     drop_junk: bool = False, B: int = 1000, seed: int = 0,
                     examples: int = 0) -> dict | None:
    rng = np.random.default_rng(seed)
    per, perm = [], np.zeros(B)
    npw = npc = ndrop = 0
    bkt_counts: Counter = Counter()
    ex = []
    for cid, by_src in groups.items():
        rows = list(by_src.values())
        if drop_junk:
            rows = [r for r in rows if bucket_of(
                task, (r.get("strata") or {}).get("subtask_short") or "") != "junk_doc"]
        n = len(rows)
        if n < 2:
            continue
        tsets = [{canon(t) for t in (r.get("key_terms") or []) if canon(t)} for r in rows]
        qsets = [qtok(r.get("quote")) for r in rows]
        bks = np.array([bucket_of(task, (r.get("strata") or {}).get("subtask_short") or "")
                        for r in rows], dtype=object)
        bkt_counts.update(bks.tolist())
        iu = np.triu_indices(n, k=1)
        jac = np.zeros(len(iu[0]))
        keep = np.zeros(len(iu[0]), bool)
        for p, (i, j) in enumerate(zip(*iu)):
            a, b = tsets[i], tsets[j]
            if not (a or b):
                continue
            if mirror is not None:
                qa, qb = qsets[i], qsets[j]
                if qa and qb and len(qa & qb) / len(qa | qb) >= mirror:
                    ndrop += 1
                    continue
            keep[p] = True
            jac[p] = len(a & b) / len(a | b)
        eq = (bks[:, None] == bks[None, :])[iu]
        w, c = keep & eq, keep & ~eq
        if w.any() and c.any():
            per.append((float(jac[w].mean()), float(jac[c].mean())))
            npw += int(w.sum()); npc += int(c.sum())
            if examples and jac[w].max() > 0.3:
                hi = int(np.argmax(np.where(w, jac, -1)))
                i, j = iu[0][hi], iu[1][hi]
                ex.append((cid, str(bks[i]), round(float(jac[hi]), 2),
                           sorted(tsets[i] & tsets[j])[:6]))
            for bi in range(B):
                pb = bks[rng.permutation(n)]
                peq = (pb[:, None] == pb[None, :])[iu]
                pw, pc = keep & peq, keep & ~peq
                if pw.any() and pc.any():
                    perm[bi] += jac[pw].mean() - jac[pc].mean()
    if not per:
        return None
    wm = float(np.mean([x[0] for x in per]))
    cm = float(np.mean([x[1] for x in per]))
    obs = sum(x[0] - x[1] for x in per)
    return dict(n_constructs=len(per), pairs_w=npw, pairs_c=npc, dropped_pairs=ndrop,
                within=round(wm, 4), cross=round(cm, 4), delta=round(wm - cm, 4),
                p=round(float(np.mean(perm >= obs)), 4),
                bucket_sizes=dict(bkt_counts.most_common()),
                examples=sorted(ex, key=lambda e: -e[2])[:examples])


ARMS = [("baseline", {}), ("mirror>=.5", {"mirror": .5}), ("mirror>=.3", {"mirror": .3}),
        ("junk-dropped", {"drop_junk": True}), ("mirror.5+junk", {"mirror": .5, "drop_junk": True})]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks", help="comma-sep task names (must have BUCKETS entries)")
    ap.add_argument("--partition", default=None,
                    help="key->construct json; default outputs/lexicon/partition_<task>.json")
    ap.add_argument("--r1", action="store_true",
                    help="use outputs/lexicon/codability/partition_key2R1_<task>.json")
    ap.add_argument("--arms", action="store_true", help="run all guard arms (default: baseline+mirror.5)")
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--examples", type=int, default=3)
    args = ap.parse_args()
    for task in args.tasks.split(","):
        task = task.strip()
        pp = args.partition or (
            os.path.join(OUT, "codability", f"partition_key2R1_{task}.json") if args.r1
            else os.path.join(OUT, f"partition_{task}.json"))
        groups = load_groups(task, pp)
        print(f"\n===== {task}  partition={os.path.basename(pp)} =====")
        arms = ARMS if args.arms else ARMS[:2]
        for name, kw in arms:
            r = dialect_contrast(task, groups, B=args.B,
                                 examples=(args.examples if name == "baseline" else 0), **kw)
            if r is None:
                print(f"  {name:14s} -> no constructs with both pair classes")
                continue
            exs = r.pop("examples"); r.pop("bucket_sizes")
            print(f"  {name:14s} n={r['n_constructs']:3d} w/c {r['pairs_w']}/{r['pairs_c']} "
                  f"drop {r['dropped_pairs']:4d}  within {r['within']:.4f} cross {r['cross']:.4f} "
                  f"DELTA {r['delta']:+.4f}  p={r['p']}")
            for e in exs:
                print(f"      ex {e[0]} [{e[1]}] jac={e[2]} shared={e[3]}")


if __name__ == "__main__":
    main()
