#!/usr/bin/env python
"""PREREG-27 (third-domain name swap) materials. Domain = argv[1] ('math' or 'news').
Reuses prereg26_build.build_adj and the PREREG-10 formality protocol; writes OUT/jobs/{formality,adj,adj_v2,check}.json,
OUT/private/keys.json, OUT/forms_manifest.json, OUT/text_pool.jsonl, OUT/scaffolds.json.
Frozen designs: notes/2026-10-12__prereg27-math-name-swap.md (+ Amendment 1); news analogue in its own note.
"""
from __future__ import annotations

import glob
import gzip
import hashlib
import itertools
import json
import os
import random
import sys

ROOT = "ROOT"
sys.path.insert(0, f"{ROOT}/methods/codability/lexicon")
import prereg26_build as B  # noqa: E402

DOM = sys.argv[1]
P10 = f"{ROOT}/outputs/lexicon/register_corpus_20260723"
N_ADJ_CONSTRUCTS = 24
N_NAMES = 10


def h(s):
    return hashlib.sha256(s.encode()).hexdigest()


CFG = {
    "math": dict(
        out=f"{ROOT}/outputs/lexicon/prereg27_math_20261012", task="math-stackexchange", tag="p27-math",
        strict_who="mathematics Q&A communities (Math StackExchange, math forums, mathematics teachers) use to judge answers to mathematics questions",
        piece="answer", better="is it a better or worse answer by the community's standards?",
        S0=("Scoring task, no investigation. You are evaluating answers posted to a mathematics question-and-answer "
            "community. The criterion named below is one used by the mathematics Q&A community to judge such answers. "
            "No definition is provided --- apply the criterion as you understand it from its name. CRITERION NAME: "
            "{name}. For EVERY item, judge how well the item satisfies the criterion named above on a 1-7 scale (1 = "
            "clearly fails it, 7 = clearly satisfies it). Judge each item independently from its text alone."),
        S1=("This is a rating job; do not research anything. The texts below are answers posted to a mathematics "
            "question-and-answer community. People who write and read answers in mathematics Q&A communities use the "
            "criterion named below when they assess this kind of answer. You will not be given a definition; rely on "
            "your own understanding of what the name means. CRITERION: {name}. Rate every text on a 1 to 7 scale for "
            "how far it meets this criterion (1 = it plainly does not meet it, 7 = it plainly meets it). Rate each "
            "text on its own, using only its text."),
        S2=("Please score the answers listed below. Do not look anything up. Each answer was posted in response to a "
            "question in an online mathematics community, and the criterion given below is a standard that people in "
            "mathematics Q&A apply to such answers. Interpret the criterion from its name alone, since no definition "
            "is supplied. Criterion to apply: {name}. Give each answer a score from 1 to 7, where 7 means the answer "
            "clearly meets the criterion and 1 means it clearly does not. Score the answers independently of one "
            "another, based on the text alone."),
        same=[("Show all steps of the derivation", "Every step of the computation is written out"),
              ("Correct final answer", "The result stated at the end is right"),
              ("Use LaTeX/MathJax formatting", "Mathematical notation typeset with MathJax"),
              ("Define your notation", "All symbols are introduced before they are used"),
              ("Answer the question that was asked", "Responsiveness to the actual question"),
              ("Cite the theorem you use", "Name the result being applied"),
              ("Give a hint rather than a full solution for homework", "For homework questions, offer hints instead of complete solutions"),
              ("Rigorous proof", "The argument is logically airtight"),
              ("Concise answer", "Brevity (no unnecessary length)"),
              ("Include a worked example", "Illustrate with a concrete example"),
              ("Check edge cases", "Treat the boundary and degenerate cases"),
              ("Explain the intuition", "Convey the idea behind the argument")],
        diff=[("Use LaTeX/MathJax formatting", "Correct final answer"),
              ("Cite the theorem you use", "Polite tone toward the asker"),
              ("Concise answer", "Handles the case n = 0"),
              ("Explain the intuition", "Proper tagging of the question"),
              ("Include a worked example", "Spelling and grammar"),
              ("Rigorous proof", "Links to external software"),
              ("Define your notation", "Answer posted quickly"),
              ("Check edge cases", "Friendly welcome to new users"),
              ("Historical context of the theorem", "Correct arithmetic"),
              ("Use of diagrams", "Proof by induction is valid"),
              ("Answer the question that was asked", "Good use of bold headings"),
              ("Generalizes the result", "Code is runnable")],
        faults=["Circular reasoning", "Arithmetic errors", "Hand-waving instead of proof", "Unexplained leaps in logic",
                "Wrong final answer", "Answering a different question than the one asked",
                "Wall of unformatted equations", "Undefined notation", "Overcomplicated solution",
                "Copy-pasted answer without attribution", "Rude or condescending tone", "Missing cases in the proof"],
    ),
    "news": dict(
        out=f"{ROOT}/outputs/lexicon/prereg27_news_20261012", task="news-homepages", tag="p27-news",
        strict_who="journalism communities (newsrooms, editors, press critics, news readers) use to judge news stories",
        piece="news story", plural="news stories", better="is it a better or worse news story by the community's standards?",
        S0=("Scoring task, no investigation. You are evaluating news stories as they appear on news outlets' "
            "homepages (headline, with surrounding homepage context). The criterion named below is one used by the "
            "journalism community to judge such stories. No definition is provided --- apply the criterion as you "
            "understand it from its name. CRITERION NAME: {name}. For EVERY item, judge how well the item satisfies "
            "the criterion named above on a 1-7 scale (1 = clearly fails it, 7 = clearly satisfies it). Judge each "
            "item independently from its text alone."),
        S1=("This is a rating job; do not research anything. The texts below are news stories as shown on news "
            "outlets' homepages (headline plus surrounding homepage context). Journalists, editors and news readers use "
            "the criterion named below when they assess this kind of story. You will not be given a definition; rely "
            "on your own understanding of what the name means. CRITERION: {name}. Rate every text on a 1 to 7 scale "
            "for how far it meets this criterion (1 = it plainly does not meet it, 7 = it plainly meets it). Rate each "
            "text on its own, using only its text."),
        S2=("Please score the news stories listed below. Do not look anything up. Each story appeared on a news "
            "outlet's homepage (you see its headline and the surrounding homepage context), and the criterion given "
            "below is a standard that people in journalism apply to such stories. Interpret the criterion from its "
            "name alone, since no definition is supplied. Criterion to apply: {name}. Give each story a score from 1 "
            "to 7, where 7 means the story clearly meets the criterion and 1 means it clearly does not. Score the "
            "stories independently of one another, based on the text alone."),
        same=[("Lead with the most important fact", "Put the key news in the first sentence"),
              ("Attribute every claim to a source", "Clear sourcing of claims"),
              ("Accuracy of facts", "Factual correctness"),
              ("Balance: include both sides", "Present opposing viewpoints fairly"),
              ("Plain, jargon-free language", "Avoid technical jargon"),
              ("Answer who, what, when, where, why", "Covers the five Ws"),
              ("Headline reflects the story", "The headline accurately represents the article"),
              ("Use direct quotes from people involved", "Quotes from participants in the story"),
              ("Explain why it matters", "Gives the significance of the news"),
              ("Timeliness", "The story covers recent events"),
              ("Concise writing", "Tight copy with no wasted words"),
              ("Separate news from opinion", "Keep commentary out of straight news")],
        diff=[("Accuracy of facts", "Attractive photo layout"),
              ("Lead with the most important fact", "Diversity of sources"),
              ("Plain, jargon-free language", "Corrections policy is visible"),
              ("Timeliness", "Use of direct quotes"),
              ("Headline reflects the story", "Local relevance"),
              ("Balance: include both sides", "Correct spelling of names"),
              ("Concise writing", "Data visualization quality"),
              ("Explain why it matters", "Byline includes the reporter's email"),
              ("Separate news from opinion", "Story length under 800 words"),
              ("Answer who, what, when, where, why", "Search-engine optimization of the URL"),
              ("Attribute every claim to a source", "Humor in the writing"),
              ("Investigative depth", "Mobile-friendly formatting")],
        faults=["Clickbait headline", "Unsourced rumors", "Sensationalism", "Factual errors", "One-sided coverage",
                "Burying the lede", "Editorializing in a news story", "Plagiarism", "Misleading statistics",
                "Anonymous sources without justification", "Typos and grammatical errors", "Padding with filler"],
    ),
}[DOM]


def main():
    OUT = CFG["out"]
    os.makedirs(f"{OUT}/jobs", exist_ok=True)
    os.makedirs(f"{OUT}/private", exist_ok=True)
    tag = CFG["tag"]
    hj = json.load(open(f"{ROOT}/outputs/hierarchy/{CFG['task']}_general_r2_expanded.json"))
    groups = []
    for i, g in enumerate(hj["merged_groups"]):
        names = sorted({l["name"].strip() for l in g.get("all_leaves", []) if l.get("name", "").strip()})
        if len(names) >= 4:
            groups.append((h(g["merged_name"] + tag), f"g{i}", g["merged_name"], names))
    groups.sort()
    groups = groups[:N_ADJ_CONSTRUCTS]
    manifest, name_of, bym = [], {}, {}
    for gh, gid, gname, names in groups:
        names = sorted(names, key=lambda n: h(n + tag))[:N_NAMES]
        bym[gid] = []
        for j, n in enumerate(names):
            fid = f"{gid}:{j}"
            name_of[fid] = n
            bym[gid].append(fid)
            manifest.append({"metric_id": gid, "form_idx": j, "rubric": n, "construct_name": gname, "construct_hash": gh})
    json.dump(manifest, open(f"{OUT}/forms_manifest.json", "w"), indent=1, ensure_ascii=False)
    pairs = sorted(tuple(sorted(p)) for m in bym for p in itertools.combinations(sorted(bym[m]), 2))
    forms = sorted(name_of)
    print(f"constructs={len(bym)} forms={len(forms)} pairs={len(pairs)}")

    # formality (same protocol / anchors as PREREG-26)
    p10 = json.load(open(f"{P10}/p10_key.json"))["truth"]
    anchor_terms = {}
    for f in sorted(glob.glob(f"{P10}/p10_chunk_*.jsonl")):
        for line in open(f):
            x = json.loads(line)
            if x["id"] in p10:
                anchor_terms[x["term"]] = p10[x["id"]]
    bank = sorted(anchor_terms.items())
    instr = open(f"{P10}/p10_judge_instructions.txt").read()
    instr = instr[:instr.index("Do NOT call any API")].rstrip() + "\n"
    form_jobs, form_key = [], {}
    for k in (1, 2):
        rng = random.Random(f"{tag}-form-pass{k}")
        fl = forms[:]
        rng.shuffle(fl)
        nch = 6
        for c in range(nch):
            items = [("real", f) for f in fl[c::nch]] + [("anchor", a) for a in rng.sample(bank, 10)]
            rng.shuffle(items)
            lines, kb = [], {}
            for kind, it in items:
                if kind == "real":
                    iid = "n" + h(f"{tag}fm|{k}|{it}")[:10]
                    kb[iid] = {"kind": "real", "form": it}
                    term = name_of[it]
                else:
                    iid = "n" + h(f"{tag}fa|{k}|{c}|{it[0]}")[:10]
                    kb[iid] = {"kind": "anchor", "term": it[0], "stratum": it[1]}
                    term = it[0]
                lines.append(json.dumps({"id": iid, "term": term}, ensure_ascii=False))
            jid = f"fp{k}c{c}"
            form_jobs.append({"job": jid, "prompt": instr + B.FORM_HEAD_TAIL + "\n".join(lines) + "\n"})
            form_key[jid] = kb
    json.dump(form_jobs, open(f"{OUT}/jobs/formality.json", "w"), indent=1, ensure_ascii=False)

    # adjudication: strict + calibrated, domain anchors
    B.PARAPHRASE_SAME, B.UNRELATED_DIFF, B.FAULT_NAMES = CFG["same"], CFG["diff"], CFG["faults"]
    strict = (B.ADJ_HEAD
              .replace("creative-writing communities (online writing-prompt communities, short-fiction workshops, fiction editors) use to judge short fiction", CFG["strict_who"])
              .replace("the same piece of short fiction", f"the same {CFG['piece']}")
              .replace("is it better or worse fiction by the community's standards?", CFG["better"]))
    v2 = open(f"{ROOT}/outputs/lexicon/prereg25_20261012/readjudication_v2/prompt_v2_head_creative_writing.txt").read()
    cal = (v2.replace("creative-writing communities (fiction, screenwriting, writing workshops) use to judge creative writing", CFG["strict_who"])
             .replace("the same piece of writing", f"the same {CFG['piece']}")
             .replace("a writing community cares about", "the community cares about")
             .replace("is it better or worse writing by the community's standards?", CFG["better"]))
    assert strict != B.ADJ_HEAD and cal.count(CFG["strict_who"]) == 1 and CFG["better"] in cal and CFG["better"] in strict
    open(f"{OUT}/prompt_strict_head.txt", "w").write(strict)
    open(f"{OUT}/prompt_calibrated_head.txt", "w").write(cal)
    rng = random.Random(f"{tag}-adj")
    pl, fl = pairs[:], forms[:]
    rng.shuffle(pl); rng.shuffle(fl)
    adj, adj_key = B.build_adj(pl, fl, name_of, strict, "b", rng)
    rng = random.Random(f"{tag}-adjv2")
    pl, fl = pairs[:], forms[:]
    rng.shuffle(pl); rng.shuffle(fl)
    adj2, adj2_key = B.build_adj(pl, fl, name_of, cal, "v", rng)
    json.dump(adj, open(f"{OUT}/jobs/adj.json", "w"), indent=1, ensure_ascii=False)
    json.dump(adj2, open(f"{OUT}/jobs/adj_v2.json", "w"), indent=1, ensure_ascii=False)
    S = {"S0": CFG["S0"], "S1": CFG["S1"], "S2": CFG["S2"]}
    json.dump(S, open(f"{OUT}/scaffolds.json", "w"), indent=1)
    chk = B.CHECK_PROMPT.replace("scores short stories", "scores " + CFG.get("plural", CFG["piece"] + "s")).format(**S)
    json.dump([{"job": "scaffold_check", "prompt": chk}], open(f"{OUT}/jobs/check.json", "w"), indent=1, ensure_ascii=False)
    json.dump({"formality": form_key, "adj": adj_key, "adj_v2": adj2_key, "name_of": name_of,
               "pairs": ["|".join(p) for p in pairs], "forms": forms,
               "construct_order": [g[1] for g in groups]},
              open(f"{OUT}/private/keys.json", "w"), indent=1, ensure_ascii=False)
    print(f"formality jobs={len(form_jobs)} adj={len(adj)} adj_v2={len(adj2)} "
          f"max chars={max(len(b['prompt']) for b in adj2)}")


if __name__ == "__main__":
    main()
