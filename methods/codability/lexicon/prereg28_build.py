#!/usr/bin/env python
"""PREREG-28 (name swap in code / press releases / academic articles) materials, one task per call.

Frozen design: notes/2026-10-12__prereg28-codepress-peer-name-swap.md. Mirrors prereg26_build.py (lane B2).
usage: prereg28_build.py <code-review|press-releases|peer-review>
Writes OUT/jobs/{formality,adj_v,adj_b,check}.json, OUT/scaffolds.json, OUT/forms_manifest.json,
OUT/item_pool.jsonl, OUT/private/keys.json (anchor answers, form maps).
"""
from __future__ import annotations

import ast
import glob
import gzip
import hashlib
import itertools
import json
import os
import random
import re
import sys
from collections import defaultdict

ROOT = "ROOT"
P10 = f"{ROOT}/outputs/lexicon/register_corpus_20260723"
A2_HEAD = f"{ROOT}/outputs/lexicon/prereg25_20261012/readjudication_v2/prompt_v2_head_creative_writing.txt"
PAIRS_PER_BATCH = 60
N_CONSTRUCTS = 20
MAX_FORMS = 8
POOL_N = 200

DOM = {
    "code-review": dict(
        short="code", community="software-engineering communities (code reviewers, open-source maintainers, style guides)",
        judge_obj="code changes", piece="piece of code", a2_comm="a software-engineering community",
        better_obj="code",
        S0=("Scoring task, no investigation. You are evaluating code changes submitted for review (excerpts of "
            "GitHub pull-request diffs). The criterion named below is one used by the software-engineering community "
            "to judge such code. No definition is provided --- apply the criterion as you understand it from its "
            "name. CRITERION NAME: {name}. For EVERY item, judge how well the item satisfies the criterion named "
            "above on a 1-7 scale (1 = clearly fails it, 7 = clearly satisfies it). Judge each item independently "
            "from its text alone."),
        S1=("This is a rating job; do not research anything. The texts below are excerpts of code changes (GitHub "
            "pull-request diffs) submitted for review. Software engineers use the criterion named below when they "
            "assess this kind of code. You will not be given a definition; rely on your own understanding of what "
            "the name means. CRITERION: {name}. Rate every text on a 1 to 7 scale for how far it meets this "
            "criterion (1 = it plainly does not meet it, 7 = it plainly meets it). Rate each text on its own, using "
            "only its text."),
        S2=("Please score the code changes listed below. Do not look anything up. Each one is an excerpt of a pull "
            "request diff submitted for review on GitHub, and the criterion given below is a standard that people "
            "in software engineering apply to such code. Interpret the criterion from its name alone, since no "
            "definition is supplied. Criterion to apply: {name}. Give each code change a score from 1 to 7, where 7 "
            "means it clearly meets the criterion and 1 means it clearly does not. Score the code changes "
            "independently of one another, based on the text alone."),
        same=[("Descriptive variable names", "Variables are given meaningful, descriptive names"),
              ("No dead code", "Unused code is removed"),
              ("Unit tests cover the change", "The change comes with unit tests"),
              ("Consistent indentation", "Indentation is uniform throughout"),
              ("Functions do one thing", "Single responsibility for each function"),
              ("Avoid magic numbers", "Use named constants instead of unexplained literals"),
              ("Handle errors explicitly", "Errors are checked and handled, not ignored"),
              ("Don't repeat yourself", "Avoid duplicated code"),
              ("Clear commit message", "The commit message explains the change clearly"),
              ("Short functions", "Functions are kept small"),
              ("Comments explain why, not what", "Comments give the reasoning behind the code"),
              ("No hard-coded credentials", "Secrets are not embedded in source code")],
        diff=[("Descriptive variable names", "Thread safety under concurrent access"),
              ("Unit tests cover the change", "Consistent brace placement"),
              ("No hard-coded credentials", "Short functions"),
              ("Clear commit message", "Efficient database queries"),
              ("Avoid magic numbers", "Accessible user interface labels"),
              ("Consistent indentation", "Correct license headers"),
              ("Handle errors explicitly", "Alphabetized import statements"),
              ("Backward-compatible API", "Spelling in comments"),
              ("Memory usage", "Descriptive pull request title"),
              ("Input validation", "Code formatting with a linter"),
              ("Documentation for public functions", "Algorithmic time complexity"),
              ("Logging at the right level", "Dependency version pinning")],
        fault=["Spaghetti code", "Copy-pasted duplicated logic", "Magic numbers everywhere", "Swallowed exceptions",
               "God object that does everything", "Hard-coded passwords", "Deeply nested if-statements",
               "Commented-out dead code", "Off-by-one errors", "Race conditions", "Memory leaks",
               "Cryptic one-letter variable names"]),
    "press-releases": dict(
        short="press", community="communications and journalism communities (PR professionals, newsroom editors, journalists)",
        judge_obj="press releases", piece="press release", a2_comm="a communications or journalism community",
        better_obj="press release",
        S0=("Scoring task, no investigation. You are evaluating press releases (opening excerpts) issued by "
            "organizations to the news media. The criterion named below is one used by the communications and "
            "journalism community to judge such press releases. No definition is provided --- apply the criterion "
            "as you understand it from its name. CRITERION NAME: {name}. For EVERY item, judge how well the item "
            "satisfies the criterion named above on a 1-7 scale (1 = clearly fails it, 7 = clearly satisfies it). "
            "Judge each item independently from its text alone."),
        S1=("This is a rating job; do not research anything. The texts below are the opening parts of press "
            "releases that organizations sent to the news media. Communications professionals and journalists use "
            "the criterion named below when they assess this kind of press release. You will not be given a "
            "definition; rely on your own understanding of what the name means. CRITERION: {name}. Rate every text "
            "on a 1 to 7 scale for how far it meets this criterion (1 = it plainly does not meet it, 7 = it plainly "
            "meets it). Rate each text on its own, using only its text."),
        S2=("Please score the press releases listed below. Do not look anything up. Each one is the opening excerpt "
            "of a release an organization issued to the news media, and the criterion given below is a standard "
            "that people in communications and journalism apply to such releases. Interpret the criterion from its "
            "name alone, since no definition is supplied. Criterion to apply: {name}. Give each press release a "
            "score from 1 to 7, where 7 means it clearly meets the criterion and 1 means it clearly does not. Score "
            "the press releases independently of one another, based on the text alone."),
        same=[("Newsworthy lead", "The opening paragraph states the news"),
              ("Includes contact information", "Media contact details are provided"),
              ("Answers who, what, when, where, why", "Covers the five Ws"),
              ("Concise headline", "A short, clear headline"),
              ("Includes a quote from a spokesperson", "Features a spokesperson's quotation"),
              ("No jargon", "Avoids technical jargon"),
              ("Inverted pyramid structure", "Most important information first"),
              ("Includes a boilerplate about the company", "Standard 'About the company' paragraph"),
              ("Factually accurate", "Free of factual errors"),
              ("Dateline at the start", "Begins with the city and date of release"),
              ("Avoid hype words", "No exaggerated promotional superlatives"),
              ("Timely news", "The news is current")],
        diff=[("Includes contact information", "Newsworthy lead"),
              ("Concise headline", "Accessible website design"),
              ("Includes a quote from a spokesperson", "Correct use of the serial comma"),
              ("Dateline at the start", "Emotional appeal of the story"),
              ("Factually accurate", "Social media hashtags"),
              ("No jargon", "Embargo date clearly marked"),
              ("Inverted pyramid structure", "High-resolution product photos"),
              ("Includes a boilerplate about the company", "Local angle for regional outlets"),
              ("Timely news", "Length under one page"),
              ("Data and statistics", "Brand logo placement"),
              ("Avoid hype words", "Distribution through a wire service"),
              ("Human-interest angle", "Stock ticker symbol included")],
        fault=["Buried lead", "Marketing fluff and hype", "Jargon-filled corporate speak", "Missing contact information",
               "Factual errors", "Typos and grammatical mistakes", "Wall of text with no structure",
               "No news value", "Vague, generic quotes", "Misleading headline", "Excessive length",
               "Overuse of superlatives"]),
    "peer-review": dict(
        short="peer", community="academic peer-review communities (conference reviewers, area chairs, journal editors)",
        judge_obj="academic papers", piece="academic paper", a2_comm="an academic peer-review community",
        better_obj="paper",
        S0=("Scoring task, no investigation. You are evaluating academic paper submissions (title and abstract) to "
            "a machine-learning conference. The criterion named below is one used by the academic peer-review "
            "community to judge such papers. No definition is provided --- apply the criterion as you understand "
            "it from its name. CRITERION NAME: {name}. For EVERY item, judge how well the item satisfies the "
            "criterion named above on a 1-7 scale (1 = clearly fails it, 7 = clearly satisfies it). Judge each item "
            "independently from its text alone."),
        S1=("This is a rating job; do not research anything. The texts below are the titles and abstracts of papers "
            "submitted to a machine-learning conference. Reviewers in the academic peer-review community use the "
            "criterion named below when they assess this kind of paper. You will not be given a definition; rely "
            "on your own understanding of what the name means. CRITERION: {name}. Rate every text on a 1 to 7 "
            "scale for how far it meets this criterion (1 = it plainly does not meet it, 7 = it plainly meets it). "
            "Rate each text on its own, using only its text."),
        S2=("Please score the paper submissions listed below. Do not look anything up. Each one is the title and "
            "abstract of a paper submitted to a machine-learning conference, and the criterion given below is a "
            "standard that academic peer reviewers apply to such papers. Interpret the criterion from its name "
            "alone, since no definition is supplied. Criterion to apply: {name}. Give each paper a score from 1 to "
            "7, where 7 means it clearly meets the criterion and 1 means it clearly does not. Score the papers "
            "independently of one another, based on the text alone."),
        same=[("Novelty of the contribution", "Originality of the work"),
              ("Clear writing", "The paper is clearly written"),
              ("Reproducibility", "Results can be reproduced from the description"),
              ("Strong baselines", "Comparison against competitive baselines"),
              ("Sound methodology", "The methods are technically sound"),
              ("Claims supported by evidence", "Conclusions follow from the results"),
              ("Thorough related work", "Adequate coverage of prior literature"),
              ("Ablation studies", "Experiments isolating each component's contribution"),
              ("Significance of the results", "The results matter to the field"),
              ("Statistical rigor", "Appropriate statistical testing of results"),
              ("Clear problem statement", "The research question is well defined"),
              ("Limitations discussed", "The paper acknowledges its limitations")],
        diff=[("Novelty of the contribution", "Correct citation formatting"),
              ("Clear writing", "Large-scale experiments"),
              ("Reproducibility", "Catchy title"),
              ("Strong baselines", "Ethical considerations statement"),
              ("Sound methodology", "Page-limit compliance"),
              ("Thorough related work", "Theoretical convergence guarantees"),
              ("Ablation studies", "Readable figure fonts"),
              ("Significance of the results", "Anonymization of the submission"),
              ("Statistical rigor", "Broader societal impact"),
              ("Clear problem statement", "Code release"),
              ("Limitations discussed", "Mathematical notation consistency"),
              ("Real-world applicability", "Grammar and spelling")],
        fault=["Overclaiming results", "Missing baselines", "Incremental contribution", "Poorly written and hard to follow",
               "Cherry-picked results", "Unsupported claims", "Flawed experimental design", "Missing related work",
               "Lack of reproducibility details", "Data leakage between train and test", "Plagiarism",
               "Statistical errors"]),
}

CHECK_PROMPT = """Meaning-equivalence check, no investigation; do not use any tools. Below are three instructions given to a rater who scores {judge_obj} on a named criterion. S0 is the reference. For S1 and for S2 separately, answer: does it ask the rater to do the same task as S0, with the same meaning (same material, same source of the criterion, no definition given, same 1-7 scale with the same anchors, items judged independently from their text alone)? Ignore wording and order. Answer "equivalent" or "not equivalent" with a one-sentence reason for each.

S0: {S0}

S1: {S1}

S2: {S2}

Reply with ONE JSON object and nothing else: {{"S1": {{"verdict": "equivalent|not equivalent", "reason": "..."}}, "S2": {{"verdict": "equivalent|not equivalent", "reason": "..."}}}}
"""

STRICT_HEAD = """Adjudication task, no investigation. Do not look anything up and do not use any tools; answer from your own understanding of the names below. They are names of criteria that {community} use to judge {judge_obj}. No definitions are given; read each name as a rater would who is handed only that name.

PART 1 - EQUIVALENCE. For each pair, answer this question: would two raters, one applying name A and one applying name B to the same {piece}, be checking the same requirement?
Answer with exactly one code: "same" (same requirement), "overlap" (overlapping but different), "different" (different requirement).

PART 2 - ORIENTATION. For each criterion name, answer this question: if a {piece} receives the top score on this criterion name, is it a better or worse {better_obj} by the community's standards?
Answer with exactly one code: "better" (higher = better), "worse" (higher = worse), "unclear".

Reply with ONE JSON object and nothing else, covering every id exactly once:
{{"pairs": {{"P01": "same|overlap|different", ...}}, "forms": {{"F01": "better|worse|unclear", ...}}}}
"""

FORM_HEAD_TAIL = """
Do NOT call any API or model and do not use any tools -- you are the judge. Do not skip any line. The INPUT lines are given below (instead of an INPUT file). Reply with one JSON line per input line and nothing else (instead of writing an OUTPUT file): {"id":"...","stratum":"...","formality":1-7,"nominalization":0|1}. Output line count MUST equal input line count.

INPUT:
"""


def h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def calibrated_head(D):
    """Lane A2's calibrated head (creative-writing version) with the domain words swapped; few-shots verbatim."""
    t = open(A2_HEAD).read()
    reps = [("creative-writing communities (fiction, screenwriting, writing workshops) use to judge creative writing",
             f"{D['community']} use to judge {D['judge_obj']}"),
            ("to the same piece of writing", f"to the same {D['piece']}"),
            ("the quality a writing community cares about", f"the quality {D['a2_comm']} cares about"),
            ("if a piece receives the top score", f"if a {D['piece']} receives the top score"),
            ("is it better or worse writing by", f"is it a better or worse {D['better_obj']} by")]
    for a, b in reps:
        assert a in t, a
        t = t.replace(a, b)
    return t


def build_adj(pairs, forms_list, name_of, head, tag, rng, D):
    nb = -(-len(pairs) // PAIRS_PER_BATCH)
    batches, key = [], {}
    for b in range(nb):
        real_pairs = pairs[b::nb]
        real_forms = forms_list[b::nb]
        p_items = [("real", k) for k in real_pairs] + [("anchor_same", a) for a in rng.sample(D["same"], 3)] + \
                  [("anchor_diff", a) for a in rng.sample(D["diff"], 3)]
        f_items = [("real", f) for f in real_forms] + [("anchor_fault", a) for a in rng.sample(D["fault"], 4)]
        rng.shuffle(p_items)
        rng.shuffle(f_items)
        lp, lf, kb = [], [], {"pairs": {}, "forms": {}}
        for i, (kind, it) in enumerate(p_items, 1):
            pid = f"P{i:02d}"
            if kind == "real":
                fa, fb = it if rng.random() < 0.5 else (it[1], it[0])
                na, nb_ = name_of[fa], name_of[fb]
                kb["pairs"][pid] = {"kind": "real", "forms": sorted(it)}
            else:
                na, nb_ = it if rng.random() < 0.5 else (it[1], it[0])
                kb["pairs"][pid] = {"kind": kind, "answer": "same" if kind == "anchor_same" else "different",
                                    "names": [na, nb_]}
            lp.append(f"{pid}: A = {json.dumps(na, ensure_ascii=False)} | B = {json.dumps(nb_, ensure_ascii=False)}")
        for i, (kind, it) in enumerate(f_items, 1):
            fid = f"F{i:02d}"
            if kind == "real":
                kb["forms"][fid] = {"kind": "real", "form": it}
                nm = name_of[it]
            else:
                kb["forms"][fid] = {"kind": kind, "answer": "worse", "name": it}
                nm = it
            lf.append(f"{fid}: {json.dumps(nm, ensure_ascii=False)}")
        prompt = head + "\nPAIRS:\n" + "\n".join(lp) + "\n\nCRITERION NAMES:\n" + "\n".join(lf) + "\n"
        bid = f"{tag}{b:02d}"
        batches.append({"job": bid, "prompt": prompt})
        key[bid] = kb
    return batches, key


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


def item_pool(task):
    """Clean document corpus under datasets/<task>/ -> list of {item_id, text} (400-1,000 chars)."""
    rows = []
    if task == "code-review":
        src = f"{ROOT}/datasets/code-review/diffs/*.diff"
        for f in sorted(glob.glob(src)):
            t = open(f, errors="replace").read()
            t = t.replace("\r", "")
            if len(t.strip()) >= 400:
                rows.append({"item_id": os.path.basename(f)[:-5], "text": t.strip()[:1000]})
    elif task == "press-releases":
        src = f"{ROOT}/datasets/press-releases/press_release_clean.jsonl.gz"
        for l in gzip.open(src, "rt"):
            r = json.loads(l)
            t = re.sub(r"[ \t]+", " ", r.get("extracted_text") or "")
            t = re.sub(r"\n\s*\n+", "\n\n", t).strip()
            if len(t) >= 400:
                rows.append({"item_id": "pr" + str(r["press_release_id"]), "text": t[:1000]})
    else:
        import pandas as pd
        src = f"{ROOT}/datasets/peer-review/berenslab-iclr-dataset/data/iclr24v2.parquet"
        d = pd.read_parquet(src, columns=["id", "title", "abstract", "year"])
        for _, r in d.iterrows():
            t = norm(str(r["title"])) + "\n\n" + norm(str(r["abstract"]))
            if len(t) >= 400 and str(r["abstract"]) not in ("", "None", "nan"):
                rows.append({"item_id": f"iclr{r['year']}_{r['id']}", "text": t[:1000]})
    rows.sort(key=lambda r: h(r["item_id"] + "p28-pool"))
    return rows[:POOL_N], src, len(rows)


def main() -> None:
    task = sys.argv[1]
    D = DOM[task]
    OUT = f"{ROOT}/outputs/lexicon/prereg28_{task}_20261012"
    os.makedirs(f"{OUT}/jobs", exist_ok=True)
    os.makedirs(f"{OUT}/private", exist_ok=True)

    # ---- constructs and attested phrasings (metric bank = general_r2_expanded merged groups)
    hier = json.load(open(f"{ROOT}/outputs/hierarchy/{task}_general_r2_expanded.json"))
    cons = []
    for gi, g in enumerate(hier["merged_groups"]):
        L = ast.literal_eval(g["all_leaves"]) if isinstance(g["all_leaves"], str) else g["all_leaves"]
        seen, names = set(), []
        for x in L:
            n = norm(x["name"])
            if len(n) < 3 or n.lower() in seen:
                continue
            seen.add(n.lower())
            names.append((n, x["key"]))
        if len(names) >= 2:
            cons.append({"mid": f"g{gi}", "merged_name": g["merged_name"], "names": names})
    n_eligible = len(cons)
    cons.sort(key=lambda c: h(task + "|" + c["merged_name"] + "|p28"))
    cons = cons[:N_CONSTRUCTS]
    man = []
    for c in cons:
        nm = sorted(c["names"], key=lambda nk: h(c["mid"] + "|" + nk[0].lower() + "|p28-form"))[:MAX_FORMS]
        for i, (n, k) in enumerate(nm):
            man.append({"metric_id": c["mid"], "form_idx": i, "rubric": n, "census_key": k,
                        "merged_name": c["merged_name"], "n_attested_names": len(c["names"])})
    json.dump(man, open(f"{OUT}/forms_manifest.json", "w"), indent=1, ensure_ascii=False)
    name_of = {f"{r['metric_id']}:{r['form_idx']}": r["rubric"] for r in man}
    bym = defaultdict(list)
    for r in man:
        bym[r["metric_id"]].append(f"{r['metric_id']}:{r['form_idx']}")
    pairs = sorted(tuple(sorted(p)) for m in bym for p in itertools.combinations(sorted(bym[m]), 2))
    forms = sorted(name_of)
    print(f"{task}: eligible constructs={n_eligible} selected={len(bym)} forms={len(forms)} pairs={len(pairs)}")

    # ---- items
    pool, src, n_elig_items = item_pool(task)
    with open(f"{OUT}/item_pool.jsonl", "w") as f:
        for r in pool:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- Step 0: formality (Claude), PREREG-10 anchor bank
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
    nch = max(1, -(-len(forms) // 40))
    for k in (1, 2):
        rng = random.Random(f"p28-{task}-form-pass{k}")
        fl = forms[:]
        rng.shuffle(fl)
        for c in range(nch):
            real = fl[c::nch]
            anchors = rng.sample(bank, 10)
            items = [("real", f) for f in real] + [("anchor", a) for a in anchors]
            rng.shuffle(items)
            lines, kb = [], {}
            for kind, it in items:
                if kind == "real":
                    iid = "n" + h(f"p28fm|{task}|{k}|{it}")[:10]
                    kb[iid] = {"kind": "real", "form": it}
                    term = name_of[it]
                else:
                    iid = "n" + h(f"p28fa|{task}|{k}|{c}|{it[0]}")[:10]
                    kb[iid] = {"kind": "anchor", "term": it[0], "stratum": it[1]}
                    term = it[0]
                lines.append(json.dumps({"id": iid, "term": term}, ensure_ascii=False))
            jid = f"fp{k}c{c}"
            form_jobs.append({"job": jid, "prompt": instr + FORM_HEAD_TAIL + "\n".join(lines) + "\n"})
            form_key[jid] = kb
    json.dump(form_jobs, open(f"{OUT}/jobs/formality.json", "w"), indent=1, ensure_ascii=False)

    # ---- Step 1: adjudication, calibrated (v, primary) and strict (b, counts only); same pairs and batching
    cal_head = calibrated_head(D)
    open(f"{OUT}/prompt_calibrated_head_USED.txt", "w").write(cal_head)
    strict_head = STRICT_HEAD.format(**D)
    open(f"{OUT}/prompt_strict_head_USED.txt", "w").write(strict_head)
    pl, fl = pairs[:], forms[:]
    random.Random(f"p28-{task}-adj-order").shuffle(pl)
    random.Random(f"p28-{task}-adj-order-forms").shuffle(fl)
    v_jobs, v_key = build_adj(pl, fl, name_of, cal_head, "v", random.Random(f"p28-{task}-adjv2"), D)
    b_jobs, b_key = build_adj(pl, fl, name_of, strict_head, "b", random.Random(f"p28-{task}-adjstrict"), D)
    json.dump(v_jobs, open(f"{OUT}/jobs/adj_v.json", "w"), indent=1, ensure_ascii=False)
    json.dump(b_jobs, open(f"{OUT}/jobs/adj_b.json", "w"), indent=1, ensure_ascii=False)

    S = {k: D[k] for k in ("S0", "S1", "S2")}
    json.dump([{"job": "scaffold_check", "prompt": CHECK_PROMPT.format(judge_obj=D["judge_obj"], **S)}],
              open(f"{OUT}/jobs/check.json", "w"), indent=1, ensure_ascii=False)
    json.dump(S, open(f"{OUT}/scaffolds.json", "w"), indent=1)
    json.dump({"formality": form_key, "adj": b_key, "adj_v2": v_key, "name_of": name_of,
               "pairs": ["|".join(p) for p in pairs], "forms": forms},
              open(f"{OUT}/private/keys.json", "w"), indent=1, ensure_ascii=False)
    meta = {"task": task, "n_bank_constructs": len(hier["merged_groups"]), "n_eligible_constructs": n_eligible,
            "n_selected_constructs": len(bym), "n_forms": len(forms), "n_pairs": len(pairs),
            "item_source": src, "n_eligible_items": n_elig_items, "pool_n": len(pool),
            "formality_jobs": len(form_jobs), "adj_batches_per_standard": len(v_jobs)}
    json.dump(meta, open(f"{OUT}/build_meta.json", "w"), indent=1)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
