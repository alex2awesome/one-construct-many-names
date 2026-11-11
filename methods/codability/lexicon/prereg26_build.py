#!/usr/bin/env python
"""PREREG-26 (creative writing name swap) materials: formality chunks, adjudication batches, scaffold check.

Frozen design: notes/2026-10-12__prereg26-cw-name-swap.md (Steps 0, 1, 4-check).
Writes OUT/jobs/{formality,adj,check}.json (prompts only) and OUT/private/keys.json (anchor answers, form maps).
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import random
from collections import defaultdict

ROOT = "ROOT"
SJ = f"{ROOT}/outputs/lexicon/silver_join"
OUT = f"{ROOT}/outputs/lexicon/prereg26_cw_20261012"
P10 = f"{ROOT}/outputs/lexicon/register_corpus_20260723"
PAIRS_PER_BATCH = 60

PARAPHRASE_SAME = [
    ("Show, don't tell", "Dramatize emotions through action rather than stating them"),
    ("Hook the reader in the first line", "A strong opening sentence that grabs attention"),
    ("Consistent point of view", "Point of view does not shift inconsistently"),
    ("Correct spelling and grammar", "Free of spelling and grammatical errors"),
    ("Believable dialogue", "Dialogue that sounds like real people talking"),
    ("A satisfying ending", "The ending resolves the story in a satisfying way"),
    ("Vivid sensory detail", "Concrete details that appeal to the senses"),
    ("Tight prose with no wasted words", "Economy of language (no unnecessary words)"),
    ("The story answers the prompt", "Responsiveness to the writing prompt"),
    ("Characters with clear motivations", "Character motivation is clear"),
    ("A twist the reader doesn't see coming", "Surprising plot twist"),
    ("Consistent tense", "Verb tense is kept consistent throughout"),
    ("Pacing that keeps the story moving", "Good pacing (the story never drags)"),
    ("An original premise", "Originality of the central idea"),
    ("Clear, readable sentences", "Clarity of the prose"),
    ("Emotional impact on the reader", "The story makes the reader feel something"),
    ("Internal consistency of the world's rules", "The setting's rules stay consistent"),
    ("Distinctive narrative voice", "A unique, recognizable narrator voice"),
    ("Proper paragraphing and formatting", "Readable formatting with sensible paragraph breaks"),
    ("Stakes the reader cares about", "Meaningful stakes for the characters"),
]

UNRELATED_DIFF = [
    ("Correct spelling and grammar", "A twist the reader doesn't see coming"),
    ("Vivid sensory detail", "The story stays within the word limit"),
    ("Believable dialogue", "Worldbuilding of the magic system"),
    ("Consistent tense", "Emotional impact on the reader"),
    ("Hook the reader in the first line", "Accurate historical research"),
    ("Proper paragraphing and formatting", "Characters with clear motivations"),
    ("Humor and comic timing", "Consistent point of view"),
    ("Title choice", "Depth of the antagonist's backstory"),
    ("Scientific plausibility of the technology", "Rhythm and sound of the sentences"),
    ("Use of foreshadowing", "Respectful portrayal of real cultures"),
    ("A satisfying ending", "Correct use of dialogue punctuation"),
    ("Originality of the premise", "Grammatical correctness"),
    ("Atmosphere and mood", "Following the subreddit's posting rules"),
    ("Strong verbs instead of adverbs", "Theme of the story"),
    ("Romance subplot chemistry", "Accurate legal procedure"),
    ("Pacing that keeps the story moving", "Content warnings for sensitive topics"),
    ("Dialogue tags kept simple ('said')", "Larger philosophical meaning"),
    ("Varied sentence length", "Satisfying character arc"),
    ("Poetic imagery", "Plot logic and causality"),
    ("Second-person narration handled well", "Strong final line"),
]

FAULT_NAMES = [
    "Purple prose (overwritten, flowery language)",
    "Info-dumping exposition",
    "Cliched 'it was all a dream' ending",
    "Mary Sue protagonist",
    "Head-hopping between points of view",
    "Overuse of adverbs",
    "Deus ex machina resolution",
    "Wall of text with no paragraph breaks",
    "Telling instead of showing",
    "Stilted, wooden dialogue",
    "Spelling and grammar errors",
    "Plot holes",
    "Rushed, unearned ending",
    "Flat, one-dimensional characters",
    "Melodrama and forced emotion",
    "Tense shifts mid-story",
    "Overused tropes and cliches",
    "Filler scenes that go nowhere",
    "Confusing timeline the reader can't follow",
    "Repetitive sentence structure",
]

ADJ_HEAD = """Adjudication task, no investigation. Do not look anything up and do not use any tools; answer from your own understanding of the names below. They are names of criteria that creative-writing communities (online writing-prompt communities, short-fiction workshops, fiction editors) use to judge short fiction. No definitions are given; read each name as a rater would who is handed only that name.

PART 1 - EQUIVALENCE. For each pair, answer this question: would two raters, one applying name A and one applying name B to the same piece of short fiction, be checking the same requirement?
Answer with exactly one code: "same" (same requirement), "overlap" (overlapping but different), "different" (different requirement).

PART 2 - ORIENTATION. For each criterion name, answer this question: if a piece receives the top score on this criterion name, is it better or worse fiction by the community's standards?
Answer with exactly one code: "better" (higher = better), "worse" (higher = worse), "unclear".

Reply with ONE JSON object and nothing else, covering every id exactly once:
{"pairs": {"P01": "same|overlap|different", ...}, "forms": {"F01": "better|worse|unclear", ...}}
"""

FORM_HEAD_TAIL = """
Do NOT call any API or model and do not use any tools -- you are the judge. Do not skip any line. The INPUT lines are given below (instead of an INPUT file). Reply with one JSON line per input line and nothing else (instead of writing an OUTPUT file): {"id":"...","stratum":"...","formality":1-7,"nominalization":0|1}. Output line count MUST equal input line count.

INPUT:
"""

S = {
    "S0": ("Scoring task, no investigation. You are evaluating short fiction written for an online writing-prompt "
           "community. The criterion named below is one used by the creative-writing community to judge such "
           "fiction. No definition is provided --- apply the criterion as you understand it from its name. "
           "CRITERION NAME: {name}. For EVERY item, judge how well the item satisfies the criterion named above on "
           "a 1-7 scale (1 = clearly fails it, 7 = clearly satisfies it). Judge each item independently from its "
           "text alone."),
    "S1": ("This is a rating job; do not research anything. The texts below are pieces of short fiction posted to "
           "an online writing-prompt community. Writers and readers in creative-writing communities use the "
           "criterion named below when they assess this kind of fiction. You will not be given a definition; rely "
           "on your own understanding of what the name means. CRITERION: {name}. Rate every text on a 1 to 7 scale "
           "for how far it meets this criterion (1 = it plainly does not meet it, 7 = it plainly meets it). Rate "
           "each text on its own, using only its text."),
    "S2": ("Please score the short stories listed below. Do not look anything up. Each story was written in "
           "response to a prompt in an online writing community, and the criterion given below is a standard that "
           "people in creative writing apply to such fiction. Interpret the criterion from its name alone, since no "
           "definition is supplied. Criterion to apply: {name}. Give each story a score from 1 to 7, where 7 means "
           "the story clearly meets the criterion and 1 means it clearly does not. Score the stories independently "
           "of one another, based on the text alone."),
}

CHECK_PROMPT = """Meaning-equivalence check, no investigation; do not use any tools. Below are three instructions given to a rater who scores short stories on a named criterion. S0 is the reference. For S1 and for S2 separately, answer: does it ask the rater to do the same task as S0, with the same meaning (same material, same source of the criterion, no definition given, same 1-7 scale with the same anchors, items judged independently from their text alone)? Ignore wording and order. Answer "equivalent" or "not equivalent" with a one-sentence reason for each.

S0: {S0}

S1: {S1}

S2: {S2}

Reply with ONE JSON object and nothing else: {{"S1": {{"verdict": "equivalent|not equivalent", "reason": "..."}}, "S2": {{"verdict": "equivalent|not equivalent", "reason": "..."}}}}
"""


def h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def build_adj(pairs, forms_list, name_of, head, tag, rng):
    nb = -(-len(pairs) // PAIRS_PER_BATCH)
    batches, key = [], {}
    for b in range(nb):
        real_pairs = pairs[b::nb]
        real_forms = forms_list[b::nb]
        p_items = [("real", k) for k in real_pairs] + [("anchor_same", a) for a in rng.sample(PARAPHRASE_SAME, 3)] + \
                  [("anchor_diff", a) for a in rng.sample(UNRELATED_DIFF, 3)]
        f_items = [("real", f) for f in real_forms] + [("anchor_fault", a) for a in rng.sample(FAULT_NAMES, 4)]
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


def main() -> None:
    os.makedirs(f"{OUT}/jobs", exist_ok=True)
    os.makedirs(f"{OUT}/private", exist_ok=True)
    man = json.load(open(f"{SJ}/cw_forms_manifest.json"))
    name_of = {f"{r['metric_id']}:{r['form_idx']}": r["rubric"] for r in man}
    assert len(name_of) == len(man)
    bym = defaultdict(list)
    for r in man:
        bym[r["metric_id"]].append(f"{r['metric_id']}:{r['form_idx']}")
    pairs = sorted(tuple(sorted(p)) for m in bym for p in itertools.combinations(sorted(bym[m]), 2))
    forms = sorted(name_of)
    print(f"constructs={len(bym)} forms={len(forms)} within-construct pairs={len(pairs)}")

    # ---- Step 0: formality chunks (2 passes x 7 chunks), PREREG-10 anchor bank (stratum truth)
    p10 = json.load(open(f"{P10}/p10_key.json"))["truth"]
    anchor_terms = {}
    import glob
    for f in sorted(glob.glob(f"{P10}/p10_chunk_*.jsonl")):
        for line in open(f):
            x = json.loads(line)
            if x["id"] in p10:
                anchor_terms[x["term"]] = p10[x["id"]]
    bank = sorted(anchor_terms.items())
    instr = open(f"{P10}/p10_judge_instructions.txt").read()
    instr = instr[:instr.index("Do NOT call any API")].rstrip() + "\n"  # delivery adapted: inline in / inline out
    form_jobs, form_key = [], {}
    for k in (1, 2):
        rng = random.Random(f"p26-form-pass{k}")
        fl = forms[:]
        rng.shuffle(fl)
        nch = 7
        for c in range(nch):
            real = fl[c::nch]
            anchors = rng.sample(bank, 10)
            items = [("real", f) for f in real] + [("anchor", a) for a in anchors]
            rng.shuffle(items)
            lines, kb = [], {}
            for kind, it in items:
                if kind == "real":
                    iid = "n" + h(f"p26fm|{k}|{it}")[:10]
                    kb[iid] = {"kind": "real", "form": it}
                    term = name_of[it]
                else:
                    iid = "n" + h(f"p26fa|{k}|{c}|{it[0]}")[:10]
                    kb[iid] = {"kind": "anchor", "term": it[0], "stratum": it[1]}
                    term = it[0]
                lines.append(json.dumps({"id": iid, "term": term}, ensure_ascii=False))
            jid = f"fp{k}c{c}"
            form_jobs.append({"job": jid, "prompt": instr + FORM_HEAD_TAIL + "\n".join(lines) + "\n"})
            form_key[jid] = kb
    json.dump(form_jobs, open(f"{OUT}/jobs/formality.json", "w"), indent=1, ensure_ascii=False)

    # ---- Step 1: adjudication batches
    rng = random.Random("p26-adj-20261012")
    pl = pairs[:]
    rng.shuffle(pl)
    fl = forms[:]
    rng.shuffle(fl)
    adj_jobs, adj_key = build_adj(pl, fl, name_of, ADJ_HEAD, "b", rng)
    json.dump(adj_jobs, open(f"{OUT}/jobs/adj.json", "w"), indent=1, ensure_ascii=False)

    # ---- scaffold check (GPT-5.6)
    json.dump([{"job": "scaffold_check", "prompt": CHECK_PROMPT.format(**S)}],
              open(f"{OUT}/jobs/check.json", "w"), indent=1, ensure_ascii=False)
    json.dump(S, open(f"{OUT}/scaffolds.json", "w"), indent=1)

    json.dump({"formality": form_key, "adj": adj_key, "name_of": name_of,
               "pairs": ["|".join(p) for p in pairs], "forms": forms},
              open(f"{OUT}/private/keys.json", "w"), indent=1, ensure_ascii=False)
    print(f"formality jobs={len(form_jobs)} (anchor bank {len(bank)} terms); adj batches={len(adj_jobs)} "
          f"max chars={max(len(b['prompt']) for b in adj_jobs)}")


if __name__ == "__main__":
    main()
