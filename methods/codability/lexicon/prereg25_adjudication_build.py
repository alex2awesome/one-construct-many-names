#!/usr/bin/env python
"""PREREG-25 Step 1 materials: equivalence + orientation adjudication batches.

Frozen design: notes/2026-10-12__prereg25-matched-meaning-name-swap.md (Step 1).
Scope (coordinator brief 2026-10-12):
  (a) every within-metric pair of forms with formality gap >= 1.042 (1 pooled SD) for the
      86 PREREG-15-v2 metrics that have a frozen 40-item set;
  (b) the 86 frozen v2 arm pairs (v2_arms.json);
  (c) the 10 published v1 Table-3 pairs (paper1_figdata.json table3; all 20 strings are in
      the v2 form pool).
Orientation is asked for every form in any of these pairs.

Each batch carries 10 camouflaged known-answer anchors: 3 synthetic paraphrase pairs
(answer: same requirement), 3 cross-metric pairs of real forms with unrelated content
(answer: different requirement), 4 explicit fault names (answer: higher = worse). Anchors
get ids from the same sequence as real items and are shuffled in. Judges see names only:
no metric ids, no definitions, no formality, no scores, no AUCs.

Writes outputs/lexicon/prereg25_20261012/adjudication/{batches.json, key_private.json}.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import random
from collections import defaultdict

ROOT = "ROOT"
V2 = f"{ROOT}/outputs/lexicon/prereg15v2_20260828"
OUT = f"{ROOT}/outputs/lexicon/prereg25_20261012/adjudication"
SEED = "p25-adj-20261012"
PAIRS_PER_BATCH = 60

PARAPHRASE_SAME = [
    ("Keep the setup short", "Setups should be brief"),
    ("The punchline comes as a surprise", "The punch line delivers an unexpected turn"),
    ("Callbacks to earlier material", "Referring back later in the set to a bit from earlier"),
    ("Original material, not borrowed", "Jokes are the comic's own, not lifted from others"),
    ("Commit to the character", "Fully inhabit the character you are playing"),
    ("No wasted words", "Economy of wording (no superfluous words)"),
    ("Observations the audience recognizes from daily life", "Relatable everyday observations"),
    ("End the set on your biggest laugh", "Strongest joke placed as the closer"),
    ("Pause right before the punchline", "Timing: a beat of silence before the punch line"),
    ("Heighten the absurdity step by step", "Escalation of the absurd premise"),
    ("Concrete specifics instead of vague generalities", "Use specific details, not generic ones"),
    ("Make yourself the butt of the joke", "Self-deprecating humor (joking at one's own expense)"),
    ("No profanity", "Absence of swear words"),
    ("Physically act out the characters", "Act-outs: embodying the characters in the story"),
    ("Jokes about current events", "Topical humor on recent news"),
    ("Play on the double meanings of words", "Wordplay exploiting words with two meanings"),
    ("Connect with the crowd", "Building rapport with the audience"),
    ("Treat a ridiculous situation with total seriousness", "Absurd premise played completely straight"),
    ("Lots of laughs per minute", "High frequency of laughs throughout the set"),
    ("Lead them to expect one thing, then give them another", "Misdirection (setting up a false expectation)"),
    ("Add extra punchlines onto the same setup", "Tagging: additional punch lines on one setup"),
    ("The audience can tell what the bit is about", "Clarity of the premise"),
    ("Stay consistent with your stage persona", "Consistency of the established stage character"),
    ("Keep it clean", "Clean material (no explicit content)"),
    ("Delivery speed: don't rush the lines", "Do not hurry the delivery"),
    ("The joke is easy to understand", "Comprehensibility of the joke"),
    ("Jokes about your own family", "Material drawn from the comic's family life"),
    ("Silly voices and impressions", "Use of impressions and funny voices"),
    ("Rhyme and rhythm in the wording", "Rhythmic, rhyming phrasing"),
    ("Exaggerate for comic effect", "Comic exaggeration (hyperbole)"),
]

CROSS_METRIC_DIFF = [
    ("Puns and word play", "Ethical and legal considerations when dealing with real historical atrocities"),
    ("Eliminate lengthy build-ups; go straight to the punch line", "Cover art: unified, sophisticated, provocative design"),
    ("No toilet humor", "Topicality / attention to current events"),
    ("Tell stories where you're the idiot — but trying", "Brevitas (brevity)"),
    ("Specificity and vivid imagery", "Heckler type: \"They don't like you\""),
    ("Recurring motifs, settings and running gags", "Sensitivity of comedic depiction of suffering (ethical considerations)"),
    ("Surrealist humor", "Mine your own life for comedy ideas"),
    ("Strong button (sketch)", "Cross-cultural breadth and comparative perspective"),
    ("Time your pause", "Subject suitability / moral boundary consideration"),
    ("Accuracy and nuance of political/media criticism", "Learn to appreciate witty wordplay and puns"),
    ("Tag your jokes", "Authenticity/realism in depicting depression"),
    ("Punch up, not down", "Create imagery"),
    ("Blend of writing and drawing ('novel-in-pictures')", "Stay in the bit / Don't step on laughs"),
    ("Plausibility", "Puns and wordplay: use with care"),
    ("Withering putdowns", "Coherence, structure, and quality control"),
    ("Narrative Complexity", "Blue comedy"),
    ("Autonomy enables risk-taking and distinctive voices", "Reduplication and playful sound structure"),
    ("iO Working Rubric — 1: Establish the universe in first scene", "Political / Social Commentary"),
    ("Tonal balance: equal parts funny and dark", "Speed"),
    ("Playing the straight man / restrained stage persona", "Factually defensible / research-vetted"),
    ("Politeness work", "Setup…punch"),
    ("Avoiding 'low-hanging fruit' topical jokes", "Compatibility between the two stories"),
    ("Mine life/pain for material", "Maxim — Manner (clarity and presentation)"),
    ("Intertextuality and cultural literacy", "Avoid cynicism; preserve warmth and wit"),
    ("Cleanness", "Require two beats, not one"),
    ("Make the fake news actually funny", "Is the structure tight (no padding)?"),
    ("Balancing atrocity and festivity; racial context sensitivity", "Go beyond the first idea — find another angle/layer"),
    ("World played straight despite its absurdity", "Knowledge-dependent humor (assumes reader background)"),
    ("Audience fit", "Use of non-sequiturs and twisted punch lines"),
    ("Don't cater to the room", "Epigram vs. Aphorism (humor requirement)"),
]

FAULT_NAMES = [
    "Hacky, recycled premises (anti-pattern)",
    "Stale airline-food clichés",
    "Punching down at vulnerable groups",
    "Telegraphed punchlines the audience sees coming",
    "Rambling, unfocused setups",
    "Overexplaining the joke",
    "Stepping on your own laughs",
    "Cheap shock value with no point",
    "Stolen or lifted material (joke theft)",
    "Dead air and lost momentum",
    "Mean-spirited crowd work that alienates the room",
    "Failure mode: punchline buried mid-sentence",
    "Padding and filler lines",
    "Lazy stereotypes as the whole joke",
    "Anti-pattern: laughing at your own jokes before the punch",
    "Predictable, on-the-nose wordplay",
    "Inconsistent persona that confuses the audience",
    "Profanity used as a crutch",
    "Pandering to the crowd for cheap applause",
    "Mistake: explaining the reference after it fails",
    "Derivative bits copied from a famous comic",
    "Muddled premise the audience can't follow",
    "Error: callback to something never set up",
    "Tonal whiplash that kills the laugh",
    "Over-long tags that drain the laugh",
    "Hackneyed 'what's the deal with' observations",
    "Bombing (material that gets no laughs)",
    "Self-pity with no comic turn",
    "Clunky, wordy phrasing",
    "Groaner puns used as filler",
]

PROMPT_HEAD = """Adjudication task, no investigation. Do not look anything up and do not use any tools; answer from your own understanding of the names below. They are names of criteria that comedy communities (stand-up, sketch, humor writing) use to judge comedy. No definitions are given; read each name as a rater would who is handed only that name.

PART 1 - EQUIVALENCE. For each pair, answer this question: would two raters, one applying name A and one applying name B to the same piece of comedy, be checking the same requirement?
Answer with exactly one code: "same" (same requirement), "overlap" (overlapping but different), "different" (different requirement).

PART 2 - ORIENTATION. For each criterion name, answer this question: if a piece receives the top score on this criterion name, is it better or worse comedy by the community's standards?
Answer with exactly one code: "better" (higher = better), "worse" (higher = worse), "unclear".

Reply with ONE JSON object and nothing else, covering every id exactly once:
{"pairs": {"P01": "same|overlap|different", ...}, "forms": {"F01": "better|worse|unclear", ...}}
"""


def h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    pool = json.load(open(f"{V2}/v2_form_pool.json"))
    fm = json.load(open(f"{V2}/v2_form_formality.json"))
    means, sd = fm["form_means"], fm["pooled_sd"]
    rub = {r["form_id"]: r["rubric"] for r in pool}
    by_rub = {r["rubric"]: r["form_id"] for r in pool}
    item_mids = sorted(f[len("v2_items_"):-5] for f in os.listdir(V2)
                       if f.startswith("v2_items_") and f.endswith(".json"))
    bym = defaultdict(list)
    for r in pool:
        for m in r["metric_ids"]:
            bym[m].append(r["form_id"])

    pairs: dict[tuple, dict] = {}   # unordered (fa, fb) -> {sources, mids}

    def add(fa: str, fb: str, source: str, mid: str) -> None:
        key = tuple(sorted((fa, fb)))
        e = pairs.setdefault(key, {"sources": set(), "mids": set()})
        e["sources"].add(source)
        e["mids"].add(mid)

    for mid in item_mids:
        for fa, fb in itertools.combinations(sorted(set(bym[mid])), 2):
            if abs(means[fa] - means[fb]) >= sd:
                add(fa, fb, "candidate", mid)
    arms = json.load(open(f"{V2}/v2_arms.json"))
    for mid, a in arms.items():
        add(a["low_form_id"], a["high_form_id"], "v2_arm", mid)
    t3 = json.load(open(f"{ROOT}/outputs/lexicon/paper1_figdata.json"))["table3"]
    for r in t3["low_wins"] + t3["high_wins"]:
        add(by_rub[r["low"]], by_rub[r["high"]], "v1_table3", r["mid"])

    forms = sorted({f for k in pairs for f in k})
    print(f"pairs={len(pairs)} (candidate={sum('candidate' in e['sources'] for e in pairs.values())}, "
          f"v2_arm={sum('v2_arm' in e['sources'] for e in pairs.values())}, "
          f"v1_table3={sum('v1_table3' in e['sources'] for e in pairs.values())}) forms={len(forms)}")

    rng = random.Random(SEED)
    pair_keys = sorted(pairs)
    rng.shuffle(pair_keys)
    nb = -(-len(pair_keys) // PAIRS_PER_BATCH)
    form_list = forms[:]
    rng.shuffle(form_list)
    batches, key = [], {}
    for b in range(nb):
        real_pairs = pair_keys[b::nb]
        real_forms = form_list[b::nb]
        anchors_same = rng.sample(PARAPHRASE_SAME, 3)
        anchors_diff = rng.sample(CROSS_METRIC_DIFF, 3)
        anchors_fault = rng.sample(FAULT_NAMES, 4)
        p_items = [("real", k) for k in real_pairs] + [("anchor_same", a) for a in anchors_same] + \
                  [("anchor_diff", a) for a in anchors_diff]
        f_items = [("real", f) for f in real_forms] + [("anchor_fault", a) for a in anchors_fault]
        rng.shuffle(p_items)
        rng.shuffle(f_items)
        lines_p, lines_f, kb = [], [], {"pairs": {}, "forms": {}}
        for i, (kind, it) in enumerate(p_items, 1):
            pid = f"P{i:02d}"
            if kind == "real":
                fa, fb = it
                if rng.random() < 0.5:
                    fa, fb = fb, fa
                na, nb_ = rub[fa], rub[fb]
                kb["pairs"][pid] = {"kind": "real", "forms": sorted(it)}
            else:
                na, nb_ = it if rng.random() < 0.5 else (it[1], it[0])
                kb["pairs"][pid] = {"kind": kind, "answer": "same" if kind == "anchor_same" else "different",
                                    "names": [na, nb_]}
            lines_p.append(f"{pid}: A = {json.dumps(na, ensure_ascii=False)} | B = {json.dumps(nb_, ensure_ascii=False)}")
        for i, (kind, it) in enumerate(f_items, 1):
            fid = f"F{i:02d}"
            if kind == "real":
                kb["forms"][fid] = {"kind": "real", "form": it}
                nm = rub[it]
            else:
                kb["forms"][fid] = {"kind": kind, "answer": "worse", "name": it}
                nm = it
            lines_f.append(f"{fid}: {json.dumps(nm, ensure_ascii=False)}")
        prompt = (PROMPT_HEAD + "\nPAIRS:\n" + "\n".join(lines_p) + "\n\nCRITERION NAMES:\n" +
                  "\n".join(lines_f) + "\n")
        bid = f"b{b:02d}"
        batches.append({"batch": bid, "prompt": prompt, "n_pairs": len(lines_p), "n_forms": len(lines_f)})
        key[bid] = kb
    json.dump(batches, open(f"{OUT}/batches.json", "w"), indent=1, ensure_ascii=False)
    json.dump({"key": key,
               "pairs": {"|".join(k): {"sources": sorted(v["sources"]), "mids": sorted(v["mids"])}
                         for k, v in pairs.items()},
               "forms": forms, "seed": SEED},
              open(f"{OUT}/key_private.json", "w"), indent=1, ensure_ascii=False)
    print(f"batches={len(batches)} max_prompt_chars={max(len(b['prompt']) for b in batches)}")


if __name__ == "__main__":
    main()
