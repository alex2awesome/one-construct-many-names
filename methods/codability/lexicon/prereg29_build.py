#!/usr/bin/env python
"""PREREG-29 (name swap in legal / patents / grants / regulatory comments) materials.

Frozen design: notes/2026-10-12__prereg29-legal-patents-grants-reg-name-swap.md.
Mirrors prereg26_build.py (creative writing). Per domain writes OUT/jobs/{formality,adj_v2,adj,check}.json,
OUT/scaffolds.json, OUT/item_pool.jsonl (+ source), OUT/constructs.json and OUT/private/keys.json.
usage: prereg29_build.py [legal patents grants reg]
"""
from __future__ import annotations

import glob
import hashlib
import io
import itertools
import json
import os
import random
import re
import sys
import zipfile
from collections import Counter, defaultdict

ROOT = "ROOT"
LEX = f"{ROOT}/outputs/lexicon"
P10 = f"{LEX}/register_corpus_20260723"
V2HEAD = f"{LEX}/prereg25_20261012/readjudication_v2/prompt_v2_head.txt"
sys.path.insert(0, f"{ROOT}/methods/codability/lexicon")
from prereg26_build import FORM_HEAD_TAIL, h  # noqa: E402

PAIRS_PER_BATCH = 60
N_CONSTRUCTS = 20
MAX_FORMS = 6
POOL = 200

# ---------------------------------------------------------------- domain configuration (written before freeze)
D = {
    "legal": dict(
        task="legal-outcome-prediction",
        strict_comm="legal communities (judges, litigators, legal-writing instructors) use to judge legal arguments",
        piece="legal argument", better="a better or worse legal argument", comm_short="a legal community",
        s0_material="legal writing (case facts and claims) from U.S. federal employment-discrimination cases",
        community="the legal community", judged="legal arguments",
        s1_texts="passages of legal writing (case facts and claims) from U.S. federal employment-discrimination cases",
        s1_people="Lawyers and judges in the legal community",
        s2_units="legal passages", s2_unit="passage",
        s2_desc="Each passage sets out the facts and claims of a U.S. federal employment-discrimination case",
        check_items="passages of legal writing",
        same=[("Cite controlling authority", "Reliance on binding precedent from the governing jurisdiction"),
              ("Clear statement of the issue", "The question presented is stated clearly"),
              ("Accurate statement of the facts", "Facts are stated accurately and without distortion"),
              ("Concise writing", "Brevity: no unnecessary words"),
              ("Address the opposing argument", "Engages with and rebuts the other side's position"),
              ("Correct citation format", "Citations follow Bluebook form"),
              ("Logical organization", "The argument is organized in a logical order"),
              ("Apply the law to the facts", "Application of legal rules to the specific facts of the case")],
        diff=[("Correct citation format", "Persuasive theme of the case"),
              ("Concise writing", "Standing to sue"),
              ("Accurate statement of the facts", "Typeface and margin requirements"),
              ("Cite controlling authority", "Emotional appeal to the jury"),
              ("Statute of limitations compliance", "Clear topic sentences"),
              ("Logical organization", "Request for oral argument"),
              ("Preservation of error for appeal", "Plain English vocabulary"),
              ("Table of contents included", "Address the opposing argument")],
        fault=["Misstating the holding of a cited case", "Citing overruled authority", "Mischaracterizing the record",
               "Ad hominem attacks on opposing counsel", "Burying the main argument", "Typos and grammatical errors",
               "Conclusory assertions without support", "Exceeding the page limit"]),
    "patents": dict(
        task="patents",
        strict_comm="patent communities (patent examiners, patent attorneys and agents, patent-drafting guides) use to judge patent applications",
        piece="patent application", better="a better or worse patent application", comm_short="a patent community",
        s0_material="abstracts of U.S. patent applications",
        community="the patent community", judged="patent applications",
        s1_texts="abstracts of U.S. patent applications",
        s1_people="Patent examiners and patent practitioners",
        s2_units="patent abstracts", s2_unit="abstract",
        s2_desc="Each abstract summarizes a U.S. patent application",
        check_items="patent abstracts",
        same=[("Claims supported by the specification", "Written description support for every claim"),
              ("Clear claim language", "Claims are definite and unambiguous"),
              ("Enablement", "The disclosure teaches a skilled person how to make and use the invention"),
              ("Novelty over prior art", "The invention is new compared with existing references"),
              ("Consistent terminology", "Terms are used consistently throughout the application"),
              ("Concise abstract", "A brief abstract summarizing the invention"),
              ("Broad independent claims", "Independent claims of wide scope"),
              ("Drawings illustrate every claimed feature", "Every claimed element is shown in the figures")],
        diff=[("Concise abstract", "Non-obviousness"),
              ("Clear claim language", "Filing fee paid"),
              ("Enablement", "Title length"),
              ("Consistent terminology", "Commercial value of the invention"),
              ("Drawings illustrate every claimed feature", "Best mode disclosure"),
              ("Novelty over prior art", "Proper margins on drawing sheets"),
              ("Broad independent claims", "Inventor oath or declaration"),
              ("Priority claim to an earlier application", "Clear claim language")],
        fault=["Indefinite claim terms", "New matter added after filing", "Claims anticipated by prior art",
               "Inconsistent reference numerals", "Vague functional language", "Typographical errors in the claims",
               "Lack of antecedent basis", "Obvious combination of known elements"]),
    "grants": dict(
        task="grant-funding",
        strict_comm="grant-review communities (study sections, program officers, grant-writing advisors) use to judge research grant proposals",
        piece="grant proposal", better="a better or worse grant proposal", comm_short="a grant-review community",
        s0_material="abstracts of research grant proposals submitted to the U.S. National Institutes of Health",
        community="the grant-review community", judged="grant proposals",
        s1_texts="abstracts of research grant proposals submitted to the U.S. National Institutes of Health",
        s1_people="Grant reviewers and research funders",
        s2_units="grant abstracts", s2_unit="abstract",
        s2_desc="Each abstract summarizes a research grant proposal submitted to the U.S. National Institutes of Health",
        check_items="grant-proposal abstracts",
        same=[("Clear specific aims", "The aims of the project are stated clearly"),
              ("Significance of the problem", "The project addresses an important problem"),
              ("Feasibility", "The work can realistically be completed as proposed"),
              ("Rigorous methods", "Sound and rigorous research design"),
              ("Qualified investigators", "The team has the expertise to do the work"),
              ("Innovation", "The project proposes novel ideas or approaches"),
              ("Realistic timeline", "The schedule for the work is achievable"),
              ("Preliminary data", "Pilot results that support the hypothesis")],
        diff=[("Clear specific aims", "Adequate institutional facilities"),
              ("Realistic timeline", "Novelty of the hypothesis"),
              ("Preliminary data", "Font size and page limits"),
              ("Innovation", "Budget justification"),
              ("Qualified investigators", "Statistical power analysis"),
              ("Rigorous methods", "Letters of support"),
              ("Significance of the problem", "Data sharing plan"),
              ("Inclusion of women and minorities", "Feasibility")],
        fault=["Overly ambitious aims", "Lack of preliminary data", "Vague hypotheses",
               "Fatal flaws in the experimental design", "Jargon-heavy, unreadable writing", "Unrealistic budget",
               "Incremental, unoriginal ideas", "Missing alternative strategies"]),
    "reg": dict(
        task="notice-and-comment",
        strict_comm="regulatory communities (agency staff who review public comments, regulatory lawyers, comment-writing guides) use to judge public comments on proposed rules",
        piece="public comment", better="a better or worse public comment", comm_short="a regulatory community",
        s0_material="public comments submitted to U.S. federal agencies on proposed rules",
        community="the regulatory community", judged="public comments on proposed rules",
        s1_texts="public comments that were submitted to U.S. federal agencies on proposed rules",
        s1_people="Agency staff and others in the regulatory community",
        s2_units="public comments", s2_unit="comment",
        s2_desc="Each comment was submitted to a U.S. federal agency in response to a proposed rule",
        check_items="public comments on proposed rules",
        same=[("Cite supporting data", "The comment backs its claims with evidence and data"),
              ("Specific suggested changes", "Proposes concrete alternative regulatory text"),
              ("Identify the commenter's interest", "The commenter explains who they are and how they are affected"),
              ("Address the agency's questions", "Responds to the questions the agency asked in the proposal"),
              ("Clear and concise", "Brief and easy to understand"),
              ("Reference the docket number", "The comment identifies the rulemaking docket"),
              ("Explain the impact of the rule", "Describes how the proposed rule would affect the commenter"),
              ("Legal authority arguments", "Arguments about the agency's statutory authority")],
        diff=[("Cite supporting data", "Polite tone"),
              ("Reference the docket number", "Cost-benefit analysis"),
              ("Clear and concise", "Small business impacts"),
              ("Specific suggested changes", "Signature of the commenter"),
              ("Identify the commenter's interest", "Environmental justice considerations"),
              ("Address the agency's questions", "Submission before the deadline"),
              ("Legal authority arguments", "Personal anecdote"),
              ("Explain the impact of the rule", "Formatting of attachments")],
        fault=["Form letter with no original content", "Off-topic remarks", "Unsupported assertions",
               "Abusive or profane language", "Misstating the proposed rule", "Duplicate submissions",
               "Vague opposition without reasons", "Factual errors"]),
}

STRICT_HEAD = """Adjudication task, no investigation. Do not look anything up and do not use any tools; answer from your own understanding of the names below. They are names of criteria that {strict_comm}. No definitions are given; read each name as a rater would who is handed only that name.

PART 1 - EQUIVALENCE. For each pair, answer this question: would two raters, one applying name A and one applying name B to the same {piece}, be checking the same requirement?
Answer with exactly one code: "same" (same requirement), "overlap" (overlapping but different), "different" (different requirement).

PART 2 - ORIENTATION. For each criterion name, answer this question: if a {piece} receives the top score on this criterion name, is it {better} by the community's standards?
Answer with exactly one code: "better" (higher = better), "worse" (higher = worse), "unclear".

Reply with ONE JSON object and nothing else, covering every id exactly once:
{{"pairs": {{"P01": "same|overlap|different", ...}}, "forms": {{"F01": "better|worse|unclear", ...}}}}
"""

S_T = {
    "S0": ("Scoring task, no investigation. You are evaluating {s0_material}. The criterion named below is one used by "
           "{community} to judge {judged}. No definition is provided --- apply the criterion as you understand it from "
           "its name. CRITERION NAME: {{name}}. For EVERY item, judge how well the item satisfies the criterion named "
           "above on a 1-7 scale (1 = clearly fails it, 7 = clearly satisfies it). Judge each item independently from "
           "its text alone."),
    "S1": ("This is a rating job; do not research anything. The texts below are {s1_texts}. {s1_people} use the "
           "criterion named below when they assess {judged}. You will not be given a definition; rely on your own "
           "understanding of what the name means. CRITERION: {{name}}. Rate every text on a 1 to 7 scale for how far it "
           "meets this criterion (1 = it plainly does not meet it, 7 = it plainly meets it). Rate each text on its own, "
           "using only its text."),
    "S2": ("Please score the {s2_units} listed below. Do not look anything up. {s2_desc}, and the criterion given below "
           "is a standard that people in {community} apply to {judged}. Interpret the criterion from its name alone, "
           "since no definition is supplied. Criterion to apply: {{name}}. Give each {s2_unit} a score from 1 to 7, "
           "where 7 means the {s2_unit} clearly meets the criterion and 1 means it clearly does not. Score the "
           "{s2_units} independently of one another, based on the text alone."),
}

CHECK_PROMPT = """Meaning-equivalence check, no investigation; do not use any tools. Below are three instructions given to a rater who scores {check_items} on a named criterion. S0 is the reference. For S1 and for S2 separately, answer: does it ask the rater to do the same task as S0, with the same meaning (same material, same source of the criterion, no definition given, same 1-7 scale with the same anchors, items judged independently from their text alone)? Ignore wording and order. Answer "equivalent" or "not equivalent" with a one-sentence reason for each.

S0: {S0}

S1: {S1}

S2: {S2}

Reply with ONE JSON object and nothing else: {{"S1": {{"verdict": "equivalent|not equivalent", "reason": "..."}}, "S2": {{"verdict": "equivalent|not equivalent", "reason": "..."}}}}
"""


def calibrated_head(c):
    t = open(V2HEAD).read()
    reps = [("comedy communities (stand-up, sketch, humor writing) use to judge comedy", c["strict_comm"]),
            ("the same piece of comedy", f"the same {c['piece']}"),
            ("the quality a comedy community cares about", f"the quality {c['comm_short']} cares about"),
            ("if a piece receives the top score", f"if a {c['piece']} receives the top score"),
            ("is it better or worse comedy by the community's standards?", f"is it {c['better']} by the community's standards?")]
    for a, b in reps:
        assert t.count(a) == 1, a
        t = t.replace(a, b)
    return t


def build_adj(pairs, forms_list, name_of, head, tag, rng, c):
    nb = max(1, -(-len(pairs) // PAIRS_PER_BATCH))
    batches, key = [], {}
    for b in range(nb):
        real_pairs = pairs[b::nb]
        real_forms = forms_list[b::nb]
        p_items = [("real", k) for k in real_pairs] + [("anchor_same", a) for a in rng.sample(c["same"], 3)] + \
                  [("anchor_diff", a) for a in rng.sample(c["diff"], 3)]
        f_items = [("real", f) for f in real_forms] + [("anchor_fault", a) for a in rng.sample(c["fault"], 4)]
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


def trunc(t, n=1000):
    t = re.sub(r"[ \t]+", " ", t.strip())
    if len(t) <= n:
        return t
    cut = t[:n]
    sp = cut.rfind(" ")
    return cut[:sp] if sp > n - 100 else cut


def pool(dom):
    """-> (list of {item_id, text}, source description)"""
    D_ = f"{ROOT}/datasets"
    if dom == "legal":
        src = "datasets/legal-outcome-prediction/title_vii_balanced_v2.jsonl (field facts; all rows; truncated to 1,000 chars at a word boundary)"
        rows = [json.loads(l) for l in open(f"{D_}/legal-outcome-prediction/title_vii_balanced_v2.jsonl")]
        cand = [{"item_id": f"op{r['opinion_id']}", "text": trunc(r["facts"])} for r in rows if len(r.get("facts") or "") >= 400]
    elif dom == "patents":
        import pandas as pd
        src = ("datasets/patents/patents_final_outcome_balanced.csv.gz, first 20,000 rows, ABSTRACT section of field "
               "text (up to the next CAPS header), kept if 400-1,000 chars")
        d = pd.read_csv(f"{D_}/patents/patents_final_outcome_balanced.csv.gz", nrows=20000)
        cand = []
        for i, t in enumerate(d["text"].fillna("")):
            m = re.match(r"\s*ABSTRACT:\s*(.*?)(?:\n\s*[A-Z][A-Z /&-]{3,}:\s*\n|$)", t, re.S)
            if not m:
                continue
            a = re.sub(r"\s+", " ", m.group(1)).strip()
            if 400 <= len(a) <= 1000:
                cand.append({"item_id": f"pat{i}", "text": a})
    elif dom == "grants":
        import pandas as pd
        src = "datasets/grant-funding/nih_exporter/RePORTER_PRJABS_C_FY2024.zip (NIH RePORTER project abstracts FY2024), kept if 400-1,000 chars"
        z = zipfile.ZipFile(f"{D_}/grant-funding/nih_exporter/RePORTER_PRJABS_C_FY2024.zip")
        d = pd.read_csv(z.open(z.namelist()[0]), encoding="latin-1")
        cand = []
        for aid, t in zip(d["APPLICATION_ID"], d["ABSTRACT_TEXT"].fillna("")):
            t = re.sub(r"\s+", " ", str(t)).strip()
            if 400 <= len(t) <= 1000:
                cand.append({"item_id": f"nih{aid}", "text": t})
    elif dom == "reg":
        import pandas as pd
        src = "datasets/notice-and-comment/v4/nc_dense8b_all.csv (field text = public comment), kept if 400-1,000 chars"
        d = pd.read_csv(f"{D_}/notice-and-comment/v4/nc_dense8b_all.csv")
        cand = [{"item_id": f"nc{i}", "text": str(t).strip()} for i, t in enumerate(d["text"].fillna(""))
                if 400 <= len(str(t).strip()) <= 1000]
    seen, uniq = set(), []
    for c in cand:
        if c["text"] in seen:
            continue
        seen.add(c["text"])
        uniq.append(c)
    uniq.sort(key=lambda c: h(c["item_id"] + f"p29-pool-{dom}"))
    return uniq[:POOL], src, len(uniq)


def constructs(c):
    t = c["task"]
    r1 = json.load(open(f"{LEX}/partition_{t}_R1.json"))
    l0 = json.load(open(f"{LEX}/partition_{t}_L0v4.json"))
    cn = json.load(open(f"{LEX}/cluster_names_{t}_L0v4.json"))
    att = Counter(str(v) for v in l0.values())
    g = defaultdict(dict)  # r1 -> lower(name) -> (name, l0 id)
    for cid, r in r1.items():
        if cid in cn and att.get(cid, 0) > 0 and len(cn[cid]["name"]) < 90:  # 90 = storage cap: truncated names out
            nm = re.sub(r"\s+", " ", cn[cid]["name"]).strip()
            key = nm.lower()
            if key not in g[r] or int(cid) < int(g[r][key][1]):
                g[r][key] = (nm, cid)
    elig = sorted([r for r in g if len(g[r]) >= 2], key=lambda r: h(f"{t}|{r}|p29"))
    chosen = elig[:N_CONSTRUCTS]
    out = []
    for k, r in enumerate(chosen):
        fl = sorted(g[r].values(), key=lambda x: h(f"{t}|{x[0]}|p29-form"))
        out.append({"mid": f"c{k:02d}", "r1_id": r, "n_forms_total": len(fl),
                    "forms": [{"form_id": f"c{k:02d}:{i}", "name": nm, "l0_cluster": cid, "n_mentions": att[cid]}
                              for i, (nm, cid) in enumerate(fl[:MAX_FORMS])]})
    return out, len(g), len(elig)


def main():
    doms = sys.argv[1:] or list(D)
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
    for dom in doms:
        c = D[dom]
        OUT = f"{LEX}/prereg29_{c['task']}_20261012"
        os.makedirs(f"{OUT}/jobs", exist_ok=True)
        os.makedirs(f"{OUT}_private", exist_ok=True)
        cons, n_r1, n_elig = constructs(c)
        name_of = {f["form_id"]: f["name"] for x in cons for f in x["forms"]}
        pairs = sorted(tuple(sorted(p)) for x in cons for p in itertools.combinations([f["form_id"] for f in x["forms"]], 2))
        forms = sorted(name_of)
        # formality: 2 passes, chunks of <= 41 forms + 10 anchors
        form_jobs, form_key = [], {}
        nch = max(1, -(-len(forms) // 41))
        for k in (1, 2):
            rng = random.Random(f"p29-{dom}-form-pass{k}")
            fl = forms[:]
            rng.shuffle(fl)
            for ch in range(nch):
                items = [("real", f) for f in fl[ch::nch]] + [("anchor", a) for a in rng.sample(bank, 10)]
                rng.shuffle(items)
                lines, kb = [], {}
                for kind, it in items:
                    if kind == "real":
                        iid = "n" + h(f"p29fm|{dom}|{k}|{it}")[:10]
                        kb[iid] = {"kind": "real", "form": it}
                        term = name_of[it]
                    else:
                        iid = "n" + h(f"p29fa|{dom}|{k}|{ch}|{it[0]}")[:10]
                        kb[iid] = {"kind": "anchor", "term": it[0], "stratum": it[1]}
                        term = it[0]
                    lines.append(json.dumps({"id": iid, "term": term}, ensure_ascii=False))
                jid = f"fp{k}c{ch}"
                form_jobs.append({"job": jid, "prompt": instr + FORM_HEAD_TAIL + "\n".join(lines) + "\n"})
                form_key[jid] = kb
        json.dump(form_jobs, open(f"{OUT}/jobs/formality.json", "w"), indent=1, ensure_ascii=False)
        # adjudication: calibrated (v, primary) and strict (b, counts only)
        head_v = calibrated_head(c)
        head_s = STRICT_HEAD.format(**c)
        rng = random.Random(f"p29-{dom}-adjv2-20261012")
        pl, fl = pairs[:], forms[:]
        rng.shuffle(pl); rng.shuffle(fl)
        adjv, keyv = build_adj(pl, fl, name_of, head_v, "v", rng, c)
        rng = random.Random(f"p29-{dom}-adj-20261012")
        pl, fl = pairs[:], forms[:]
        rng.shuffle(pl); rng.shuffle(fl)
        adjs, keys = build_adj(pl, fl, name_of, head_s, "b", rng, c)
        json.dump(adjv, open(f"{OUT}/jobs/adj_v2.json", "w"), indent=1, ensure_ascii=False)
        json.dump(adjs, open(f"{OUT}/jobs/adj.json", "w"), indent=1, ensure_ascii=False)
        open(f"{OUT}/prompt_calibrated_head_USED.txt", "w").write(head_v)
        open(f"{OUT}/prompt_strict_head_USED.txt", "w").write(head_s)
        S = {k: v.format(**c) for k, v in S_T.items()}
        json.dump(S, open(f"{OUT}/scaffolds.json", "w"), indent=1)
        json.dump([{"job": "scaffold_check", "prompt": CHECK_PROMPT.format(check_items=c["check_items"], **S)}],
                  open(f"{OUT}/jobs/check.json", "w"), indent=1, ensure_ascii=False)
        items, src, n_cand = pool(dom)
        with open(f"{OUT}/item_pool.jsonl", "w") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        json.dump({"task": c["task"], "domain": dom, "n_r1_constructs_attested": n_r1, "n_eligible_ge2_forms": n_elig,
                   "n_selected": len(cons), "n_forms": len(forms), "n_pairs": len(pairs),
                   "item_source": src, "n_candidate_items": n_cand, "pool_size": len(items),
                   "constructs": cons}, open(f"{OUT}/constructs.json", "w"), indent=1, ensure_ascii=False)
        json.dump({"formality": form_key, "adj": keys, "adj_v2": keyv, "name_of": name_of,
                   "pairs": ["|".join(p) for p in pairs], "forms": forms},
                  open(f"{OUT}_private/keys.json", "w"), indent=1, ensure_ascii=False)
        print(f"{dom}: r1={n_r1} elig={n_elig} cons={len(cons)} forms={len(forms)} pairs={len(pairs)} "
              f"form_jobs={len(form_jobs)} adj_v2={len(adjv)} adj={len(adjs)} pool={len(items)}/{n_cand}")


if __name__ == "__main__":
    main()
