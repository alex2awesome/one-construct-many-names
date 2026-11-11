"""Planted anchor items for the extraction pass (blinded known-answer checks).

House rule: blinded known-label anchors ride in EVERY judged/extracted batch — resumed or lazy
annotators produce degenerate passes that only anchors catch. Anchors are synthetic source docs
whose correct extraction is unambiguous; their keys are disguised to look like ordinary items
(``_strata``/layer/doc fields mimic real rows). Expected answers live ONLY here, never in payloads.

Scoring per anchor: found match; named_in_source match; head_term match (casefold, null-safe);
required_terms ⊆ extracted key_terms (normalized substring match); forbidden terms absent
(catches annotators substituting the canonical field term for the author's words).
"""
from __future__ import annotations

from typing import List

from .sources import norm_text

_A = [
    dict(
        key="humor::raw::comedy_craft_column_17.html::3",
        name="Targets of jokes and power dynamics",
        description="Jokes should direct mockery at those with more power or status, not at "
                    "vulnerable or marginalized people.",
        doc_text=(
            "Notes from the writers' room, part 17. A thing older comics drill into younger ones: "
            "pick your targets. Comics call this punching up: aim the joke at the powerful — the "
            "boss, the senator, the network — never at the intern, never at the guy who cleans the "
            "club at 2am. Audiences read target choice instantly. A room will forgive a sloppy tag; "
            "it will not forgive a set that keeps swinging down. Punching up is not politeness, it "
            "is physics: status is the fulcrum a joke pivots on, and a joke pivoted off someone "
            "beneath you has nowhere to fall. Separately: keep your openers short. And tip your "
            "bartenders."),
        expect=dict(found=True, named_in_source=True, head_term="punching up",
                    required_terms=["punching up"], forbidden_terms=[]),
    ),
    dict(
        key="creative-writing::raw::prose_rhythm_workshop_notes.html::1",
        name="Sentence length variation",
        description="Vary sentence lengths to create rhythm; avoid monotonous strings of "
                    "same-length sentences.",
        doc_text=(
            "Workshop handout, week 4. Read your draft aloud. If every sentence runs the same "
            "distance before the period, the paragraph flattens into a drone. Let some sentences "
            "run long, stacking clause on clause until the reader leans forward, and then stop "
            "short. Two words. The contrast is what the ear keeps. We are not giving this a name; "
            "just listen for the drone and break it. Also this week: bring three verbs you are "
            "proud of, and cut every adverb you can live without."),
        expect=dict(found=True, named_in_source=False, head_term=None,
                    required_terms=["drone"], forbidden_terms=["sentence variety", "rhythm control"]),
    ),
    dict(
        key="news-homepages::raw::copydesk_memo_endings.html::0",
        name="Story endings that resonate",
        description="End stories with a line that lands — a resonant final detail or quote rather "
                    "than a trailing summary.",
        doc_text=(
            "Copy desk memo. On endings: the last line of a story is called the kicker, and it is "
            "not a summary. A kicker is the detail you saved on purpose — the quote that closes the "
            "circle, the image that keeps arguing after the page is closed. If your final paragraph "
            "begins with 'In conclusion' or restates the nut graf, cut it and look one paragraph up; "
            "the kicker is usually already there, buried. File by 4pm. The vending machine is still "
            "broken."),
        expect=dict(found=True, named_in_source=True, head_term="kicker",
                    required_terms=["kicker"], forbidden_terms=[]),
    ),
    dict(
        key="humor::raw::open_mic_survival_guide.html::2",
        name="Citation formatting consistency",
        description="References must follow a consistent citation style with complete "
                    "bibliographic fields.",
        doc_text=(
            "Open mic survival guide. Get there early and put your name in the bucket yourself. "
            "Watch the room before you go up: what died for the guy before you will die for you "
            "too. Keep a notebook in your pocket, not your phone — the light reads as boredom from "
            "stage. If the mic stand squeaks, take the mic out first thing; wrestling it mid-bit "
            "costs you the laugh you were building. Buy something. Rooms remember who buys "
            "something."),
        expect=dict(found=False, named_in_source=False, head_term=None,
                    required_terms=[], forbidden_terms=[]),
    ),
    dict(
        key="peer-review::raw::methods_seminar_handout_9.html::1",
        name="Reproducibility of analyses",
        description="Analyses should be independently repeatable: others can rerun them and "
                    "obtain the same results.",
        doc_text=(
            "Methods seminar, handout 9. Before you submit, hand your scripts and your data folder "
            "to the person at the next desk and ask them to rerun the whole analysis on their own "
            "machine and get the same numbers, without asking you anything. If they cannot, the "
            "paper is not done. Most failures are boring: a hard-coded path, a package version, a "
            "cell run out of order, a figure edited by hand after the fact. Write down every step "
            "the moment you take it. Your future self counts as another person."),
        expect=dict(found=True, named_in_source=False, head_term=None,
                    required_terms=["rerun", "same numbers"], forbidden_terms=["reproducibility", "replicability"]),
    ),
    dict(
        key="humor::raw::sketch_structure_lecture.html::0",
        name="Escalation in threes",
        description="Comic escalation works in patterns of three: two beats establish, the third "
                    "breaks the pattern.",
        doc_text=(
            "Sketch structure lecture. The oldest tool in the drawer is the rule of three: the "
            "first beat teaches the audience the pattern, the second confirms it, the third betrays "
            "it. Two is not enough to build the expectation; four is one more than the surprise "
            "needed. When a list in your sketch runs flat, count it — nine times out of ten it has "
            "four items, and the fix is deletion. The rule of three is why 'a priest, a rabbi, and "
            "a duck' works and the duck goes last."),
        expect=dict(found=True, named_in_source=True, head_term="rule of three",
                    required_terms=["rule of three"], forbidden_terms=[]),
    ),
    dict(
        key="grant-funding::raw::program_officer_blog_12.html::2",
        name="Budget realism",
        description="Requested budgets should match the actual scope of the proposed work, "
                    "neither padded nor implausibly lean.",
        doc_text=(
            "From a program officer's notebook, post 12. On money: I can tell within a page whether "
            "the numbers were built from the work or the work was built to the numbers. A budget "
            "that lists a full-time postdoc for a six-week analysis, or promises three field sites "
            "on one graduate stipend, tells me the applicants have not walked through their own "
            "study day by day. Cost the study you actually described. Reviewers forgive ambition; "
            "they do not forgive arithmetic that assumes nobody will check. Also, call it what it "
            "is: overhead is overhead."),
        expect=dict(found=True, named_in_source=False, head_term=None,
                    required_terms=[["budget", "cost"]], forbidden_terms=["budget realism"]),
    ),
    dict(
        key="creative-writing::raw::flash_fiction_editor_notes.html::4",
        name="Trimming to essentials",
        description="Cut everything the story can survive without; compression is the craft.",
        doc_text=(
            "Editor's notes for the flash issue. People ask what makes a piece feel inevitable at "
            "800 words. My honest answer: the writer removed everything the story could survive "
            "without, and then removed one more thing. I have no better word for it than surgery "
            "without anesthesia. Elsewhere in these notes I use the term 'voice' a lot — voice is "
            "the thing I will never cut for length — but that is a different matter from the "
            "cutting itself, which has no name in my shop, only a practice: read it again, "
            "lose a sentence, read it again."),
        expect=dict(found=True, named_in_source=False, head_term=None,
                    required_terms=["remove"], forbidden_terms=["compression", "concision"]),
    ),
]


def anchor_contexts() -> List[dict]:
    """Payload-shaped anchor rows (no expected answers; disguised as ordinary items)."""
    out = []
    for a in _A:
        t = a["key"].split("::")[0]
        out.append({
            "key": a["key"], "task": t, "bucket": "general", "layer": "raw",
            "doc": a["key"].split("::")[2], "item_idx": int(a["key"].split("::")[3]),
            "name": a["name"], "description": a["description"], "guidance": None,
            "canonical": a["description"], "source_id": f"anchor/{a['key'].split('::')[2]}",
            "strata": {}, "doc_text": a["doc_text"], "n_doc_chars": len(a["doc_text"]),
        })
    return out


def anchor_keys() -> set:
    return {a["key"] for a in _A}


def score_anchor(key: str, rec: dict) -> dict:
    """Score one extraction record against the anchor's expectation."""
    exp = next(a["expect"] for a in _A if a["key"] == key)
    head = rec.get("head_term")
    pool = " || ".join(norm_text(t) for t in (rec.get("key_terms") or []) + ([head] if head else []))

    def _req_ok(req) -> bool:  # str = must appear; list = any-of alternatives
        alts = req if isinstance(req, list) else [req]
        return any(norm_text(t) in pool for t in alts)

    def _head_ok() -> bool:
        if not (exp["head_term"] or head):
            return True
        a, b = norm_text(head or ""), norm_text(exp["head_term"] or "")
        return a.removeprefix("the ").strip() == b.removeprefix("the ").strip()

    checks = {
        "found": bool(rec.get("found")) == exp["found"],
        "named": bool(rec.get("named_in_source")) == exp["named_in_source"],
        "head": _head_ok(),
        "required": all(_req_ok(r) for r in exp["required_terms"]),
        "forbidden": not any(norm_text(t) in pool for t in exp["forbidden_terms"]),
    }
    return {"key": key, "pass": all(checks.values()), **checks}


def score_anchor_batch(records: dict) -> dict:
    """records: {key: extraction record} — returns per-anchor results + pass rate over PRESENT
    anchors and n_missing (an absent anchor is a failed batch, count it)."""
    res = [score_anchor(k, records[k]) for k in sorted(anchor_keys() & set(records))]
    missing = sorted(anchor_keys() - set(records))
    n = len(res) + len(missing)
    return {"results": res, "missing": missing,
            "pass_rate": (sum(r["pass"] for r in res) / n) if n else 0.0}
