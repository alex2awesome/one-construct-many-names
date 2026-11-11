#!/usr/bin/env python3
"""PREREG-21 Leg 3b splice redesign: deterministic material builder and audit.

This module deliberately does not run either the realism judge or the extractor.  It
freezes the all-real splice materials and their provenance before either measurement.
Re-running ``build`` refuses to overwrite an existing freeze.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from bs4 import BeautifulSoup
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[3]
OLD = ROOT / "outputs/lexicon/extraction_validity_20260724"
OUT = ROOT / "outputs/lexicon/extraction_validity_20260726"
PAGES_PATH = OUT / "leg3b_splice_pages.json"
KEY_PATH = OUT / "leg3b_planting_key_private.json"

HOST_IDS = [
    "l3_10", "l3_04", "l3_44", "l3_06", "l3_34",
    "l3_14", "l3_16", "l3_18", "l3_20", "l3_21",
    "l3_24", "l3_25", "l3_40", "l3_27", "l3_29",
    "l3_36", "l3_37", "l3_38", "l3_43", "l3_17",
]

# Separate held-out real sources for the realism gate.  These are kept private with
# the planting key so the public splice file cannot reveal the gate labels.
CONTROL_IDS = [
    "l3_03", "l3_07", "l3_08", "l3_09", "l3_00",
    "l3_11", "extra_vogler", "l3_35", "extra_devops", "extra_kiss",
    "extra_morgridge", "l3_02", "l3_31", "l3_33", "l3_32",
    "l3_39", "l3_26", "l3_41", "l3_46", "l3_49",
]

EXTRA_CONTROLS = {
    "extra_vogler": {
        "page_id": "extra_vogler", "task": "creative-writing",
        "doc": "wiki_christopher_vogler.html",
    },
    "extra_devops": {
        "page_id": "extra_devops", "task": "code-review",
        "doc": "waveb_wikipedia_devops.html",
    },
    "extra_kiss": {
        "page_id": "extra_kiss", "task": "code-review",
        "doc": "existing_42_en_wikipedia_org_wiki_KISS_principle.html",
    },
    "extra_morgridge": {
        "page_id": "extra_morgridge", "task": "grant-funding",
        "doc": "morgridge_news.html",
    },
}

# Every quote is copied from the specified real subreddit page.  "formal" and
# "casual" are the registered register conditions; some casual items are deliberately
# oblique rather than merely contractions of formal rules.
CONSTRUCTS = [
    {
        "construct": "civil interaction",
        "formal": ("p09", 0, "Please ensure that all your posts and comments in this subreddit are civil and use appropriate and respectful language at all times."),
        "casual": ("p01", 4, "Don't be an asshole."),
    },
    {
        "construct": "avoid reposting",
        "formal": ("p18", 1, "Do not repost content that has previously been posted on this subreddit."),
        "casual": ("p14", 2, "Don't post content that was already submitted in the near past."),
    },
    {
        "construct": "no unsolicited advertising",
        "formal": ("p23", 4, "Posts and comments made to advertise and promote a company are strictly prohibited and will result in a ban."),
        "casual": ("p25", 3, "Please keep any unsolicited advertisements of discord servers, youtube channels, or anything else similar to that out of this subreddit."),
    },
    {
        "construct": "use an appropriate flair",
        "formal": ("p24", 4, "Correct flair has to be added to every post."),
        "casual": ("p13", 0, "Please try to select the appropriate flair for your submission"),
    },
    {
        "construct": "hide spoilers",
        "formal": ("p08", 3, "Hide all spoilers except in threads that have already been marked with an official Spoiler tag."),
        "casual": ("p49", 0, "People listen out of order and catch up on old stuff all the time, don't ruin things for newbies."),
    },
    {
        "construct": "do not facilitate piracy",
        "formal": ("p49", 3, "This subreddit has a strict rule against links to unauthorized streams or downloads of commercial content of any kind."),
        "casual": ("p24", 2, "Do not enable piracy"),
    },
    {
        "construct": "tag NSFW material",
        "formal": ("p39", 3, "NSFW posts are allowed, but MUST be tagged."),
        "casual": ("p24", 0, "If your post is NSFW please mark it as such"),
    },
    {
        "construct": "credit the original source",
        "formal": ("p02", 7, "Media Content (photos, videos, etc.) and Fanart should always link the primary source in a comment directly after submission."),
        "casual": ("p37", 0, "Credit your memes if you are reposting someone else's meme from a popular website."),
    },
    {
        "construct": "do not spam",
        "formal": ("p29", 2, "Spam in all forms will result in content removal and likely a ban."),
        "casual": ("p24", 8, "Do not spam numerous posts at the same time."),
    },
    {
        "construct": "do not harass other users",
        "formal": ("p17", 1, "No harassment will be tolerated in /r/oddlyspecific."),
        "casual": ("p07", 7, "Do not troll, harass people, or be an asshole."),
    },
    {
        "construct": "substantive post quality",
        "formal": ("p09", 4, "Low-effort posts will be removed."),
        "casual": ("p03", 1, "Posts should be of decent quality."),
    },
    {
        "construct": "informative titles",
        "formal": ("p35", 4, "titles must contain clear summary of post (vague or editorialized titles will be removed)."),
        "casual": ("p17", 2, "No click-bait or \"haha so true\" titles are allowed."),
    },
    {
        "construct": "no selling or trading",
        "formal": ("p02", 8, "Any posts trying to sell/trade will be removed."),
        "casual": ("p14", 1, "Please, don't try to make money with C&H merchandise or help others to do so."),
    },
    {
        "construct": "no political discussion",
        "formal": ("p19", 2, "Posting any political/Religious comment is strictly prohibited on this Sub."),
        "casual": ("p15", 0, "We do not allow political discussions that aren't about YouTube - take that to r/Politics."),
    },
    {
        "construct": "keep submissions on topic",
        "formal": ("p06", 1, "All posts must be related to AMD, their products or technologies."),
        "casual": ("p21", 1, "Posts must be about Parkitect. No off topic posts."),
    },
    {
        "construct": "clear and visible photographs",
        "formal": ("p04", 0, "Images must have visible land."),
        "casual": ("p31", 1, "Please post clear, full photos that show the area you're looking for help/giving tips about."),
    },
    {
        "construct": "no hate speech",
        "formal": ("p18", 0, "Posting any content that encourages hate is strictly prohibited."),
        "casual": ("p00", 4, "No hate speech. No racism. No sexism. No homophobia. No transphobia. No antisemitism."),
    },
    {
        "construct": "no meme-only submissions",
        "formal": ("p18", 4, "Image-based submissions in which the humor can be conveyed via text alone are not allowed."),
        "casual": ("p05", 8, "Do not post memes in posts or comments."),
    },
    {
        "construct": "route designated content to a megathread",
        "formal": ("p06", 0, "PC build questions, purchase advice and technical support posts are only allowed in the PC Build Questions, Purchase Advice and Technical Support Megathread."),
        "casual": ("p24", 6, "All tier lists go in the megathread."),
    },
    {
        "construct": "verify leaks before posting",
        "formal": ("p24", 7, "If you are to post about leaks and rumors there must be a source or a link to a article that backs the information up."),
        "casual": ("p25", 7, "Before you post images or information about upcoming content, make sure to check whether they've been officially revealed or released yet."),
    },
]

# These are real, verbatim statements about quality or value that do not assert a
# requirement.  They are deliberately close enough to the target construct to tempt
# false extraction.  Provenance is checked against leg3_pages.json at build time.
DISTRACTORS = [
    ("l3_21", "The as-prepared PANI-g-EPVA dispersion can be directly used as the conductive ink or coatings for cellulose fibre paper to prepare conductive paper with high conductivity and mechanical property, which is also suitable for large scalable production."),
    ("l3_05", "Funny comes from violation of conventional logic and expectation while maintaining internal consistency."),
    ("l3_16", "Detachment from a violation is another factor that will affect how funny a situation may be perceived."),
    ("l3_16", "Relief laughter occurs when something bad is presented ( V ) followed by something that is perceived as good ( V+N )."),
    ("l3_09", "Regardless, it is important that this API be clear and precise."),
    ("l3_09", "By giving a name and clear definition to the above ideas, it becomes easy to communicate your intentions to the users of your software."),
    ("l3_10", "However, trial protocols and existing protocol guidelines vary greatly in content and quality."),
    ("l3_10", "These limitations may partly explain why an opportunity exists to improve the quality of protocols."),
    ("l3_10", "The problems that underlie these protocol deficiencies may in turn lead to avoidable protocol amendments, poor trial conduct, and inadequate reporting in trial publications ( 15 , 30 )."),
    ("l3_14", "Software quality is difficult to measure in a way that can practically be applied to developers' decision-making."),
    ("l3_14", "This non-compliance can be detected by measuring the static quality attributes of an application."),
    ("l3_14", "Functional quality is typically assessed dynamically but it is also possible to use static tests (such as software reviews )."),
    ("l3_18", "The investigating committee has tentatively confirmed that there are concerns about the quality of the western blot images reported in this article and that the underlying data are no longer available."),
    ("l3_20", "Sometimes I find subplots exactly like that in novels I read—usually the ones that aren’t very good."),
    ("l3_20", "I prefer to think of story strands rather than subplots as that better explains how they work."),
    ("l3_20", "(Incidentally, the better your book is plotted, the harder it is for analysts to tease apart the various strands of the story.) How many strands does a book need?"),
    ("l3_20", "However, longer novels are more interesting if they have more than one strand and the longer the book, the more strands you can include."),
    ("l3_20", "But you can also achieve a strong plot by making two strands of your story complement or mirror each other."),
    ("l3_24", "[ 8 ] Humor has for a long time been the most frequently used communication tool within advertising, and according to branch active people it is considered to be the most effective."),
    ("l3_24", "Using shocking pictures could affect the way consumers perceive a brand and quality of their product."),
    ("l3_25", "There are many different factors to keep in mind when writing a Comedy Pilot: Am I using a strong comedic voice?"),
    ("l3_25", "The Roadmap Promise Roadmap Writers prides itself on the quality of executives we bring to our programs and we work hard to get you the best feedback possible."),
    ("l3_28", "Quality fiction makes the reader feel before they think."),
    ("l3_28", "Defenders argue it's a useful diagnostic for stories that \"feel like nothing's happening\" despite events."),
    ("l3_29", "A successful TV show pitch determines if a TV series will move from an initial idea into development and production."),
    ("l3_29", "But if these format details are not clear, it stands less of a chance with Netflix and co."),
    ("l3_30", "What makes a good turning point, what makes a good resolution etc?"),
    ("l3_30", "I found this book (along with Robert McKee's 'Story') the most useful out of the many (screenwriting) books I've read because he gets into the nitty gritty hard stuff."),
    ("l3_30", "Even though she was the one looking to buy a good book on writing, I was the one who walked out with this book.For me, all writing falls into one of two categories - focused and revealing, or unfocused and confusing."),
    ("l3_31", "Each item of the guidelines includes examples of good reporting from the published literature, extracted from different types of studies, in model organisms ranging from mammals to invertebrates."),
    ("l3_37", "Les Liaisons dangereuses , written by French author Choderlos de Laclos , is a strong example of polylogic epistolary writing, because the alternating letters work well with the mood of the novel."),
    ("l3_37", "Reading a novel in epistolary form can be much more fun and engaging than sticking with a single, omniscient narrator."),
    ("l3_38", "His writing style has been described as \"realist\" or \" phenomenological \" (in the Husserlian sense) or \"a theory of pure surface\"."),
    ("l3_43", "Setting is more interesting when the familiar becomes unfamiliar."),
    ("l3_43", "But it is absolutely useful and beautiful to work person versus nature as one of your big arcs."),
    ("l3_44", "Perceptions of conflict of interest are as important as actual conflicts of interest."),
    ("l3_01", "A form of humor based on commonplace aspects of everyday life — small details and shared experiences."),
    ("l3_12", "Best puns activate two valid readings simultaneously."),
    ("l3_48", "Comedy writing relies on building tension to a comedic release."),
    ("l3_04", "In general, I find puns to land in the Dad joke category of a good chuckle."),
]

NORMATIVE = re.compile(
    r"(?i)\b(must|should|shall|required|requirement|need to|needs to|have to|"
    r"do not|don't|avoid|ensure|recommend|please|never|always|prohibited|"
    r"not allowed|only allowed|ought|best practice|guideline|instructions?|"
    r"urges?|advises?|recommends?|encourages?|responsible for|requires?|"
    r"calls for|watch the gap)\b"
)
SENTENCE = re.compile(r"(?<=[.!?])\s+")
BOILER = re.compile(
    r"(?i)(cookie|privacy policy|all rights reserved|subscribe|sign up|source_url|"
    r"javascript is disabled|skip to content|advertisement|newsletter)"
)
RUBRIC_PARA = re.compile(
    r"(?i)(^|\s)(rubric|judging criteria|quality heuristics|quality tests|"
    r"process steps|what distinguishes quality|what makes it work)(\s|:)"
)


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def source_path(meta: dict) -> Path:
    base = ROOT / "datasets" / meta["task"] / "online-rubrics"
    found = list(base.rglob(meta["doc"]))
    if len(found) != 1:
        raise RuntimeError(f"{meta['page_id']}: expected one raw source, found {found}")
    return found[0]


def raw_paragraphs(meta: dict) -> list[str]:
    path = source_path(meta)
    if path.suffix.lower() == ".pdf":
        paras = []
        for page in PdfReader(path).pages:
            text = page.extract_text() or ""
            page_paras = re.split(r"\n\s*\n", text)
            if len(page_paras) == 1:
                page_paras = text.splitlines()
            paras.extend(page_paras)
        return [" ".join(p.split()) for p in paras]
    raw = path.read_text(errors="replace")
    if path.suffix.lower() in {".html", ".htm"} or raw.lstrip().startswith("<"):
        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
            tag.decompose()
        pre = soup.find("pre")
        if pre is not None and len(pre.get_text(" ", strip=True).split()) > 500:
            paras = re.split(r"\n\s*\n", pre.get_text("\n", strip=True))
        else:
            paras = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    else:
        paras = re.split(r"\n\s*\n", raw)
    return [" ".join(p.split()) for p in paras]


def descriptive_host(meta: dict, target_words: int = 430) -> tuple[list[str], list[str]]:
    """Take source-order real paragraphs, conservatively deleting norm sentences."""
    kept, removed = [], []
    for para in raw_paragraphs(meta):
        wc = len(para.split())
        if wc < 20 or wc > 330 or BOILER.search(para) or RUBRIC_PARA.search(para):
            continue
        sentences = SENTENCE.split(para)
        good = []
        for sentence in sentences:
            # Imperatives are also excluded when their first token is a common directive.
            first = re.sub(r"^[^A-Za-z]+", "", sentence).split(" ", 1)[0].casefold()
            imperative = first in {
                "add", "ask", "avoid", "be", "check", "choose", "click", "consider",
                "credit", "do", "ensure", "give", "hide", "include", "keep", "make",
                "mark", "never", "post", "provide", "read", "remember", "report",
                "search", "select", "share", "show", "stay", "submit", "tag", "try",
                "use", "write", "discover", "join",
            }
            if NORMATIVE.search(sentence) or imperative:
                removed.append(sentence)
            else:
                good.append(sentence)
        cleaned = " ".join(good).strip()
        if len(cleaned.split()) >= 18:
            kept.append(cleaned)
        if sum(len(x.split()) for x in kept) >= target_words:
            break
    if sum(len(x.split()) for x in kept) < 120:
        raise RuntimeError(f"{meta['page_id']}: insufficient descriptive host prose")
    return kept, removed


def real_control(meta: dict, target_words: int) -> list[str]:
    """Source-order paragraph excerpt with no sentence-level editing."""
    kept = []
    for para in raw_paragraphs(meta):
        wc = len(para.split())
        if wc < 20 or wc > 330 or BOILER.search(para):
            continue
        kept.append(para)
        if sum(len(x.split()) for x in kept) >= target_words:
            break
    if sum(len(x.split()) for x in kept) < 150:
        # Short source pages are valid untouched controls; retain all usable prose.
        kept = [
            p for p in raw_paragraphs(meta)
            if len(p.split()) >= 12 and not BOILER.search(p)
        ]
    return kept


def locate(text: str, value: str) -> dict:
    start = text.find(value)
    if start < 0:
        raise RuntimeError(f"inserted value missing: {value!r}")
    if text.find(value, start + 1) >= 0:
        raise RuntimeError(f"inserted value occurs more than once: {value!r}")
    return {"char_start": start, "char_end": start + len(value)}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def build() -> None:
    if PAGES_PATH.exists() or KEY_PATH.exists():
        raise SystemExit("freeze exists; refusing to overwrite or tune preregistered materials")
    OUT.mkdir(parents=True, exist_ok=True)
    leg1 = {x["page_id"]: x for x in load_json(OLD / "leg1_pages.json")}
    leg3 = {x["page_id"]: x for x in load_json(OLD / "leg3_pages.json")}
    leg3.update(EXTRA_CONTROLS)

    # Verify that every planted quote is truly verbatim in its registered real page/rule.
    for cid, construct in enumerate(CONSTRUCTS):
        for register in ("formal", "casual"):
            page_id, gid, quote = construct[register]
            page = leg1[page_id]
            rule = next(r for r in page["gold_rules"] if r["gid"] == gid)
            if quote not in page["page_text"] or (
                quote not in rule["sn"] and quote not in rule["desc"]
            ):
                raise RuntimeError(f"non-verbatim planting c{cid} {register}: {quote}")

    for source_id, quote in DISTRACTORS:
        full_source = " ".join(raw_paragraphs(leg3[source_id]))
        if quote not in leg3[source_id]["doc_text"] and quote not in full_source:
            raise RuntimeError(f"non-verbatim distractor {source_id}: {quote}")
        if NORMATIVE.search(quote):
            raise RuntimeError(f"distractor contains directive marker: {quote}")

    pages, page_keys, controls = [], [], []
    for i, host_id in enumerate(HOST_IDS):
        meta = leg3[host_id]
        host_paras, removed = descriptive_host(meta)
        # Cyclic Latin arrangement: page i has formal i and casual i-1.  Thus every
        # construct occurs exactly twice, registers balance within page, and no page
        # contains both variants of one construct.
        planting_specs = [(i, "formal"), ((i - 1) % len(CONSTRUCTS), "casual")]
        # A fixed half-rotation prevents a page from receiving distractors taken from
        # its own host passage; the permutation was frozen before judging.
        distractor_specs = [
            DISTRACTORS[(2 * i + 20) % len(DISTRACTORS)],
            DISTRACTORS[(2 * i + 21) % len(DISTRACTORS)],
        ]
        insertions = [
            {
                "kind": "planting",
                "construct_id": cid,
                "construct": CONSTRUCTS[cid]["construct"],
                "register": register,
                "sentence": CONSTRUCTS[cid][register][2],
            }
            for cid, register in planting_specs
        ] + [
            {
                "kind": "distractor",
                "source_page_id": sid,
                "sentence": sentence,
            }
            for sid, sentence in distractor_specs
        ]
        # Fixed spread across paragraph boundaries.  No semantic placement tuning.
        slots = [[], [], [], [], []]
        slots[1].append(insertions[0]["sentence"])
        slots[2].append(insertions[2]["sentence"])
        slots[3].append(insertions[1]["sentence"])
        slots[4].append(insertions[3]["sentence"])
        chunks = []
        for j, para in enumerate(host_paras):
            if j < len(slots):
                chunks.extend(slots[j])
            chunks.append(para)
        for j in range(len(host_paras), len(slots)):
            chunks.extend(slots[j])
        text = "\n\n".join(chunks)
        for insertion in insertions:
            insertion.update(locate(text, insertion["sentence"]))
        page_id = f"splice{i:02d}"
        pages.append({
            "page_id": page_id,
            "page_text": text,
        })
        page_keys.append({
            "page_id": page_id,
            "host_source": {
                "leg3_page_id": host_id,
                "task": meta["task"],
                "doc": meta["doc"],
                "raw_path": str(source_path(meta).relative_to(ROOT)),
            },
            "host_paragraphs_after_norm_removal": host_paras,
            "preexisting_norm_sentences_removed": removed,
            "insertions": insertions,
            "page_sha256": sha256_text(text),
            "word_count": len(text.split()),
        })

    for i, control_id in enumerate(CONTROL_IDS):
        meta = leg3[control_id]
        target = page_keys[i]["word_count"]
        paras = real_control(meta, target)
        text = "\n\n".join(paras)
        controls.append({
            "control_id": f"real{i:02d}",
            "leg3_page_id": control_id,
            "task": meta["task"],
            "doc": meta["doc"],
            "raw_path": str(source_path(meta).relative_to(ROOT)),
            "page_text": text,
            "page_sha256": sha256_text(text),
            "word_count": len(text.split()),
            "content_policy": "source-order paragraph excerpt; no sentence editing",
        })

    public = {
        "schema": "prereg21-leg3b-splice-pages-v1",
        "created_date": "2026-07-26",
        "frozen_before_realism_gate": True,
        "construction": {
            "n_pages": len(pages),
            "n_constructs": len(CONSTRUCTS),
            "plantings_per_page": 2,
            "distractors_per_page": 2,
            "register_balance": "one formal and one casual per page",
            "pair_separation": "formal and casual variants of a construct are never on the same page",
        },
        "pages": pages,
    }
    public_text = json.dumps(public, ensure_ascii=False, indent=2) + "\n"
    key = {
        "schema": "prereg21-leg3b-private-key-v1",
        "created_date": "2026-07-26",
        "freeze_status": "frozen before realism gate and extraction",
        "public_pages_sha256": sha256_text(public_text),
        "source_inputs": {
            "leg1_pages": str((OLD / "leg1_pages.json").relative_to(ROOT)),
            "leg3_pages": str((OLD / "leg3_pages.json").relative_to(ROOT)),
        },
        "constructs": [
            {
                "construct_id": cid,
                "construct": c["construct"],
                **{
                    register: {
                        "source_page_id": c[register][0],
                        "source_gid": c[register][1],
                        "verbatim_sentence": c[register][2],
                    }
                    for register in ("formal", "casual")
                },
            }
            for cid, c in enumerate(CONSTRUCTS)
        ],
        "pages": page_keys,
        "realism_controls_private": controls,
        "design_note": (
            "No page was tuned after seeing gate labels. Gate controls use disjoint real "
            "sources and were fixed at the same time as the splice pages."
        ),
    }
    PAGES_PATH.write_text(public_text)
    KEY_PATH.write_text(json.dumps(key, ensure_ascii=False, indent=2) + "\n")
    print(f"froze {len(pages)} splice pages and {len(controls)} real controls")
    print(f"public sha256 {key['public_pages_sha256']}")


def audit() -> None:
    public = load_json(PAGES_PATH)
    key = load_json(KEY_PATH)
    assert len(public["pages"]) == len(key["pages"]) == 20
    assert len(key["constructs"]) == 20
    assert len(key["realism_controls_private"]) == 20
    current = PAGES_PATH.read_text()
    assert sha256_text(current) == key["public_pages_sha256"]
    seen = {(cid, reg): 0 for cid in range(20) for reg in ("formal", "casual")}
    for page, pkey in zip(public["pages"], key["pages"]):
        assert page["page_id"] == pkey["page_id"]
        assert sha256_text(page["page_text"]) == pkey["page_sha256"]
        plants = [x for x in pkey["insertions"] if x["kind"] == "planting"]
        dists = [x for x in pkey["insertions"] if x["kind"] == "distractor"]
        assert len(plants) == len(dists) == 2
        assert {x["register"] for x in plants} == {"formal", "casual"}
        assert len({x["construct_id"] for x in plants}) == 2
        for x in plants:
            seen[(x["construct_id"], x["register"])] += 1
        for x in pkey["insertions"]:
            assert page["page_text"][x["char_start"]:x["char_end"]] == x["sentence"]
    assert all(n == 1 for n in seen.values())
    print("audit ok: 20 pages, 40 source-verbatim plantings, 40 distractors")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["build", "audit"])
    args = parser.parse_args()
    if args.command == "build":
        build()
    else:
        audit()


if __name__ == "__main__":
    main()
