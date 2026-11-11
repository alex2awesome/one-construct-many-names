#!/usr/bin/env python3
"""CRP-style streaming ingestion: new sources -> extracted criteria -> seat into the
certified L0->R3 hierarchy (join an existing table or open a new one).

Chinese-restaurant framing: each new criterion either sits at an existing L0 table
(same criterion, CONFIRM_PROTOCOL_L0_V2 strictness), or opens a new L0 table and then
either joins an existing R1 construct (same construct, STRICT_BUILD_PROTOCOL_R1
strictness) or opens a new construct. The realized new-table rates on genuinely fresh
sources are the empirical check of the Good-Turing predictions in
outputs/lexicon/coverage_census_20260719.json (L0 ~.40-.59, R1 ~.07-.21).

Division of labor (repo standing rules): this module only orchestrates — fetch, shard
payloads, shortlist candidates (TF-IDF is a SHORTLIST, never a decision), validate,
ingest, and keep an append-only ledger. Every extraction and every seating decision is
made by an LLM judge (Sonnet+ subagents run in-session over the emitted payload files).
Canonical partition_* files are NEVER modified; all seatings live in the sidecar
outputs/lexicon/crp_ingest/<task>/.

Wave lifecycle (one "wave" = one batch of new sources):
  1. fetch          urls.txt -> fetched/<slug>.txt (+ meta)   [or drop files in by hand]
  2. emit-extract   fetched/ -> extract_payload_<i>.txt        (subagents write extract_out_<i>.jsonl)
  3. ingest-extract validate + assemble new_criteria.jsonl     (then SPOT-CHECK before seating)
  4. emit-seat      shortlists + hidden anchors -> seat_payload_<i>.txt
                                                               (subagents write seat_out_<i>.jsonl)
  5. apply          anchor gates -> seating ledger + partition delta + GT comparison

Usage: python -m methods.codability.lexicon.crp_ingest <stage> --task T --wave DIR [...]
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import html.parser
import json
import os
import re
import urllib.request
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
LEX = os.path.join(ROOT, "outputs", "lexicon")

EXTRACT_INSTRUCTIONS = """You are extracting EVALUATION CRITERIA from a source document about {task_desc}.
A criterion is a quality standard the author uses or recommends for judging work in this domain
(what makes it good/bad), NOT a step-by-step instruction, fact, or anecdote.
For EACH distinct criterion the author articulates, emit ONE JSON line:
{{"doc": "<doc filename>", "item_idx": <int, 0-based within doc>,
  "name": "<3-8 word title>",
  "canonical": "<one sentence: 'The {task_word} should ...' normative form>",
  "description": "<1-3 sentences in the author's terms>",
  "evidence": "<short verbatim phrase (<=15 words) copied exactly from the source that grounds it>"}}
Rules: extract only criteria genuinely present in the text (empty output for a doc with none is fine,
but do not default to empty — most of these documents contain several); evidence MUST be verbatim;
do not invent; do not merge distinct criteria; skip pure how-to mechanics with no quality standard.
Write JSON lines to the output file named in the payload header. WRITE THE FILE; the file is the deliverable."""

SEAT_INSTRUCTIONS = """You are seating NEW evaluation criteria into an existing hierarchy (Chinese-restaurant style).
For each ITEM you get candidate L0 clusters (existing criteria) and candidate R1 constructs.
Decide, in order:
1. L0: does any candidate express the SAME CRITERION — same quality standard, same facet, judged
   the same way? Paraphrase = SAME; a different facet of one quality = NOT same (that is R1's job).
   Uncertain -> NOT same. This is a strict gate.
2. If L0=NEW: does any R1 candidate name the SAME CONSTRUCT — operationally interchangeable, or a
   distinct facet of one latent quality? Uncertain -> NEW.
Output ONE JSON line per item: {"item_id": "<id>", "l0": "<candidate cluster id or NEW>",
"r1": "<candidate construct id or NEW or INHERIT>", "why": "<=12 words"}
(r1=INHERIT when l0 is an existing cluster.) Judge every item. Candidates are shortlists from a
lexical net — trust the text shown, not the ranking. Write JSON lines to the output file named in
the payload header. WRITE THE FILE; the file is the deliverable."""

TASK_DESC = {
    "humor": ("comedy/humor writing and performance", "humor"),
    "creative-writing": ("creative writing craft", "writing"),
    "news-homepages": ("journalism and news editing", "coverage"),
    "math-stackexchange": ("mathematical writing and exposition", "mathematics"),
}


class _Text(html.parser.HTMLParser):
    SKIP = {"script", "style", "nav", "footer", "header", "noscript", "svg", "form"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self._skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip:
            self._skip -= 1
        if tag in {"p", "li", "h1", "h2", "h3", "h4", "br", "div", "tr"}:
            self.parts.append("\n")

    def handle_data(self, d):
        if not self._skip:
            self.parts.append(d)


def html_to_text(raw: str) -> str:
    p = _Text()
    try:
        p.feed(raw)
    except Exception:
        pass
    t = re.sub(r"[ \t]+", " ", "".join(p.parts))
    return re.sub(r"\n{3,}", "\n\n", t).strip()


def slug(url: str) -> str:
    h = hashlib.sha1(url.encode()).hexdigest()[:10]
    tail = re.sub(r"[^a-z0-9]+", "-", url.split("//")[-1].lower())[:60].strip("-")
    return f"{tail}_{h}"


def visited_urls(task: str) -> set:
    """Normalized URLs already in the scraped corpus (dedup guard)."""
    seen = set()
    p = os.path.join(ROOT, "datasets", task, "online-rubrics", "urls-visited.csv")
    if os.path.exists(p):
        for ln in open(p, errors="ignore"):
            u = ln.strip().split(",")[0].strip('"')
            if u.startswith("http"):
                seen.add(norm_url(u))
    return seen


def norm_url(u: str) -> str:
    u = re.sub(r"^https?://(www\.)?", "", u.strip().lower()).rstrip("/")
    return u.split("#")[0].split("?")[0]


def cmd_fetch(a):
    os.makedirs(f"{a.wave}/fetched", exist_ok=True)
    seen = visited_urls(a.task)
    urls = [u.strip() for u in open(a.urls) if u.strip() and not u.startswith("#")]
    meta = []
    for u in urls:
        if norm_url(u) in seen:
            print(f"SKIP (already in corpus): {u}")
            continue
        s = slug(u)
        out = f"{a.wave}/fetched/{s}.txt"
        if os.path.exists(out):
            meta.append({"url": u, "slug": s, "status": "cached"})
            continue
        try:
            req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0 (research corpus builder)"})
            raw = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "ignore")
            txt = html_to_text(raw)
            if len(txt) < 800:
                meta.append({"url": u, "slug": s, "status": f"thin({len(txt)}ch)"})
                print(f"THIN  {u} ({len(txt)} chars) — fetch manually (e.g. WebFetch) if wanted")
                continue
            open(out, "w").write(txt[:40000])
            meta.append({"url": u, "slug": s, "status": "ok", "chars": len(txt)})
            print(f"OK    {u} -> {s}.txt ({len(txt)} chars)")
        except Exception as e:
            meta.append({"url": u, "slug": s, "status": f"error:{e}"})
            print(f"FAIL  {u}: {e}")
    _append(f"{a.wave}/fetch_meta.jsonl", meta)


def cmd_emit_extract(a):
    docs = sorted(glob.glob(f"{a.wave}/fetched/*.txt"))
    assert docs, f"no fetched docs in {a.wave}/fetched/"
    td, tw = TASK_DESC.get(a.task, (a.task, "work"))
    per = max(1, a.docs_per_shard)
    shards = [docs[i:i + per] for i in range(0, len(docs), per)]
    for i, sh in enumerate(shards):
        with open(f"{a.wave}/extract_payload_{i:02d}.txt", "w") as f:
            f.write(f"OUTPUT FILE: {a.wave}/extract_out_{i:02d}.jsonl\n\n")
            f.write(EXTRACT_INSTRUCTIONS.format(task_desc=td, task_word=tw) + "\n")
            for d in sh:
                name = os.path.basename(d)[:-4]
                f.write(f"\n===== DOC: {name} =====\n{open(d).read()[:28000]}\n")
    print(f"emitted {len(shards)} extraction payload(s) for {len(docs)} docs -> {a.wave}/extract_payload_*.txt")


def cmd_ingest_extract(a):
    rows, bad = [], Counter()
    doc_text = {os.path.basename(p)[:-4]: open(p).read() for p in glob.glob(f"{a.wave}/fetched/*.txt")}

    def _norm(s):
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", s.lower())).strip()

    for p in sorted(glob.glob(f"{a.wave}/extract_out_*.jsonl")):
        for ln in open(p):
            ln = ln.strip()
            if not ln:
                continue
            try:
                r = json.loads(ln)
            except Exception:
                bad["unparseable"] += 1
                continue
            if not all(r.get(k) for k in ("doc", "name", "canonical", "description", "evidence")):
                bad["missing-field"] += 1
                continue
            src = doc_text.get(r["doc"])
            if src is None:
                bad["unknown-doc"] += 1
                continue
            if _norm(r["evidence"]) not in _norm(src):
                bad["evidence-not-verbatim"] += 1
                continue
            r["key"] = f"{a.task}::crp::{r['doc']}::{r.get('item_idx', len(rows))}"
            rows.append(r)
    # dedup within wave by (doc, normalized name)
    out, seen = [], set()
    for r in rows:
        k = (r["doc"], _norm(r["name"]))
        if k not in seen:
            seen.add(k)
            out.append(r)
    with open(f"{a.wave}/new_criteria.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"ingested {len(out)} criteria ({len(rows) - len(out)} within-wave dups) | rejects: {dict(bad)}")
    print(f"-> {a.wave}/new_criteria.jsonl   SPOT-CHECK A SAMPLE BEFORE SEATING (validate-before-scaling).")


def _rep_l0(task):
    names = json.load(open(f"{LEX}/cluster_names_{task}_L0v4.json"))
    return {str(k): f"{v.get('name', '')}. {v.get('gloss', '')}" for k, v in names.items()}


def _rep_r1(task):
    names = json.load(open(f"{LEX}/node_names_{task}_R1.json"))
    part = json.load(open(f"{LEX}/partition_{task}_R1.json"))
    live = set(map(str, part.values()))
    return {str(k): f"{v.get('name', '')}. {v.get('gloss', '')}"
            for k, v in names.items() if str(k) in live}


def cmd_emit_seat(a):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import random
    rows = [json.loads(l) for l in open(f"{a.wave}/new_criteria.jsonl") if l.strip()]
    reps0, reps1 = _rep_l0(a.task), _rep_r1(a.task)
    items = [{"item_id": r["key"], "text": f"{r['name']}. {r['canonical']} {r['description']}"} for r in rows]

    # hidden anchors: recall anchors (existing criteria; truth = own L0) + novelty anchors (off-domain; truth = NEW/NEW)
    rng = random.Random(a.seed)
    l0map = {str(k): str(v) for k, v in json.load(open(f"{LEX}/partition_{a.task}_L0v4.json")).items()}
    ctx = {}
    for ln in open(f"{LEX}/contexts_{a.task}.jsonl"):
        r = json.loads(ln)
        ctx[r["key"]] = r
    multi = [k for k, v in Counter(l0map.values()).items() if v >= 2]
    truth, anchors = {}, []
    pool = [k for k in l0map if l0map[k] in set(multi) and k in ctx and ctx[k].get("canonical")]
    for k in rng.sample(pool, min(a.n_anchors, len(pool))):
        r = ctx[k]
        aid = f"ANCHOR_R{len(anchors)}"
        anchors.append({"item_id": aid, "text": f"{r['name']}. {r['canonical']} {r.get('description', '')}",
                        "_force_l0": l0map[k]})
        truth[aid] = {"kind": "recall", "l0": l0map[k]}
    NOVEL = ["Kerning consistency. The typeface kerning should be optically consistent across weights.",
             "Hull displacement. The boat hull should displace water proportional to load.",
             "Soil drainage. The garden bed should drain within two hours of heavy rain.",
             "Rebar spacing. The reinforced concrete should keep uniform rebar spacing."]
    for i, t in enumerate(NOVEL[:max(2, a.n_anchors // 3)]):
        aid = f"ANCHOR_N{i}"
        anchors.append({"item_id": aid, "text": t})
        truth[aid] = {"kind": "novel"}
    all_items = items + anchors
    rng.shuffle(all_items)
    json.dump(truth, open(f"{a.wave}/seat_anchor_truth.json", "w"), indent=1)

    ids0, texts0 = list(reps0), list(reps0.values())
    ids1, texts1 = list(reps1), list(reps1.values())
    vec = TfidfVectorizer(sublinear_tf=True, stop_words="english", max_features=60000)
    m_all = vec.fit_transform(texts0 + texts1 + [it["text"] for it in all_items])
    m0, m1 = m_all[: len(ids0)], m_all[len(ids0): len(ids0) + len(ids1)]
    mq = m_all[len(ids0) + len(ids1):]
    s0, s1 = cosine_similarity(mq, m0), cosine_similarity(mq, m1)

    per = max(1, a.items_per_shard)
    shards = [all_items[i:i + per] for i in range(0, len(all_items), per)]
    pos = 0
    for si, sh in enumerate(shards):
        with open(f"{a.wave}/seat_payload_{si:02d}.txt", "w") as f:
            f.write(f"OUTPUT FILE: {a.wave}/seat_out_{si:02d}.jsonl\n\n{SEAT_INSTRUCTIONS}\n")
            for it in sh:
                qi = pos
                pos += 1
                top0 = sorted(range(len(ids0)), key=lambda j: -s0[qi, j])[: a.k_l0]
                cand0 = [ids0[j] for j in top0]
                if it.get("_force_l0") and it["_force_l0"] not in cand0:
                    cand0[-1] = it["_force_l0"]  # judge-anchor tests the judge, not the net
                top1 = sorted(range(len(ids1)), key=lambda j: -s1[qi, j])[: a.k_r1]
                f.write(f"\nITEM {it['item_id']}\n  NEW CRITERION: {it['text']}\n  L0 CANDIDATES:\n")
                for c in cand0:
                    f.write(f"    [{c}] {reps0[c][:220]}\n")
                f.write("  R1 CANDIDATES:\n")
                for j in top1:
                    f.write(f"    [{ids1[j]}] {reps1[ids1[j]][:220]}\n")
    print(f"emitted {len(shards)} seat payload(s): {len(items)} new + {len(anchors)} hidden anchors "
          f"-> {a.wave}/seat_payload_*.txt (anchor truth in seat_anchor_truth.json)")


def cmd_apply(a):
    truth = json.load(open(f"{a.wave}/seat_anchor_truth.json"))
    votes = {}
    for p in sorted(glob.glob(f"{a.wave}/seat_out_*.jsonl")):
        for ln in open(p):
            ln = ln.strip()
            if ln:
                try:
                    r = json.loads(ln)
                    votes[r["item_id"]] = r
                except Exception:
                    pass
    # anchor gates. Recall credit = exact cluster OR a same-R1-construct twin: L0v4 retains
    # residual same-criterion duplicate tables (v6 under-merge tail), and seating into a twin
    # is not an instrument failure. Both exact and credited rates are reported.
    r1_of = {str(k): str(v) for k, v in json.load(open(f"{LEX}/partition_{a.task}_R1.json")).items()}
    rec = [(truth[i], votes.get(i)) for i in truth if truth[i]["kind"] == "recall"]
    nov = [(truth[i], votes.get(i)) for i in truth if truth[i]["kind"] == "novel"]
    rec_exact = sum(1 for t, v in rec if v and str(v.get("l0")) == str(t["l0"]))
    rec_ok = sum(1 for t, v in rec if v and (str(v.get("l0")) == str(t["l0"])
                 or (str(v.get("l0")) in r1_of and r1_of.get(str(v.get("l0"))) == r1_of.get(str(t["l0"])))))
    nov_ok = sum(1 for t, v in nov if v and str(v.get("l0", "")).upper() == "NEW"
                 and str(v.get("r1", "")).upper() == "NEW")
    rec_rate = rec_ok / len(rec) if rec else 1.0
    nov_rate = nov_ok / len(nov) if nov else 1.0
    gate = rec_rate >= a.anchor_gate and nov_rate >= a.anchor_gate
    print(f"anchors: recall {rec_ok}/{len(rec)} ({rec_rate:.2f}; exact {rec_exact}) | novelty "
          f"{nov_ok}/{len(nov)} ({nov_rate:.2f}) | gate(>={a.anchor_gate}) {'PASS' if gate else 'FAIL'}")
    if not gate and not a.force:
        print("anchor gate FAILED — not applying (use --force to override after inspection).")
        return
    rows = {r["key"]: r for r in map(json.loads, open(f"{a.wave}/new_criteria.jsonl")) if r}
    l0_live = set(_rep_l0(a.task))
    r1_live = set(_rep_r1(a.task))
    r1_of_l0 = {str(k): str(v) for k, v in json.load(open(f"{LEX}/partition_{a.task}_R1.json")).items()}
    ledger, c = [], Counter()
    n_new_l0 = n_new_r1 = 0
    for key, r in rows.items():
        v = votes.get(key)
        if not v:
            c["unjudged"] += 1
            continue
        l0v, r1v = str(v.get("l0", "NEW")), str(v.get("r1", "NEW"))
        if l0v != "NEW" and l0v not in l0_live:
            c["bad-l0-id"] += 1
            l0v = "NEW"
        if l0v != "NEW":
            seat = {"l0": l0v, "r1": r1_of_l0.get(l0v), "verdict": "seated-L0"}
            c["seated-L0"] += 1
        elif r1v not in ("NEW", "INHERIT") and r1v in r1_live:
            n_new_l0 += 1
            seat = {"l0": f"crpL0_{a.tag}_{n_new_l0:04d}", "r1": r1v, "verdict": "newL0-seated-R1"}
            c["newL0-seated-R1"] += 1
        else:
            n_new_l0 += 1
            n_new_r1 += 1
            seat = {"l0": f"crpL0_{a.tag}_{n_new_l0:04d}", "r1": f"crpR1_{a.tag}_{n_new_r1:04d}",
                    "verdict": "newL0-newR1"}
            c["newL0-newR1"] += 1
        ledger.append({"key": key, "doc": r["doc"], "name": r["name"], **seat, "why": v.get("why", "")})
    sd = os.path.join(LEX, "crp_ingest", a.task)
    os.makedirs(sd, exist_ok=True)
    _append(os.path.join(sd, "seating_ledger.jsonl"), ledger)
    n = sum(c[k] for k in ("seated-L0", "newL0-seated-R1", "newL0-newR1"))
    new_l0_rate = (c["newL0-seated-R1"] + c["newL0-newR1"]) / n if n else float("nan")
    new_r1_rate = c["newL0-newR1"] / n if n else float("nan")
    gt = {}
    cov = f"{LEX}/coverage_census_20260719.json"
    if os.path.exists(cov):
        g = json.load(open(cov)).get(a.task, {})
        gt = {"gt_pred_new_L0": g.get("L0", {}).get("gt_missing_mass"),
              "gt_pred_new_R1": g.get("R1", {}).get("gt_missing_mass")}
    summary = {"wave": a.tag, "task": a.task, "n_seated": n, "counts": dict(c),
               "realized_new_L0_rate": round(new_l0_rate, 3), "realized_new_R1_rate": round(new_r1_rate, 3),
               **gt, "anchor_recall": round(rec_rate, 2), "anchor_recall_exact": round(rec_exact / len(rec), 2) if rec else None,
               "anchor_novelty": round(nov_rate, 2)}
    _append(os.path.join(sd, "wave_summaries.jsonl"), [summary])
    print(json.dumps(summary, indent=1))
    print(f"ledger appended -> {sd}/seating_ledger.jsonl (canonical partitions untouched)")


def _append(path, rows):
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="stage", required=True)
    for st in ("fetch", "emit-extract", "ingest-extract", "emit-seat", "apply"):
        p = sub.add_parser(st)
        p.add_argument("--task", required=True)
        p.add_argument("--wave", required=True, help="wave working directory")
        if st == "fetch":
            p.add_argument("--urls", required=True, help="txt file, one URL per line")
        if st == "emit-extract":
            p.add_argument("--docs-per-shard", type=int, default=3)
        if st == "emit-seat":
            p.add_argument("--items-per-shard", type=int, default=40)
            p.add_argument("--k-l0", type=int, default=8)
            p.add_argument("--k-r1", type=int, default=6)
            p.add_argument("--n-anchors", type=int, default=10)
            p.add_argument("--seed", type=int, default=0)
        if st == "apply":
            p.add_argument("--tag", required=True, help="wave tag for new-table ids, e.g. 20260719a")
            p.add_argument("--anchor-gate", type=float, default=0.8)
            p.add_argument("--force", action="store_true")
    a = ap.parse_args()
    {"fetch": cmd_fetch, "emit-extract": cmd_emit_extract, "ingest-extract": cmd_ingest_extract,
     "emit-seat": cmd_emit_seat, "apply": cmd_apply}[a.stage.replace("_", "-")](a)


if __name__ == "__main__":
    main()
