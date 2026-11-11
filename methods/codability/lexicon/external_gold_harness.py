#!/usr/bin/env python
"""PREREG-23: parse every external human-coded hierarchy into ONE node schema, then build
frozen pair sets for the L0 / R1 / R2 / R3 "same-at-level" prompts.

Node schema (per corpus):  {"id", "text", "p1", "p2", "p3"}
  text = the surface string a judge sees at that node's own level
  p1/p2/p3 = the gold parent id at R1 / R2 / R3 (None where the corpus has no such level)

Rungs.  A rung is a PAIRWISE test of one of our frozen prompts against one gold link:
  L0 rung : two surface items      -> gold same = share the same R1 parent   (SAME CRITERION)
  R1 rung : two R1 node names      -> gold same = share the same R2 parent   (SAME CONSTRUCT)
  R2 rung : two R2 node names      -> gold same = share the same R3 parent   (SAME THEME)
  R3 rung : two R3 node names      -> gold same = share the same R4 parent   (SAME CATEGORY)
Negatives are stratified: HARD = different parent but SAME grandparent (the case that
separates a real instrument from a topic detector); EASY = different grandparent.
Random negatives alone make AUC trivially high, so hard share is fixed at 50%.

Readout is threshold-free (AUC) per the standing rule; P/R at the prompt's own cut is
reported alongside but never as the headline.
"""
from __future__ import annotations

import csv
import glob
import hashlib
import html
import json
import os
import random
import re
from collections import defaultdict

ROOT = "ROOT"
GOLD = f"{ROOT}/datasets/prior_norms/cluster_gold"
OUT = f"{ROOT}/outputs/lexicon/prereg23"
SEED = 20260725


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(str(s or ""))).strip()


# ---------------------------------------------------------------- parsers
def onet() -> dict:
    """L0 task statement -> R1 DWA -> R2 IWA -> R3 GWA (4 real levels, 23,851 links)."""
    d = f"{GOLD}/onet/db_30_0_text"
    T = lambda f: list(csv.DictReader(open(f"{d}/{f}", encoding="utf-8", errors="replace"), delimiter="\t"))
    dwa = {r["DWA ID"]: r for r in T("DWA Reference.txt")}
    iwa = {r["IWA ID"]: r for r in T("IWA Reference.txt")}
    gwa = {r["Element ID"]: _norm(r["Element Name"]) for r in T("Work Activities.txt")}
    task_text = {r["Task ID"]: _norm(r["Task"]) for r in T("Task Statements.txt")}
    # a task may map to several DWAs; keep only single-DWA tasks so the gold parent is unambiguous
    t2d = defaultdict(list)
    for r in T("Tasks to DWAs.txt"):
        t2d[r["Task ID"]].append(r["DWA ID"])
    items = []
    for tid, ds in t2d.items():
        if len(ds) != 1 or ds[0] not in dwa or tid not in task_text:
            continue
        w = dwa[ds[0]]
        items.append({"id": f"t{tid}", "level": "L0", "text": task_text[tid],
                      "p1": w["DWA ID"], "p2": w["IWA ID"], "p3": w["Element ID"]})
    for k, w in dwa.items():
        items.append({"id": k, "level": "R1", "text": _norm(w["DWA Title"]),
                      "p1": None, "p2": w["IWA ID"], "p3": w["Element ID"]})
    for k, w in iwa.items():
        items.append({"id": k, "level": "R2", "text": _norm(w["IWA Title"]),
                      "p1": None, "p2": None, "p3": w["Element ID"]})
    names = {**{k: _norm(v["DWA Title"]) for k, v in dwa.items()},
             **{k: _norm(v["IWA Title"]) for k, v in iwa.items()}, **gwa}
    return {"corpus": "onet", "domain": "work activities (US DOL)", "items": items, "names": names,
            "rungs": ["L0", "R1", "R2"]}


def codereview() -> dict:
    """L0 review comment -> R1 category (19) -> R2 comment_group (5). ON-DOMAIN."""
    import pandas as pd
    df = pd.read_excel(f"{GOLD}/codereview_esem23/labeled.xlsx")
    items, names = [], {}
    for i, r in df.iterrows():
        msg = _norm(r["message"])
        if len(msg) < 8:
            continue
        cat, grp = _norm(r["category"]), _norm(r["comment_group"])
        items.append({"id": f"c{i}", "level": "L0", "text": msg, "p1": cat, "p2": grp, "p3": None})
        names[cat] = cat
        names[grp] = grp
    for cat in sorted({it["p1"] for it in items}):
        grp = next(it["p2"] for it in items if it["p1"] == cat)
        items.append({"id": cat, "level": "R1", "text": cat, "p1": None, "p2": grp, "p3": None})
    return {"corpus": "codereview", "domain": "code review feedback", "items": items,
            "names": names, "rungs": ["L0", "R1"]}


def pdtb() -> dict:
    """L0 relation instance -> R1 sense subtype -> R2 type -> R3 class."""
    base = f"{GOLD}/pdtb3/PDTB-3.0/data"
    items, names, seen = [], {}, set()
    for f in sorted(glob.glob(f"{base}/gold/*/wsj_*")):
        raw_p = f.replace("/gold/", "/raw/")
        if not os.path.exists(raw_p):
            continue
        raw = open(raw_p, encoding="utf-8", errors="replace").read()
        for line in open(f, encoding="utf-8", errors="replace"):
            fl = line.rstrip("\n").split("|")
            if len(fl) < 21 or not fl[8].strip():
                continue
            sense = fl[8].strip().split(".")
            if len(sense) < 2:
                continue
            span = lambda s: " ".join(raw[int(a):int(b)] for a, b in
                                      (x.split("..") for x in s.split(";")) if a.isdigit())
            try:
                a1, a2 = _norm(span(fl[14])), _norm(span(fl[20]))
            except Exception:
                continue
            if len(a1) < 15 or len(a2) < 15:
                continue
            conn = _norm(fl[7]) or _norm(fl[1])
            cls, typ = sense[0], f"{sense[0]}.{sense[1]}"
            sub = ".".join(sense[:3]) if len(sense) >= 3 else typ
            uid = hashlib.sha1(f"{a1}|{a2}".encode()).hexdigest()[:12]
            if uid in seen:
                continue
            seen.add(uid)
            items.append({"id": uid, "level": "L0",
                          "text": f"{a1} [{conn}] {a2}"[:700], "p1": sub, "p2": typ, "p3": cls})
    # LEAF TEXT ONLY. Rendering the full dotted sense path ("Comparison / Contrast") leaks the
    # gold: the R1 rung's gold link IS "shares the first two path segments", so a judge that
    # merely compares prefixes scores a perfect AUC without reading anything. Show the last
    # segment alone, exactly as the other corpora show a bare code name.
    for sub in sorted({it["p1"] for it in items}):
        p = sub.split(".")
        items.append({"id": sub, "level": "R1", "text": p[-1].replace("+", " + "),
                      "p1": None, "p2": f"{p[0]}.{p[1]}", "p3": p[0]})
    for typ in sorted({it["p2"] for it in items if it["level"] == "R1"}):
        items.append({"id": typ, "level": "R2", "text": typ.split(".")[-1],
                      "p1": None, "p2": None, "p3": typ.split(".")[0]})
    for it in items:
        names[it["id"]] = it["text"]
    return {"corpus": "pdtb", "domain": "discourse relations", "items": items, "names": names,
            "rungs": ["L0", "R1", "R2"]}


def scrum() -> dict:
    """L0 interview segment -> R1 code (35) -> R2 theme (14)."""
    g = json.load(open(f"{ROOT}/outputs/lexicon/cluster_gold_validation_20260724/scrum_gold.json"))
    items = [{"id": f"s{i}", "level": "L0", "text": _norm(r["segment"]).strip('"'),
              "p1": _norm(r["code"]), "p2": _norm(r["theme"]), "p3": None} for i, r in enumerate(g)]
    for code in sorted({it["p1"] for it in items}):
        th = next(it["p2"] for it in items if it["p1"] == code)
        items.append({"id": code, "level": "R1", "text": code, "p1": None, "p2": th, "p3": None})
    return {"corpus": "scrum", "domain": "software process interviews", "items": items,
            "names": {it["id"]: it["text"] for it in items}, "rungs": ["L0", "R1"]}


def disapere() -> dict:
    """L0 review sentence -> R1 aspect (8). One rung only."""
    s = json.load(open(f"{ROOT}/outputs/lexicon/cluster_gold_validation_20260724/disapere_sample.json"))
    rows = s if isinstance(s, list) else s.get("items", [])
    items = []
    for i, r in enumerate(rows):
        a = _norm(r.get("aspect") or r.get("label") or "")
        t = _norm(r.get("text") or r.get("sentence") or "")
        if not a or a.startswith("ANCHOR") or len(t) < 15:
            continue
        items.append({"id": f"d{i}", "level": "L0", "text": t, "p1": a, "p2": None, "p3": None})
    return {"corpus": "disapere", "domain": "peer review", "items": items,
            "names": {it["id"]: it["text"] for it in items}, "rungs": ["L0"]}


def ucsb() -> dict:
    """TREE ONLY (no segment links): leaf code -> depth-2 node -> depth-1 root.
    Registered exclusion: the Course\\Demographics attribute branch is dropped (facet coding,
    not thematic). Unfiltered variant is emitted as corpus 'ucsb_all' for the reported contrast."""
    paths = [r["Code"] for r in csv.DictReader(open(f"{GOLD}/ucsb_dryad_ithaka/coodebook.csv"))
             if r["Code"].strip()]
    S = set(paths)
    out = {}
    for tag, keep in (("ucsb", lambda p: not p.startswith("Course\\Demographics")), ("ucsb_all", lambda p: True)):
        items = []
        for p in paths:
            if any(q.startswith(p + "\\") for q in S) or not keep(p):
                continue
            parts = p.split("\\")
            if len(parts) < 2:
                continue
            items.append({"id": p, "level": "R1", "text": parts[-1],
                          "p2": "\\".join(parts[:2]), "p3": parts[0], "p1": None})
        for mid in sorted({it["p2"] for it in items}):
            items.append({"id": mid, "level": "R2", "text": mid.split("\\")[-1],
                          "p1": None, "p2": None, "p3": mid.split("\\")[0]})
        out[tag] = {"corpus": tag, "domain": "qualitative codebook (library instruction)",
                    "items": items, "names": {it["id"]: it["text"] for it in items},
                    "rungs": ["R1", "R2"]}
    return out


PARSERS = {"onet": onet, "codereview": codereview, "pdtb": pdtb, "scrum": scrum,
           "disapere": disapere}


# ---------------------------------------------------------------- pair builder
PARENT_OF = {"L0": ("p1", "p2"), "R1": ("p2", "p3"), "R2": ("p3", None)}


def build_pairs(c: dict, rung: str, n_pos: int, n_neg: int, rng: random.Random) -> list:
    """Positives share the gold parent; negatives do not. Half the negatives are HARD
    (different parent, same grandparent) when the corpus has a grandparent level."""
    pk, gk = PARENT_OF[rung]
    pool = [it for it in c["items"] if it["level"] == rung and it.get(pk)]
    by_parent = defaultdict(list)
    for it in pool:
        by_parent[it[pk]].append(it)
    gp = {it[pk]: it.get(gk) for it in pool} if gk else {}

    pos = []
    multi = [p for p, v in by_parent.items() if len(v) >= 2]
    rng.shuffle(multi)
    while multi and len(pos) < n_pos:
        for p in list(multi):
            if len(pos) >= n_pos:
                break
            a, b = rng.sample(by_parent[p], 2)
            pos.append((a, b, 1, "pos"))

    hard, easy = [], []
    parents = list(by_parent)
    # HARD: draw the two parents from inside one grandparent, so the pair is topically adjacent
    # but gold-different. Sampling parent pairs at random almost never lands here when the
    # parent inventory is large (O*NET has 2,087 DWAs), hence the explicit by-grandparent index.
    if gk:
        by_gp = defaultdict(list)
        for p in parents:
            if gp.get(p):
                by_gp[gp[p]].append(p)
        gps = [g for g, v in by_gp.items() if len(v) >= 2]
        for _ in range(n_neg * 60):
            if len(hard) >= n_neg // 2 or not gps:
                break
            p, q = rng.sample(by_gp[rng.choice(gps)], 2)
            hard.append((rng.choice(by_parent[p]), rng.choice(by_parent[q]), 0, "neg_hard"))
    cap_easy = n_neg - (n_neg // 2 if gk else 0)
    for _ in range(n_neg * 60):
        if len(easy) >= cap_easy or len(parents) < 2:
            break
        p, q = rng.sample(parents, 2)
        if gk and gp.get(p) and gp.get(p) == gp.get(q):
            continue
        easy.append((rng.choice(by_parent[p]), rng.choice(by_parent[q]), 0, "neg_easy"))

    # DEDUPE. At label grain the node inventory is small (PDTB has ~30 subtypes, ESEM'23 has
    # 19 categories), so independent draws repeatedly hit the same unordered {a,b}. Left in,
    # those become duplicate pair_ids: the advertised n is inflated and any pair that happened
    # to be drawn twice would carry double weight. Keep one row per unordered node pair and
    # report the achievable n instead of the requested n.
    rows, seen = [], set()
    for a, b, y, strat in pos + hard + easy:
        if a["id"] == b["id"]:
            continue
        key = tuple(sorted((a["id"], b["id"])))
        if key in seen:
            continue
        seen.add(key)
        pid = hashlib.sha1(f"{c['corpus']}|{rung}|{key[0]}|{key[1]}".encode()).hexdigest()[:16]
        rows.append({"pair_id": pid, "corpus": c["corpus"], "rung": rung,
                     "a_id": a["id"], "b_id": b["id"], "a": a["text"], "b": b["text"],
                     "gold": y, "stratum": strat})
    rng.shuffle(rows)
    return rows


def main(n_pos=140, n_neg=140):
    os.makedirs(OUT, exist_ok=True)
    corpora = {k: f() for k, f in PARSERS.items()}
    corpora.update(ucsb())
    summary = []
    for name, c in corpora.items():
        json.dump(c, open(f"{OUT}/nodes_{name}.json", "w"))
        for rung in c["rungs"]:
            rng = random.Random(f"{SEED}|{name}|{rung}")
            rows = build_pairs(c, rung, n_pos, n_neg, rng)
            if not rows:
                continue
            with open(f"{OUT}/pairs_{name}_{rung}.jsonl", "w") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
            st = defaultdict(int)
            for r in rows:
                st[r["stratum"]] += 1
            summary.append({"corpus": name, "domain": c["domain"], "rung": rung,
                            "n_pairs": len(rows), **st})
    json.dump(summary, open(f"{OUT}/pair_summary.json", "w"), indent=1)
    print(f"{'corpus':<12}{'rung':<6}{'pairs':>7}{'pos':>7}{'hard':>7}{'easy':>7}   domain")
    for s in summary:
        print(f"{s['corpus']:<12}{s['rung']:<6}{s['n_pairs']:>7}{s.get('pos',0):>7}"
              f"{s.get('neg_hard',0):>7}{s.get('neg_easy',0):>7}   {s['domain']}")
    return summary


if __name__ == "__main__":
    main()
