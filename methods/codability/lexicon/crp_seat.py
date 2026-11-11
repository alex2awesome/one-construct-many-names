#!/usr/bin/env python3
"""CRP hierarchical seating cascade — reuses the EXACT machinery of the original L0->R3 build,
applied incrementally to one new criterion at a time.

The cascade (user spec 2026-07-19):
  L0  retrieve existing-cluster candidates with the SAME union net as the original build
      (BGE semantic  UNION  TF-IDF lexical  UNION  shared-rare-name), judge each candidate
      with the frozen CONFIRM_PROTOCOL_L0_V2 (score 0/1/2). If any candidate == 2 -> SEAT at
      that L0 cluster and STOP (inherit its R1->R3 path; names are never rewritten).
  R1  else open a NEW L0 (rep = the criterion itself). Retrieve construct candidates over
      L0-cluster reps (name+gloss), judge with STRICT_BUILD_PROTOCOL_R1 (score 0/1/2). If any
      == 2 -> SEAT under that construct and STOP.
  R2  else open a NEW R1 construct. CLASSIFY it into the frozen Opus R2 theme taxonomy (same
      derive-then-classify machinery). If primary != OTHER -> SEAT under that theme and STOP.
  R3  else open a NEW R2 theme. CLASSIFY into the frozen R3 category taxonomy. If primary !=
      OTHER -> SEAT under that category and STOP. Else a genuinely novel category (a finding).

Every level's realized new-node rate is the empirical Good-Turing missing mass at that grain,
compared step-by-step to outputs/lexicon/coverage_census_20260719.json.

Division of labor (repo rules): retrieval/TF-IDF/BGE are a SHORTLIST only. Every seat decision
is an LLM judge (Sonnet+) applying a frozen protocol verbatim. Canonical partitions are never
mutated; all seatings live in outputs/lexicon/crp_ingest/<task>/ (append-only).

Staged because each level depends on the previous level's judge output:
  stage l0-emit / l0-ingest -> stage r1-emit / r1-ingest -> r2-emit / r2-ingest -> r3-emit / r3-ingest -> finalize
Between an emit and its ingest, the caller runs Sonnet subagents over the emitted payload files.
"""
from __future__ import annotations

import argparse
import functools
import glob
import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
LEX = os.path.join(ROOT, "outputs", "lexicon")
DTC = os.path.join(LEX, "derive_then_classify_v1")
L0_PROTO = open(os.path.join(LEX, "CONFIRM_PROTOCOL_L0_V2.txt")).read()
R1_PROTO = open(os.path.join(LEX, "STRICT_BUILD_PROTOCOL_R1.txt")).read()
_W = re.compile(r"[a-z0-9']+")


def norm_term(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", (t or "").lower())).strip()


# ---- reps -------------------------------------------------------------------------------------
def l0_reps(task: str) -> dict:
    d = json.load(open(f"{LEX}/cluster_names_{task}_L0v4.json"))
    return {str(k): f"{v.get('name', '')}. {v.get('gloss', '')}".strip() for k, v in d.items()}


def r1_reps(task: str) -> dict:
    names = json.load(open(f"{LEX}/node_names_{task}_R1.json"))
    live = set(map(str, json.load(open(f"{LEX}/partition_{task}_R1.json")).values()))
    return {str(k): f"{v.get('name', '')}. {v.get('gloss', '')}".strip()
            for k, v in names.items() if str(k) in live}


def l0_to_r1(task: str) -> dict:
    return {str(k): str(v) for k, v in json.load(open(f"{LEX}/partition_{task}_R1.json")).items()}


# ---- blinded in-batch anchors (standing rule: every judging batch carries known-truth items) ---
ANCHOR_DONORS = ["patents", "legal-outcome-prediction", "math-stackexchange", "humor",
                 "press-releases", "grant-funding"]
_SLUGW = ["notes", "essay", "guide", "column", "review", "fieldnotes", "digest", "primer"]


def _fake_key(task, seed_text, idx):
    h = hashlib.sha1(seed_text.encode()).hexdigest()
    return f"{task}::crp::{_SLUGW[int(h[:2], 16) % len(_SLUGW)]}-{h[2:12]}::{idx}"


def _ctx_rows(task):
    rows = {}
    for ln in open(f"{LEX}/contexts_{task}.jsonl"):
        r = json.loads(ln)
        rows[r["key"]] = r
    return rows


def _crit_text(r):
    return f"{r.get('name', '')}. {r.get('canonical', '')} {(r.get('description') or '')[:200]}".strip()


def _make_l0_anchors(task, n_recall=5, n_novel=3, seed=0):
    """Recall anchor = a REAL member of a multi-member L0 cluster (its raw text, truth = its own
    cluster) whose name differs from the cluster rep head — a genuine paraphrase by construction.
    Novel anchor = a criterion from a donor task (truth = must NOT seat)."""
    part = {str(k): str(v) for k, v in json.load(open(f"{LEX}/partition_{task}_L0v4.json")).items()}
    members = defaultdict(list)
    for k, c in part.items():
        members[c].append(k)
    reps = l0_reps(task)
    ctx = _ctx_rows(task)
    rng = random.Random(seed)
    multi = [c for c, ms in members.items() if len(ms) >= 2 and c in reps]
    rng.shuffle(multi)
    items, truth = {}, {}
    for c in multi:
        if sum(1 for t in truth.values() if t["kind"] == "recall") >= n_recall:
            break
        rep_head = norm_term(reps[c].split(".")[0])
        ms = [m for m in members[c] if m in ctx and norm_term(ctx[m].get("name", "")) != rep_head]
        if not ms:
            continue
        m = rng.choice(sorted(ms))
        key = _fake_key(task, m, len(items))
        items[key] = {"name": ctx[m].get("name", ""), "canonical": ctx[m].get("canonical", ""),
                      "text": _crit_text(ctx[m]), "status": "pending"}
        truth[key] = {"kind": "recall", "l0": c, "src": m}
    donors = [d for d in ANCHOR_DONORS if d != task and os.path.exists(f"{LEX}/contexts_{d}.jsonl")]
    dctx = _ctx_rows(donors[seed % len(donors)])
    for m in rng.sample(sorted(dctx), min(n_novel, len(dctx))):
        key = _fake_key(task, m, len(items))
        items[key] = {"name": dctx[m].get("name", ""), "canonical": dctx[m].get("canonical", ""),
                      "text": _crit_text(dctx[m]), "status": "pending"}
        truth[key] = {"kind": "novel", "l0": None, "src": m}
    return items, truth


def _make_r1_anchors(task, n_recall=4, seed=0):
    """R1 recall anchor = an existing L0-cluster rep whose construct holds >=2 clusters, presented
    as a 'new L0'; its OWN cluster is excluded from candidates so seating must go via a sibling.
    Truth = its construct."""
    l0r1 = l0_to_r1(task)
    reps = l0_reps(task)
    per_con = defaultdict(list)
    for cl, con in l0r1.items():
        if cl in reps:
            per_con[con].append(cl)
    rng = random.Random(seed + 1)
    # >=3 clusters: after self-exclusion the construct still has 2+ sibling surfaces for retrieval
    cons = [c for c, cls in per_con.items() if len(cls) >= 3]
    rng.shuffle(cons)
    items, truth = {}, {}
    for con in cons[:n_recall]:
        cl = rng.choice(sorted(per_con[con]))
        key = _fake_key(task, f"r1anchor::{cl}", len(items))
        head = reps[cl].split(".")[0]
        items[key] = {"name": head, "canonical": reps[cl],
                      "text": reps[cl], "status": "r1-await"}
        truth[key] = {"kind": "recall", "r1": con, "self_l0": cl}
    return items, truth


def _make_classify_anchors(task, level, n_recall=4, seed=0):
    """R2 anchor = existing construct rep (truth = its theme). R3 anchor = existing theme
    (name+definition from the frozen R2 taxonomy; truth = its category)."""
    rng = random.Random(seed + 2)
    items, truth = {}, {}
    if level == "R2":
        part = {str(k): str(v) for k, v in json.load(open(f"{LEX}/partition_{task}_R2.json")).items()}
        reps = r1_reps(task)
        pool = sorted(c for c in part if c in reps)
        for con in rng.sample(pool, min(n_recall, len(pool))):
            key = _fake_key(task, f"r2anchor::{con}", len(items))
            items[key] = {"name": reps[con].split(".")[0], "canonical": reps[con],
                          "text": reps[con], "status": "r2-await"}
            truth[key] = {"kind": "recall", "seat": part[con], "src": con}
    else:
        part = {str(k): str(v) for k, v in json.load(open(f"{LEX}/partition_{task}_R3.json")).items()}
        tax = {t["id"]: t for t in _taxonomy(task, "R2")}
        pool = sorted(t for t in part if t in tax)
        for th in rng.sample(pool, min(n_recall, len(pool))):
            key = _fake_key(task, f"r3anchor::{th}", len(items))
            txt = f"{tax[th]['name']}. {tax[th].get('definition', '')[:300]}"
            items[key] = {"name": tax[th]["name"], "canonical": txt,
                          "text": txt, "status": "r3-await"}
            truth[key] = {"kind": "recall", "seat": part[th], "src": th}
    return items, truth


@functools.lru_cache(maxsize=4)
def _bge():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("BAAI/bge-small-en-v1.5")


def _embed(texts):
    return _bge().encode(list(texts), normalize_embeddings=True, batch_size=256,
                         show_progress_bar=False)


def union_retrieve(query_texts, query_names, cand_ids, cand_texts, k_bge=15, k_tfidf=15,
                   min_cos=0.20, max_name_df=12):
    """The original union net (BGE semantic UNION TF-IDF lexical UNION shared-rare-name), adapted
    to retrieve existing candidates for NEW query nodes. Returns per-query list of (cand_id, cos)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors
    C = len(cand_ids)
    # (1) BGE semantic
    cvec = _embed(cand_texts)
    qvec = _embed(query_texts)
    bge_nn = NearestNeighbors(n_neighbors=min(k_bge, C), metric="cosine").fit(cvec)
    bd, bi = bge_nn.kneighbors(qvec)
    # (2) TF-IDF lexical (fit on candidates, transform queries)
    vec = TfidfVectorizer(min_df=1, max_features=40000, sublinear_tf=True)
    Cx = vec.fit_transform(cand_texts)
    Qx = vec.transform(query_texts)
    tf_nn = NearestNeighbors(n_neighbors=min(k_tfidf, C), metric="cosine").fit(Cx)
    td, ti = tf_nn.kneighbors(Qx)
    # (3) shared-rare-name token -> candidate ids
    tok2c = defaultdict(list)
    for cid, ct in zip(cand_ids, cand_texts):
        for w in set(norm_term(ct.split(".")[0][:80]).split()):
            if len(w) > 5:
                tok2c[w].append(cid)
    out = []
    for qi in range(len(query_texts)):
        pool = {}
        for jj, dd in zip(bi[qi], bd[qi]):
            pool[cand_ids[int(jj)]] = max(pool.get(cand_ids[int(jj)], 0.0), 1.0 - float(dd))
        for jj, dd in zip(ti[qi], td[qi]):
            cos = 1.0 - float(dd)
            if cos >= min_cos:
                pool[cand_ids[int(jj)]] = max(pool.get(cand_ids[int(jj)], 0.0), cos)
        for w in set(norm_term((query_names[qi] or "")[:80]).split()):
            if len(w) > 5 and 2 <= len(tok2c.get(w, [])) <= max_name_df:
                for cid in tok2c[w]:
                    pool.setdefault(cid, 0.30)
        out.append(sorted(pool.items(), key=lambda x: -x[1]))
    return out


# ---- state ------------------------------------------------------------------------------------
def _state_path(task, wave):
    return f"{wave}/cascade_state.json"


def _load_state(task, wave):
    p = _state_path(task, wave)
    if os.path.exists(p):
        return json.load(open(p))
    rows = [json.loads(l) for l in open(f"{wave}/new_criteria.jsonl") if l.strip()]
    return {"task": task, "items": {r["key"]: {"name": r["name"], "canonical": r["canonical"],
            "text": f"{r['name']}. {r['canonical']} {r['description']}", "status": "pending"}
            for r in rows}, "log": []}


def _save_state(task, wave, st):
    json.dump(st, open(_state_path(task, wave), "w"), indent=1)


def _pending(st, stage):
    return [k for k, v in st["items"].items() if v["status"] == stage]


# ---- L0 stage ---------------------------------------------------------------------------------
def cmd_l0_emit(a):
    st = _load_state(a.task, a.wave)
    if getattr(a, "anchors", 0) and "_anchor_l0" not in st:
        anc, truth = _make_l0_anchors(a.task, n_recall=max(1, a.anchors - a.anchors // 3),
                                      n_novel=a.anchors // 3, seed=a.anchor_seed)
        st["items"].update(anc)
        st["_anchor_l0"] = truth
        json.dump(truth, open(f"{a.wave}/anchor_truth_l0.json", "w"), indent=1)
    for v in st["items"].values():
        if v["status"] == "pending":
            v["status"] = "l0-await"
    reps = l0_reps(a.task)
    cids, ctexts = list(reps), list(reps.values())
    items = [(k, v) for k, v in st["items"].items() if v["status"] == "l0-await"]
    random.Random(11).shuffle(items)   # blind: anchors interleaved with real items
    ret = union_retrieve([v["text"] for _, v in items], [v["name"] for _, v in items],
                         cids, ctexts, k_bge=a.k, k_tfidf=a.k)
    st["_l0_cand"] = {items[i][0]: [c for c, _ in ret[i][:a.k]] for i in range(len(items))}
    _emit_pairwise(a, st, items, reps, ret, L0_PROTO, "l0", level_label="L0 clusters (SAME CRITERION)")
    _save_state(a.task, a.wave, st)


def cmd_r1_emit(a):
    st = _load_state(a.task, a.wave)
    if getattr(a, "anchors", 0) and "_anchor_r1" not in st:
        anc, truth = _make_r1_anchors(a.task, n_recall=a.anchors, seed=a.anchor_seed)
        st["items"].update(anc)
        st["_anchor_r1"] = truth
        json.dump(truth, open(f"{a.wave}/anchor_truth_r1.json", "w"), indent=1)
    reps = r1_reps(a.task)                 # construct reps
    l0reps = l0_reps(a.task)
    l0r1 = l0_to_r1(a.task)
    # retrieve over L0-cluster reps, then map candidate cluster -> its construct
    cids, ctexts = list(l0reps), list(l0reps.values())
    items = [(k, v) for k, v in st["items"].items() if v["status"] == "r1-await"]
    if not items:
        print("no items awaiting R1"); return
    random.Random(13).shuffle(items)       # blind: anchors interleaved with real items
    ret = union_retrieve([v["text"] for _, v in items], [v["name"] for _, v in items],
                         cids, ctexts, k_bge=a.k, k_tfidf=a.k)
    # collapse to construct candidates (best cluster rep per construct as the shown member)
    anch = st.get("_anchor_r1", {})
    cand_by_item = {}
    for i, (k, v) in enumerate(items):
        self_cl = str(anch.get(k, {}).get("self_l0", ""))   # anchor must seat via a sibling
        seen = {}
        for cl, cos in ret[i]:
            if cl == self_cl:
                continue
            con = l0r1.get(cl)
            if con and con in reps and (con not in seen or cos > seen[con][1]):
                seen[con] = (cl, cos)
        cand_by_item[k] = sorted(seen.items(), key=lambda x: -x[1][1])[:a.k]
    st["_r1_cand"] = {k: [c for c, _ in v] for k, v in cand_by_item.items()}
    _emit_construct(a, st, items, reps, cand_by_item, R1_PROTO, "r1")
    _save_state(a.task, a.wave, st)


def _emit_pairwise(a, st, items, reps, ret, proto, tag, level_label):
    per = a.items_per_shard
    shards = [items[i:i + per] for i in range(0, len(items), per)]
    for si, sh in enumerate(shards):
        with open(f"{a.wave}/{tag}_payload_{si:02d}.txt", "w") as f:
            f.write(f"OUTPUT FILE: {a.wave}/{tag}_out_{si:02d}.jsonl\n\n")
            f.write("You apply the following frozen protocol to score each NEW criterion against each\n"
                    f"candidate existing {level_label}. For EACH (item, candidate) pair emit the score.\n\n")
            f.write(proto + "\n\nPAIRS (each item is a NEW criterion; candidates are existing nodes):\n")
            for k, v in sh:
                f.write(f"\nITEM {k}\n  NEW: {v['text'][:400]}\n")
                cand = st[f"_{tag}_cand"][k]
                for c in cand:
                    f.write(f"    CANDIDATE [{c}] {reps[c][:260]}\n")
            f.write('\nOutput one JSON line per (item,candidate): {"pair_id":"<ITEM>||<CAND>","score":0|1|2}\n')
    print(f"{tag}: emitted {len(shards)} payload(s) for {len(items)} items -> {a.wave}/{tag}_payload_*.txt")


def _emit_construct(a, st, items, reps, cand_by_item, proto, tag):
    per = a.items_per_shard
    shards = [items[i:i + per] for i in range(0, len(items), per)]
    for si, sh in enumerate(shards):
        with open(f"{a.wave}/{tag}_payload_{si:02d}.txt", "w") as f:
            f.write(f"OUTPUT FILE: {a.wave}/{tag}_out_{si:02d}.jsonl\n\n")
            f.write("You apply the following frozen protocol to score each NEW concept against each\n"
                    "candidate existing CONSTRUCT. For EACH (item, candidate) pair emit the score.\n\n")
            f.write(proto + "\n\nPAIRS:\n")
            for k, v in sh:
                f.write(f"\nITEM {k}\n  NEW: {v['text'][:400]}\n")
                for con, (cl, cos) in cand_by_item[k]:
                    f.write(f"    CANDIDATE [{con}] {reps[con][:260]}\n")
            f.write('\nOutput one JSON line per (item,candidate): {"pair_id":"<ITEM>||<CAND>","score":0|1|2}\n')
    print(f"{tag}: emitted {len(shards)} payload(s) for {len(items)} items -> {a.wave}/{tag}_payload_*.txt")


def _ingest_scores(wave, tag):
    """pair_id 'ITEM||CAND' -> score; return {item: [(cand,score)...]}."""
    by_item = defaultdict(list)
    for p in sorted(glob.glob(f"{wave}/{tag}_out_*.jsonl")):
        for ln in open(p):
            ln = ln.strip()
            if not ln:
                continue
            try:
                r = json.loads(ln)
            except Exception:
                continue
            pid = str(r.get("pair_id", ""))
            if "|" not in pid or type(r.get("score")) is not int or r["score"] not in (0, 1, 2):
                continue
            if "||" in pid:
                it, cand = pid.split("||", 1)
            else:                       # tolerate single-pipe outputs (item keys contain no '|')
                it, cand = pid.rsplit("|", 1)
            by_item[it].append((cand, r["score"]))
    return by_item


def _anchor_report_seat(st, truth_key, scores, twin_map=None):
    """Evaluate seat-stage anchors. Returns (report, anchor_keys). twin_map (L0 only) credits a
    seat into a same-construct twin cluster (residual v6 under-merge tail, not judge error)."""
    truth = st.get(truth_key, {})
    rep = {"n_recall": 0, "recall_exact": 0, "recall_credited": 0, "n_novel": 0, "novel_pass": 0,
           "misses": []}
    for k, t in truth.items():
        v = st["items"].get(k)
        if v is None or v["status"] not in ("l0-await", "r1-await"):
            continue
        twos = [c for c, s in scores.get(k, []) if s == 2]
        top = Counter(twos).most_common(1)[0][0] if twos else None
        if t["kind"] == "recall":
            rep["n_recall"] += 1
            want = str(t.get("l0") or t.get("r1"))
            exact = want in [str(c) for c in twos]     # truth scored 2 = recall success
            credited = exact or (twin_map is not None and top is not None
                                 and twin_map.get(str(top)) == twin_map.get(want))
            rep["recall_exact"] += exact
            rep["recall_credited"] += credited
            if not credited:
                rep["misses"].append({"anchor": k, "want": want, "got": top})
        else:
            rep["n_novel"] += 1
            rep["novel_pass"] += top is None
            if top is not None:
                rep["misses"].append({"anchor": k, "want": None, "got": top})
        v["status"], v["verdict"] = "anchor-done", "anchor"
    return rep, set(truth)


def _coverage_guard(st, scores, await_key, stage):
    """Items awaiting this stage with ZERO judged pairs would silently default to new-node.
    Refuse to ingest them; caller re-emits a completion shard instead."""
    missing = [k for k, v in st["items"].items() if v["status"] == await_key and not scores.get(k)]
    if missing:
        print(f"WARNING {stage}: {len(missing)} awaiting item(s) have NO judged pairs — "
              f"left in {await_key} (run a completion shard): {[m[:60] for m in missing[:5]]}")
    return set(missing)


def cmd_l0_ingest(a):
    st = _load_state(a.task, a.wave)
    scores = _ingest_scores(a.wave, "l0")
    rep, anchor_keys = _anchor_report_seat(st, "_anchor_l0", scores, twin_map=l0_to_r1(a.task))
    unjudged = _coverage_guard(st, scores, "l0-await", "l0")
    cand_cos = st.get("_l0_cand", {})
    seated = newl0 = 0
    for k, v in st["items"].items():
        if v["status"] != "l0-await" or k in anchor_keys or k in unjudged:
            continue
        twos = [(c, s) for c, s in scores.get(k, []) if s == 2]
        if twos:
            best = max(twos, key=lambda cs: cand_cos.get(k, []).index(cs[0]) * -1)  # earliest = highest cos
            v["status"], v["seat_l0"] = "seated", best[0]
            v["verdict"] = "seated-L0"
            seated += 1
        else:
            v["status"] = "r1-await"
            newl0 += 1
    st["log"].append({"stage": "l0", "seated_L0": seated, "new_L0": newl0, "anchor_gate": rep})
    _save_state(a.task, a.wave, st)
    n = seated + newl0
    if rep["n_recall"] or rep["n_novel"]:
        print(f"ANCHOR GATE l0: recall {rep['recall_credited']}/{rep['n_recall']} credited "
              f"({rep['recall_exact']} exact), novel {rep['novel_pass']}/{rep['n_novel']} pass"
              + (f"; misses {rep['misses']}" if rep["misses"] else ""))
    if n:
        print(f"L0: {seated} seated / {newl0} new-L0 (of {n}); new-L0 rate {newl0/n:.3f}")
    else:
        print("L0: no non-anchor items awaited ingest")


def cmd_r1_ingest(a):
    st = _load_state(a.task, a.wave)
    scores = _ingest_scores(a.wave, "r1")
    rep, anchor_keys = _anchor_report_seat(st, "_anchor_r1", scores)
    unjudged = _coverage_guard(st, scores, "r1-await", "r1")
    seated = newr1 = 0
    for k, v in st["items"].items():
        if v["status"] != "r1-await" or k in anchor_keys or k in unjudged:
            continue
        twos = [c for c, s in scores.get(k, []) if s == 2]
        if twos:
            v["status"], v["seat_r1"] = "seated", Counter(twos).most_common(1)[0][0]
            v["verdict"] = "newL0-seated-R1"
            seated += 1
        else:
            v["status"] = "r2-await"
            newr1 += 1
    st["log"].append({"stage": "r1", "seated_R1": seated, "new_R1": newr1, "anchor_gate": rep})
    _save_state(a.task, a.wave, st)
    if rep["n_recall"]:
        print(f"ANCHOR GATE r1: recall {rep['recall_credited']}/{rep['n_recall']}"
              + (f"; misses {rep['misses']}" if rep["misses"] else ""))
    print(f"R1: {seated} seated-construct / {newr1} new-construct")


# ---- R2 / R3 classify stages ------------------------------------------------------------------
def _taxonomy(task, level):
    tx = json.load(open(glob.glob(f"{DTC}/{task}/{level}/taxonomy_*_{level}.json")[0]))
    key = "themes" if level == "R2" else "categories"
    return tx[key]


def cmd_classify_emit(a):
    level = a.level
    tag = level.lower()
    await_key = "r2-await" if level == "R2" else "r3-await"
    st = _load_state(a.task, a.wave)
    akey = f"_anchor_{tag}"
    if getattr(a, "anchors", 0) and akey not in st:
        anc, truth = _make_classify_anchors(a.task, level, n_recall=a.anchors, seed=a.anchor_seed)
        st["items"].update(anc)
        st[akey] = truth
        json.dump(truth, open(f"{a.wave}/anchor_truth_{tag}.json", "w"), indent=1)
    items = [(k, v) for k, v in st["items"].items() if v["status"] == await_key]
    if not items:
        print(f"no items awaiting {level}"); return
    random.Random(17).shuffle(items)   # blind: anchors interleaved with real items
    tax = _taxonomy(a.task, level)
    unit = "theme" if level == "R2" else "category"
    per = a.items_per_shard
    shards = [items[i:i + per] for i in range(0, len(items), per)]
    for si, sh in enumerate(shards):
        with open(f"{a.wave}/{tag}_payload_{si:02d}.txt", "w") as f:
            f.write(f"OUTPUT FILE: {a.wave}/{tag}_out_{si:02d}.jsonl\n\n")
            f.write(f"Classify each NEW evaluation concept into exactly ONE {unit} from the frozen "
                    f"{level} taxonomy below (the same taxonomy the hierarchy was built with). Choose the "
                    f"single best {unit} by its definition. If NONE genuinely fits, output OTHER — do not "
                    f"force a poor fit. Judge each item independently.\n\n=== {level} TAXONOMY ===\n")
            for t in tax:
                f.write(f"  [{t['id']}] {t['name']}: {t.get('definition', '')[:400]}\n")
            f.write("\n=== ITEMS ===\n")
            for k, v in sh:
                f.write(f"ITEM {k}: {v['text'][:400]}\n")
            f.write(f'\nOutput one JSON line per item: {{"item_id":"<ITEM>","primary":"<{unit} id or OTHER>",'
                    '"conf":0.0-1.0}\n')
    _save_state(a.task, a.wave, st)   # anchors injected above must persist for ingest
    print(f"{level}: emitted {len(shards)} payload(s) for {len(items)} items -> {a.wave}/{tag}_payload_*.txt")


def cmd_classify_ingest(a):
    level = a.level
    tag = level.lower()
    await_key = "r2-await" if level == "R2" else "r3-await"
    nxt = "r3-await" if level == "R2" else "novel-category"
    seat_key = "seat_r2" if level == "R2" else "seat_r3"
    verd = "newR1-seated-R2" if level == "R2" else "newR2-seated-R3"
    valid = {t["id"] for t in _taxonomy(a.task, level)}
    st = _load_state(a.task, a.wave)
    prim = {}
    for p in sorted(glob.glob(f"{a.wave}/{tag}_out_*.jsonl")):
        for ln in open(p):
            ln = ln.strip()
            if ln:
                try:
                    r = json.loads(ln); prim[str(r["item_id"])] = str(r.get("primary", "OTHER"))
                except Exception:
                    pass
    truth = st.get(f"_anchor_{tag}", {})
    arep = {"n_recall": 0, "recall_ok": 0, "misses": []}
    for k, t in truth.items():
        v = st["items"].get(k)
        if v is None or v["status"] != await_key:
            continue
        got = prim.get(k, "OTHER")
        arep["n_recall"] += 1
        arep["recall_ok"] += got == str(t["seat"])
        if got != str(t["seat"]):
            arep["misses"].append({"anchor": k, "want": t["seat"], "got": got})
        v["status"], v["verdict"] = "anchor-done", "anchor"
    unjudged = [k for k, v in st["items"].items()
                if v["status"] == await_key and k not in truth and k not in prim]
    if unjudged:
        print(f"WARNING {tag}: {len(unjudged)} awaiting item(s) unclassified — left in {await_key}")
    seated = newnode = 0
    for k, v in st["items"].items():
        if v["status"] != await_key or k in truth or k in unjudged:
            continue
        p = prim.get(k, "OTHER")
        if p in valid:
            v["status"], v[seat_key], v["verdict"] = "seated", p, verd
            seated += 1
        else:
            v["status"] = nxt
            newnode += 1
    st["log"].append({"stage": level, "seated": seated, "new": newnode, "anchor_gate": arep})
    _save_state(a.task, a.wave, st)
    if arep["n_recall"]:
        print(f"ANCHOR GATE {tag}: {arep['recall_ok']}/{arep['n_recall']}"
              + (f"; misses {arep['misses']}" if arep["misses"] else ""))
    print(f"{level}: {seated} seated-{tag} / {newnode} new-{tag}")


# ---- finalize + GT ----------------------------------------------------------------------------
def cmd_finalize(a):
    st = _load_state(a.task, a.wave)
    anchor_keys = set()
    for kk in ("_anchor_l0", "_anchor_r1", "_anchor_r2", "_anchor_r3"):
        anchor_keys |= set(st.get(kk, {}))
    real = {k: v for k, v in st["items"].items() if k not in anchor_keys}
    n = len(real)
    verd = Counter(v.get("verdict", v["status"]) for v in real.values())
    new_l0 = sum(1 for v in real.values() if v.get("verdict") != "seated-L0")
    new_r1 = sum(1 for v in real.values() if v.get("verdict") not in ("seated-L0", "newL0-seated-R1"))
    new_r2 = sum(1 for v in real.values() if v.get("verdict") in ("newR2-seated-R3",) or v["status"] == "novel-category")
    new_r3 = sum(1 for v in real.values() if v["status"] == "novel-category")
    cov = json.load(open(f"{LEX}/coverage_census_20260719.json")).get(a.task, {})
    summary = {
        "wave": a.tag, "task": a.task, "n": n, "verdicts": dict(verd),
        "realized": {"new_L0": round(new_l0 / n, 3), "new_construct": round(new_r1 / n, 3),
                     "new_theme": round(new_r2 / n, 3), "new_category": round(new_r3 / n, 3)},
        "gt_predicted": {"new_L0": cov.get("L0", {}).get("gt_missing_mass"),
                         "new_construct": cov.get("R1", {}).get("gt_missing_mass"),
                         "new_theme": cov.get("R2", {}).get("gt_missing_mass"),
                         "new_category": cov.get("R3", {}).get("gt_missing_mass")},
        "anchor_gates": {e["stage"]: e["anchor_gate"] for e in st["log"] if e.get("anchor_gate")},
    }
    sd = os.path.join(LEX, "crp_ingest", a.task)
    os.makedirs(sd, exist_ok=True)
    ledger = [{"key": k, **{kk: v.get(kk) for kk in
               ("name", "verdict", "status", "seat_l0", "seat_r1", "seat_r2", "seat_r3")}}
              for k, v in real.items()]
    with open(os.path.join(sd, "cascade_seating_ledger.jsonl"), "a") as f:
        for r in ledger:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(sd, "cascade_wave_summaries.jsonl"), "a") as f:
        f.write(json.dumps(summary) + "\n")
    print(json.dumps(summary, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="stage", required=True)
    def common(p):
        p.add_argument("--task", required=True)
        p.add_argument("--wave", required=True)
    for name in ("l0-emit", "l0-ingest", "r1-emit", "r1-ingest"):
        p = sub.add_parser(name); common(p)
        p.add_argument("--k", type=int, default=12)
        p.add_argument("--items-per-shard", type=int, default=12)
        p.add_argument("--anchors", type=int, default=8 if name == "l0-emit" else 4)
        p.add_argument("--anchor-seed", type=int, default=0)
    for name in ("classify-emit", "classify-ingest"):
        p = sub.add_parser(name); common(p)
        p.add_argument("--level", required=True, choices=["R2", "R3"])
        p.add_argument("--items-per-shard", type=int, default=25)
        p.add_argument("--anchors", type=int, default=4)
        p.add_argument("--anchor-seed", type=int, default=0)
    p = sub.add_parser("finalize"); common(p); p.add_argument("--tag", required=True)
    a = ap.parse_args()
    fn = {"l0-emit": cmd_l0_emit, "l0-ingest": cmd_l0_ingest, "r1-emit": cmd_r1_emit,
          "r1-ingest": cmd_r1_ingest, "classify-emit": cmd_classify_emit,
          "classify-ingest": cmd_classify_ingest, "finalize": cmd_finalize}[a.stage]
    fn(a)


if __name__ == "__main__":
    main()
