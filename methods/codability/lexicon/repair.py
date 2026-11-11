#!/usr/bin/env python
"""Packaged L0 under-merge repair — generalizes the humor round-2 (recall .785->.857) to any task.

Pipeline (mirrors notes/2026-07-06__undermerge-repair-program.md):
  1. build_candidates  — cross-cluster under-merge suspects: TF-IDF kNN (high cos) UNION shared-rare
     -name UNION extraction-lexeme net. Embedding-free, local, no CE required (CE was only a router).
  2. emit_screen_payloads — blind pair chunks (+ ride-along blinded anchors) for an LLM screen fleet.
  3. ingest_verified   — permissive screen -> independent strict L0 same-criterion confirm ->
     verified SAME edges. New confirms must use CONFIRM_PROTOCOL_L0_V2.txt; historical L0v2/v3
     confirms used the broader R1 same-construct prompt and are preserved but audited separately.
  4. apply_merges      — >=2 verified independent edges per cluster-pair => STAR-1-ROUND merge
     (smaller into larger, one hop, moved-guard). NEVER union-find (chaining tanks precision .69->.60).
  5. score_vs_truth    — realized recall P(co|same) / precision P(same|co) vs adjudicated_truth_<task>.json.

Base partition = numerically latest partition_<task>_L0vN.json, falling back to the legacy r1 file.
Repairs are append-only; the base is never mutated. Score is the FROZEN adjudicated truth.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from . import audit
from .census import norm_term
from .judge import canon_map
from .sources import ROOT

OUT = os.path.join(ROOT, "outputs", "lexicon")


def _pid(a: str, b: str) -> str:
    return hashlib.sha1("||".join(sorted((a, b))).encode()).hexdigest()[:16]


def load_base_partition(task: str) -> Dict[str, str]:
    """Current partition to repair: latest append-only L0vN, then legacy r1, then audit output."""
    versions = []
    for path in glob.glob(os.path.join(OUT, f"partition_{task}_L0v*.json")):
        m = re.search(r"_L0v(\d+)\.json$", path)
        if m:
            versions.append((int(m.group(1)), path))
    p = max(versions)[1] if versions else os.path.join(OUT, f"partition_{task}_r1.json")
    if os.path.exists(p):
        d = json.load(open(p))
        return {k: str(v) for k, v in (d.get("partition", d)).items()}
    return {k: str(v) for k, v in audit.repaired_partition(task)["partition"].items()}


def load_extractions(task: str) -> Dict[str, dict]:
    """Author key-term extractions if present (extract_bundles/sonnet_out_<task>_*.jsonl or
    extract_<task>_*.jsonl). Keyed by canon key. Empty dict if none — lexeme net just drops out."""
    out: Dict[str, dict] = {}
    for pat in (f"extract_bundles/sonnet_out_{task}_*.jsonl", f"extract_{task}_*.jsonl"):
        for f in sorted(glob.glob(os.path.join(OUT, pat))):
            for line in open(f):
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("key"):
                    out[r["key"]] = r
    return out


def build_candidates(task: str, partition: Dict[str, str], knn: int = 20, min_cos: float = 0.2,
                     max_name_df: int = 12, cap: int = 15000,
                     block_score0: bool = True) -> List[dict]:
    """Cross-cluster under-merge suspects from ALL modalities, so no true under-merge is out of
    reach (net_ceiling.py: TF-IDF(0.2) UNION name UNION BGE hits ceiling 1.0 on adjudicated eval).

    The net sets the RECALL CEILING; the verifier + >=2-edge star gate set precision. So go WIDE:
      (1) TF-IDF kNN cross-cluster pairs (cos >= min_cos) — lexical net.
      (2) shared-rare-name cross-cluster pairs — surface-name net.
      (3) extraction-lexeme cross-cluster pairs — author key-term net (if extractions present).
      (4) v6 BGE candidate universe (all_verdicts pairs) — SEMANTIC net, catches paraphrases the
          lexical nets miss (~22-33% of under-merges are lexically invisible). v6-score-2 cross
          pairs (found-but-not-merged) are the highest-yield and float to the top of the ranking.
    Ranked: known-v6-SAME first, then cosine desc. cap keeps the top band for a bounded screen."""
    cmap = canon_map(task)
    keys = sorted(k for k in cmap if k in partition)
    texts = [cmap[k] for k in keys]
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors
    X = TfidfVectorizer(min_df=2, max_features=40000, sublinear_tf=True).fit_transform(texts)
    nn = NearestNeighbors(n_neighbors=min(knn + 1, len(keys)), metric="cosine").fit(X)
    dist, idx = nn.kneighbors(X)

    cand: Dict[frozenset, dict] = {}

    def add(a: str, b: str, source: str, cos: Optional[float], v6: Optional[int] = None):
        if a == b or partition.get(a) is None or partition.get(b) is None:
            return
        if partition[a] == partition[b]:
            return
        fs = frozenset((a, b))
        if fs in cand:
            if cos is not None and (cand[fs]["cos"] is None or cos > cand[fs]["cos"]):
                cand[fs]["cos"] = round(cos, 3)
            if v6 is not None:
                cand[fs]["v6_score"] = max(cand[fs].get("v6_score", -1), v6)
            cand[fs]["source"] = "+".join(sorted(set(cand[fs]["source"].split("+")) | {source}))
            return
        cand[fs] = {"pair_id": _pid(a, b), "task": task, "key_a": a, "key_b": b,
                    "canonical_a": cmap[a], "canonical_b": cmap[b],
                    "cos": round(cos, 3) if cos is not None else None, "source": source,
                    "v6_score": v6 if v6 is not None else -1}

    # (1) TF-IDF kNN
    for i in range(len(keys)):
        for jp in range(1, idx.shape[1]):
            j = int(idx[i][jp])
            cos = 1.0 - float(dist[i][jp])
            if cos >= min_cos and i < j:
                add(keys[i], keys[j], "knn", cos)
    # (2) shared-rare-name-token
    tok2keys: Dict[str, List[str]] = defaultdict(list)
    for k in keys:
        name = cmap[k].split(".")[0][:80]
        for w in set(norm_term(name).split()):
            if len(w) > 5:
                tok2keys[w].append(k)
    for w, ks in tok2keys.items():
        if 2 <= len(ks) <= max_name_df:
            for x in range(len(ks) - 1):
                add(ks[x], ks[x + 1], "name", None)
    # (3) extraction-lexeme net
    ex = load_extractions(task)
    if ex:
        for a, b in audit.lexeme_overlap_candidates(ex, partition, top=8000):
            add(a, b, "lexeme", None)
    # (4) v6 BGE candidate universe (semantic net); carry v6 score + BGE cos
    verd = audit.load_verdicts(task)
    for r in verd:
        ka, kb = r.get("key_a"), r.get("key_b")
        if ka and kb:
            add(ka, kb, "bge", r.get("cos"), r.get("score") if r.get("score") in (0, 1, 2) else None)
    # drop score-0 blocked cluster-pairs
    if block_score0:
        blocked: Set[Tuple[str, str]] = set()
        for r in verd:
            if r.get("score") == 0:
                ca, cb = partition.get(r.get("key_a")), partition.get(r.get("key_b"))
                if ca and cb and ca != cb:
                    blocked.add(tuple(sorted((ca, cb))))
        cand = {fs: v for fs, v in cand.items()
                if tuple(sorted((partition[v["key_a"]], partition[v["key_b"]]))) not in blocked}

    # rank: known-v6-SAME (score 2) first, then cosine desc
    ranked = sorted(cand.values(), key=lambda r: (-(1 if r.get("v6_score") == 2 else 0),
                                                  -(r["cos"] if r["cos"] is not None else 0.0),
                                                  r["pair_id"]))
    return ranked[:cap]


def emit_screen_payloads(task: str, candidates: List[dict], per_agent: int = 120,
                         n_anchor: int = 8) -> List[str]:
    """Blind screen payloads + ride-along blinded anchors (known 0/2) per shard."""
    pv = [json.loads(l) for l in open(os.path.join(OUT, f"arbiter_eval_{task}.jsonl"))]
    anc_path = os.path.join(OUT, f"arbiter_anchors_{task}.json")
    anchors = json.load(open(anc_path)) if os.path.exists(anc_path) else {}
    eval_by_pid = {r["pair_id"]: r for r in pv}
    anchor_rows = [{"pair_id": pid, "task": task, "canonical_a": eval_by_pid[pid]["canonical_a"],
                    "canonical_b": eval_by_pid[pid]["canonical_b"]}
                   for pid in anchors if pid in eval_by_pid]
    pd = os.path.join(OUT, "repair_payloads")
    os.makedirs(pd, exist_ok=True)
    outs = []
    import random
    rng = random.Random(0)
    for a in range(0, len(candidates), per_agent):
        i = a // per_agent
        chunk = candidates[a: a + per_agent]
        rows = [{"pair_id": r["pair_id"], "task": task, "canonical_a": r["canonical_a"],
                 "canonical_b": r["canonical_b"]} for r in chunk]
        ride = rng.sample(anchor_rows, min(n_anchor, len(anchor_rows))) if anchor_rows else []
        rows = rows + ride
        rng.shuffle(rows)
        p = os.path.join(pd, f"{task}_screen{i:03d}.jsonl")
        with open(p, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        outs.append(p)
    json.dump({"n_candidates": len(candidates), "n_shards": len(outs), "per_agent": per_agent,
               "anchors_ride": n_anchor}, open(os.path.join(OUT, f"repair_manifest_{task}.json"), "w"))
    print(f"[{task}] {len(candidates)} candidates -> {len(outs)} screen shards @ {per_agent} (+{n_anchor} anchors) -> {pd}")
    return outs


def _load_scored(pattern: str, *, diagnostics: bool = False):
    rows: Dict[str, List[int]] = defaultdict(list)
    malformed = 0
    for f in sorted(glob.glob(os.path.join(OUT, pattern))):
        for line in open(f):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            s = r.get("score")
            # strict int: isinstance(True, int) is True (bool subclasses int), so a JSON `true`
            # vote would otherwise be silently accepted as score 1. type(s) is int rejects bool/float.
            if type(s) is int and s in (0, 1, 2) and r.get("pair_id"):
                rows[r["pair_id"]].append(s)
            else:
                malformed += 1
    duplicate = sorted(p for p, scores in rows.items() if len(scores) > 1)
    conflicting = sorted(p for p, scores in rows.items() if len(set(scores)) > 1)
    out = {p: scores[0] for p, scores in rows.items() if len(scores) == 1}
    diag = {"malformed": malformed, "duplicate_pair_ids": duplicate,
            "conflicting_pair_ids": conflicting}
    return (out, diag) if diagnostics else out


def screen_advance(task: str, candidates: List[dict], screen_glob: str,
                   screen_min: int = 1) -> List[dict]:
    """Permissive screen: advance candidates the (stricter) Sonnet-5 screen scored >= screen_min
    (default 1 = related-or-same) to the Opus confirm gate. Drops only score-0 (unrelated). This
    keeps Sonnet-5's conservatism from capping recall; Opus makes the actual SAME call."""
    by_pid = {c["pair_id"]: c for c in candidates}
    screen = _load_scored(screen_glob)
    return [by_pid[pid] for pid, s in screen.items() if s >= screen_min and pid in by_pid]


def ingest_verified(task: str, candidates: List[dict], screen_glob: str,
                    confirm_glob: Optional[str] = None, screen_min: int = 1, *,
                    require_complete: bool = True,
                    allow_provisional: bool = False) -> List[Tuple[str, str]]:
    """Two-family verified SAME edges: Sonnet screen advances (score >= screen_min) -> Opus confirm
    score==2. If no confirm pass yet, returns the strict Sonnet-SAME (score==2) edges provisionally."""
    by_pid = {c["pair_id"]: (c["key_a"], c["key_b"]) for c in candidates}
    screen, screen_diag = _load_scored(screen_glob, diagnostics=True)
    relevant_dups = set(screen_diag["duplicate_pair_ids"]) & set(by_pid)
    if require_complete and (screen_diag["malformed"] or relevant_dups):
        raise RuntimeError(f"[{task}] invalid screen votes: malformed={screen_diag['malformed']}, "
                           f"duplicate candidate ids={len(relevant_dups)}")
    if confirm_glob:
        advanced = {pid for pid, s in screen.items() if s >= screen_min and pid in by_pid}
        confirm, confirm_diag = _load_scored(confirm_glob, diagnostics=True)
        missing_confirm = advanced - set(confirm)
        relevant_confirm_dups = set(confirm_diag["duplicate_pair_ids"]) & advanced
        if require_complete and (confirm_diag["malformed"] or missing_confirm
                                 or relevant_confirm_dups):
            raise RuntimeError(f"[{task}] incomplete/invalid confirm votes: "
                               f"missing={len(missing_confirm)}, malformed={confirm_diag['malformed']}, "
                               f"duplicate advanced ids={len(relevant_confirm_dups)}")
        same = {pid for pid in advanced if confirm.get(pid) == 2}
    else:
        if not allow_provisional:
            raise RuntimeError(f"[{task}] no confirm pass supplied; set allow_provisional=True "
                               f"only for a result that will not be frozen as L0vN")
        same = {pid for pid, s in screen.items() if s == 2 and pid in by_pid}
    return [by_pid[pid] for pid in same]


def apply_merges(base: Dict[str, str], verified_edges: List[Tuple[str, str]],
                 min_edges: int = 2, task: Optional[str] = None) -> dict:
    """>=min_edges independent verified edges per cluster-pair => star-1-round merge (smaller into
    larger, one hop, moved-guard). NEVER union-find. Optionally block score-0 cluster-pairs."""
    cross: Dict[Tuple[str, str], int] = defaultdict(int)
    for a, b in verified_edges:
        ca, cb = base.get(a), base.get(b)
        if ca and cb and ca != cb:
            cross[tuple(sorted((ca, cb)))] += 1
    blocked: Set[Tuple[str, str]] = set()
    if task:
        for r in audit.load_verdicts(task):
            if r.get("score") == 0:
                ca, cb = base.get(r.get("key_a")), base.get(r.get("key_b"))
                if ca and cb and ca != cb:
                    blocked.add(tuple(sorted((ca, cb))))
    members: Dict[str, int] = defaultdict(int)
    for k, c in base.items():
        members[c] += 1
    adopt: Dict[str, str] = {}
    moved: Set[str] = set()
    heads: Set[str] = set()
    n_merges = 0
    for (ca, cb), n in sorted(cross.items(), key=lambda kv: -kv[1]):
        if n < min_edges or (ca, cb) in blocked or ca in moved or cb in moved:
            continue
        tail, head = (ca, cb) if members[ca] <= members[cb] else (cb, ca)
        # STAR-1-ROUND invariant: a head must never become a tail (else A->B, B->C chains and the
        # A->B merge is not realized). moved-guard alone only locks tails; also lock heads.
        if tail in heads:
            continue
        adopt[tail] = head
        moved.add(tail)
        heads.add(head)
        n_merges += 1
    part = {k: adopt.get(c, c) for k, c in base.items()}
    return {"partition": part, "n_merges": n_merges,
            "n_cluster_pairs_with_edges": len(cross),
            "n_clusters_before": len(set(base.values())),
            "n_clusters_after": len(set(part.values()))}


def score_vs_truth(task: str, partition: Dict[str, str]) -> dict:
    """Realized recall P(co|same) / precision P(same|co) vs the FROZEN adjudicated truth."""
    tp = os.path.join(OUT, f"adjudicated_truth_{task}.json")
    if not os.path.exists(tp):
        return {"task": task, "error": "no adjudicated truth"}
    truth = {k: bool(v) for k, v in json.load(open(tp)).items()}
    eval_by_pid = {r["pair_id"]: r for r in
                   (json.loads(l) for l in open(os.path.join(OUT, f"arbiter_eval_{task}.jsonl")))}
    n_same = same_co = n_co = co_same = miss = 0
    for pid, same in truth.items():
        r = eval_by_pid.get(pid)
        if not r:
            miss += 1
            continue
        ca, cb = partition.get(r["key_a"]), partition.get(r["key_b"])
        if ca is None or cb is None:
            miss += 1
            continue
        co = ca == cb
        if same:
            n_same += 1
            same_co += int(co)
        if co:
            n_co += 1
            co_same += int(same)
    rec = (same_co / n_same) if n_same else None
    prec = (co_same / n_co) if n_co else None
    f1 = (2 * rec * prec / (rec + prec)) if (rec is not None and prec is not None
                                               and rec + prec > 0) else (0.0 if rec == 0 or prec == 0 else None)
    total = len(truth)
    return {"task": task, "n_same": n_same, "recall": round(rec, 3) if rec is not None else None,
            "n_colabeled": n_co, "precision": round(prec, 3) if prec is not None else None,
            "f1": round(f1, 3) if f1 is not None else None, "unmapped": miss,
            "coverage": round((total - miss) / total, 3) if total else None,
            "complete": miss == 0}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["baseline", "candidates", "apply"])
    ap.add_argument("--task", required=True)
    ap.add_argument("--per-agent", type=int, default=120)
    ap.add_argument("--min-edges", type=int, default=2)
    ap.add_argument("--screen-glob", default=None)
    ap.add_argument("--confirm-glob", default=None)
    ap.add_argument("--allow-provisional", action="store_true")
    a = ap.parse_args()
    base = load_base_partition(a.task)
    if a.cmd == "baseline":
        print(json.dumps(score_vs_truth(a.task, base), indent=1))
    elif a.cmd == "candidates":
        cand = build_candidates(a.task, base)
        json.dump(cand, open(os.path.join(OUT, f"repair_candidates_{a.task}.json"), "w"))
        emit_screen_payloads(a.task, cand, per_agent=a.per_agent)
    elif a.cmd == "apply":
        cand = json.load(open(os.path.join(OUT, f"repair_candidates_{a.task}.json")))
        edges = ingest_verified(a.task, cand, a.screen_glob or f"repair_votes/screen_{a.task}_*.jsonl",
                                a.confirm_glob, allow_provisional=a.allow_provisional)
        res = apply_merges(base, edges, min_edges=a.min_edges, task=a.task)
        before = score_vs_truth(a.task, base)
        after = score_vs_truth(a.task, res["partition"])
        if not before.get("complete") or not after.get("complete"):
            raise RuntimeError(f"[{a.task}] frozen truth does not map completely; refusing final apply")
        versions = [int(m.group(1)) for path in glob.glob(os.path.join(OUT, f"partition_{a.task}_L0v*.json"))
                    if (m := re.search(r"_L0v(\d+)\.json$", path))]
        next_version = max(versions, default=1) + 1
        out_path = os.path.join(OUT, f"partition_{a.task}_L0v{next_version}.json")
        if os.path.exists(out_path):
            raise FileExistsError(f"append-only destination already exists: {out_path}")
        json.dump(res["partition"], open(out_path, "w"))
        print(json.dumps({"n_verified_edges": len(edges), **{k: v for k, v in res.items() if k != "partition"},
                          "before": before, "after": after, "partition_path": out_path}, indent=1))


if __name__ == "__main__":
    main()
