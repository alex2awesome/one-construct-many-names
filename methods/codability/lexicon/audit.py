"""Hierarchy trust audit — finish the deferred re-clustering audit "beyond v6 llama".

Everything here is judge-grounded and chain-proof (the 2026-06-12 lessons):
  - "same rule" (v6 score=2) is NOT transitive -> no union-find; merges happen ONLY via
    (a) adopt_v2 cluster adoptions already fresh-majority-confirmed (readjudicate_verdicts.jsonl,
        3-vote 70B), or (b) one-round tail adoptions over trusted edges, no chains.
  - single greedy v6 score=2 labels overturn ~55% on re-adjudication -> pair-level trust tiers:
      T1  fresh-majority confirmed (the new fresh-judge pass feeds this)
      T2  triangle-supported (a-b score=2 AND >=1 common neighbor with score=2 to both)
      T3  single greedy label (UNTRUSTED; fresh-judge candidate, never a merge)
  - repairs are append-only: base partition (clusters_<task>.json, locked) + adoption maps.

Products: (1) trusted repaired partition {key -> concept_id}; (2) fresh-judge candidate payloads
(cross-cluster T3 same-edges + lexeme-overlap pairs from the extraction pass); (3) per-level
FP/FN certificates once fresh labels exist.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from .census import norm_term

CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))), ".cache", "norm_embed")


def load_verdicts(task: str, path: Optional[str] = None) -> List[dict]:
    path = path or os.path.join(CACHE, "all_verdicts.jsonl")
    out = []
    with open(path) as f:
        for line in f:
            if f'"task": "{task}"' not in line:
                continue
            r = json.loads(line)
            if r.get("task") == task:
                out.append(r)
    return out


def load_cluster_readjudications(task: str, path: Optional[str] = None) -> Dict[Tuple[str, str], bool]:
    """adopt_v2 3-vote verdicts on (absorbed, target) CLUSTER pairs -> confirmed."""
    path = path or os.path.join(CACHE, "readjudicate_verdicts.jsonl")
    out: Dict[Tuple[str, str], bool] = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("task") != task or "absorbed" not in r or "target" not in r:
                continue
            out[(str(r["absorbed"]), str(r["target"]))] = bool(r.get("confirmed"))
    return out


def load_partition(task: str, path: Optional[str] = None) -> Dict[str, str]:
    """key -> base cluster id (locked tau-0.825 artifact, read-only)."""
    path = path or os.path.join(CACHE, "match_out", f"clusters_{task}.json")
    d = json.load(open(path))
    return {k: str(v) for k, v in d.items()}


def trust_edges(verdicts: List[dict],
                fresh_pairs: Optional[Dict[frozenset, bool]] = None) -> dict:
    """Pair-level tiers over score-2 edges; score-0 blockers kept separately."""
    fresh_pairs = fresh_pairs or {}
    same_adj: Dict[str, Set[str]] = defaultdict(set)
    pairs2: Set[frozenset] = set()
    pairs0: Set[frozenset] = set()
    for r in verdicts:
        ka, kb = r.get("key_a"), r.get("key_b")
        if not (ka and kb) or ka == kb or r.get("score") is None:
            continue
        s = int(r["score"])
        if s == 2:
            pairs2.add(frozenset((ka, kb)))
            same_adj[ka].add(kb)
            same_adj[kb].add(ka)
        elif s == 0:
            pairs0.add(frozenset((ka, kb)))
    t1, t2, t3 = set(), set(), set()
    for p in pairs2:
        ka, kb = tuple(p)
        if p in fresh_pairs:
            (t1 if fresh_pairs[p] else t3).add(p)
        elif same_adj[ka] & same_adj[kb]:
            t2.add(p)
        else:
            t3.add(p)
    return {"t1": t1, "t2": t2, "t3": t3, "score0": pairs0, "same_adj": same_adj}


def repaired_partition(task: str, verdicts: Optional[List[dict]] = None,
                       fresh_pairs: Optional[Dict[frozenset, bool]] = None) -> dict:
    """Base partition + (a) adopt_v2 confirmed cluster adoptions (fresh-majority, apply
    directly) + (b) one-round chain-proof tail adoptions over trusted pair edges. Larger-cluster
    trusted cross-edges are RECORDED as merge candidates, never auto-merged."""
    verdicts = verdicts if verdicts is not None else load_verdicts(task)
    base = load_partition(task)
    members: Dict[str, List[str]] = defaultdict(list)
    for k, c in base.items():
        members[c].append(k)

    # (a) adopt_v2 fresh-confirmed cluster adoptions — one hop, no chaining
    readj = load_cluster_readjudications(task)
    cluster_map: Dict[str, str] = {}
    targets = {t for (_, t), ok in readj.items() if ok}
    for (absorbed, target), ok in readj.items():
        if ok and absorbed not in targets and absorbed in members and target in members:
            cluster_map[absorbed] = target

    # (b) pair-level trust over the remaining structure
    T = trust_edges(verdicts, fresh_pairs)
    trusted = T["t1"] | T["t2"]

    def cur(c: str) -> str:
        return cluster_map.get(c, c)

    blocked: Set[Tuple[str, str]] = set()
    for p in T["score0"]:
        ka, kb = tuple(p)
        ca, cb = base.get(ka), base.get(kb)
        if ca and cb and cur(ca) != cur(cb):
            blocked.add(tuple(sorted((cur(ca), cur(cb)))))
    cross_votes: Dict[Tuple[str, str], int] = defaultdict(int)
    cross_votes_t1: Dict[Tuple[str, str], int] = defaultdict(int)
    for p in trusted:
        ka, kb = tuple(p)
        ca, cb = base.get(ka), base.get(kb)
        if ca and cb and cur(ca) != cur(cb):
            pair = tuple(sorted((cur(ca), cur(cb))))
            cross_votes[pair] += 1
            if p in T["t1"]:
                cross_votes_t1[pair] += 1

    adoptions: Dict[str, str] = {}
    merge_candidates: List[dict] = []
    moved: Set[str] = set()
    heads: Set[str] = set()
    for (ca, cb), n in sorted(cross_votes.items(), key=lambda kv: -kv[1]):
        if (ca, cb) in blocked or ca in moved or cb in moved:
            continue
        sa, sb = len(members.get(ca, [])), len(members.get(cb, []))
        tail, head = (ca, cb) if sa <= sb else (cb, ca)
        if tail in heads:
            continue
        if min(sa, sb) <= 2:
            adoptions[tail] = head
            moved.add(tail)
            heads.add(head)
        elif cross_votes_t1[(ca, cb)] >= 2:
            # >=2 independent fresh-majority-confirmed edges = the adopt_v2 evidence standard;
            # apply as a one-round merge, chain-proof via both moved-tail and heads guards.
            adoptions[tail] = head
            moved.add(tail)
            heads.add(head)
        elif n >= 2:
            merge_candidates.append({"a": ca, "b": cb, "trusted_edges": n, "sizes": [sa, sb]})

    def final(c: str) -> str:
        c = cur(c)
        return adoptions.get(c, c)

    part = {k: final(c) for k, c in base.items()}
    fresh_cands = []
    for p in T["t3"]:
        ka, kb = tuple(p)
        if base.get(ka) and base.get(kb) and final(base[ka]) != final(base[kb]):
            fresh_cands.append(sorted((ka, kb)))
    return {"partition": part,
            "adopt_v2_applied": len(cluster_map),
            "tail_adoptions": len(adoptions),
            "merge_candidates": merge_candidates,
            "n_trusted_edges": len(trusted), "n_untrusted_t3": len(T["t3"]),
            "n_clusters_base": len(set(base.values())),
            "n_clusters_repaired": len(set(part.values())),
            "fresh_judge_candidates": fresh_cands}


def lexeme_overlap_candidates(extractions: Dict[str, dict], partition: Dict[str, str],
                              max_df: int = 20, top: int = 4000) -> List[List[str]]:
    """Embedding-free under-merge net from the extraction pass: cross-cluster key pairs sharing
    a RARE author lexeme (document frequency <= max_df). These go to the fresh judge."""
    by_term: Dict[str, List[str]] = defaultdict(list)
    for k, r in extractions.items():
        if r.get("status") != "ok" or not r.get("found"):
            continue
        terms = {norm_term(t) for t in (r.get("key_terms") or [])}
        if r.get("head_term"):
            terms.add(norm_term(r["head_term"]))
        for t in terms:
            if t and len(t) > 3:
                by_term[t].append(k)
    scored: Dict[frozenset, int] = defaultdict(int)
    for t, keys in by_term.items():
        if not (2 <= len(keys) <= max_df):
            continue
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = keys[i], keys[j]
                if partition.get(a) and partition.get(b) and partition[a] != partition[b]:
                    scored[frozenset((a, b))] += 1
    ranked = sorted(scored.items(), key=lambda kv: -kv[1])[:top]
    return [sorted(p) for p, _ in ranked]


def leaf_certificate(task: str, partition: Dict[str, str],
                     fresh_pairs: Dict[frozenset, bool],
                     verdicts: Optional[List[dict]] = None) -> dict:
    """Realized precision/recall of a partition against trusted + fresh labels.

    recall    = P(co-clustered | labeled SAME)   — the quantity the pipeline has always tracked
                (realized, per the 2026-06-12 correction; NOT pair-affinity recall).
    precision = P(labeled SAME | co-clustered)   — over labeled co-clustered pairs only.
    Caveats carried by the caller: labels live on the kNN candidate net (in-net recall), and
    the SAME pool is trusted+fresh edges only (single-greedy T3 excluded until adjudicated)."""
    verdicts = verdicts if verdicts is not None else load_verdicts(task)
    labels: Dict[frozenset, bool] = {}
    T = trust_edges(verdicts, fresh_pairs)
    for p in T["t1"] | T["t2"]:
        labels[p] = True
    for p in T["score0"]:
        labels[p] = False
    labels.update(fresh_pairs)
    n_same = same_co = n_co = co_same = 0
    for p, same in labels.items():
        ka, kb = tuple(p)
        ca, cb = partition.get(ka), partition.get(kb)
        if not (ca and cb):
            continue
        co = ca == cb
        if same:
            n_same += 1
            same_co += int(co)
        if co:
            n_co += 1
            co_same += int(same)
    return {"task": task,
            "n_same_labeled": n_same, "recall": (same_co / n_same) if n_same else None,
            "n_colabeled": n_co, "precision": (co_same / n_co) if n_co else None}
