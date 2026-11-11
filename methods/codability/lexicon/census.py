"""The conventionalization census: naming agreement / name entropy / synonymy / subfield control.

Inputs: (1) extraction JSONL (author lexemes per metric, verbatim-validated), (2) a concept
partition {key -> concept_id} from audit.py (judge-grounded; NEVER the verdict-trained blend
clusters), (3) optional {concept_id -> higher node} maps for the R3-subtree readout.

Independence unit = source_id (normalized URL): repeats of a concept inside one source never
count as independent conventionalization evidence. Term normalization is deliberately
conservative (casefold, punctuation strip, plural-s squeeze) — over-normalization would HIDE
real lexical variation, which is the quantity under study.
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Optional

_PUNCT = re.compile(r"[^\w\s'-]")
_WS = re.compile(r"\s+")


def norm_term(t: str) -> str:
    s = _WS.sub(" ", _PUNCT.sub(" ", (t or "").casefold())).strip()
    words = [w[:-1] if (len(w) > 3 and w.endswith("s") and not w.endswith("ss")) else w
             for w in s.split()]
    return " ".join(words)


def _entropy(counts: Iterable[int]) -> float:
    tot = sum(counts)
    if tot <= 0:
        return 0.0
    return -sum((c / tot) * math.log2(c / tot) for c in counts if c > 0)


def concept_census(recs: List[dict]) -> Optional[dict]:
    """One concept's conventionalization profile from its member extraction records."""
    by_src: Dict[str, List[dict]] = defaultdict(list)
    for r in recs:
        if r.get("status") == "ok" and r.get("found"):
            by_src[r.get("source_id") or r["key"]].append(r)
    if not by_src:
        return None
    heads: Dict[str, int] = defaultdict(int)
    named_flags, term_sets = [], []
    for src, rows in by_src.items():
        named = [norm_term(r["head_term"]) for r in rows
                 if r.get("named_in_source") and r.get("head_term")]
        named_flags.append(bool(named))
        if named:
            heads[max(set(named), key=named.count)] += 1
        ts = set()
        for r in rows:
            ts |= {norm_term(t) for t in (r.get("key_terms") or []) if norm_term(t)}
        term_sets.append(ts)
    n_src = len(by_src)
    n_named = sum(named_flags)
    pair_j, disjoint = [], 0
    for i in range(len(term_sets)):
        for j in range(i + 1, len(term_sets)):
            a, b = term_sets[i], term_sets[j]
            if a or b:
                jc = len(a & b) / max(1, len(a | b))
                pair_j.append(jc)
                disjoint += int(jc == 0.0)
    top = max(heads.values()) if heads else 0
    return {
        "n_sources": n_src,
        "n_named_sources": n_named,
        "unnamed_rate": 1.0 - (n_named / n_src),
        "n_head_lexicalizations": len(heads),
        "head_terms": dict(sorted(heads.items(), key=lambda kv: -kv[1])),
        "naming_agreement": (top / n_named) if n_named else None,
        "name_entropy_bits": _entropy(heads.values()) if heads else None,
        "mean_pair_jaccard": (sum(pair_j) / len(pair_j)) if pair_j else None,
        "n_pairs": len(pair_j),
        "disjoint_pair_rate": (disjoint / len(pair_j)) if pair_j else None,
    }


def task_census(extractions: Dict[str, dict], partition: Dict[str, str],
                min_sources: int = 2) -> dict:
    """Per-concept census + task-level headline numbers.

    partition: key -> concept_id (judge-grounded). Keys absent from the partition are skipped
    (they are counted, not silently dropped)."""
    groups: Dict[str, List[dict]] = defaultdict(list)
    n_unpartitioned = 0
    for k, r in extractions.items():
        cid = partition.get(k)
        if cid is None:
            n_unpartitioned += 1
            continue
        groups[cid].append(r)
    per: Dict[str, dict] = {}
    for cid, recs in groups.items():
        c = concept_census(recs)
        if c is not None:
            per[cid] = c
    multi = {cid: c for cid, c in per.items() if c["n_sources"] >= min_sources}
    # Synonymy requires multiple AUTHOR-USED head labels. Disjoint descriptive key terms indicate
    # lexical divergence, not necessarily alternate names for the same concept.
    syn = [cid for cid, c in multi.items() if c["n_head_lexicalizations"] >= 2]
    divergent = [cid for cid, c in multi.items() if (c["disjoint_pair_rate"] or 0) > 0]
    named_multi = [c for c in multi.values() if c["n_named_sources"] >= 2]
    return {
        "n_keys": len(extractions), "n_unpartitioned": n_unpartitioned,
        "n_concepts": len(per), "n_concepts_multi_source": len(multi),
        "synonymy_rate": (len(syn) / len(multi)) if multi else None,
        "descriptive_divergence_rate": (len(divergent) / len(multi)) if multi else None,
        "mean_unnamed_rate": (sum(c["unnamed_rate"] for c in multi.values()) / len(multi))
                             if multi else None,
        "mean_naming_agreement_named": (sum(c["naming_agreement"] for c in named_multi)
                                        / len(named_multi)) if named_multi else None,
        "mean_name_entropy_named": (sum(c["name_entropy_bits"] for c in named_multi)
                                    / len(named_multi)) if named_multi else None,
        "concepts": per,
    }


def subtree_census(extractions: Dict[str, dict], partition: Dict[str, str],
                   up_map: Dict[str, str], min_sources: int = 2) -> dict:
    """The user's R3 readout: group leaves by their higher node (concept_id -> R3 node via
    up_map) and census each subtree — 'the different ways subtrees are coded underneath'."""
    lifted = {k: up_map.get(cid) for k, cid in partition.items() if up_map.get(cid)}
    return task_census(extractions, {k: v for k, v in lifted.items() if v}, min_sources)


def strata_decomposition(extractions: Dict[str, dict], partition: Dict[str, str],
                         stratum_field: str = "subtask_short") -> dict:
    """Within- vs cross-subfield lexical divergence over same-concept source pairs — the control
    that keeps genre-driven wording variation from reading as low conventionalization."""
    groups: Dict[str, List[dict]] = defaultdict(list)
    for k, r in extractions.items():
        cid = partition.get(k)
        if cid is not None and r.get("status") == "ok" and r.get("found"):
            groups[cid].append(r)
    within, cross = [], []
    for recs in groups.values():
        by_source: Dict[str, List[dict]] = defaultdict(list)
        for r in recs:
            by_source[r.get("source_id") or r["key"]].append(r)
        rows = []
        for source_id, source_rows in by_source.items():
            terms = set()
            strata_seen = set()
            for r in source_rows:
                terms |= {norm_term(t) for t in r.get("key_terms") or [] if norm_term(t)}
                s = (r.get("strata") or {}).get(stratum_field)
                if s:
                    strata_seen.add(str(s))
            rows.append({"source_id": source_id, "terms": terms,
                         "stratum": next(iter(strata_seen)) if len(strata_seen) == 1 else "unknown"})
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a, b = rows[i]["terms"], rows[j]["terms"]
                if not (a or b):
                    continue
                jc = len(a & b) / max(1, len(a | b))
                sa, sb = rows[i]["stratum"], rows[j]["stratum"]
                if "unknown" in (sa, sb):
                    continue
                (within if sa == sb else cross).append(jc)
    return {"stratum_field": stratum_field,
            "n_within": len(within), "n_cross": len(cross),
            "mean_jaccard_within": (sum(within) / len(within)) if within else None,
            "mean_jaccard_cross": (sum(cross) / len(cross)) if cross else None,
            "delta_within_minus_cross": ((sum(within) / len(within)) - (sum(cross) / len(cross)))
                                        if (within and cross) else None}


def main():
    import argparse
    from .extract import load_extractions
    p = argparse.ArgumentParser()
    p.add_argument("--extractions", required=True)
    p.add_argument("--partition", required=True, help="json {key: concept_id}")
    p.add_argument("--up-map", default="", help="json {concept_id: r3_node} for subtree census")
    p.add_argument("--out", required=True)
    p.add_argument("--min-sources", type=int, default=2)
    a = p.parse_args()
    ex = load_extractions(a.extractions, only_ok=False)
    part = json.load(open(a.partition))
    rep = {"concept_level": task_census(ex, part, a.min_sources),
           "strata": strata_decomposition(ex, part)}
    if a.up_map:
        rep["r3_subtree"] = subtree_census(ex, part, json.load(open(a.up_map)), a.min_sources)
    json.dump(rep, open(a.out, "w"), indent=1)
    c = rep["concept_level"]
    print(f"concepts={c['n_concepts']} multi-source={c['n_concepts_multi_source']} "
          f"synonymy_rate={c['synonymy_rate']} unnamed={c['mean_unnamed_rate']}")


if __name__ == "__main__":
    main()
