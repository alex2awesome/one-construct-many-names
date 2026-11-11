"""GLM-vs-Sonnet equivalence report for the extraction pass (user gate: GLM-4.7 preferred iff
equivalent to Sonnet; both must clear the blinded anchors).

Agreement is computed on the shared ok-status keys; anchors are scored per model against the
planted expectations. The verdict thresholds are stated in the report, not hidden in code review:
equivalent iff mean key-term Jaccard >= 0.55 AND head-term agreement >= 0.75 AND found agreement
>= 0.90 AND both anchor pass rates >= 0.85.
"""
from __future__ import annotations

import json
from typing import Dict, List

from .anchors import anchor_keys, score_anchor_batch
from .sources import norm_text


def _term_set(rec: dict) -> set:
    return {norm_text(t) for t in (rec.get("key_terms") or []) if norm_text(t)}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def compare(recs_a: Dict[str, dict], recs_b: Dict[str, dict],
            name_a: str = "glm", name_b: str = "sonnet") -> dict:
    ak = anchor_keys()
    shared = sorted((set(recs_a) & set(recs_b)) - ak)
    rows: List[dict] = []
    for k in shared:
        ra, rb = recs_a[k], recs_b[k]
        ha, hb = norm_text(ra.get("head_term") or ""), norm_text(rb.get("head_term") or "")
        rows.append({
            "key": k,
            "jaccard": _jaccard(_term_set(ra), _term_set(rb)),
            "head_match": ha == hb,
            "named_match": bool(ra.get("named_in_source")) == bool(rb.get("named_in_source")),
            "found_match": bool(ra.get("found")) == bool(rb.get("found")),
        })
    n = max(1, len(rows))
    agg = {
        "n_shared_ok": len(rows),
        "mean_jaccard": sum(r["jaccard"] for r in rows) / n,
        "head_agreement": sum(r["head_match"] for r in rows) / n,
        "named_agreement": sum(r["named_match"] for r in rows) / n,
        "found_agreement": sum(r["found_match"] for r in rows) / n,
    }
    anch = {name_a: score_anchor_batch({k: v for k, v in recs_a.items() if k in ak}),
            name_b: score_anchor_batch({k: v for k, v in recs_b.items() if k in ak})}
    ok_rate = {name_a: _ok_rate(recs_a, ak), name_b: _ok_rate(recs_b, ak)}
    verdict = (agg["mean_jaccard"] >= 0.55 and agg["head_agreement"] >= 0.75
               and agg["found_agreement"] >= 0.90
               and anch[name_a]["pass_rate"] >= 0.85 and anch[name_b]["pass_rate"] >= 0.85)
    return {"models": [name_a, name_b], "agreement": agg, "anchors": anch,
            "validation_ok_rate_incl_rejects": ok_rate,
            "equivalent": bool(verdict),
            "thresholds": {"jaccard": 0.55, "head": 0.75, "found": 0.90, "anchor": 0.85},
            "low_agreement_examples": sorted(rows, key=lambda r: r["jaccard"])[:12]}


def _ok_rate(recs: Dict[str, dict], exclude: set) -> float:
    ks = [k for k in recs if k not in exclude]
    if not ks:
        return 0.0
    return sum(recs[k].get("status") == "ok" for k in ks) / len(ks)


def main():
    import argparse
    from .extract import load_extractions
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True, help="extraction jsonl (model A)")
    p.add_argument("--b", required=True, help="extraction jsonl (model B)")
    p.add_argument("--name-a", default="glm-4.7")
    p.add_argument("--name-b", default="sonnet")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    rep = compare(load_extractions(args.a, only_ok=False),
                  load_extractions(args.b, only_ok=False), args.name_a, args.name_b)
    json.dump(rep, open(args.out, "w"), indent=1)
    a = rep["agreement"]
    print(f"n={a['n_shared_ok']} jaccard={a['mean_jaccard']:.3f} head={a['head_agreement']:.3f} "
          f"named={a['named_agreement']:.3f} found={a['found_agreement']:.3f}")
    for m, r in rep["anchors"].items():
        print(f"anchors[{m}]: pass {r['pass_rate']:.2f} missing {len(r['missing'])}")
    print("EQUIVALENT" if rep["equivalent"] else "NOT EQUIVALENT")


if __name__ == "__main__":
    main()
