"""Decode the L0/R1/R2/R3 taxonomy into key-level maps + build its audit payloads.

Verified chain (2026-07-06, humor general): complete-linkage clusters (members carry canon
keys; 96% singletons — the real merging is the LLM's) -> R1 = parented_trees (0..P-1) ++
merged_trees (P..P+M-1), meta_merge ordering -> R2 = merged_groups (all_leaves carry keys) +
grandparents (children r2_cluster_id -> R1 concat index) -> R3 input = R2 merged_groups in
enumeration order (r2_to_r3_input.py) -> R3 = merged_groups (all_leaves) + grandparents
(children r2_cluster_id -> R2 merged_groups index).

Level semantics for the audit: leaf partition = same-CRITERION (0/1/2 judge); R2/R3 nodes are
umbrella CONCEPTS, so their certificate is COHERENCE (does the leaf fall under the node's
name+description?) not sameness.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from .sources import ROOT

HIER = os.path.join(ROOT, "outputs", "hierarchy")


def _load(task: str, bucket: str, stem: str) -> Optional[dict]:
    p = os.path.join(HIER, f"{task}_{bucket}_{stem}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def _keys(leaves: List) -> List[str]:
    out = []
    for l in leaves or []:
        k = l.get("key") if isinstance(l, dict) else None
        if k:
            out.append(k)
    return out


def _r1_concat(r1: dict) -> List[dict]:
    """R1 nodes in meta_merge input order: parented then merged; each -> {label, keys}."""
    nodes = []
    for p in r1.get("parented_trees", []):
        ks = []
        for ch in p.get("children", []):
            ks += _keys(ch.get("rubrics"))
        nodes.append({"label": p.get("parent_name", ""), "keys": ks})
    for m in r1.get("merged_trees", []):
        nodes.append({"label": m.get("merged_name", ""), "keys": _keys(m.get("all_rubrics"))})
    return nodes


def level_maps(task: str, bucket: str) -> Optional[dict]:
    """{'r2': {key: label}, 'r3': {key: label}, 'nodes_r2': [...], 'nodes_r3': [...], stats}."""
    r1 = _load(task, bucket, "r1_refined")
    d2 = _load(task, bucket, "r2_expanded")
    if not d2:
        return None
    d3 = _load(task, bucket, "r3_expanded")
    r1_nodes = _r1_concat(r1) if r1 else []

    nodes_r2, r2map, coll2 = [], {}, 0
    for g in d2.get("merged_groups", []):
        lab = f"{bucket}::R2m::{g['merged_name']}"
        ks = _keys(g.get("all_leaves"))
        nodes_r2.append({"label": lab, "name": g["merged_name"],
                         "description": g.get("merged_description", ""), "keys": ks})
    for gp in d2.get("grandparents", []):
        ks = []
        for ch in gp.get("children", []):
            i = ch.get("r2_cluster_id")
            if isinstance(i, int) and 0 <= i < len(r1_nodes):
                ks += r1_nodes[i]["keys"]
        nodes_r2.append({"label": f"{bucket}::R2g::{gp['grandparent_name']}",
                         "name": gp["grandparent_name"],
                         "description": gp.get("grandparent_description", ""), "keys": ks})
    for n in nodes_r2:
        for k in n["keys"]:
            if k in r2map:
                coll2 += 1
            else:
                r2map[k] = n["label"]

    nodes_r3, r3map, coll3 = [], {}, 0
    if d3:
        r2_groups = d2.get("merged_groups", [])
        for g in d3.get("merged_groups", []):
            nodes_r3.append({"label": f"{bucket}::R3m::{g['merged_name']}",
                             "name": g["merged_name"],
                             "description": g.get("merged_description", ""),
                             "keys": _keys(g.get("all_leaves"))})
        for gp in d3.get("grandparents", []):
            ks = []
            for ch in gp.get("children", []):
                i = ch.get("r2_cluster_id")
                if isinstance(i, int) and 0 <= i < len(r2_groups):
                    ks += _keys(r2_groups[i].get("all_leaves"))
            nodes_r3.append({"label": f"{bucket}::R3g::{gp['grandparent_name']}",
                             "name": gp["grandparent_name"],
                             "description": gp.get("grandparent_description", ""), "keys": ks})
        for n in nodes_r3:
            for k in n["keys"]:
                if k in r3map:
                    coll3 += 1
                else:
                    r3map[k] = n["label"]

    return {"r2": r2map, "r3": r3map, "nodes_r2": nodes_r2, "nodes_r3": nodes_r3,
            "stats": {"task": task, "bucket": bucket,
                      "n_r2_nodes": len(nodes_r2), "n_r3_nodes": len(nodes_r3),
                      "keys_r2": len(r2map), "keys_r3": len(r3map),
                      "collisions_r2": coll2, "collisions_r3": coll3}}


def build_upmaps(task: str, buckets: Tuple[str, ...] = ("general", "specific",
                                                        "hyper_specific", "vague")) -> dict:
    """Merge buckets into key->label maps; save upmap (R3) + upmap_r2 + node inventories."""
    out2: Dict[str, str] = {}
    out3: Dict[str, str] = {}
    nodes = {"r2": [], "r3": []}
    stats = []
    for b in buckets:
        lm = level_maps(task, b)
        if not lm:
            continue
        stats.append(lm["stats"])
        for k, v in lm["r2"].items():
            out2.setdefault(k, v)
        for k, v in lm["r3"].items():
            out3.setdefault(k, v)
        nodes["r2"] += lm["nodes_r2"]
        nodes["r3"] += lm["nodes_r3"]
    od = os.path.join(ROOT, "outputs", "lexicon")
    json.dump(out3, open(os.path.join(od, f"upmap_{task}.json"), "w"))
    json.dump(out2, open(os.path.join(od, f"upmap_r2_{task}.json"), "w"))
    json.dump({lvl: [{"label": n["label"], "name": n["name"], "description": n["description"],
                      "n_keys": len(n["keys"])} for n in ns] for lvl, ns in nodes.items()},
              open(os.path.join(od, f"hier_nodes_{task}.json"), "w"), indent=1)
    return {"task": task, "buckets": stats, "keys_r2": len(out2), "keys_r3": len(out3),
            "nodes": nodes}


def _h(*parts: str) -> str:
    return hashlib.sha1("||".join(parts).encode()).hexdigest()[:16]


def containment_payload(task: str, nodes: List[dict], canon: Dict[str, str], level: str,
                        per_node: int = 4, n_anchors: int = 12) -> List[dict]:
    """Coherence audit rows: (leaf text, node name+description) -> belongs? Positive anchors =
    a node's own medoid-ish first leaf vs its node; negatives = leaf vs a far node (stable-hash
    pick from the other half of the node list)."""
    rows, seen = [], set()

    def add(key, node, kind):
        pid = _h(key, node["label"], kind)
        if pid in seen or key not in canon:
            return
        seen.add(pid)
        rows.append({"pair_id": pid, "task": task, "level": level, "kind": kind,
                     "key": key, "leaf_text": canon[key], "node_label": node["label"],
                     "node_name": node["name"], "node_description": node["description"]})

    for n in nodes:
        ks = sorted(n["keys"], key=lambda k: _h(k, n["label"]))
        for k in ks[:per_node]:
            add(k, n, "member")
    half = max(1, len(nodes) // 2)
    for i, n in enumerate(nodes[:n_anchors]):
        if n["keys"]:
            add(sorted(n["keys"], key=lambda k: _h(k, "anc"))[0], n, "anchor_pos")
            far = nodes[(i + half) % len(nodes)]
            add(sorted(n["keys"], key=lambda k: _h(k, "anc"))[0], far, "anchor_neg")
    return rows


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", required=True)
    a = p.parse_args()
    for task in a.tasks.split(","):
        task = task.strip()
        rep = build_upmaps(task)
        print(f"{task}: r3-mapped keys {rep['keys_r3']}, r2-mapped {rep['keys_r2']}, "
              f"r3 nodes {len(rep['nodes']['r3'])}, r2 nodes {len(rep['nodes']['r2'])}")
        for s in rep["buckets"]:
            print(f"   {s['bucket']}: R2 {s['n_r2_nodes']} nodes/{s['keys_r2']} keys "
                  f"(coll {s['collisions_r2']}), R3 {s['n_r3_nodes']}/{s['keys_r3']} "
                  f"(coll {s['collisions_r3']})")


if __name__ == "__main__":
    main()
