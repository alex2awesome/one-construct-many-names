"""Version-safe R2 materialization and blind LLM R3 preparation.

The historical canonical R2/R3 artifacts remain untouched.  This module converts a validated
versioned R2 LLM partition into flat partition/name artifacts, prepares a complete global R3 node
inventory, and validates an LLM-produced top-level category partition.  It performs no semantic
grouping itself.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict

from .build_level import OUT, _file_sha256
from .r2_recluster import ROOT as R2_ROOT, validate_variant


ROOT = Path(OUT) / "upper_variants"
R3_DEFINITION = """# R3-v1: same top-level evaluative category

R3 is the coarsest useful level in the domain hierarchy. Two focused R2 themes share an R3
category when they belong to the same broad evaluative subsystem or top-level review question and
one domain-meaningful category label would help a reviewer navigate both.

R3 is broader than an operational theme but narrower than the entire domain. A category should
usually contain several distinct R2 themes, yet remain informative enough to distinguish a major
kind of evaluation from other major kinds.

Use these tests:

1. Top-level navigation: would a domain reviewer expect both themes beneath the same first-level
   heading in a serious evaluation guide?
2. Shared subsystem: do they evaluate the same broad component, function, stakeholder concern, or
   stage of work?
3. Contrast: is there a clear reason this heading is distinct from the other top-level headings?
4. Non-genericity: is the heading more informative than quality, correctness, rigor, clarity,
   effectiveness, or the domain name itself?

Do not merge themes merely because they interact, share a generic virtue, or are both important.
Do not reproduce R2-level narrowness either: several focused operational families may belong under
one R3 category. Assign each R2 theme to one dominant top-level category for this disjoint hierarchy.
Do not force a predetermined number of categories and do not consult historical R3 partitions.
"""


def materialize_r2_variant(task: str, tag: str, llm_partition_path: str | os.PathLike[str]) -> dict:
    report = validate_variant(task, tag, llm_partition_path)
    payload = json.loads(Path(llm_partition_path).read_text())
    assignment = {str(k): str(v) for k, v in
                  (payload.get("assignment") or payload.get("partition") or {}).items()}
    themes = payload.get("themes") or {}
    used = set(assignment.values())
    names = {}
    for theme in used:
        row = themes[theme]
        if not isinstance(row, dict) or not str(row.get("name") or "").strip():
            raise ValueError(f"[{task}/{tag}] theme {theme} lacks an LLM name")
        names[theme] = {"name": str(row["name"])[:90],
                        "gloss": str(row.get("gloss") or "")}
    part_path = Path(OUT) / f"partition_{task}_R2_{tag}.json"
    names_path = Path(OUT) / f"node_names_{task}_R2_{tag}.json"
    part_path.write_text(json.dumps(assignment) + "\n")
    names_path.write_text(json.dumps(names) + "\n")
    return {**report, "llm_partition_path": str(llm_partition_path),
            "llm_partition_sha256": report["partition_sha256"],
            "partition_path": str(part_path),
            "partition_sha256": _file_sha256(str(part_path)),
            "names_path": str(names_path), "names_sha256": _file_sha256(str(names_path))}


def prepare_r3(task: str, tag: str, r2_partition_path: str | os.PathLike[str],
               r2_names_path: str | os.PathLike[str], r1_nodes_path: str | os.PathLike[str]) -> dict:
    """Write one complete R3 inventory whose nodes are the supplied R2 themes."""
    ROOT.mkdir(parents=True, exist_ok=True)
    protocol_path = ROOT / "R3_V1_PROTOCOL.md"
    protocol_path.write_text(R3_DEFINITION)
    r2_raw = json.loads(Path(r2_partition_path).read_text())
    r2 = {str(k): str(v) for k, v in r2_raw.get("partition", r2_raw).items()}
    names = json.loads(Path(r2_names_path).read_text())
    r1_nodes = {row["node_id"]: row for row in
                (json.loads(x) for x in Path(r1_nodes_path).read_text().splitlines()) if row.get("node_id")}
    if set(r2) != set(r1_nodes):
        raise ValueError(f"[{task}/{tag}] R2/R1 inventory mismatch: "
                         f"missing={len(set(r1_nodes)-set(r2))} extra={len(set(r2)-set(r1_nodes))}")
    members: Dict[str, list[str]] = {}
    for node, theme in r2.items():
        row = r1_nodes[node]
        rep = " | ".join(x for x in [str(row.get("name") or "").strip(),
                                      str(row.get("gloss") or "").strip()] if x)
        members.setdefault(theme, []).append(rep)
    node_path = ROOT / f"{task}_{tag}_R3_nodes.jsonl"
    with node_path.open("w") as out:
        for theme in sorted(set(r2.values())):
            row = names.get(theme) or {}
            out.write(json.dumps({"node_id": theme, "name": row.get("name") or theme,
                                  "gloss": row.get("gloss") or "",
                                  "member_examples": members.get(theme, [])[:5]},
                                 ensure_ascii=False) + "\n")
    manifest = {"task": task, "tag": tag, "version": "r3-top-category-v1",
                "protocol_path": str(protocol_path),
                "protocol_sha256": hashlib.sha256(R3_DEFINITION.encode()).hexdigest(),
                "input_path": str(node_path), "input_sha256": _file_sha256(str(node_path)),
                "n_nodes": len(set(r2.values())), "r2_partition_path": str(r2_partition_path),
                "r2_partition_sha256": _file_sha256(str(r2_partition_path)),
                "r2_names_path": str(r2_names_path),
                "r2_names_sha256": _file_sha256(str(r2_names_path)),
                "blind_instruction": "Do not read historical R3 partitions.",
                "semantic_decider": "LLM only; code performs representation and validation"}
    manifest_path = ROOT / f"{task}_{tag}_R3_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def validate_r3(task: str, tag: str, partition_path: str | os.PathLike[str]) -> dict:
    manifest = json.loads((ROOT / f"{task}_{tag}_R3_manifest.json").read_text())
    expected = {json.loads(x)["node_id"] for x in
                (ROOT / f"{task}_{tag}_R3_nodes.jsonl").read_text().splitlines() if x.strip()}
    payload = json.loads(Path(partition_path).read_text())
    assignment = payload.get("assignment") or payload.get("partition") or {}
    categories = payload.get("categories") or payload.get("themes") or {}
    missing, extra = expected - set(assignment), set(assignment) - expected
    if missing or extra:
        raise ValueError(f"[{task}/{tag}] R3 coverage failure: missing={len(missing)} extra={len(extra)}")
    if any(not isinstance(v, str) or not v.strip() for v in assignment.values()):
        raise ValueError("every R2 theme must have a non-empty category_id")
    used = set(assignment.values())
    if not isinstance(categories, dict) or not used.issubset(categories):
        raise ValueError(f"category metadata missing for {sorted(used-set(categories))[:5]}")
    counts: Dict[str, int] = {}
    for category in assignment.values():
        counts[category] = counts.get(category, 0) + 1
    return {"task": task, "tag": tag, "version": manifest["version"],
            "n_nodes": len(expected), "n_categories": len(used),
            "n_singleton_categories": sum(n == 1 for n in counts.values()),
            "partition_sha256": _file_sha256(str(partition_path)),
            "semantic_quality": "not measured here; requires independent LLM judgment"}


def emit_composed_r3_comparison(task: str, tag: str,
                                new_r2_path: str | os.PathLike[str],
                                new_r3_path: str | os.PathLike[str],
                                r1_nodes_path: str | os.PathLike[str],
                                per_disagreement: int = 80,
                                per_agreement: int = 40) -> dict:
    """Blindly compare historical vs variant R2->R3 assignments on shared R1 constructs.

    R2 theme IDs differ between systems, so a theme-pair comparison is undefined. Composing each
    hierarchy down to the exact shared R1 inventory yields a directly comparable co-category
    decision without code making any semantic judgment.
    """
    hist_r2_raw = json.loads((Path(OUT) / f"partition_{task}_R2.json").read_text())
    hist_r3_raw = json.loads((Path(OUT) / f"partition_{task}_R3.json").read_text())
    hist_r2 = {str(k): str(v) for k, v in hist_r2_raw.get("partition", hist_r2_raw).items()}
    hist_r3 = {str(k): str(v) for k, v in hist_r3_raw.get("partition", hist_r3_raw).items()}
    new_r2_raw = json.loads(Path(new_r2_path).read_text())
    new_r3_raw = json.loads(Path(new_r3_path).read_text())
    new_r2 = {str(k): str(v) for k, v in new_r2_raw.get("partition", new_r2_raw).items()}
    new_r3 = {str(k): str(v) for k, v in
              (new_r3_raw.get("assignment") or new_r3_raw.get("partition") or {}).items()}
    nodes = {row["node_id"]: row for row in
             (json.loads(x) for x in Path(r1_nodes_path).read_text().splitlines()) if row.get("node_id")}
    ids = set(nodes)
    if ids - set(hist_r2) or ids != set(new_r2):
        raise ValueError(f"[{task}/{tag}] R1 coverage mismatch in composed R3 comparison")
    if set(hist_r2.values()) - set(hist_r3) or set(new_r2.values()) - set(new_r3):
        raise ValueError(f"[{task}/{tag}] R3 does not cover all R2 themes")
    historical = {node: hist_r3[hist_r2[node]] for node in ids}
    variant = {node: new_r3[new_r2[node]] for node in ids}
    pools = {"both_same": [], "sonnet_only": [], "codex_only": [], "both_different": []}
    ordered_ids = sorted(ids)
    for i, a in enumerate(ordered_ids):
        for b in ordered_ids[i + 1:]:
            hs, vs = historical[a] == historical[b], variant[a] == variant[b]
            stratum = ("both_same" if hs and vs else "sonnet_only" if hs else
                       "codex_only" if vs else "both_different")
            pools[stratum].append((a, b))
    selected = []
    protocol_sha = hashlib.sha256(R3_DEFINITION.encode()).hexdigest()
    for stratum, pairs in pools.items():
        limit = per_agreement if stratum in ("both_same", "both_different") else per_disagreement
        pairs.sort(key=lambda ab: hashlib.sha256(
            f"{protocol_sha}||{task}||{tag}||{stratum}||{ab[0]}||{ab[1]}".encode()).hexdigest())
        selected.extend((stratum, a, b) for a, b in pairs[:limit])
    selected.sort(key=lambda x: hashlib.sha1(
        f"{task}||{tag}||R3||{x[1]}||{x[2]}".encode()).hexdigest())
    stem = f"{task}_{tag}_R3_composed"
    blind_path, key_path = ROOT / f"{stem}_blind.jsonl", ROOT / f"{stem}_key.json"
    key = {}
    with blind_path.open("w") as out:
        for stratum, a, b in selected:
            pid = hashlib.sha1(f"{stem}||{'||'.join(sorted((a,b)))}".encode()).hexdigest()[:16]
            def rep(node_id: str) -> str:
                row = nodes[node_id]
                bits = [row.get("name") or "", row.get("gloss") or ""]
                bits.extend((row.get("member_examples") or [])[:3])
                return " | ".join(str(x).strip() for x in bits if str(x).strip())
            out.write(json.dumps({"pair_id": pid, "task": task,
                                  "construct_a": rep(a), "construct_b": rep(b)},
                                 ensure_ascii=False) + "\n")
            key[pid] = {"stratum": stratum, "node_a": a, "node_b": b}
    key_path.write_text(json.dumps(key, indent=2) + "\n")
    manifest = {"task": task, "tag": tag, "version": "r3-composed-comparison-v1",
                "n_pairs": len(selected),
                "counts": {s: sum(x[0] == s for x in selected) for s in pools},
                "protocol_path": str(ROOT / "R3_V1_PROTOCOL.md"),
                "protocol_sha256": protocol_sha,
                "historical_r2_sha256": _file_sha256(str(Path(OUT) / f"partition_{task}_R2.json")),
                "historical_r3_sha256": _file_sha256(str(Path(OUT) / f"partition_{task}_R3.json")),
                "new_r2_sha256": _file_sha256(str(new_r2_path)),
                "new_r3_sha256": _file_sha256(str(new_r3_path)),
                "blind_path": str(blind_path), "blind_sha256": _file_sha256(str(blind_path)),
                "key_path": str(key_path), "key_sha256": _file_sha256(str(key_path)),
                "instruction": "Judge blind rows under R3_V1_PROTOCOL; do not read the key.",
                "status": "not semantic until an independent blind LLM supplies complete votes"}
    (ROOT / f"{stem}_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def summarize_composed_r3(task: str, tag: str, votes_path: str | os.PathLike[str]) -> dict:
    stem = f"{task}_{tag}_R3_composed"
    key = json.loads((ROOT / f"{stem}_key.json").read_text())
    votes = {}
    malformed = 0
    for line in Path(votes_path).read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            continue
        pid, score = row.get("pair_id"), row.get("score")
        if (set(row) != {"pair_id", "score"} or pid not in key or pid in votes
                or type(score) is not int or score not in (0, 1, 2)):
            malformed += 1
            continue
        votes[pid] = score
    missing = set(key) - set(votes)
    if missing or malformed:
        raise ValueError(f"[{task}/{tag}] invalid R3 votes: missing={len(missing)} malformed={malformed}")
    by = {s: {"n": 0, "judge_same": 0} for s in
          ("both_same", "sonnet_only", "codex_only", "both_different")}
    sonnet = codex = 0
    for pid, score in votes.items():
        stratum, same = key[pid]["stratum"], score == 2
        by[stratum]["n"] += 1; by[stratum]["judge_same"] += int(same)
        if stratum == "sonnet_only":
            sonnet += int(same); codex += int(not same)
        elif stratum == "codex_only":
            sonnet += int(not same); codex += int(same)
    n = by["sonnet_only"]["n"] + by["codex_only"]["n"]
    result = {"task": task, "tag": tag, "judge": "independent blind LLM",
              "n_judged": len(votes), "strata": by,
              "disagreement_support": {"sonnet": sonnet, "codex": codex, "n": n,
                                        "sonnet_rate": round(sonnet/n, 3) if n else None,
                                        "codex_rate": round(codex/n, 3) if n else None},
              "unit": "R1 construct pairs after composing each R2->R3 hierarchy"}
    (ROOT / f"{stem}_report.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
