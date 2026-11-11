"""Versioned R2-v2 reclustering inputs and validation.

This module does bookkeeping only.  It never decides semantic similarity, assigns a node to a
theme, or treats a string/embedding comparison as an R2 measurement.  Those decisions are made by
LLM judges using the frozen protocol written by :func:`prepare`.

The historical ``partition_<task>_R2.json`` files remain immutable.  New artifacts live below
``outputs/lexicon/r2_recluster_v2`` and carry hashes of their R1 input and historical Sonnet R2
partition so blind Codex clustering can later be compared without confusing artifact vintages.
"""
from __future__ import annotations

import hashlib
import heapq
import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List

from .build_level import OUT, nodes_from_level
from .judge import canon_map


VERSION = "r2-operational-theme-v2"
ROOT = Path(OUT) / "r2_recluster_v2"

R2_V2_DEFINITION = """# R2-v2: same operational evaluative theme

R2 sits strictly between R1 (one construct) and R3 (a top-level category).  Group two R1
constructs at R2 only when they share one operational evaluative theme: the same primary object of
evaluation and the same practical evaluative purpose or failure mode, such that one concise,
informative heading would help a reviewer find and act on both.

Use all of these tests:

1. Primary object: What aspect of the work is inspected?
2. Evaluative purpose: What decision, intervention, or failure does the criterion inform?
3. Heading test: Can one concise theme label cover both without becoming a generic value word or a
   top-level category?
4. Sibling-prediction test: Knowing that label, could a reviewer reasonably predict that both
   constructs would appear beneath it?

Score/group as SAME R2 when the constructs may remain distinct at R1 but pass all four tests.

Do NOT group constructs merely because they share a broad value or topic such as quality, rigor,
clarity, fairness, relevance, engagement, safety, or impact.  Do NOT group them merely because they
are causally related, occur next to each other in a workflow, use similar words, or belong to the
same R3 category.  The same subject matter with different evaluative functions is not enough, and
the same abstract virtue applied to different objects is not enough.

Examples of the boundary:

- Choosing appropriate statistical tests and reporting uncertainty can share a focused
  "statistical inference and uncertainty" theme.
- Statistical inference and data/code availability do not become one R2 theme merely because both
  support research rigor; "research rigor" is too broad and belongs at R3.
- Headline accuracy and respectful identity terminology do not become one R2 theme merely because
  both support responsible journalism; they inspect different objects and failures.
- Caption clarity and article organization do not become one R2 theme merely because both support
  clarity; one concerns local visual explanation and the other global structure.

R2 is represented as a disjoint primary-theme partition for this experiment.  If a construct is
genuinely cross-cutting, assign its dominant evaluative function; if no non-generic dominant theme
is defensible, leave it singleton.  Never force a target number of themes, and never use the
historical Sonnet R2 partition while clustering.
"""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _historical_r2(task: str) -> tuple[str | None, str | None]:
    """Return an optional historical Sonnet comparator without blocking a first R2 build."""
    path = Path(OUT) / f"partition_{task}_R2.json"
    return (str(path), _sha256(path)) if path.exists() else (None, None)


def _opaque(node: dict) -> bool:
    if str(node.get("gloss") or "").strip():
        return False
    name, node_id = str(node.get("name") or "").strip(), str(node["node_id"])
    return (not name or name == node_id or name.isdigit()
            or bool(re.fullmatch(r"[\w-]*_R[0-9]+_(?:g|solo_)?[\w-]+", name)))


def _examples(cmap: Dict[str, str], keys: Iterable[str], limit: int = 4) -> List[str]:
    rows = [cmap[k].strip() for k in sorted(keys) if k in cmap and cmap[k].strip()]
    # Stable descendant examples are representation, not a semantic similarity decision.
    return rows[:limit]


def prepare(task: str) -> dict:
    """Write the blind node inventory and provenance manifest for one R2-v2 LLM run."""
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "R2_V2_PROTOCOL.md").write_text(R2_V2_DEFINITION)

    nodes, keys_of = nodes_from_level(task, "R2")
    cmap = canon_map(task)
    input_path = ROOT / f"{task}_nodes.jsonl"
    with input_path.open("w") as out:
        for node in nodes:
            examples = _examples(cmap, keys_of.get(node["node_id"], ()))
            name = str(node.get("name") or "").strip()
            gloss = str(node.get("gloss") or "").strip()
            # Opaque historical names are not allowed to become semantic evidence.  Descendant
            # examples preserve the node's actual content without inventing an automatic label.
            row = {
                "node_id": node["node_id"],
                "name": "" if _opaque(node) else name,
                "gloss": gloss,
                "member_examples": examples,
                "representation_warning": "historical name was opaque; judge member examples"
                if _opaque(node) else None,
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    r1_path = Path(OUT) / f"partition_{task}_R1.json"
    sonnet_path, sonnet_sha = _historical_r2(task)
    manifest = {
        "version": VERSION,
        "task": task,
        "protocol_path": str((ROOT / "R2_V2_PROTOCOL.md").relative_to(Path(OUT).parents[1])),
        "protocol_sha256": hashlib.sha256(R2_V2_DEFINITION.encode()).hexdigest(),
        "input_path": str(input_path),
        "input_sha256": _sha256(input_path),
        "n_nodes": len(nodes),
        "r1_partition_path": str(r1_path),
        "r1_partition_sha256": _sha256(r1_path),
        "historical_sonnet_r2_path": sonnet_path,
        "historical_sonnet_r2_sha256": sonnet_sha,
        "blind_instruction": "Do not read the historical Sonnet R2 partition while clustering.",
        "semantic_decider": "LLM only; code performs representation and validation bookkeeping",
    }
    manifest_path = ROOT / f"{task}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def prepare_from_r1(task: str, r1_partition_path: str | os.PathLike[str],
                    r1_names_path: str | os.PathLike[str], tag: str,
                    l0_partition_path: str | os.PathLike[str] | None = None,
                    protocol_path: str | os.PathLike[str] | None = None) -> dict:
    """Prepare a blind R2 inventory from a versioned/rebased R1 artifact.

    This is the variant-safe counterpart to :func:`prepare`: it never mutates canonical R1/R2 files
    or silently reads their node inventory.  Original rubric descendants are composed through the
    supplied L0 and R1 partitions only to give the LLM enough semantic evidence.
    """
    ROOT.mkdir(parents=True, exist_ok=True)
    r1_path, names_path = Path(r1_partition_path), Path(r1_names_path)
    l0_path = (Path(l0_partition_path) if l0_partition_path else
               Path(OUT) / f"partition_{task}_L0v3.json")
    proto_path = (Path(protocol_path) if protocol_path else ROOT / "R2_V2_1_PROTOCOL.md")
    protocol_text = proto_path.read_text()
    r1_raw, l0_raw = json.loads(r1_path.read_text()), json.loads(l0_path.read_text())
    r1 = {str(k): str(v) for k, v in r1_raw.get("partition", r1_raw).items()}
    l0 = {str(k): str(v) for k, v in l0_raw.get("partition", l0_raw).items()}
    names = json.loads(names_path.read_text())
    keys_of: Dict[str, set[str]] = {}
    for key, l0_id in l0.items():
        if l0_id not in r1:
            raise ValueError(f"[{task}/{tag}] R1 misses active L0 node {l0_id}")
        keys_of.setdefault(r1[l0_id], set()).add(key)
    cmap = canon_map(task)
    stem = f"{task}_{tag}"
    input_path = ROOT / f"{stem}_nodes.jsonl"
    with input_path.open("w") as out:
        for node_id in sorted(keys_of):
            row = names.get(node_id) or {}
            name, gloss = str(row.get("name") or "").strip(), str(row.get("gloss") or "").strip()
            node = {"node_id": node_id, "name": name, "gloss": gloss,
                    "member_examples": _examples(cmap, keys_of[node_id]),
                    "representation_warning": None}
            if _opaque(node):
                node["name"] = ""
                node["representation_warning"] = "name was opaque; judge member examples"
            out.write(json.dumps(node, ensure_ascii=False) + "\n")
    sonnet_path, sonnet_sha = _historical_r2(task)
    manifest = {"version": "r2-focused-operational-theme-v2.1", "task": task, "tag": tag,
                "protocol_path": str(proto_path),
                "protocol_sha256": hashlib.sha256(protocol_text.encode()).hexdigest(),
                "input_path": str(input_path), "input_sha256": _sha256(input_path),
                "n_nodes": len(keys_of), "r1_partition_path": str(r1_path),
                "r1_partition_sha256": _sha256(r1_path), "r1_names_path": str(names_path),
                "r1_names_sha256": _sha256(names_path), "l0_partition_path": str(l0_path),
                "l0_partition_sha256": _sha256(l0_path),
                "historical_sonnet_r2_path": sonnet_path,
                "historical_sonnet_r2_sha256": sonnet_sha,
                "blind_instruction": "Do not read any historical R2 partition while clustering.",
                "semantic_decider": "LLM only; code performs representation and validation bookkeeping"}
    manifest_path = ROOT / f"{stem}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def validate_variant(task: str, tag: str, partition_path: str | os.PathLike[str]) -> dict:
    """Coverage/schema validation for an LLM partition produced from :func:`prepare_from_r1`."""
    manifest = json.loads((ROOT / f"{task}_{tag}_manifest.json").read_text())
    expected = {json.loads(line)["node_id"] for line in
                (ROOT / f"{task}_{tag}_nodes.jsonl").read_text().splitlines() if line.strip()}
    payload = json.loads(Path(partition_path).read_text())
    assignment: Dict[str, str] = payload.get("assignment") or payload.get("partition") or {}
    themes = payload.get("themes") or {}
    missing, extra = expected - set(assignment), set(assignment) - expected
    if missing or extra:
        raise ValueError(f"coverage failure: missing={len(missing)} extra={len(extra)}")
    if any(not isinstance(v, str) or not v.strip() for v in assignment.values()):
        raise ValueError("every node must have a non-empty string theme_id")
    used = set(assignment.values())
    if not isinstance(themes, dict) or not used.issubset(themes):
        raise ValueError(f"themes metadata missing for {sorted(used-set(themes))[:5]}")
    counts: Dict[str, int] = {}
    for theme in assignment.values():
        counts[theme] = counts.get(theme, 0) + 1
    return {"task": task, "tag": tag, "version": manifest["version"],
            "n_nodes": len(expected), "n_themes": len(used),
            "n_singletons": sum(n == 1 for n in counts.values()),
            "input_sha256": manifest["input_sha256"],
            "partition_sha256": _sha256(Path(partition_path)),
            "semantic_quality": "not measured here; requires independent LLM judgment"}


def emit_variant_comparison(task: str, tag: str,
                            codex_partition_path: str | os.PathLike[str],
                            per_disagreement: int = 80, per_agreement: int = 40) -> dict:
    """Blind LLM comparison on a variant node inventory against historical Sonnet where defined.

    Historical partitions may contain retired R1 nodes.  They are restricted to the exact variant
    inventory; missing historical assignments fail closed.  This restriction is structural and does
    not decide whether any pair is semantically SAME.
    """
    manifest = json.loads((ROOT / f"{task}_{tag}_manifest.json").read_text())
    nodes = {row["node_id"]: row for row in
             (json.loads(x) for x in (ROOT / f"{task}_{tag}_nodes.jsonl").read_text().splitlines())
             if row.get("node_id")}
    historical_path = manifest.get("historical_sonnet_r2_path")
    if not historical_path:
        raise FileNotFoundError(
            f"[{task}/{tag}] no historical Sonnet R2 exists; semantic comparison is unavailable")
    sonnet_raw = json.loads(Path(historical_path).read_text())
    sonnet_all = sonnet_raw.get("partition", sonnet_raw)
    if set(nodes) - set(sonnet_all):
        raise ValueError(f"[{task}/{tag}] historical Sonnet misses variant nodes")
    sonnet = {node: str(sonnet_all[node]) for node in nodes}
    codex_payload = json.loads(Path(codex_partition_path).read_text())
    codex = codex_payload.get("assignment") or codex_payload.get("partition") or {}
    if set(codex) != set(nodes):
        raise ValueError(f"[{task}/{tag}] Codex coverage differs from variant nodes")
    protocol_sha = manifest["protocol_sha256"]
    pools = {"both_same": [], "sonnet_only": [], "codex_only": [], "both_different": []}
    ids = sorted(nodes)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            ss, cs = sonnet[a] == sonnet[b], codex[a] == codex[b]
            stratum = ("both_same" if ss and cs else "sonnet_only" if ss else
                       "codex_only" if cs else "both_different")
            pools[stratum].append((a, b))
    selected = []
    for stratum, pairs in pools.items():
        limit = per_agreement if stratum in ("both_same", "both_different") else per_disagreement
        pairs.sort(key=lambda ab: hashlib.sha256(
            f"{protocol_sha}||{tag}||{task}||{stratum}||{ab[0]}||{ab[1]}".encode()).hexdigest())
        selected.extend((stratum, a, b) for a, b in pairs[:limit])
    selected.sort(key=lambda x: hashlib.sha1(
        f"{tag}||{task}||{x[1]}||{x[2]}".encode()).hexdigest())
    stem = f"{task}_{tag}"
    blind_path, key_path = ROOT / f"{stem}_comparison_blind.jsonl", ROOT / f"{stem}_comparison_key.json"
    key = {}
    with blind_path.open("w") as out:
        for stratum, a, b in selected:
            pid = hashlib.sha1(f"{tag}||{task}||{'||'.join(sorted((a,b)))}".encode()).hexdigest()[:16]
            out.write(json.dumps({"pair_id": pid, "task": task, "node_a": a, "node_b": b,
                                  "concept_a": _rep(nodes[a]), "concept_b": _rep(nodes[b])},
                                 ensure_ascii=False) + "\n")
            key[pid] = {"stratum": stratum, "node_a": a, "node_b": b}
    key_path.write_text(json.dumps(key, indent=2) + "\n")
    result = {"task": task, "tag": tag, "n_pairs": len(selected),
              "counts": {s: sum(x[0] == s for x in selected) for s in pools},
              "protocol_path": manifest["protocol_path"], "protocol_sha256": protocol_sha,
              "historical_sonnet_partition_sha256": _sha256(Path(historical_path)),
              "codex_partition_path": str(codex_partition_path),
              "codex_partition_sha256": _sha256(Path(codex_partition_path)),
              "blind_path": str(blind_path), "blind_sha256": _sha256(blind_path),
              "key_path": str(key_path), "key_sha256": _sha256(key_path),
              "instruction": "Judge blind_path under the frozen v2.1 protocol; do not read key_path.",
              "status": "not a semantic comparison until an independent LLM supplies complete votes"}
    (ROOT / f"{stem}_comparison_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def _partition_payload(path: str | os.PathLike[str]) -> Dict[str, str]:
    payload = json.loads(Path(path).read_text())
    raw = payload.get("partition") if isinstance(payload, dict) else None
    if raw is None and isinstance(payload, dict):
        raw = payload.get("assignment")
    if raw is None:
        raw = payload
    if not isinstance(raw, dict):
        raise ValueError(f"partition artifact must contain an object: {path}")
    return {str(k): str(v) for k, v in raw.items()}


def emit_rebased_key_comparison(
        task: str, tag: str, codex_partition_path: str | os.PathLike[str],
        current_l0_path: str | os.PathLike[str], current_r1_path: str | os.PathLike[str],
        historical_l0_path: str | os.PathLike[str],
        historical_r1_path: str | os.PathLike[str],
        historical_r2_path: str | os.PathLike[str],
        protocol_path: str | os.PathLike[str], per_disagreement: int = 100,
        per_agreement: int = 50) -> dict:
    """Emit a blind R2 comparison when corrected R1 node identities no longer align.

    Both hierarchies are composed down to their shared original rubric keys.  Pairs that either
    hierarchy already co-labels at R1 are excluded, so the audit measures R2 decisions rather than
    silently rewarding or penalizing the upstream R1 rebuild.  Code only performs composition,
    stratified deterministic sampling, and provenance bookkeeping; an independent LLM supplies all
    semantic judgments under ``protocol_path``.
    """
    ROOT.mkdir(parents=True, exist_ok=True)
    paths = {"codex_r2": Path(codex_partition_path), "current_l0": Path(current_l0_path),
             "current_r1": Path(current_r1_path), "historical_l0": Path(historical_l0_path),
             "historical_r1": Path(historical_r1_path),
             "historical_r2": Path(historical_r2_path), "protocol": Path(protocol_path)}
    maps = {name: _partition_payload(path) for name, path in paths.items() if name != "protocol"}
    current_l0, current_r1, codex_r2 = (maps["current_l0"], maps["current_r1"],
                                        maps["codex_r2"])
    old_l0, old_r1, old_r2 = (maps["historical_l0"], maps["historical_r1"],
                              maps["historical_r2"])
    keys = sorted(set(current_l0) & set(old_l0))
    if set(current_l0) != set(old_l0):
        raise ValueError("current and historical L0 partitions must cover the same rubric keys")

    def compose(l0: Dict[str, str], r1: Dict[str, str], r2: Dict[str, str], key: str):
        l0_id = l0[key]
        if l0_id not in r1:
            raise ValueError(f"R1 misses L0 node {l0_id}")
        r1_id = r1[l0_id]
        if r1_id not in r2:
            raise ValueError(f"R2 misses R1 node {r1_id}")
        return r1_id, r2[r1_id]

    current = {key: compose(current_l0, current_r1, codex_r2, key) for key in keys}
    historical = {key: compose(old_l0, old_r1, old_r2, key) for key in keys}
    limits = {"both_same": per_agreement, "sonnet_only": per_disagreement,
              "codex_only": per_disagreement, "both_different": per_agreement}
    heaps: dict[str, list[tuple[int, str, str]]] = {s: [] for s in limits}
    populations = {s: 0 for s in limits}
    protocol_sha = _sha256(paths["protocol"])
    for i, a in enumerate(keys):
        current_r1_a, current_r2_a = current[a]
        old_r1_a, old_r2_a = historical[a]
        for b in keys[i + 1:]:
            current_r1_b, current_r2_b = current[b]
            old_r1_b, old_r2_b = historical[b]
            if current_r1_a == current_r1_b or old_r1_a == old_r1_b:
                continue
            ss, cs = old_r2_a == old_r2_b, current_r2_a == current_r2_b
            stratum = ("both_same" if ss and cs else "sonnet_only" if ss else
                       "codex_only" if cs else "both_different")
            populations[stratum] += 1
            limit = limits[stratum]
            if not limit:
                continue
            rank = int(hashlib.sha256(
                f"{protocol_sha}||{tag}||{task}||{stratum}||{a}||{b}".encode()
            ).hexdigest(), 16)
            item = (-rank, a, b)
            heap = heaps[stratum]
            if len(heap) < limit:
                heapq.heappush(heap, item)
            elif rank < -heap[0][0]:
                heapq.heapreplace(heap, item)

    selected = []
    for stratum, heap in heaps.items():
        selected.extend((stratum, -neg_rank, a, b) for neg_rank, a, b in heap)
    selected.sort(key=lambda x: (x[1], x[2], x[3]))
    cmap = canon_map(task)
    missing_text = [key for key in keys if not str(cmap.get(key) or "").strip()]
    if missing_text:
        raise ValueError(f"canonical text missing for {len(missing_text)} keys")
    stem = f"{task}_{tag}"
    blind_path = ROOT / f"{stem}_comparison_blind.jsonl"
    key_path = ROOT / f"{stem}_comparison_key.json"
    key_payload = {}
    with blind_path.open("w") as out:
        for stratum, _, a, b in selected:
            pid = hashlib.sha1(
                f"rebased-key||{tag}||{task}||{a}||{b}".encode()).hexdigest()[:16]
            out.write(json.dumps({"pair_id": pid, "task": task,
                                  "concept_a": cmap[a].strip(),
                                  "concept_b": cmap[b].strip()}, ensure_ascii=False) + "\n")
            key_payload[pid] = {"stratum": stratum, "node_a": a, "node_b": b,
                                "historical_r1_a": historical[a][0],
                                "historical_r1_b": historical[b][0],
                                "current_r1_a": current[a][0], "current_r1_b": current[b][0]}
    key_path.write_text(json.dumps(key_payload, indent=2) + "\n")
    result = {
        "task": task, "tag": tag,
        "version": "r2-focused-operational-theme-v2.1-rebased-key-comparison-v1",
        "n_shared_rubric_keys": len(keys), "n_pairs": len(selected),
        "counts": {s: sum(row[0] == s for row in selected) for s in limits},
        "population_counts_excluding_pairs_colabeled_at_r1_by_either_hierarchy": populations,
        "conditioning": "both hierarchies place the two rubric keys in distinct R1 constructs",
        "protocol_path": str(paths["protocol"]), "protocol_sha256": protocol_sha,
        **{f"{name}_path": str(path) for name, path in paths.items()},
        **{f"{name}_sha256": _sha256(path) for name, path in paths.items()},
        "blind_path": str(blind_path), "blind_sha256": _sha256(blind_path),
        "key_path": str(key_path), "key_sha256": _sha256(key_path),
        "instruction": "Judge blind_path under the frozen v2.1 protocol; do not read key_path.",
        "semantic_truth": "independent LLM only; code performs composition, sampling, and arithmetic",
        "status": "not a semantic comparison until an independent LLM supplies complete votes",
    }
    (ROOT / f"{stem}_comparison_manifest.json").write_text(
        json.dumps(result, indent=2) + "\n")
    return result


def validate(task: str, partition_path: str | os.PathLike[str]) -> dict:
    """Fail closed on malformed LLM output; do not assess semantic quality in code."""
    manifest = json.loads((ROOT / f"{task}_manifest.json").read_text())
    expected = {json.loads(line)["node_id"] for line in
                (ROOT / f"{task}_nodes.jsonl").read_text().splitlines() if line.strip()}
    payload = json.loads(Path(partition_path).read_text())
    assignment: Dict[str, str] = payload.get("assignment") or payload.get("partition") or {}
    if not isinstance(assignment, dict):
        raise ValueError("partition must contain an assignment object")
    got = set(assignment)
    missing, extra = expected - got, got - expected
    if missing or extra:
        raise ValueError(f"coverage failure: missing={len(missing)} extra={len(extra)} "
                         f"sample_missing={sorted(missing)[:5]} sample_extra={sorted(extra)[:5]}")
    if any(not isinstance(v, str) or not v.strip() for v in assignment.values()):
        raise ValueError("every node must have a non-empty string theme_id")
    themes = payload.get("themes") or {}
    used = set(assignment.values())
    if not isinstance(themes, dict) or not used.issubset(themes):
        raise ValueError(f"themes metadata missing for {sorted(used - set(themes))[:5]}")
    report = {
        "task": task,
        "version": manifest["version"],
        "n_nodes": len(expected),
        "n_themes": len(used),
        "n_singletons": sum(1 for t in used if list(assignment.values()).count(t) == 1),
        "input_sha256": manifest["input_sha256"],
        "partition_sha256": _sha256(Path(partition_path)),
        "semantic_quality": "not measured here; requires LLM judgment",
    }
    return report


def _pair_id(task: str, a: str, b: str) -> str:
    return hashlib.sha1(f"{VERSION}||{task}||{'||'.join(sorted((a, b)))}".encode()).hexdigest()[:16]


def _rep(row: dict) -> str:
    bits = [row.get("name") or "", row.get("gloss") or ""]
    bits.extend(row.get("member_examples") or [])
    return " | ".join(x.strip() for x in bits if isinstance(x, str) and x.strip())


def emit_comparison(task: str, per_disagreement: int = 80, per_agreement: int = 40,
                    codex_partition_path: str | os.PathLike[str] | None = None,
                    tag: str = "", protocol_path: str | os.PathLike[str] | None = None) -> dict:
    """Emit an identity-blind, stratified pair audit for an independent LLM judge.

    Strata are selected mechanically from partition co-membership.  They are hidden in a separate
    key file so the judge sees neither partition identity nor which system grouped the pair.  The
    comparison is not complete until the blind rows receive LLM R2-v2 judgments.
    """
    manifest = json.loads((ROOT / f"{task}_manifest.json").read_text())
    nodes = {r["node_id"]: r for r in
             (json.loads(line) for line in (ROOT / f"{task}_nodes.jsonl").read_text().splitlines())
             if r.get("node_id")}
    historical_path = manifest.get("historical_sonnet_r2_path")
    if not historical_path:
        raise FileNotFoundError(
            f"[{task}] no historical Sonnet R2 exists; semantic comparison is unavailable")
    sonnet_raw = json.loads(Path(historical_path).read_text())
    sonnet = sonnet_raw.get("partition", sonnet_raw)
    codex_path = (Path(codex_partition_path) if codex_partition_path else
                  ROOT / f"{task}_codex_partition.json")
    codex_payload = json.loads(codex_path.read_text())
    codex = codex_payload.get("assignment") or codex_payload.get("partition") or {}
    if set(nodes) != set(sonnet) or set(nodes) != set(codex):
        raise ValueError(f"[{task}] comparison coverage differs: nodes={len(nodes)} "
                         f"sonnet={len(sonnet)} codex={len(codex)}")

    pools = {"both_same": [], "sonnet_only": [], "codex_only": [], "both_different": []}
    ids = sorted(nodes)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            ss, cs = sonnet[a] == sonnet[b], codex[a] == codex[b]
            stratum = ("both_same" if ss and cs else "sonnet_only" if ss else
                       "codex_only" if cs else "both_different")
            pools[stratum].append((a, b))

    protocol_text = Path(protocol_path).read_text() if protocol_path else R2_V2_DEFINITION
    protocol_sha = hashlib.sha256(protocol_text.encode()).hexdigest()
    selected = []
    for stratum, pairs in pools.items():
        limit = per_agreement if stratum in ("both_same", "both_different") else per_disagreement
        pairs.sort(key=lambda ab: hashlib.sha256(
            f"{protocol_sha}||{tag}||{task}||{stratum}||{ab[0]}||{ab[1]}".encode()).hexdigest())
        selected.extend((stratum, a, b) for a, b in pairs[:limit])
    selected.sort(key=lambda x: _pair_id(task, x[1], x[2]))

    stem = f"{task}_{tag}" if tag else task
    blind_path = ROOT / f"{stem}_comparison_blind.jsonl"
    key = {}
    with blind_path.open("w") as out:
        for stratum, a, b in selected:
            pid = _pair_id(task, a, b)
            out.write(json.dumps({"pair_id": pid, "task": task, "node_a": a, "node_b": b,
                                  "concept_a": _rep(nodes[a]), "concept_b": _rep(nodes[b])},
                                 ensure_ascii=False) + "\n")
            key[pid] = {"stratum": stratum, "node_a": a, "node_b": b}
    key_path = ROOT / f"{stem}_comparison_key.json"
    key_path.write_text(json.dumps(key, indent=2) + "\n")
    manifest = {
        "task": task, "version": VERSION, "n_pairs": len(selected),
        "counts": {s: sum(x[0] == s for x in selected) for s in pools},
        "protocol_path": str(protocol_path) if protocol_path else str(ROOT / "R2_V2_PROTOCOL.md"),
        "protocol_sha256": protocol_sha,
        "historical_sonnet_partition_sha256": _sha256(Path(OUT) / f"partition_{task}_R2.json"),
        "codex_partition_path": str(codex_path),
        "codex_partition_sha256": _sha256(codex_path),
        "blind_path": str(blind_path), "blind_sha256": _sha256(blind_path),
        "key_path": str(key_path), "key_sha256": _sha256(key_path),
        "instruction": (f"Judge blind_path under {protocol_path}; do not read key_path first."
                        if protocol_path else
                        "Judge blind_path under R2_V2_PROTOCOL.md; do not read key_path first."),
        "status": "not a semantic comparison until an independent LLM supplies complete votes",
    }
    (ROOT / f"{stem}_comparison_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def summarize_comparison(task: str, votes_path: str | os.PathLike[str], tag: str = "") -> dict:
    """Aggregate complete blind LLM judgments into support for Sonnet/Codex decisions."""
    stem = f"{task}_{tag}" if tag else task
    key = json.loads((ROOT / f"{stem}_comparison_key.json").read_text())
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
        if pid in votes or pid not in key or type(score) is not int or score not in (0, 1, 2):
            malformed += 1
            continue
        votes[pid] = score
    missing = set(key) - set(votes)
    if missing or malformed:
        raise ValueError(f"[{task}] invalid comparison votes: missing={len(missing)} "
                         f"malformed_or_duplicate={malformed}")

    by = {s: {"n": 0, "judge_same": 0} for s in
          ("both_same", "sonnet_only", "codex_only", "both_different")}
    sonnet_support = codex_support = sonnet_opportunities = codex_opportunities = 0
    for pid, score in votes.items():
        stratum, same = key[pid]["stratum"], score == 2
        by[stratum]["n"] += 1
        by[stratum]["judge_same"] += int(same)
        if stratum == "sonnet_only":
            sonnet_opportunities += 1; codex_opportunities += 1
            sonnet_support += int(same); codex_support += int(not same)
        elif stratum == "codex_only":
            sonnet_opportunities += 1; codex_opportunities += 1
            sonnet_support += int(not same); codex_support += int(same)
    manifest_path = ROOT / f"{stem}_manifest.json"
    comparison_manifest_path = ROOT / f"{stem}_comparison_manifest.json"
    version = VERSION
    for path in (manifest_path, comparison_manifest_path):
        if path.exists():
            version = json.loads(path.read_text()).get("version", version)
            if version != VERSION:
                break
    report = {
        "task": task, "version": version, "judge": "independent blind LLM",
        "n_judged": len(votes), "strata": by,
        "disagreement_support": {
            "sonnet": sonnet_support,
            "codex": codex_support,
            "n": sonnet_opportunities,
            "sonnet_rate": round(sonnet_support / sonnet_opportunities, 3)
            if sonnet_opportunities else None,
            "codex_rate": round(codex_support / codex_opportunities, 3)
            if codex_opportunities else None,
        },
        "note": ("Historical Sonnet used the looser R2-v1 definition; judge uses the frozen "
                 f"{version} protocol."),
    }
    out = ROOT / f"{stem}_comparison_report.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def summarize_replicated_comparison(task: str, tag: str,
                                    votes_a_path: str | os.PathLike[str],
                                    votes_b_path: str | os.PathLike[str]) -> dict:
    """Score a stratified Sonnet/Codex comparison with two independent LLM judges.

    The report gives each judge's metrics and a conservative dual-confirmed view.  Because the
    comparison deliberately balances decision strata, these are audit-sample diagnostics rather
    than population prevalence estimates.
    """
    stem = f"{task}_{tag}"
    key_path = ROOT / f"{stem}_comparison_key.json"
    manifest_path = ROOT / f"{stem}_comparison_manifest.json"
    key = json.loads(key_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    for field in ("key", "blind", "protocol"):
        if _sha256(Path(manifest[f"{field}_path"])) != manifest[f"{field}_sha256"]:
            raise ValueError(f"[{task}/{tag}] frozen {field} changed")

    def load(path: str | os.PathLike[str]) -> Dict[str, int]:
        votes: Dict[str, int] = {}
        malformed = 0
        for line in Path(path).read_text().splitlines():
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
            raise ValueError(f"invalid comparison votes: missing={len(missing)} "
                             f"malformed_or_duplicate={malformed}")
        return votes

    a, b = load(votes_a_path), load(votes_b_path)

    def metric(pred_same: set[str], truth_same: set[str]) -> dict:
        universe = set(key)
        tp = len(pred_same & truth_same)
        fp = len(pred_same - truth_same)
        fn = len(truth_same - pred_same)
        tn = len(universe - pred_same - truth_same)
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = (2 * precision * recall / (precision + recall)
              if precision is not None and recall is not None and precision + recall else None)
        return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "precision": round(precision, 3) if precision is not None else None,
                "recall": round(recall, 3) if recall is not None else None,
                "f1": round(f1, 3) if f1 is not None else None}

    sonnet_pred = {pid for pid, row in key.items()
                   if row["stratum"] in ("both_same", "sonnet_only")}
    codex_pred = {pid for pid, row in key.items()
                  if row["stratum"] in ("both_same", "codex_only")}

    def judge_view(truth_same: set[str]) -> dict:
        return {"n_same": len(truth_same), "n": len(key),
                "sonnet": metric(sonnet_pred, truth_same),
                "codex": metric(codex_pred, truth_same)}

    a_same = {pid for pid, score in a.items() if score == 2}
    b_same = {pid for pid, score in b.items() if score == 2}
    dual_same = a_same & b_same
    av = [pid in a_same for pid in key]
    bv = [pid in b_same for pid in key]
    po = sum(x == y for x, y in zip(av, bv)) / len(key) if key else None
    pa = sum(av) / len(key) if key else None
    pb = sum(bv) / len(key) if key else None
    pe = pa * pb + (1 - pa) * (1 - pb) if key else None
    kappa = (po - pe) / (1 - pe) if key and pe < 1 else None
    result = {
        "task": task, "tag": tag, "version": manifest.get("version"),
        "n": len(key), "judge_a": judge_view(a_same), "judge_b": judge_view(b_same),
        "dual_confirmed": judge_view(dual_same),
        "binary_same_agreement": round(po, 3) if po is not None else None,
        "binary_same_kappa": round(kappa, 3) if kappa is not None else None,
        "sampling_warning": ("Decision strata were deliberately balanced; precision, recall, and "
                             "F1 are matched audit diagnostics, not population prevalence estimates."),
        "semantic_truth": "two independent LLM judges; dual_confirmed requires both score 2",
    }
    out = ROOT / f"{stem}_comparison_replicated_report.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["prepare", "validate", "compare-emit", "compare-summary"])
    p.add_argument("--tasks", required=True)
    p.add_argument("--partition")
    p.add_argument("--codex-partition")
    p.add_argument("--protocol")
    p.add_argument("--tag", default="")
    a = p.parse_args()
    for task in (x.strip() for x in a.tasks.split(",") if x.strip()):
        if a.cmd == "prepare":
            print(json.dumps(prepare(task), indent=2))
        elif a.cmd == "compare-emit":
            print(json.dumps(emit_comparison(task, codex_partition_path=a.codex_partition,
                                             tag=a.tag, protocol_path=a.protocol), indent=2))
        elif a.cmd in ("validate", "compare-summary"):
            if not a.partition or len(a.tasks.split(",")) != 1:
                raise SystemExit(f"{a.cmd} requires one --tasks value and --partition")
            if a.cmd == "validate":
                result = validate(task, a.partition)
            else:
                result = summarize_comparison(task, a.partition, tag=a.tag)
            print(json.dumps(result, indent=2))
