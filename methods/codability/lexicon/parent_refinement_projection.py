"""Project an LLM-built upper partition through a strict parent refinement.

This is provenance routing, not semantic inference.  If every node in a new parent
partition is wholly contained in exactly one node of the old parent partition, an
existing LLM-assigned upper-group label can be inherited by each new child.  The
projected candidate must still pass independent LLM recall, precision, and large-group
certification gates before promotion.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from .build_level import OUT, _file_sha256


VERSION = "parent-refinement-upper-projection-v1"


def _load_mapping(path: str | Path) -> dict[str, str]:
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"partition must be a JSON object: {source}")
    raw = payload.get("assignment") or payload.get("partition") or payload
    if (not isinstance(raw, dict) or not raw
            or any(not str(key) or not isinstance(value, (str, int)) or not str(value)
                   for key, value in raw.items())):
        raise ValueError(f"invalid partition mapping: {source}")
    mapped = {str(key): str(value) for key, value in raw.items()}
    if len(mapped) != len(raw):
        raise ValueError(f"partition keys collide after string normalization: {source}")
    return mapped


def project_parent_refinement(
    task: str,
    level: str,
    old_parent_partition_path: str | Path,
    new_parent_partition_path: str | Path,
    source_upper_partition_path: str | Path,
    output_partition_path: str | Path,
    manifest_path: str | Path,
) -> dict:
    """Inherit upper labels only when the new parent is a provable pure refinement."""
    old_path = Path(old_parent_partition_path).expanduser().resolve()
    new_path = Path(new_parent_partition_path).expanduser().resolve()
    source_path = Path(source_upper_partition_path).expanduser().resolve()
    output_path = Path(output_partition_path).expanduser().resolve()
    frozen_manifest_path = Path(manifest_path).expanduser().resolve()

    old_parent = _load_mapping(old_path)
    new_parent = _load_mapping(new_path)
    source_upper = _load_mapping(source_path)
    if set(old_parent) != set(new_parent):
        raise ValueError(
            f"[{task}/{level}] parent leaf inventories differ: "
            f"old_only={len(set(old_parent)-set(new_parent))} "
            f"new_only={len(set(new_parent)-set(old_parent))}"
        )
    old_nodes = set(old_parent.values())
    if set(source_upper) != old_nodes:
        raise ValueError(
            f"[{task}/{level}] source upper partition does not exactly cover old parent nodes: "
            f"missing={len(old_nodes-set(source_upper))} "
            f"extra={len(set(source_upper)-old_nodes)}"
        )

    old_parents_by_new: dict[str, set[str]] = defaultdict(set)
    for leaf_id, new_node in new_parent.items():
        old_parents_by_new[new_node].add(old_parent[leaf_id])
    non_refinement = {
        new_node: sorted(parents)
        for new_node, parents in old_parents_by_new.items()
        if len(parents) != 1
    }
    if non_refinement:
        sample = dict(list(sorted(non_refinement.items()))[:5])
        raise ValueError(
            f"[{task}/{level}] new parent is not a pure refinement: "
            f"violating_nodes={len(non_refinement)} sample={sample}"
        )

    parent_bridge = {
        new_node: next(iter(old_nodes_for_child))
        for new_node, old_nodes_for_child in old_parents_by_new.items()
    }
    projected = {
        new_node: source_upper[old_node]
        for new_node, old_node in parent_bridge.items()
    }
    if set(projected) != set(new_parent.values()):
        raise AssertionError("projected partition lost new parent nodes")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(dict(sorted(projected.items()))) + "\n")
    sizes = Counter(projected.values())
    report = {
        "version": VERSION,
        "task": task,
        "level": level,
        "semantic_source": (
            "existing LLM-built old-parent upper partition; no new similarity labels inferred"
        ),
        "old_parent_partition_path": str(old_path),
        "old_parent_partition_sha256": _file_sha256(str(old_path)),
        "new_parent_partition_path": str(new_path),
        "new_parent_partition_sha256": _file_sha256(str(new_path)),
        "source_upper_partition_path": str(source_path),
        "source_upper_partition_sha256": _file_sha256(str(source_path)),
        "projection_invariant": (
            "every new parent node is a subset of exactly one old parent node"
        ),
        "n_leaves": len(new_parent),
        "n_old_parent_nodes": len(old_nodes),
        "n_new_parent_nodes": len(projected),
        "n_non_refinement_nodes": 0,
        "n_upper_groups": len(sizes),
        "n_singletons": sum(size == 1 for size in sizes.values()),
        "n_groups_over_30": sum(size > 30 for size in sizes.values()),
        "max_group_size": max(sizes.values(), default=0),
        "output_partition_path": str(output_path),
        "output_partition_sha256": _file_sha256(str(output_path)),
        "evaluation_blinding": (
            "candidate materialized without reading evaluation pairs, votes, or truth"
        ),
        "promotion_rule": (
            "independent LLM recall and global precision gates plus whole-group LLM "
            "certification for every group over 30 nodes"
        ),
    }
    frozen_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    frozen_manifest_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def project_r1(task: str, source_upper_partition_path: str | Path,
               tag: str = "projected_L0v3_semantics_to_L0v4") -> dict:
    """Project a task's L0v3-parent R1 partition onto strict L0v4."""
    return project_parent_refinement(
        task=task,
        level="R1",
        old_parent_partition_path=Path(OUT) / f"partition_{task}_L0v3.json",
        new_parent_partition_path=Path(OUT) / f"partition_{task}_L0v4.json",
        source_upper_partition_path=source_upper_partition_path,
        output_partition_path=Path(OUT) / f"partition_{task}_R1_{tag}.json",
        manifest_path=Path(OUT) / f"{task}_R1_{tag}_manifest.json",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task")
    parser.add_argument("source_upper_partition_path")
    parser.add_argument("--tag", default="projected_L0v3_semantics_to_L0v4")
    args = parser.parse_args()
    print(json.dumps(project_r1(args.task, args.source_upper_partition_path, args.tag), indent=2))


if __name__ == "__main__":
    main()
