"""Two-LLM reconsideration of ambiguous semantic-recovery pairs.

The first semantic screen remains authoritative for clear score-0/score-2 decisions.  This module
routes a frozen subset (normally score-1, related-but-distinct/uncertain pairs) to two fresh LLM
judges.  An ambiguous edge is added only when *both* reconsideration judges independently score it
2.  Retrieval scores and string comparisons never authorize an edge.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from .build_level import OUT, _file_sha256, _load_partition
from .semantic_group_merge import (
    VERSION,
    FrozenInputError,
    _DisjointSet,
    _directory_sha256,
    _jsonl_rows,
    _load_manifest,
    _load_vote_directory,
    _payload_candidates,
    _validate_frozen_file,
)


ROOT = Path(OUT) / "semantic_group_merge"
RECONSIDER_VERSION = "semantic-reconsider-v1"


def _stem(task: str, level: str, tag: str, reconsider_tag: str) -> str:
    bits = [task, level]
    if tag:
        bits.append(tag)
    bits.append(reconsider_tag)
    return "_".join(bits)


def stage(task: str, level: str, tag: str, screen_vote_dir: str, *,
          source_manifest_path: str | None = None, reconsider_tag: str = "score1",
          select_score: int = 1, cap: int | None = None, per_agent: int = 100) -> dict:
    """Freeze and emit ambiguous pairs for two new direct semantic judgments."""
    if select_score not in (0, 1, 2) or per_agent < 1 or (cap is not None and cap < 0):
        raise ValueError("invalid select_score, cap, or per_agent")
    if source_manifest_path:
        source_path = Path(source_manifest_path).expanduser().resolve()
    else:
        source_path = (ROOT / (f"{task}_{level}" + (f"_{tag}" if tag else "")
                               + "_manifest.json")).resolve()
    manifest = _load_manifest(task, level, str(source_path))
    candidates = _payload_candidates(manifest)
    screen_path = Path(screen_vote_dir).expanduser().resolve()
    screen = _load_vote_directory(str(screen_path), set(candidates), "screen")
    selected = [pair_id for pair_id in candidates if screen[pair_id] == select_score]
    if cap is not None:
        selected = selected[:cap]

    payload_by_id = {}
    for path in manifest.get("payload_paths") or []:
        for row in _jsonl_rows(Path(path)):
            payload_by_id[row["pair_id"]] = row
    if set(payload_by_id) != set(candidates):
        raise FrozenInputError("reconsideration could not recover every frozen payload")

    stem = _stem(task, level, tag, reconsider_tag)
    payload_dir = ROOT / "reconsider_payloads"
    payload_dir.mkdir(parents=True, exist_ok=True)
    for old in payload_dir.glob(f"{stem}_*.jsonl"):
        old.unlink()
    paths, fingerprints = [], []
    for shard, start in enumerate(range(0, len(selected), per_agent)):
        path = payload_dir / f"{stem}_{shard:03d}.jsonl"
        path.write_text("".join(json.dumps(payload_by_id[pair_id], ensure_ascii=False) + "\n"
                                for pair_id in selected[start:start + per_agent]))
        paths.append(str(path.resolve()))
        fingerprints.append({"path": str(path.resolve()), "sha256": _file_sha256(str(path))})

    result = {
        "version": RECONSIDER_VERSION,
        "semantic_source_version": VERSION,
        "task": task,
        "level": level,
        "tag": tag,
        "reconsider_tag": reconsider_tag,
        "select_score": select_score,
        "cap": cap,
        "source_manifest_path": str(source_path),
        "source_manifest_sha256": _file_sha256(str(source_path)),
        "screen_vote_dir": str(screen_path),
        "screen_vote_sha256": _directory_sha256(str(screen_path)),
        "n_source_candidates": len(candidates),
        "n_selected": len(selected),
        "selected_pair_ids": selected,
        "payload_paths": paths,
        "payload_fingerprints": fingerprints,
        "rule": "selected first-pass ambiguities require two fresh independent score-2 votes",
    }
    destination = ROOT / f"{stem}_reconsider_manifest.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    return result


def _load_stage(task: str, level: str, tag: str, reconsider_tag: str) -> tuple[dict, dict]:
    path = ROOT / f"{_stem(task, level, tag, reconsider_tag)}_reconsider_manifest.json"
    try:
        staged = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenInputError(f"cannot read reconsideration manifest: {path}") from exc
    if (staged.get("version") != RECONSIDER_VERSION or staged.get("task") != task
            or staged.get("level") != level or staged.get("tag") != tag
            or staged.get("reconsider_tag") != reconsider_tag):
        raise FrozenInputError("reconsideration manifest identity/version mismatch")
    source_path = str(staged["source_manifest_path"])
    if _file_sha256(source_path) != staged.get("source_manifest_sha256"):
        raise FrozenInputError("reconsideration source manifest changed")
    if _directory_sha256(staged["screen_vote_dir"]) != staged.get("screen_vote_sha256"):
        raise FrozenInputError("reconsideration screen votes changed")
    for row in staged.get("payload_fingerprints") or []:
        _validate_frozen_file(row, "reconsideration payload")
    source = _load_manifest(task, level, source_path)
    candidates = _payload_candidates(source)
    screen = _load_vote_directory(staged["screen_vote_dir"], set(candidates), "screen")
    expected_all = [pair_id for pair_id in candidates
                    if screen[pair_id] == staged["select_score"]]
    cap = staged.get("cap")
    expected = expected_all if cap is None else expected_all[:cap]
    recorded = staged.get("selected_pair_ids")
    if (not isinstance(recorded, list) or len(recorded) != len(set(recorded))
            or recorded != expected or staged.get("n_selected") != len(expected)):
        raise FrozenInputError("reconsideration selection no longer matches frozen screen")
    return staged, candidates


def apply(task: str, level: str, tag: str, reconsider_tag: str,
          base_partition_path: str, votes_b_dir: str, votes_c_dir: str,
          output_path: str) -> dict:
    """Add only doubly-upgraded ambiguous edges to an already authorized base partition."""
    staged, candidates = _load_stage(task, level, tag, reconsider_tag)
    expected = set(staged["selected_pair_ids"])
    b = _load_vote_directory(votes_b_dir, expected, "reconsider-b")
    c = _load_vote_directory(votes_c_dir, expected, "reconsider-c")
    source_manifest = _load_manifest(task, level, staged["source_manifest_path"])
    source = {str(node): str(group) for node, group in
              _load_partition(source_manifest["partition_path"]).items()}
    base_path = Path(base_partition_path).expanduser().resolve()
    base = {str(node): str(group) for node, group in _load_partition(str(base_path)).items()}
    if set(base) != set(source):
        raise FrozenInputError("base reconsideration partition does not cover source nodes")

    # Every original source group must remain intact in the authorized base.  Recover which source
    # groups the base already composed, then add only newly double-confirmed ambiguous edges.
    source_to_base: dict[str, set[str]] = defaultdict(set)
    for node, source_group in source.items():
        source_to_base[source_group].add(base[node])
    if any(len(groups) != 1 for groups in source_to_base.values()):
        raise FrozenInputError("base partition split an original semantic source group")
    source_groups = set(source.values())
    dsu = _DisjointSet(source_groups)
    by_base: dict[str, list[str]] = defaultdict(list)
    for source_group, base_groups in source_to_base.items():
        by_base[next(iter(base_groups))].append(source_group)
    for groups in by_base.values():
        for other in groups[1:]:
            dsu.union(groups[0], other)

    accepted = []
    for pair_id in staged["selected_pair_ids"]:
        if b[pair_id] == 2 and c[pair_id] == 2:
            group_a, group_b = candidates[pair_id]
            dsu.union(group_a, group_b)
            accepted.append(pair_id)
    components: dict[str, list[str]] = defaultdict(list)
    for group in sorted(source_groups):
        components[dsu.find(group)].append(group)
    composed_group = {}
    for groups in components.values():
        groups.sort()
        new = (groups[0] if len(groups) == 1 else
               f"{task}_{level}_reconsider_{hashlib.sha1('||'.join(groups).encode()).hexdigest()[:12]}")
        for group in groups:
            composed_group[group] = new
    output = {node: composed_group[group] for node, group in source.items()}
    destination = Path(output_path).expanduser().resolve()
    destination.write_text(json.dumps(output, sort_keys=True) + "\n")
    report = {
        "version": RECONSIDER_VERSION,
        "task": task,
        "level": level,
        "tag": tag,
        "reconsider_tag": reconsider_tag,
        "base_partition_path": str(base_path),
        "base_partition_sha256": _file_sha256(str(base_path)),
        "votes_b_sha256": _directory_sha256(votes_b_dir),
        "votes_c_sha256": _directory_sha256(votes_c_dir),
        "n_reconsidered": len(expected),
        "n_double_score2_added": len(accepted),
        "accepted_pair_ids": accepted,
        "groups_before": len(set(base.values())),
        "groups_after": len(set(output.values())),
        "partition_path": str(destination),
        "partition_sha256": _file_sha256(str(destination)),
        "semantic_rule": "ambiguous screen pair added only when both fresh LLMs score 2",
    }
    report_path = ROOT / f"{_stem(task, level, tag, reconsider_tag)}_reconsider_apply_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report
