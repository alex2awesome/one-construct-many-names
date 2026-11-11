"""Three-judge majority application for semantic group-pair recovery."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from .build_level import OUT, _file_sha256, _load_partition
from .semantic_group_merge import (
    VERSION,
    _DisjointSet,
    _directory_sha256,
    _load_manifest,
    _load_vote_directory,
    _payload_candidates,
)


ROOT = Path(OUT) / "semantic_group_merge"


def majority_same(a: int, b: int, c: int | None = None) -> bool:
    votes = [a, b] + ([] if c is None else [c])
    return sum(score == 2 for score in votes) >= 2


def stage(task: str, level: str, tag: str, manifest_path: str,
          votes_a_dir: str, votes_b_dir: str, per_agent: int = 100) -> dict:
    manifest_file = Path(manifest_path).resolve()
    manifest = _load_manifest(task, level, str(manifest_file))
    candidates = _payload_candidates(manifest)
    expected = set(candidates)
    a = _load_vote_directory(votes_a_dir, expected, "judge-a")
    b = _load_vote_directory(votes_b_dir, expected, "judge-b")
    disputed = [pid for pid in candidates if a[pid] != b[pid] and (a[pid] == 2 or b[pid] == 2)]

    payload_rows = {}
    for path in manifest["payload_paths"]:
        for line in Path(path).read_text().splitlines():
            if line.strip():
                row = json.loads(line); payload_rows[row["pair_id"]] = row
    payload_dir = ROOT / "panel_adjudicate_payloads"
    payload_dir.mkdir(exist_ok=True)
    stem = f"{task}_{level}_{tag}"
    for old in payload_dir.glob(f"{stem}_*.jsonl"):
        old.unlink()
    paths = []
    for start in range(0, len(disputed), per_agent):
        path = payload_dir / f"{stem}_{start // per_agent:03d}.jsonl"
        path.write_text("".join(json.dumps(payload_rows[pid], ensure_ascii=False) + "\n"
                                for pid in disputed[start:start + per_agent]))
        paths.append(str(path.resolve()))
    report = {
        "version": VERSION, "task": task, "level": level, "tag": tag,
        "manifest_path": str(manifest_file), "manifest_sha256": _file_sha256(str(manifest_file)),
        "votes_a_dir": str(Path(votes_a_dir).resolve()),
        "votes_a_sha256": _directory_sha256(votes_a_dir),
        "votes_b_dir": str(Path(votes_b_dir).resolve()),
        "votes_b_sha256": _directory_sha256(votes_b_dir),
        "n_candidates": len(candidates), "n_direct_dual2": sum(
            a[pid] == 2 and b[pid] == 2 for pid in candidates),
        "n_adjudicate": len(disputed), "adjudicate_pair_ids": disputed,
        "payload_paths": paths,
        "rule": "accept group edge when at least two of three independent LLMs score 2",
    }
    (ROOT / f"{stem}_panel_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def apply(task: str, level: str, tag: str, votes_c_dir: str,
          output_path: str | None = None) -> dict:
    panel_path = ROOT / f"{task}_{level}_{tag}_panel_manifest.json"
    panel = json.loads(panel_path.read_text())
    if (_file_sha256(panel["manifest_path"]) != panel["manifest_sha256"]
            or _directory_sha256(panel["votes_a_dir"]) != panel["votes_a_sha256"]
            or _directory_sha256(panel["votes_b_dir"]) != panel["votes_b_sha256"]):
        raise ValueError("frozen semantic-panel inputs changed")
    manifest = _load_manifest(task, level, panel["manifest_path"])
    candidates = _payload_candidates(manifest)
    expected = set(candidates)
    a = _load_vote_directory(panel["votes_a_dir"], expected, "judge-a")
    b = _load_vote_directory(panel["votes_b_dir"], expected, "judge-b")
    disputed = set(panel["adjudicate_pair_ids"])
    c = _load_vote_directory(votes_c_dir, disputed, "judge-c") if disputed else {}

    source = {str(k): str(v) for k, v in _load_partition(manifest["partition_path"]).items()}
    source_groups = set(source.values()); dsu = _DisjointSet(source_groups)
    accepted = []
    for pid, (ga, gb) in candidates.items():
        third = c.get(pid)
        if majority_same(a[pid], b[pid], third):
            dsu.union(ga, gb); accepted.append(pid)
    components: dict[str, list[str]] = defaultdict(list)
    for group in sorted(source_groups): components[dsu.find(group)].append(group)
    composed_group = {}
    for groups in components.values():
        groups.sort()
        new = groups[0] if len(groups) == 1 else f"{task}_{level}_panel_{hashlib.sha1('||'.join(groups).encode()).hexdigest()[:12]}"
        for group in groups: composed_group[group] = new
    out = {node: composed_group[group] for node, group in source.items()}
    destination = Path(output_path).resolve() if output_path else Path(OUT) / f"partition_{task}_{level}_{tag}_panel.json"
    destination.write_text(json.dumps(out, sort_keys=True) + "\n")
    report = {"task": task, "level": level, "tag": tag, "n_candidates": len(candidates),
              "n_accepted_edges": len(accepted), "groups_before": len(source_groups),
              "groups_after": len(set(out.values())), "partition_path": str(destination),
              "partition_sha256": _file_sha256(str(destination)),
              "rule": "at least two of three independent LLM score-2 votes"}
    (ROOT / f"{task}_{level}_{tag}_panel_apply_report.json").write_text(
        json.dumps(report, indent=2) + "\n")
    return report
