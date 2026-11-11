"""Frozen selective LLM re-audit of proposed semantic edges.

This module is for a calibrated judge discontinuity discovered after a complete verifier pass.
It never promotes a previously unselected pair: every pair receiving score 2 anywhere in the
selected shards must receive one fresh LLM judgment, and all repeated occurrences of that pair
then receive the same audited score.  The consolidated stream retains an edge only when the fresh
judge also scores it 2.
"""
from __future__ import annotations

import glob
import json
from collections import Counter
from pathlib import Path

from .build_level import OUT, _file_sha256
from .semantic_group_merge import (
    FrozenInputError,
    _directory_sha256,
    _jsonl_rows,
    _load_vote_directory,
    _validate_frozen_file,
)


ROOT = Path(OUT) / "selective_vote_reaudit"
VERSION = "selective-vote-reaudit-v1"


def _files(pattern: str, label: str) -> list[Path]:
    paths = [Path(path).expanduser().resolve() for path in sorted(glob.glob(pattern))]
    if not paths:
        raise FileNotFoundError(f"no {label} files match {pattern}")
    return paths


def _fingerprint(path: Path) -> dict:
    return {"path": str(path), "sha256": _file_sha256(str(path))}


def _paired_rows(payload_path: Path, vote_path: Path) -> list[tuple[dict, dict]]:
    payload = list(_jsonl_rows(payload_path))
    votes = list(_jsonl_rows(vote_path))
    if len(payload) != len(votes):
        raise FrozenInputError(
            f"payload/vote length mismatch: {payload_path.name}={len(payload)} "
            f"{vote_path.name}={len(votes)}")
    result = []
    for index, (candidate, vote) in enumerate(zip(payload, votes)):
        pair_id = candidate.get("pair_id")
        if (not isinstance(pair_id, str) or set(vote) != {"pair_id", "score"}
                or vote.get("pair_id") != pair_id or type(vote.get("score")) is not int
                or vote["score"] not in (0, 1, 2)):
            raise FrozenInputError(
                f"invalid or misordered vote at row {index} of {vote_path.name}")
        result.append((candidate, vote))
    return result


def stage(task: str, level: str, tag: str, payload_glob: str, vote_glob: str, *,
          select_score: int = 2, per_agent: int = 100) -> dict:
    """Freeze score-selected rows and emit de-duplicated payloads for one fresh LLM."""
    if select_score not in (0, 1, 2) or per_agent < 1:
        raise ValueError("invalid select_score or per_agent")
    payload_paths, vote_paths = _files(payload_glob, "payload"), _files(vote_glob, "vote")
    if len(payload_paths) != len(vote_paths):
        raise FrozenInputError(
            f"source shard count mismatch: payload={len(payload_paths)} vote={len(vote_paths)}")

    selected: dict[str, dict] = {}
    seen_payloads: dict[str, dict] = {}
    n_occurrences = 0
    n_source_rows = 0
    for payload_path, vote_path in zip(payload_paths, vote_paths):
        for candidate, vote in _paired_rows(payload_path, vote_path):
            n_source_rows += 1
            pair_id, score = vote["pair_id"], vote["score"]
            if pair_id in seen_payloads and seen_payloads[pair_id] != candidate:
                raise FrozenInputError(f"repeated pair {pair_id} has inconsistent payloads")
            seen_payloads[pair_id] = candidate
            if score == select_score:
                n_occurrences += 1
                selected[pair_id] = candidate

    ROOT.mkdir(parents=True, exist_ok=True)
    payload_dir = ROOT / f"{task}_{level}_{tag}_payloads"
    payload_dir.mkdir(parents=True, exist_ok=True)
    for old in payload_dir.glob("*.jsonl"):
        old.unlink()
    rows = list(selected.values())
    emitted = []
    for shard, start in enumerate(range(0, len(rows), per_agent)):
        path = payload_dir / f"{task}_{level}_{tag}_{shard:03d}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                for row in rows[start:start + per_agent]))
        emitted.append(_fingerprint(path))

    manifest = {
        "version": VERSION,
        "task": task,
        "level": level,
        "tag": tag,
        "select_score": select_score,
        "payload_glob": payload_glob,
        "vote_glob": vote_glob,
        "source_payloads": [_fingerprint(path) for path in payload_paths],
        "source_votes": [_fingerprint(path) for path in vote_paths],
        "n_source_rows": n_source_rows,
        "n_selected_occurrences": n_occurrences,
        "n_selected_unique": len(rows),
        "selected_pair_ids": list(selected),
        "reaudit_payloads": emitted,
        "rule": ("a pair scored 2 in any source occurrence survives only when the fresh LLM "
                 "also scores it 2; repeated occurrences receive one consistent audited score"),
    }
    manifest_path = ROOT / f"{task}_{level}_{tag}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _load_stage(task: str, level: str, tag: str) -> tuple[dict, list[Path], list[Path]]:
    path = ROOT / f"{task}_{level}_{tag}_manifest.json"
    try:
        manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenInputError(f"cannot read selective re-audit manifest: {path}") from exc
    if (manifest.get("version") != VERSION or manifest.get("task") != task
            or manifest.get("level") != level or manifest.get("tag") != tag):
        raise FrozenInputError("selective re-audit manifest identity/version mismatch")
    for row in manifest.get("source_payloads") or []:
        _validate_frozen_file(row, "source payload")
    for row in manifest.get("source_votes") or []:
        _validate_frozen_file(row, "source vote")
    for row in manifest.get("reaudit_payloads") or []:
        _validate_frozen_file(row, "re-audit payload")
    payload_paths = [Path(row["path"]) for row in manifest["source_payloads"]]
    vote_paths = [Path(row["path"]) for row in manifest["source_votes"]]
    selected = []
    occurrences = 0
    for payload_path, vote_path in zip(payload_paths, vote_paths):
        for candidate, vote in _paired_rows(payload_path, vote_path):
            if vote["score"] == manifest["select_score"]:
                occurrences += 1
                if vote["pair_id"] not in selected:
                    selected.append(vote["pair_id"])
    if (selected != manifest.get("selected_pair_ids")
            or occurrences != manifest.get("n_selected_occurrences")
            or len(selected) != manifest.get("n_selected_unique")):
        raise FrozenInputError("selective re-audit selection no longer matches frozen sources")
    return manifest, payload_paths, vote_paths


def apply(task: str, level: str, tag: str, audit_vote_dir: str,
          output_dir: str) -> dict:
    """Write a complete consolidated copy; never overwrite the frozen source votes."""
    manifest, payload_paths, vote_paths = _load_stage(task, level, tag)
    expected = set(manifest["selected_pair_ids"])
    audit = _load_vote_directory(audit_vote_dir, expected, "selective re-audit")
    destination = Path(output_dir).expanduser().resolve()
    source_vote_parents = {path.parent.resolve() for path in vote_paths}
    if destination in source_vote_parents:
        raise FrozenInputError("consolidated output directory must not overwrite frozen votes")
    destination.mkdir(parents=True, exist_ok=True)
    for vote_path in vote_paths:
        old = destination / vote_path.name
        if old.exists():
            old.unlink()

    before, after = Counter(), Counter()
    output_fingerprints = []
    for payload_path, vote_path in zip(payload_paths, vote_paths):
        output_path = destination / vote_path.name
        lines = []
        for _, vote in _paired_rows(payload_path, vote_path):
            pair_id, original = vote["pair_id"], vote["score"]
            # A repeated anchor/pair can have inconsistent source votes.  If it was ever selected,
            # one fresh judgment governs every occurrence; a never-selected pair stays frozen.
            final = audit[pair_id] if pair_id in expected else original
            before[original] += 1
            after[final] += 1
            lines.append(json.dumps({"pair_id": pair_id, "score": final}) + "\n")
        output_path.write_text("".join(lines))
        output_fingerprints.append(_fingerprint(output_path))

    retained = sum(audit[pair_id] == manifest["select_score"] for pair_id in expected)
    report = {
        "version": VERSION,
        "task": task,
        "level": level,
        "tag": tag,
        "audit_vote_dir": str(Path(audit_vote_dir).expanduser().resolve()),
        "audit_vote_sha256": _directory_sha256(audit_vote_dir),
        "n_reaudited_unique": len(expected),
        "n_retained_unique_score2": retained,
        "n_vetoed_unique_score2": len(expected) - retained,
        "score_counts_before": dict(sorted(before.items())),
        "score_counts_after": dict(sorted(after.items())),
        "output_dir": str(destination),
        "output_files": output_fingerprints,
        "rule": manifest["rule"],
    }
    report_path = ROOT / f"{task}_{level}_{tag}_apply_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def relocate_frozen_source_votes(task: str, level: str, tag: str,
                                 archive_dir: str) -> dict:
    """Record a content-preserving source-vote move after a successful consolidation.

    Promotion normally replaces the canonical verifier filenames with the consolidated stream.
    The original bytes must remain independently verifiable, so this helper permits only an exact
    hash-preserving relocation into an archive and rewrites no score.  It refuses to run before a
    complete ``apply`` report exists and keeps each original path in the manifest provenance.
    """
    manifest_path = ROOT / f"{task}_{level}_{tag}_manifest.json"
    report_path = ROOT / f"{task}_{level}_{tag}_apply_report.json"
    try:
        manifest = json.loads(manifest_path.read_text())
        report = json.loads(report_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenInputError("cannot relocate before a valid selective re-audit apply") from exc
    if (manifest.get("version") != VERSION or report.get("version") != VERSION
            or any(record.get(key) != value for record in (manifest, report)
                   for key, value in (("task", task), ("level", level), ("tag", tag)))):
        raise FrozenInputError("selective re-audit relocation identity/version mismatch")
    for row in report.get("output_files") or []:
        _validate_frozen_file(row, "consolidated output")

    archive = Path(archive_dir).expanduser().resolve()
    relocated = []
    for row in manifest.get("source_votes") or []:
        original = str(row.get("original_path") or row["path"])
        target = archive / Path(original).name
        expected = {"path": str(target), "sha256": row["sha256"]}
        _validate_frozen_file(expected, "relocated source vote")
        relocated.append({"path": str(target), "sha256": row["sha256"],
                          "original_path": original})
    manifest["source_votes"] = relocated
    manifest["source_vote_relocation"] = {
        "archive_dir": str(archive),
        "rule": "every archived file must exactly match its staged source SHA-256",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest["source_vote_relocation"]
