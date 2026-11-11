import json
from pathlib import Path

import pytest

from methods.codability.lexicon import selective_vote_reaudit as svr
from methods.codability.lexicon.semantic_group_merge import FrozenInputError, VoteCoverageError


def _jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _toy(tmp_path, monkeypatch):
    monkeypatch.setattr(svr, "ROOT", tmp_path / "reaudit")
    payload_dir, vote_dir = tmp_path / "payload", tmp_path / "votes"
    rows = [{"pair_id": "a", "canonical_a": "x", "canonical_b": "y"},
            {"pair_id": "b", "canonical_a": "x", "canonical_b": "z"},
            {"pair_id": "anchor", "canonical_a": "p", "canonical_b": "q"}]
    _jsonl(payload_dir / "p_000.jsonl", rows)
    _jsonl(vote_dir / "v_000.jsonl", [
        {"pair_id": "a", "score": 2}, {"pair_id": "b", "score": 1},
        {"pair_id": "anchor", "score": 2}])
    _jsonl(payload_dir / "p_001.jsonl", [rows[2]])
    # Repeated QC anchors can expose the very calibration drift being audited.
    _jsonl(vote_dir / "v_001.jsonl", [{"pair_id": "anchor", "score": 1}])
    manifest = svr.stage("toy", "R1", "drift", str(payload_dir / "*.jsonl"),
                         str(vote_dir / "*.jsonl"), per_agent=1)
    return manifest, payload_dir, vote_dir


def test_selective_reaudit_vetoes_only_original_score2(tmp_path, monkeypatch):
    manifest, _, _ = _toy(tmp_path, monkeypatch)
    assert manifest["n_selected_occurrences"] == 2
    assert manifest["selected_pair_ids"] == ["a", "anchor"]
    audit = tmp_path / "audit"
    _jsonl(audit / "votes.jsonl", [
        {"pair_id": "a", "score": 1}, {"pair_id": "anchor", "score": 2}])
    output = tmp_path / "output"
    report = svr.apply("toy", "R1", "drift", str(audit), str(output))
    first = [json.loads(line) for line in (output / "v_000.jsonl").read_text().splitlines()]
    assert [row["score"] for row in first] == [1, 1, 2]
    repeated = [json.loads(line) for line in (output / "v_001.jsonl").read_text().splitlines()]
    assert repeated == [{"pair_id": "anchor", "score": 2}]
    assert report["n_retained_unique_score2"] == 1
    assert report["n_vetoed_unique_score2"] == 1


def test_selective_reaudit_requires_exact_audit_coverage(tmp_path, monkeypatch):
    _toy(tmp_path, monkeypatch)
    audit = tmp_path / "audit"
    _jsonl(audit / "votes.jsonl", [{"pair_id": "a", "score": 2}])
    with pytest.raises(VoteCoverageError, match="coverage is not exact"):
        svr.apply("toy", "R1", "drift", str(audit), str(tmp_path / "output"))


def test_selective_reaudit_rejects_changed_source(tmp_path, monkeypatch):
    _, _, vote_dir = _toy(tmp_path, monkeypatch)
    with (vote_dir / "v_000.jsonl").open("a") as out:
        out.write(json.dumps({"pair_id": "extra", "score": 2}) + "\n")
    audit = tmp_path / "audit"
    _jsonl(audit / "votes.jsonl", [
        {"pair_id": "a", "score": 2}, {"pair_id": "anchor", "score": 2}])
    with pytest.raises(FrozenInputError, match="source vote changed"):
        svr.apply("toy", "R1", "drift", str(audit), str(tmp_path / "output"))


def test_selective_reaudit_refuses_source_overwrite(tmp_path, monkeypatch):
    _, _, vote_dir = _toy(tmp_path, monkeypatch)
    audit = tmp_path / "audit"
    _jsonl(audit / "votes.jsonl", [
        {"pair_id": "a", "score": 2}, {"pair_id": "anchor", "score": 2}])
    with pytest.raises(FrozenInputError, match="must not overwrite"):
        svr.apply("toy", "R1", "drift", str(audit), str(vote_dir))
    assert (vote_dir / "v_000.jsonl").exists()


def test_source_relocation_preserves_frozen_hashes(tmp_path, monkeypatch):
    _, _, vote_dir = _toy(tmp_path, monkeypatch)
    audit = tmp_path / "audit"
    _jsonl(audit / "votes.jsonl", [
        {"pair_id": "a", "score": 1}, {"pair_id": "anchor", "score": 2}])
    svr.apply("toy", "R1", "drift", str(audit), str(tmp_path / "output"))
    archive = tmp_path / "archive"
    archive.mkdir()
    for path in vote_dir.glob("*.jsonl"):
        path.rename(archive / path.name)

    relocation = svr.relocate_frozen_source_votes(
        "toy", "R1", "drift", str(archive))
    manifest, _, relocated_votes = svr._load_stage("toy", "R1", "drift")

    assert relocation["archive_dir"] == str(archive.resolve())
    assert all(path.parent == archive.resolve() for path in relocated_votes)
    assert all("original_path" in row for row in manifest["source_votes"])


def test_source_relocation_rejects_changed_archive(tmp_path, monkeypatch):
    _, _, vote_dir = _toy(tmp_path, monkeypatch)
    audit = tmp_path / "audit"
    _jsonl(audit / "votes.jsonl", [
        {"pair_id": "a", "score": 1}, {"pair_id": "anchor", "score": 2}])
    svr.apply("toy", "R1", "drift", str(audit), str(tmp_path / "output"))
    archive = tmp_path / "archive"
    archive.mkdir()
    for path in vote_dir.glob("*.jsonl"):
        path.rename(archive / path.name)
    with (archive / "v_000.jsonl").open("a") as out:
        out.write('{"pair_id":"tampered","score":2}\n')

    with pytest.raises(FrozenInputError, match="relocated source vote changed"):
        svr.relocate_frozen_source_votes("toy", "R1", "drift", str(archive))
