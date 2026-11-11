import json
from pathlib import Path

import pytest

from methods.codability.lexicon import large_group_cert as cert
from methods.codability.lexicon.large_group_cert import (
    _apply_decisions,
    _common_refinement,
    _partition_diagnostics,
)


def _sandbox(monkeypatch, tmp_path):
    out = tmp_path / "out"
    root = out / "large_group_cert"
    out.mkdir()
    (out / "ARBITER_PROTOCOL_R1.txt").write_text("semantic protocol\n")
    nodes = [
        {"node_id": node, "name": f"Name {node}", "gloss": f"Gloss {node}"}
        for node in ("a", "b", "c", "d")
    ]
    monkeypatch.setattr(cert, "OUT", str(out))
    monkeypatch.setattr(cert, "ROOT", root)
    monkeypatch.setattr(cert, "nodes_from_level", lambda task, level: (nodes, {}))
    return out, root


def _partition(path: Path, group: str) -> Path:
    path.write_text(json.dumps({node: group for node in ("a", "b", "c", "d")}) + "\n")
    return path


def _vote(root: Path, manifest: dict, directory: str,
          groups: list[list[str]] | None = None) -> Path:
    payload_path = Path(manifest["payload_paths"][0])
    payload = json.loads(payload_path.read_text())
    vote = {
        "group_id": payload["group_id"],
        "certified": groups is None,
        "shared_concept": "A narrow shared construct",
        "rationale": "The complete membership was assessed semantically.",
    }
    if groups is not None:
        vote["groups"] = groups
    destination = root / directory / payload_path.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(vote) + "\n")
    return destination


def test_common_refinement_preserves_every_requested_split():
    expected = {"a", "b", "c", "d"}
    left = [["a", "b"], ["c", "d"]]
    right = [["a", "c"], ["b", "d"]]

    assert sorted(_common_refinement(expected, left, right)) == [["a"], ["b"], ["c"], ["d"]]


def test_certificate_cannot_erase_other_judges_partition():
    split = [["a", "b"], ["c"]]

    assert _common_refinement({"a", "b", "c"}, None, split) == split
    assert _common_refinement({"a", "b", "c"}, split, None) == split
    assert _common_refinement({"a", "b", "c"}, None, None) is None


def test_apply_decisions_retains_certified_group_and_splits_requested_group():
    part = {"a": "g1", "b": "g1", "c": "g2", "d": "g2"}
    out = _apply_decisions(part, {"g1": None, "g2": [["c"], ["d"]]})

    assert out["a"] == out["b"] == "g1"
    assert out["c"] != out["d"]
    assert _partition_diagnostics(out, threshold=1) == {
        "n_groups": 3, "remaining_over_threshold": 1, "max_group_size": 2}


def test_repartition_may_reject_shared_concept_with_null(monkeypatch, tmp_path):
    _, root = _sandbox(monkeypatch, tmp_path)
    source = _partition(tmp_path / "candidate.json", "group")
    manifest = cert.prepare("demo", "R1", str(source), threshold=2)
    payload_path = Path(manifest["payload_paths"][0])
    payload = json.loads(payload_path.read_text())
    destination = root / "votes" / payload_path.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({
        "group_id": payload["group_id"],
        "certified": False,
        "shared_concept": None,
        "rationale": "No single narrow construct covers the complete membership.",
        "groups": [["a", "b"], ["c", "d"]],
    }) + "\n")

    report = cert.apply("demo", "R1")

    assert report["repartitioned_oversized"] == 1
    assert report["groups_after"] == 2


def test_intact_certificate_requires_named_shared_concept(monkeypatch, tmp_path):
    _, root = _sandbox(monkeypatch, tmp_path)
    source = _partition(tmp_path / "candidate.json", "group")
    manifest = cert.prepare("demo", "R1", str(source), threshold=2)
    payload_path = Path(manifest["payload_paths"][0])
    payload = json.loads(payload_path.read_text())
    destination = root / "votes" / payload_path.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({
        "group_id": payload["group_id"],
        "certified": True,
        "shared_concept": None,
        "rationale": "Claims coherence without naming the construct.",
    }) + "\n")

    with pytest.raises(ValueError, match="incomplete certificates"):
        cert.apply("demo", "R1")


def test_empty_tag_preserves_legacy_artifact_names(monkeypatch, tmp_path):
    out, root = _sandbox(monkeypatch, tmp_path)
    source = _partition(tmp_path / "legacy.json", "legacy_group")

    manifest = cert.prepare("demo", "R1", str(source), threshold=2)

    assert "tag" not in manifest
    assert (root / "demo_R1_manifest.json").exists()
    assert [Path(path).name for path in manifest["payload_paths"]] == ["demo_R1_000.json"]
    _vote(root, manifest, "votes")
    report = cert.apply("demo", "R1")
    assert Path(report["partition_path"]) == out / "partition_demo_R1_certified.json"
    assert (root / "demo_R1_apply_report.json").exists()


def test_two_tags_coexist_and_freeze_independently(monkeypatch, tmp_path):
    out, root = _sandbox(monkeypatch, tmp_path)
    source_a = _partition(tmp_path / "candidate_a.json", "group_a")
    source_b = _partition(tmp_path / "candidate_b.json", "group_b")

    manifest_a = cert.prepare(
        "demo", "R1", str(source_a), threshold=2, required_judges=2, tag="candidate_a")
    payload_a = Path(manifest_a["payload_paths"][0])
    frozen_payload_a = payload_a.read_bytes()
    manifest_b = cert.prepare(
        "demo", "R1", str(source_b), threshold=2, required_judges=2, tag="candidate_b")
    # Preparing the legacy stem must also leave both tagged payload sets untouched.
    cert.prepare("demo", "R1", str(source_a), threshold=2)

    assert payload_a.read_bytes() == frozen_payload_a
    assert manifest_a["tag"] == "candidate_a"
    assert manifest_b["tag"] == "candidate_b"
    assert (root / "demo_R1_candidate_a_manifest.json").exists()
    assert (root / "demo_R1_candidate_b_manifest.json").exists()
    assert Path(manifest_a["payload_paths"][0]).name != Path(manifest_b["payload_paths"][0]).name

    vote_a = _vote(root, manifest_a, "votes")
    vote_b = _vote(root, manifest_b, "votes")
    replicate_a = _vote(root, manifest_a, "replicate_votes")
    replicate_b = _vote(root, manifest_b, "replicate_votes")
    report_a = cert.apply("demo", "R1", tag="candidate_a")
    report_b = cert.apply("demo", "R1", tag="candidate_b")
    assert vote_a != vote_b and vote_a.exists() and vote_b.exists()
    assert replicate_a != replicate_b and replicate_a.exists() and replicate_b.exists()
    assert Path(report_a["partition_path"]) == out / "partition_demo_R1_candidate_a_certified.json"
    assert Path(report_b["partition_path"]) == out / "partition_demo_R1_candidate_b_certified.json"
    assert (root / "demo_R1_candidate_a_apply_report.json").exists()
    assert (root / "demo_R1_candidate_b_apply_report.json").exists()

    # A changed candidate-B input invalidates only B's frozen manifest; A remains applicable.
    source_b.write_text(json.dumps({node: "changed" for node in ("a", "b", "c", "d")}) + "\n")
    cert.apply("demo", "R1", tag="candidate_a")
    with pytest.raises(ValueError, match="certification inputs changed"):
        cert.apply("demo", "R1", tag="candidate_b")


def test_tagged_two_judge_apply_uses_common_refinement(monkeypatch, tmp_path):
    out, root = _sandbox(monkeypatch, tmp_path)
    source = _partition(tmp_path / "crossed.json", "group")
    manifest = cert.prepare(
        "demo", "R1", str(source), threshold=2, required_judges=2, tag="crossed")
    _vote(root, manifest, "votes", [["a", "b"], ["c", "d"]])
    _vote(root, manifest, "replicate_votes", [["a", "c"], ["b", "d"]])
    _vote(root, manifest, "tiebreak_votes", [["a", "b"], ["c", "d"]])

    report = cert.apply("demo", "R1", tag="crossed")
    consensus = json.loads(Path(report["partition_path"]).read_text())

    assert len(set(consensus.values())) == 4
    assert report["common_refinements"] == 1
    assert report["tiebreaks_used"] == 1
    assert report["candidate_partitions"]["judge_a"]["n_groups"] == 2
    assert report["candidate_partitions"]["judge_b"]["n_groups"] == 2
    assert Path(report["partition_path"]) == out / "partition_demo_R1_crossed_certified.json"
    assert Path(report["candidate_partitions"]["judge_a"]["partition_path"]) == (
        out / "partition_demo_R1_crossed_certified_judge_a.json")
    assert Path(report["candidate_partitions"]["judge_b"]["partition_path"]) == (
        out / "partition_demo_R1_crossed_certified_judge_b.json")


def test_two_judge_disagreement_requires_tiebreak(monkeypatch, tmp_path):
    _, root = _sandbox(monkeypatch, tmp_path)
    source = _partition(tmp_path / "candidate.json", "group")
    manifest = cert.prepare(
        "demo", "R1", str(source), threshold=2, required_judges=2, tag="needs_tie")
    _vote(root, manifest, "votes")
    _vote(root, manifest, "replicate_votes", [["a", "b"], ["c", "d"]])

    with pytest.raises(ValueError, match="incomplete certificates"):
        cert.apply("demo", "R1", tag="needs_tie")

    _vote(root, manifest, "tiebreak_votes")
    report = cert.apply("demo", "R1", tag="needs_tie")
    assert report["tiebreaks_used"] == 1
    assert report["certified_oversized"] == 1
