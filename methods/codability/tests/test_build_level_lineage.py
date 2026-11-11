"""Fail-closed lineage and candidate-promotion tests (no model calls or real artifacts)."""
import inspect
import json
from pathlib import Path

import pytest

from methods.codability.lexicon import build_level as build


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


@pytest.fixture
def isolated_lexicon(tmp_path, monkeypatch):
    out = tmp_path / "outputs" / "lexicon"
    out.mkdir(parents=True)
    monkeypatch.setattr(build, "ROOT", str(tmp_path))
    monkeypatch.setattr(build, "OUT", str(out))
    return out


def _seed_l0(out: Path, task: str = "demo") -> None:
    _write(out / f"partition_{task}_L0v2.json", {"leaf-a": "l0-a", "leaf-b": "l0-b"})
    _write(out / f"cluster_names_{task}_L0v2.json", {
        "l0-a": {"name": "A", "gloss": "First construct"},
        "l0-b": {"name": "B", "gloss": "Second construct"},
    })


def test_pairwise_requires_explicit_noncanonical_output(isolated_lexicon):
    assert inspect.signature(build.apply_pairwise).parameters["output_path"].default \
        is inspect.Parameter.empty
    with pytest.raises(TypeError, match="output_path"):
        build.apply_pairwise("demo", "R1")
    with pytest.raises(ValueError, match="canonical-looking"):
        build.apply_pairwise(
            "demo", "R1", output_path=isolated_lexicon / "partition_demo_R1.json")


def test_freeze_parent_chain_pins_every_rung(isolated_lexicon):
    out = isolated_lexicon
    _seed_l0(out)
    _write(out / "partition_demo_R1.json", {"l0-a": "r1-a", "l0-b": "r1-b"})
    _write(out / "node_names_demo_R1.json", {
        "r1-a": {"name": "R1 A", "gloss": "First R1 group"},
        "r1-b": {"name": "R1 B", "gloss": "Second R1 group"},
    })
    _write(out / "partition_demo_R2.json", {"r1-a": "r2-a", "r1-b": "r2-b"})
    _write(out / "node_names_demo_R2.json", {
        "r2-a": {"name": "R2 A", "gloss": "First R2 group"},
        "r2-b": {"name": "R2 B", "gloss": "Second R2 group"},
    })

    build._freeze_parent_for_new_build("demo", "R3")

    for level in ("R1", "R2", "R3"):
        manifest = json.loads((out / f"level_manifest_demo_{level}.json").read_text())
        assert all(manifest.get(field) for field in build._PARENT_MANIFEST_FIELDS)
        build._validate_level_manifest("demo", level)


def test_manifest_parent_drift_fails_closed(isolated_lexicon):
    out = isolated_lexicon
    _seed_l0(out)
    build._freeze_parent_for_new_build("demo", "R1")
    _write(out / "partition_demo_L0v2.json", {"leaf-a": "changed", "leaf-b": "l0-b"})

    with pytest.raises(RuntimeError, match="frozen parent partition changed"):
        build._validate_level_manifest("demo", "R1")


def test_partial_manifest_is_rejected(isolated_lexicon):
    out = isolated_lexicon
    _seed_l0(out)
    _write(out / "level_manifest_demo_R1.json", {
        "task": "demo", "level": "R1",
        "parent_partition_path": "outputs/lexicon/partition_demo_L0v2.json",
    })

    with pytest.raises(build.LevelManifestError, match="partial parent partition pin"):
        build._freeze_parent_for_new_build("demo", "R1")


def test_only_promotion_writes_canonical_and_banks_replacement(isolated_lexicon):
    out = isolated_lexicon
    _seed_l0(out)
    build._freeze_parent_for_new_build("demo", "R1")
    first = out / "partition_demo_R1_candidate_first.json"
    second = out / "partition_demo_R1_candidate_second.json"
    _write(first, {"l0-a": "group-a", "l0-b": "group-b"})
    _write(second, {"l0-a": "group-ab", "l0-b": "group-ab"})

    promoted = build.promote_partition("demo", "R1", str(first))
    canonical = out / "partition_demo_R1.json"
    assert json.loads(canonical.read_text()) == json.loads(first.read_text())
    assert promoted["replaced"] is False

    with pytest.raises(FileExistsError, match="--replace-canonical"):
        build.promote_partition("demo", "R1", str(second))
    replaced = build.promote_partition("demo", "R1", str(second), replace=True)
    assert replaced["replaced"] is True
    assert Path(replaced["backup_path"]).is_file()
    assert json.loads(Path(replaced["backup_path"]).read_text()) == json.loads(first.read_text())
    manifest = json.loads((out / "level_manifest_demo_R1.json").read_text())
    assert manifest["canonical_partition_sha256"] == build._file_sha256(str(canonical))


def test_promotion_rejects_incomplete_node_inventory(isolated_lexicon):
    out = isolated_lexicon
    _seed_l0(out)
    build._freeze_parent_for_new_build("demo", "R1")
    candidate = out / "partition_demo_R1_candidate_incomplete.json"
    _write(candidate, {"l0-a": "group-a"})

    with pytest.raises(ValueError, match="node inventory mismatch"):
        build.promote_partition("demo", "R1", str(candidate))
    assert not (out / "partition_demo_R1.json").exists()


def test_freeze_upper_parent_rejects_stale_inventory_and_missing_names(isolated_lexicon):
    out = isolated_lexicon
    _seed_l0(out)
    _write(out / "partition_demo_R1.json", {"l0-a": "r1-a", "obsolete": "r1-b"})
    _write(out / "node_names_demo_R1.json", {
        "r1-a": {"name": "R1 A", "gloss": "First R1 group"},
        "r1-b": {"name": "R1 B", "gloss": "Second R1 group"},
    })

    with pytest.raises(build.LevelManifestError, match="node inventory mismatch"):
        build._freeze_parent_for_new_build("demo", "R2")
    assert not (out / "level_manifest_demo_R2.json").exists()

    _write(out / "partition_demo_R1.json", {"l0-a": "r1-a", "l0-b": "r1-b"})
    _write(out / "node_names_demo_R1.json", {
        "r1-a": {"name": "R1 A", "gloss": "First R1 group"},
    })
    with pytest.raises(build.LevelManifestError, match="unnamed groups"):
        build._freeze_parent_for_new_build("demo", "R2")
    assert not (out / "level_manifest_demo_R2.json").exists()
