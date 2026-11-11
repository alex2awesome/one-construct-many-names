"""Partition isolation for score matrices."""

import json
from pathlib import Path

import numpy as np
import pytest

from methods.codability.experiments.build_fresh_item_partitions import sha256_file
from methods.codability.experiments.shard_fresh_score_artifact import shard_artifact


def test_sharding_preserves_rows_and_isolates_columns(tmp_path):
    source = tmp_path / "source.npz"
    np.savez_compressed(
        source, scores=np.arange(12).reshape(2, 6),
        meta=np.asarray([json.dumps({"row": 0}), json.dumps({"row": 1})], dtype=object),
        probe_sha256=np.asarray([f"h{i}" for i in range(6)]),
        probe_partition=np.asarray(["dev", "dev", "dev", "lock", "lock", "lock"]),
        probe_set_sha256="old", reader="reader")

    reports = shard_artifact(source, tmp_path / "shards")

    assert {row["partition"] for row in reports} == {"dev", "lock"}
    with np.load(tmp_path / "shards/dev/source.npz", allow_pickle=True) as shard:
        assert shard["scores"].shape == (2, 3)
        assert set(shard["probe_partition"]) == {"dev"}
        assert str(shard["source_artifact_sha256"])


def test_job_scope_prevents_same_basename_target_view_overwrite(tmp_path):
    sources = []
    for job in ("llama70_n_target", "llama70_g_target"):
        source_dir = tmp_path / job
        source_dir.mkdir()
        source = source_dir / "grid_humor_same_model_rep0.npz"
        np.savez_compressed(
            source, scores=np.arange(6).reshape(1, 6),
            meta=np.asarray([json.dumps({"row": 0})], dtype=object),
            probe_sha256=np.asarray([f"h{i}" for i in range(6)]),
            probe_partition=np.asarray(["dev"] * 3 + ["lock"] * 3),
            model_job_id=job)
        sources.append(source)

    for source in sources:
        shard_artifact(source, tmp_path / "shards")

    n_path = tmp_path / "shards/dev/llama70_n_target/grid_humor_same_model_rep0.npz"
    g_path = tmp_path / "shards/dev/llama70_g_target/grid_humor_same_model_rep0.npz"
    assert n_path.exists()
    assert g_path.exists()
    with np.load(n_path, allow_pickle=True) as n, np.load(g_path, allow_pickle=True) as g:
        assert str(n["model_job_id"]) == "llama70_n_target"
        assert str(g["model_job_id"]) == "llama70_g_target"


def test_sharding_keeps_same_legacy_gi_distinct_across_r1_r2_r3(tmp_path):
    source = tmp_path / "breadth.npz"
    levels = ("R1", "R2", "R3")
    meta = []
    for level in levels:
        meta.append(json.dumps({
            "cell_id": f"TB::demo::{level}::node",
            "domain": "demo",
            "task": "demo",
            "level": level,
            "bucket": "general",
            "node_id": f"demo::{level}::node",
            "metric_id": f"demo::{level}::node",
            "gi": 7,
            "construct": f"construct {level}",
            "arm_id": "name",
            "form": "canonical",
        }))
    np.savez_compressed(
        source,
        scores=np.arange(18).reshape(3, 6),
        meta=np.asarray(meta, dtype=object),
        probe_sha256=np.asarray([f"h{i}" for i in range(6)]),
        probe_partition=np.asarray(["dev"] * 6),
        model_job_id="small",
    )

    report = shard_artifact(source, tmp_path / "shards")[0]

    assert report["n_cells"] == 3
    assert {row["level"] for row in report["cell_identities"]} == set(levels)
    assert {row["gi"] for row in report["cell_identities"]} == {7}
    assert len({row["cell_id"] for row in report["cell_identities"]}) == 3
    with np.load(report["path"], allow_pickle=True) as shard:
        frozen = json.loads(str(shard["cell_identity_manifest"]))
        assert frozen == report["cell_identities"]
        assert str(shard["cell_identity_sha256"]) == report["cell_identity_sha256"]


def test_sharding_rejects_one_cell_id_with_conflicting_hierarchy_identity(tmp_path):
    source = tmp_path / "collision.npz"
    rows = []
    for level in ("R1", "R2"):
        rows.append(json.dumps({
            "cell_id": "same-cell",
            "domain": "demo",
            "task": "demo",
            "level": level,
            "bucket": "general",
            "node_id": f"demo::{level}::node",
            "gi": 7,
            "construct": "construct",
        }))
    np.savez_compressed(
        source,
        scores=np.arange(8).reshape(2, 4),
        meta=np.asarray(rows, dtype=object),
        probe_sha256=np.asarray([f"h{i}" for i in range(4)]),
        probe_partition=np.asarray(["dev"] * 4),
    )
    with pytest.raises(ValueError, match="rows disagree on cell identity"):
        shard_artifact(source, tmp_path / "shards")


def test_frozen_sharding_authenticates_execution_manifest(tmp_path):
    root = Path(__file__).parents[3]
    implementation_path = root / "methods/codability/experiments/policy_data.py"
    execution = tmp_path / "current_execution_manifest.json"
    execution.write_text(json.dumps({
        "implementation": {
            "scoring": {
                "semantics": "test-local current implementation binding",
                "files": [{
                    "path": str(implementation_path),
                    "sha256": sha256_file(implementation_path),
                }],
            },
        },
    }))
    execution_sha256 = sha256_file(execution)
    source = tmp_path / "source.npz"

    np.savez_compressed(
        source,
        scores=np.arange(4).reshape(1, 4),
        probe_sha256=np.asarray([f"h{i}" for i in range(4)]),
        probe_partition=np.asarray(["calibration"] * 4),
        execution_manifest_sha256=np.asarray(execution_sha256),
    )
    reports = shard_artifact(
        source, tmp_path / "shards", execution_manifest_path=execution)
    assert reports[0]["execution_manifest_sha256"] == execution_sha256

    np.savez_compressed(
        source,
        scores=np.arange(4).reshape(1, 4),
        probe_sha256=np.asarray([f"h{i}" for i in range(4)]),
        probe_partition=np.asarray(["calibration"] * 4),
        execution_manifest_sha256=np.asarray("wrong"),
    )
    with pytest.raises(ValueError, match="frozen execution manifest"):
        shard_artifact(
            source, tmp_path / "shards", execution_manifest_path=execution)
