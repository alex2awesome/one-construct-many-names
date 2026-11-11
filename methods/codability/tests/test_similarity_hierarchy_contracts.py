from __future__ import annotations

import json
from pathlib import Path

import pytest

from methods.codability.lexicon_distill.dataset import PROTOCOL_IDS
from methods.codability.lexicon_distill.hierarchy_contracts import (
    CANDIDATE_MANIFEST_SCHEMA,
    NAMES_SCHEMA,
    NODE_INVENTORY_SCHEMA,
    PAIR_INPUT_SCHEMA,
    PAIR_OUTPUT_SCHEMA,
    PARTITION_SCHEMA,
    build_candidate_manifest,
    pair_input_sha256,
    sha256_file,
    validate_pair_files,
    validate_pair_input,
    validate_pair_output,
)


HASH_A = "a" * 64
HASH_B = "b" * 64
ADAPTER_HASH = "c" * 64
PROTOCOL_HASH = "d" * 64


def _pair(pair_id: str = "p1", protocol: str = PROTOCOL_IDS["R2_V2_1"]) -> dict:
    return {
        "schema_version": PAIR_INPUT_SCHEMA,
        "pair_id": pair_id,
        "task": "humor",
        "level": "R2",
        "protocol_id": protocol,
        "node_a": "a",
        "node_b": "b",
        "text_a": "Comedic timing",
        "text_b": "Surprise delivery",
        "source_node_a_sha256": HASH_A,
        "source_node_b_sha256": HASH_B,
    }


def _output(pair: dict) -> dict:
    return {
        "schema_version": PAIR_OUTPUT_SCHEMA,
        "pair_id": pair["pair_id"],
        "task": pair["task"],
        "level": pair["level"],
        "protocol_id": pair["protocol_id"],
        "input_sha256": pair_input_sha256(pair),
        "prediction": "SAME",
        "probabilities": {"DIFFERENT": 0.15, "RELATED": 0.25, "SAME": 0.6},
        "order_views": {
            "ab": {"prediction": "SAME", "probabilities": {"DIFFERENT": 0.1, "RELATED": 0.2, "SAME": 0.7}},
            "ba": {"prediction": "SAME", "probabilities": {"DIFFERENT": 0.2, "RELATED": 0.3, "SAME": 0.5}},
        },
        "order_consistent": True,
        "adapter_sha256": ADAPTER_HASH,
        "protocol_sha256": PROTOCOL_HASH,
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[object]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def test_pair_contract_is_strict_and_checks_order_average() -> None:
    pair = _pair()
    validate_pair_input(pair)
    output = _output(pair)
    validate_pair_output(output)
    output["probabilities"]["SAME"] = 0.61
    output["probabilities"]["DIFFERENT"] = 0.14
    with pytest.raises(ValueError, match="not the mean"):
        validate_pair_output(output)
    pair["unexpected"] = True
    with pytest.raises(ValueError, match="keys mismatch"):
        validate_pair_input(pair)


def test_r2_protocols_cannot_be_pooled_in_one_scoring_batch(tmp_path: Path) -> None:
    current = _pair("current")
    legacy = _pair("legacy", PROTOCOL_IDS["R2"])
    inputs = tmp_path / "inputs.jsonl"
    _write_jsonl(inputs, [current, legacy])
    with pytest.raises(ValueError, match="exactly one task/level/protocol cell"):
        validate_pair_files(inputs)


def test_output_must_cover_and_hash_exact_inputs(tmp_path: Path) -> None:
    pair = _pair()
    output = _output(pair)
    inputs = tmp_path / "inputs.jsonl"
    outputs = tmp_path / "outputs.jsonl"
    _write_jsonl(inputs, [pair])
    output["input_sha256"] = "0" * 64
    _write_jsonl(outputs, [output])
    with pytest.raises(ValueError, match="input hash mismatch"):
        validate_pair_files(inputs, outputs)


def test_scoring_batch_rejects_duplicate_unordered_pair(tmp_path: Path) -> None:
    first = _pair("first")
    second = _pair("second")
    second["node_a"], second["node_b"] = second["node_b"], second["node_a"]
    second["text_a"], second["text_b"] = second["text_b"], second["text_a"]
    second["source_node_a_sha256"], second["source_node_b_sha256"] = (
        second["source_node_b_sha256"], second["source_node_a_sha256"]
    )
    inputs = tmp_path / "inputs.jsonl"
    _write_jsonl(inputs, [first, second])
    with pytest.raises(ValueError, match="duplicate unordered node pair"):
        validate_pair_files(inputs)


def _candidate_files(tmp_path: Path) -> dict[str, Path]:
    paths = {name: tmp_path / f"{name}.json" for name in (
        "parent", "inventory", "retrieval", "scorer", "threshold", "graph", "partition", "names"
    )}
    paths.update(inputs=tmp_path / "pairs.jsonl", outputs=tmp_path / "scores.jsonl", manifest=tmp_path / "candidate.json")
    _write_json(paths["parent"], {"partition": {"x": "a", "y": "b"}})
    pair = _pair()
    _write_jsonl(paths["inputs"], [pair])
    _write_jsonl(paths["outputs"], [_output(pair)])
    _write_json(paths["inventory"], {
        "schema_version": NODE_INVENTORY_SCHEMA,
        "task": "humor", "level": "R2", "protocol_id": PROTOCOL_IDS["R2_V2_1"],
        "parent_partition_sha256": sha256_file(paths["parent"]),
        "nodes": [
            {"node_id": "a", "text": "Comedic timing", "source_node_sha256": HASH_A},
            {"node_id": "b", "text": "Surprise delivery", "source_node_sha256": HASH_B},
        ],
    })
    for kind in ("retrieval", "threshold", "graph"):
        _write_json(paths[kind], {
            "schema_version": f"gemma-hierarchy-{kind}-config-v1",
            "config_id": f"{kind}-fixture-v1",
        })
    _write_json(paths["scorer"], {
        "schema_version": "gemma-hierarchy-scorer-config-v1",
        "config_id": "scorer-fixture-v1",
        "protocol_id": PROTOCOL_IDS["R2_V2_1"],
        "adapter_sha256": ADAPTER_HASH,
        "protocol_sha256": PROTOCOL_HASH,
    })
    _write_json(paths["partition"], {
        "schema_version": PARTITION_SCHEMA,
        "task": "humor", "level": "R2", "protocol_id": PROTOCOL_IDS["R2_V2_1"],
        "partition": {"a": "g0", "b": "g0"},
    })
    _write_json(paths["names"], {
        "schema_version": NAMES_SCHEMA,
        "task": "humor", "level": "R2", "protocol_id": PROTOCOL_IDS["R2_V2_1"],
        "partition_sha256": sha256_file(paths["partition"]),
        "names": {"g0": {"name": "Comic delivery", "gloss": "How a joke is delivered."}},
    })
    return paths


def _build(paths: dict[str, Path]) -> dict:
    return build_candidate_manifest(
        parent_partition_path=paths["parent"], node_inventory_path=paths["inventory"],
        pair_inputs_path=paths["inputs"], pair_outputs_path=paths["outputs"],
        retrieval_config_path=paths["retrieval"], scorer_config_path=paths["scorer"],
        threshold_config_path=paths["threshold"], graph_config_path=paths["graph"],
        partition_path=paths["partition"], names_path=paths["names"], output_path=paths["manifest"],
    )


def test_candidate_manifest_binds_every_artifact_and_never_overwrites(tmp_path: Path) -> None:
    paths = _candidate_files(tmp_path)
    manifest = _build(paths)
    assert manifest["schema_version"] == CANDIDATE_MANIFEST_SCHEMA
    assert manifest["canonical"] is False
    assert manifest["n_nodes"] == 2
    assert set(manifest["artifacts"]) == {
        "parent_partition", "node_inventory", "pair_inputs", "pair_outputs",
        "retrieval_config", "scorer_config", "threshold_config", "graph_config",
        "partition", "names",
    }
    with pytest.raises(FileExistsError):
        _build(paths)


def test_candidate_manifest_rejects_stale_parent_and_incomplete_partition(tmp_path: Path) -> None:
    paths = _candidate_files(tmp_path)
    _write_json(paths["parent"], {"partition": {"changed": "after-inventory-freeze"}})
    with pytest.raises(ValueError, match="not bound to the supplied parent"):
        _build(paths)


def test_candidate_manifest_rejects_partition_coverage_gap(tmp_path: Path) -> None:
    paths = _candidate_files(tmp_path)
    partition = json.loads(paths["partition"].read_text())
    partition["partition"].pop("b")
    _write_json(paths["partition"], partition)
    names = json.loads(paths["names"].read_text())
    names["partition_sha256"] = sha256_file(paths["partition"])
    _write_json(paths["names"], names)
    with pytest.raises(ValueError, match="does not exactly cover node inventory"):
        _build(paths)


def test_candidate_manifest_rejects_scorer_protocol_drift(tmp_path: Path) -> None:
    paths = _candidate_files(tmp_path)
    scorer = json.loads(paths["scorer"].read_text())
    scorer["protocol_id"] = PROTOCOL_IDS["R2"]
    _write_json(paths["scorer"], scorer)
    with pytest.raises(ValueError, match="scorer config protocol"):
        _build(paths)
