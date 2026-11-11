import hashlib
import json

import pytest

from methods.codability.lexicon_distill.build_hierarchy_candidate import (
    build_partition,
    prepare_pairs,
)
from methods.codability.lexicon_distill.hierarchy_contracts import (
    NODE_INVENTORY_SCHEMA,
    PAIR_OUTPUT_SCHEMA,
    pair_input_sha256,
)


def _write_json(path, payload):
    path.write_text(json.dumps(payload) + "\n")


def _inventory(tmp_path):
    path = tmp_path / "nodes.json"
    nodes = []
    for node in "abcd":
        text = f"Concept {node}"
        nodes.append({
            "node_id": node, "text": text,
            "source_node_sha256": hashlib.sha256(
                json.dumps({"node_id": node, "text": text}, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        })
    _write_json(path, {
        "schema_version": NODE_INVENTORY_SCHEMA,
        "task": "math-stackexchange", "level": "R2",
        "protocol_id": "r2-focused-operational-family-v2.1",
        "parent_partition_sha256": hashlib.sha256(b"parent").hexdigest(),
        "nodes": nodes,
    })
    return path


def test_exhaustive_retrieval_and_calibrated_candidate_build(tmp_path):
    inventory = _inventory(tmp_path)
    pairs = tmp_path / "pairs.jsonl"
    retrieval = prepare_pairs(
        inventory_path=inventory, pairs_path=pairs, report_path=tmp_path / "retrieval.json",
        exhaustive_limit=10,
    )
    assert retrieval["n_pairs"] == 6
    inputs = [json.loads(line) for line in pairs.read_text().splitlines()]
    adapter_sha = hashlib.sha256(b"adapter").hexdigest()
    protocol_sha = hashlib.sha256(b"protocol").hexdigest()
    outputs = []
    for row in inputs:
        same = set((row["node_a"], row["node_b"])) in ({"a", "b"}, {"c", "d"})
        probabilities = {"DIFFERENT": 0.05 if same else 0.8,
                         "RELATED": 0.05 if same else 0.1,
                         "SAME": 0.9 if same else 0.1}
        outputs.append({
            "schema_version": PAIR_OUTPUT_SCHEMA,
            "pair_id": row["pair_id"], "task": row["task"], "level": row["level"],
            "protocol_id": row["protocol_id"], "input_sha256": pair_input_sha256(row),
            "prediction": "SAME" if same else "DIFFERENT", "probabilities": probabilities,
            "order_views": {
                "ab": {"prediction": "SAME" if same else "DIFFERENT", "probabilities": probabilities},
                "ba": {"prediction": "SAME" if same else "DIFFERENT", "probabilities": probabilities},
            },
            "order_consistent": True, "adapter_sha256": adapter_sha,
            "protocol_sha256": protocol_sha,
        })
    output_path = tmp_path / "scores.jsonl"
    output_path.write_text("".join(json.dumps(row) + "\n" for row in outputs))
    calibration = tmp_path / "calibration.json"
    _write_json(calibration, {
        "schema_version": "gemma4-similarity-threshold-calibration-v1",
        "certified": True, "selected_same_threshold": 0.6, "level": "R2",
        "protocol_id": "r2-focused-operational-family-v2.1",
        "adapter_sha256": adapter_sha, "protocol_sha256": protocol_sha,
        "selected_related_weight": 0.0,
    })
    partition_path = tmp_path / "partition_math_R2_gemma_candidate.json"
    report = build_partition(
        inventory_path=inventory, pair_inputs_path=pairs, pair_outputs_path=output_path,
        calibration_path=calibration, partition_path=partition_path,
        report_path=tmp_path / "build.json",
    )
    assignment = json.loads(partition_path.read_text())["partition"]
    assert assignment["a"] == assignment["b"]
    assert assignment["c"] == assignment["d"]
    assert assignment["a"] != assignment["c"]
    assert report["groups"]["n"] == 2
    assert report["graph"]["same_edges"] == 2


def test_build_rejects_uncertified_threshold_and_canonical_name(tmp_path):
    inventory = _inventory(tmp_path)
    pairs = tmp_path / "pairs.jsonl"
    prepare_pairs(inventory_path=inventory, pairs_path=pairs,
                  report_path=tmp_path / "retrieval.json", exhaustive_limit=10)
    # Contract validation reaches the threshold only with an aligned output file; copying no rows
    # deliberately fails earlier and still demonstrates fail-closed behavior.
    outputs = tmp_path / "scores.jsonl"
    outputs.write_text("")
    calibration = tmp_path / "calibration.json"
    _write_json(calibration, {
        "schema_version": "gemma4-similarity-threshold-calibration-v1",
        "certified": False, "selected_same_threshold": 0.5, "level": "R2",
        "protocol_id": "r2-focused-operational-family-v2.1",
        "adapter_sha256": hashlib.sha256(b"adapter").hexdigest(),
        "protocol_sha256": hashlib.sha256(b"protocol").hexdigest(),
        "selected_related_weight": 0.0,
    })
    with pytest.raises(ValueError):
        build_partition(
            inventory_path=inventory, pair_inputs_path=pairs, pair_outputs_path=outputs,
            calibration_path=calibration,
            partition_path=tmp_path / "partition_math-stackexchange_R2.json",
            report_path=tmp_path / "build.json",
        )


def test_build_rejects_pair_text_or_hash_drift_from_inventory(tmp_path):
    inventory = _inventory(tmp_path)
    pairs = tmp_path / "pairs.jsonl"
    prepare_pairs(inventory_path=inventory, pairs_path=pairs,
                  report_path=tmp_path / "retrieval.json", exhaustive_limit=10)
    inputs = [json.loads(line) for line in pairs.read_text().splitlines()]
    inputs[0]["text_a"] = "A different semantic representation"
    pairs.write_text("".join(json.dumps(row) + "\n" for row in inputs))
    adapter_sha = hashlib.sha256(b"adapter").hexdigest()
    protocol_sha = hashlib.sha256(b"protocol").hexdigest()
    outputs = []
    for row in inputs:
        probabilities = {"DIFFERENT": 0.8, "RELATED": 0.1, "SAME": 0.1}
        outputs.append({
            "schema_version": PAIR_OUTPUT_SCHEMA, "pair_id": row["pair_id"],
            "task": row["task"], "level": row["level"], "protocol_id": row["protocol_id"],
            "input_sha256": pair_input_sha256(row), "prediction": "DIFFERENT",
            "probabilities": probabilities,
            "order_views": {
                "ab": {"prediction": "DIFFERENT", "probabilities": probabilities},
                "ba": {"prediction": "DIFFERENT", "probabilities": probabilities},
            },
            "order_consistent": True, "adapter_sha256": adapter_sha,
            "protocol_sha256": protocol_sha,
        })
    output_path = tmp_path / "scores.jsonl"
    output_path.write_text("".join(json.dumps(row) + "\n" for row in outputs))
    calibration = tmp_path / "calibration.json"
    _write_json(calibration, {
        "schema_version": "gemma4-similarity-threshold-calibration-v1", "certified": True,
        "selected_same_threshold": 0.6, "selected_related_weight": 0.0, "level": "R2",
        "protocol_id": "r2-focused-operational-family-v2.1",
        "adapter_sha256": adapter_sha, "protocol_sha256": protocol_sha,
    })
    with pytest.raises(ValueError, match="text differs from inventory"):
        build_partition(
            inventory_path=inventory, pair_inputs_path=pairs, pair_outputs_path=output_path,
            calibration_path=calibration, partition_path=tmp_path / "candidate.json",
            report_path=tmp_path / "build.json",
        )
