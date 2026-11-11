import json
import hashlib

import pytest

from methods.codability.lexicon_distill.calibrate_threshold import calibrate


ADAPTER_SHA = hashlib.sha256(b"adapter").hexdigest()
PROTOCOL_SHA = hashlib.sha256(b"protocol").hexdigest()


def _write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _row(index, truth, probabilities, split="pair_dev"):
    return {
        "example_id": str(index), "truth": truth, "prediction": max(range(3), key=probabilities.__getitem__),
        "probabilities": probabilities, "target_probs": [int(i == truth) for i in range(3)],
        "split": split, "order_consistent": True, "level": "R1",
        "protocol_id": "r1-narrow-construct-v1", "task": "math-stackexchange",
    }


def test_calibration_uses_precision_constrained_recall(tmp_path):
    source = tmp_path / "dev.jsonl"
    rows = [
        _row(1, 2, [0.05, 0.05, 0.90]),
        _row(2, 2, [0.10, 0.15, 0.75]),
        _row(3, 2, [0.20, 0.25, 0.55]),
        _row(4, 0, [0.40, 0.10, 0.50]),
        _row(5, 1, [0.10, 0.55, 0.35]),
        _row(6, 0, [0.70, 0.10, 0.20]),
    ]
    _write(source, rows)
    report = calibrate(
        source, tmp_path / "report.json", target_precision=0.70, minimum_recall=0.50,
        adapter_sha256=ADAPTER_SHA, protocol_sha256=PROTOCOL_SHA,
    )
    assert report["certified"] is True
    assert report["selected_same_threshold"] == 0.55
    assert report["selected_metrics"]["same_precision"] == 1
    assert report["selected_metrics"]["same_recall"] == 1
    assert report["adapter_sha256"] == ADAPTER_SHA
    assert report["protocol_sha256"] == PROTOCOL_SHA


def test_calibration_without_model_lineage_is_diagnostic_only(tmp_path):
    source = tmp_path / "dev.jsonl"
    _write(source, [_row(1, 2, [0.1, 0.1, 0.8]), _row(2, 0, [0.8, 0.1, 0.1])])
    report = calibrate(source, tmp_path / "report.json")
    assert report["certified"] is False
    assert "lineage" in report["reason"]


def test_calibration_derives_runtime_lineage_from_frozen_files(tmp_path):
    source = tmp_path / "dev.jsonl"
    _write(source, [_row(1, 2, [0.1, 0.1, 0.8]), _row(2, 0, [0.8, 0.1, 0.1])])
    adapter = tmp_path / "adapter_model.safetensors"
    adapter.write_bytes(b"adapter")
    protocols = tmp_path / "protocols.json"
    protocol_text = "Judge narrow construct similarity."
    protocols.write_text(json.dumps({
        "r1-narrow-construct-v1": {
            "text": protocol_text,
            "sha256": hashlib.sha256(protocol_text.encode()).hexdigest(),
        }
    }))
    report = calibrate(
        source, tmp_path / "report.json", adapter_file=adapter, protocols_path=protocols,
    )
    assert report["certified"] is True
    assert report["adapter_sha256"] == hashlib.sha256(b"adapter").hexdigest()
    assert report["protocol_sha256"] == hashlib.sha256(protocol_text.encode()).hexdigest()


def test_calibration_rejects_test_data_and_marks_impossible_floor(tmp_path):
    source = tmp_path / "test.jsonl"
    _write(source, [_row(1, 2, [0.1, 0.1, 0.8], split="pair_test")])
    with pytest.raises(ValueError, match="development splits only"):
        calibrate(source, tmp_path / "report.json")

    source = tmp_path / "dev.jsonl"
    _write(source, [_row(1, 2, [0.4, 0.4, 0.2]), _row(2, 0, [0.1, 0.1, 0.8])])
    report = calibrate(source, tmp_path / "uncertified.json", target_precision=1.0, minimum_recall=1.0)
    assert report["certified"] is False
