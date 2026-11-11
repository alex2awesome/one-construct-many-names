from __future__ import annotations

import argparse
import json

from methods.codability.lexicon_distill.evaluate_similarity_lora import (
    cohen_kappa,
    compare_variants,
    metrics,
)


def _row(truth: int, prediction: int, consistent: bool = True) -> dict:
    probabilities = [0.0, 0.0, 0.0]
    probabilities[prediction] = 1.0
    target = [0.0, 0.0, 0.0]
    target[truth] = 1.0
    return {
        "truth": truth,
        "prediction": prediction,
        "probabilities": probabilities,
        "target_probs": target,
        "order_consistent": consistent,
    }


def test_metrics_perfect_predictions() -> None:
    rows = [_row(0, 0), _row(1, 1), _row(2, 2)]
    result = metrics(rows)
    assert result["accuracy"] == 1.0
    assert result["macro_f1"] == 1.0
    assert result["cohen_kappa"] == 1.0
    assert result["brier"] == 0.0


def test_kappa_penalizes_constant_prediction() -> None:
    assert cohen_kappa([0, 1, 2], [1, 1, 1]) == 0.0


def test_paired_variant_comparison_reports_supported_improvement(tmp_path) -> None:
    reference_path = tmp_path / "reference.jsonl"
    candidate_path = tmp_path / "candidate.jsonl"
    reference_rows = []
    candidate_rows = []
    for index in range(120):
        truth = index % 3
        base = {"example_id": str(index), **_row(truth, 0)}
        reference_rows.append(base)
        candidate_rows.append({"example_id": str(index), **_row(truth, truth)})
    reference_path.write_text(
        "".join(json.dumps(row) + "\n" for row in reference_rows), encoding="utf-8",
    )
    candidate_path.write_text(
        "".join(json.dumps(row) + "\n" for row in candidate_rows), encoding="utf-8",
    )
    report_path = tmp_path / "report.json"
    args = argparse.Namespace(
        reference_predictions=str(reference_path),
        candidate_predictions=str(candidate_path),
        reference_label="base", candidate_label="full",
        report=str(report_path), bootstrap_samples=50, seed=17,
    )

    compare_variants(args)

    report = json.loads(report_path.read_text())
    assert report["improvement_gate"]["supported"] is True
    assert report["delta"]["cohen_kappa"] == 1.0
    assert report["paired_bootstrap_95ci"]["macro_f1"][0] > 0
