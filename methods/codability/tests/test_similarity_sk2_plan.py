from __future__ import annotations

import json
from pathlib import Path

from methods.codability.lexicon_distill import freeze_sk2_jobs


def test_frozen_plan_uses_only_sk2_paths(tmp_path: Path, monkeypatch) -> None:
    inventory = tmp_path / "inventory.json"
    manifest = tmp_path / "manifest.json"
    inventory.write_text(json.dumps({"powered_cells": []}), encoding="utf-8")
    manifest.write_text("{}", encoding="utf-8")
    (tmp_path / "sk2_model_inventory.json").write_text("{}", encoding="utf-8")
    output = tmp_path / "jobs.json"
    monkeypatch.setattr(
        "sys.argv",
        ["freeze", "--inventory", str(inventory), "--dataset-manifest", str(manifest),
         "--output", str(output)],
    )
    freeze_sk2_jobs.main()
    plan = json.loads(output.read_text())
    assert plan["sk3_forbidden"] is True
    assert "gpuhost" in plan["model"]
    assert len(plan["model_inventory"]["sha256"]) == 64
    assert set(plan["implementation_files"]) == {
        "methods/codability/lexicon_distill/dataset.py",
        "methods/codability/lexicon_distill/train_gemma4_similarity_lora.py",
        "methods/codability/lexicon_distill/evaluate_similarity_lora.py",
        "methods/codability/lexicon_distill/calibrate_threshold.py",
        "methods/codability/lexicon_distill/hierarchy_contracts.py",
        "methods/codability/lexicon_distill/score_hierarchy_pairs.py",
        "methods/codability/lexicon_distill/build_hierarchy_candidate.py",
        "methods/codability/lexicon_distill/frontier_calibration.py",
        "methods/codability/lexicon_distill/freeze_sk2_jobs.py",
        "methods/codability/lexicon_distill/run_sk2_jobs.py",
    }
    assert all(len(reference["sha256"]) == 64 for reference in plan["implementation_files"].values())
    assert all("gpuhost" not in " ".join(job["argv"]) for job in plan["jobs"])
    assert {job["job_id"] for job in plan["jobs"]}.issuperset({"preflight_R1", "preflight_R2", "preflight_R3"})
    jobs = {job["job_id"]: job for job in plan["jobs"]}
    # The empty fixture has no auxiliary rows, so the headline adapter is a
    # direct primary-family fit and no meaningless ablation is scheduled.
    assert jobs["pooled_R1_full"]["depends_on"] == ["preflight_R1"]
    assert "--primary-only" in jobs["pooled_R1_full"]["argv"]
    assert "--init-adapter" not in jobs["pooled_R1_full"]["argv"]
    batch_index = jobs["pooled_R1_full"]["argv"].index("--batch-size")
    accumulation_index = jobs["pooled_R1_full"]["argv"].index("--gradient-accumulation-steps")
    assert jobs["pooled_R1_full"]["argv"][batch_index + 1] == "8"
    assert jobs["pooled_R1_full"]["argv"][accumulation_index + 1] == "2"
    learning_rate_index = jobs["pooled_R1_full"]["argv"].index("--learning-rate")
    assert jobs["pooled_R1_full"]["argv"][learning_rate_index + 1] == "2e-5"
    assert jobs["pooled_R1_full"]["gpu"] == 0
    assert jobs["eval_R1_base"]["gpu"] == 2
    assert jobs["eval_R2_base"]["gpu"] == 3
    assert jobs["eval_R1_full_dev"]["depends_on"] == ["pooled_R1_full"]
    assert jobs["calibrate_R1_full"]["depends_on"] == ["eval_R1_full_dev"]
    assert "calibrate_R2_full" not in jobs  # focused R2-v2.1 needs its own fresh calibration panel


def test_r1_primary_ablation_uses_parallel_training_lane(tmp_path: Path, monkeypatch) -> None:
    inventory = tmp_path / "inventory.json"
    manifest = tmp_path / "manifest.json"
    inventory.write_text(json.dumps({"powered_cells": []}), encoding="utf-8")
    manifest.write_text("{}", encoding="utf-8")
    (tmp_path / "sk2_model_inventory.json").write_text("{}", encoding="utf-8")
    (tmp_path / "R1_train.jsonl").write_text(
        json.dumps({"family_distributions": {"sonnet": [0, 0, 1], "opus": [0, 1, 0]}}) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "jobs.json"
    monkeypatch.setattr(
        "sys.argv",
        ["freeze", "--inventory", str(inventory), "--dataset-manifest", str(manifest),
         "--output", str(output), "--include-r1-auxiliary"],
    )

    freeze_sk2_jobs.main()

    jobs = {job["job_id"]: job for job in json.loads(output.read_text())["jobs"]}
    assert jobs["pooled_R1_auxiliary"]["gpu"] == 0
    assert jobs["pooled_R1_full"]["gpu"] == 0
    assert jobs["pooled_R1_primary"]["gpu"] == 6
    assert jobs["pooled_R1_full"]["depends_on"] == ["pooled_R1_auxiliary"]
    assert jobs["pooled_R1_primary"]["depends_on"] == ["preflight_R1", "pooled_R3_full"]
    cap_index = jobs["pooled_R1_primary"]["argv"].index("--max-nonfinite-fraction")
    assert jobs["pooled_R1_primary"]["argv"][cap_index + 1] == "0.05"
    auxiliary_rates = [
        jobs["pooled_R1_auxiliary"]["argv"][index + 1]
        for index, value in enumerate(jobs["pooled_R1_auxiliary"]["argv"])
        if value == "--learning-rate"
    ]
    assert auxiliary_rates[-1] == "5e-6"


def test_underpowered_trainable_cells_get_descriptive_task_comparisons(
    tmp_path: Path, monkeypatch,
) -> None:
    inventory = tmp_path / "inventory.json"
    manifest = tmp_path / "manifest.json"
    inventory.write_text(
        json.dumps(
            {
                "powered_cells": [
                    {
                        "task": "math-stackexchange", "level": "R2",
                        "weighted_train_pairs": 1500, "test_pairs": 900,
                        "test_same": 100, "powered": True,
                    },
                    {
                        "task": "humor", "level": "R2",
                        "weighted_train_pairs": 500, "test_pairs": 900,
                        "test_same": 100, "powered": False,
                    },
                    {
                        "task": "peer-review", "level": "R1",
                        "weighted_train_pairs": 0, "test_pairs": 900,
                        "test_same": 100, "powered": False,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    manifest.write_text("{}", encoding="utf-8")
    (tmp_path / "sk2_model_inventory.json").write_text("{}", encoding="utf-8")
    output = tmp_path / "jobs.json"
    monkeypatch.setattr(
        "sys.argv",
        ["freeze", "--inventory", str(inventory), "--dataset-manifest", str(manifest), "--output", str(output)],
    )

    freeze_sk2_jobs.main()

    plan = json.loads(output.read_text())
    jobs = {job["job_id"]: job for job in plan["jobs"]}
    assert len(plan["powered_cells"]) == 1
    assert len(plan["task_cells"]) == 2
    assert "task_math_stackexchange_R2" in jobs
    assert "task_humor_R2" in jobs
    assert "task_peer_review_R1" not in jobs
    assert "--descriptive-only" not in jobs["compare_math_stackexchange_R2"]["argv"]
    assert "--descriptive-only" in jobs["compare_humor_R2"]["argv"]


def test_r1_task_work_uses_primary_only_headline_when_auxiliary_fails_independently(
    tmp_path: Path, monkeypatch,
) -> None:
    inventory = tmp_path / "inventory.json"
    manifest = tmp_path / "manifest.json"
    inventory.write_text(
        json.dumps({"powered_cells": [{
            "task": "math-stackexchange", "level": "R1", "weighted_train_pairs": 1000,
            "test_pairs": 900, "test_same": 100, "powered": True,
        }]}), encoding="utf-8")
    manifest.write_text("{}", encoding="utf-8")
    (tmp_path / "sk2_model_inventory.json").write_text("{}", encoding="utf-8")
    (tmp_path / "R1_train.jsonl").write_text(
        json.dumps({"family_distributions": {"sonnet": [0, 0, 1], "opus": [0, 1, 0]}}) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "jobs.json"
    monkeypatch.setattr(
        "sys.argv",
        ["freeze", "--inventory", str(inventory), "--dataset-manifest", str(manifest),
         "--output", str(output)],
    )

    freeze_sk2_jobs.main()

    plan = json.loads(output.read_text())
    jobs = {job["job_id"]: job for job in plan["jobs"]}
    assert plan["headline_variant_by_level"]["R1"] == "primary"
    assert "compare_R1_full_vs_base" not in jobs
    assert "compare_R1_primary_vs_base" in jobs
    assert all(
        dependency in jobs
        for job in jobs.values()
        for dependency in job["depends_on"]
    )
    assert jobs["task_math_stackexchange_R1"]["depends_on"] == ["pooled_R1_primary"]
    assert jobs["compare_math_stackexchange_R1"]["depends_on"] == [
        "eval_math_stackexchange_R1_task", "eval_R1_primary"]
    pooled_index = jobs["compare_math_stackexchange_R1"]["argv"].index("--pooled-predictions")
    assert jobs["compare_math_stackexchange_R1"]["argv"][pooled_index + 1].endswith(
        "eval_R1_primary.jsonl")
