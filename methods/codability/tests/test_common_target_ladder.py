"""Tests for fixed-target ladder validation and censoring discipline."""

import copy
import hashlib
import json

import pytest

from methods.codability.experiments.common_target_ladder import (
    build_ladder_report,
    build_policy_executor_ladder,
    build_policy_executor_response_surface,
    validate_common_target,
)


def _artifact(small, big, target="8B", *, substitution=False):
    row = {
        "domain": "taste", "gi": 0,
        "target": {"target_id": f"name:taste:0:{target}"},
        "probe_split": {"seed": 4, "n_heldout": 100},
        "heldout": {
            "valid": True,
            "gates": {
                "baseline_gap_confirmed": True,
                "articulation_improvement_confirmed": True,
                "noninferior_to_big_sparse": substitution,
                "equivalent_to_big_sparse": substitution,
                "signature_improved": True,
                "signature_noninferior_to_big": substitution,
            },
            "methodological_substitution": substitution,
            "equivalent_methodological_substitution": substitution,
        },
    }
    return {
        "schema": "fixed_target_name_substitution/v2",
        "config": {"small_tag": small, "big_tag": big, "target_tag": target},
        "by_domain": {"taste": {"per_metric": [row],
                                  "inputs": {"target_grid": {"sha256": "a" * 64}}}},
    }


def test_validator_rejects_pairwise_moving_targets():
    with pytest.raises(ValueError, match="common target tag"):
        validate_common_target([_artifact("1B", "3B", target="3B"),
                                _artifact("3B", "8B", target="8B")])


def test_ladder_keeps_potential_censored_and_word_cost_ineligible():
    report = build_ladder_report([_artifact("1B", "3B"), _artifact("3B", "8B")],
                                 labels=["1_to_3", "3_to_8"])
    assert report["validation"]["valid"]
    assert report["hops"]["1_to_3"]["debt_status"]["right_censored_within_bank"] == 1
    assert not report["potential_test"]["triangle_evaluable"]
    assert any("not composable" in reason for reason in report["potential_test"]["reasons"])


def test_validator_rejects_metric_level_target_drift():
    left, right = _artifact("1B", "3B"), _artifact("3B", "8B")
    right = copy.deepcopy(right)
    right["by_domain"]["taste"]["per_metric"][0]["target"]["target_id"] = "other"
    with pytest.raises(ValueError, match="validation failed"):
        validate_common_target([left, right])


def test_common_target_rows_do_not_collapse_same_gi_across_hierarchy_levels():
    left = _artifact("1B", "3B")
    right = _artifact("3B", "8B")
    rows = []
    for level in ("R1", "R2", "R3"):
        row = copy.deepcopy(left["by_domain"]["taste"]["per_metric"][0])
        row.update({
            "cell_id": f"TB::taste::{level}::node",
            "task": "taste",
            "level": level,
            "bucket": "general",
            "node_id": f"taste::{level}::node",
            "metric_id": f"taste::{level}::node",
            "gi": 7,
        })
        row["target"]["target_id"] = f"target::{level}"
        rows.append(row)
    left["by_domain"]["taste"]["per_metric"] = copy.deepcopy(rows)
    right["by_domain"]["taste"]["per_metric"] = copy.deepcopy(rows)

    result = validate_common_target([left, right])

    assert result["n_common_cells"] == 3
    assert result["common_cells_by_domain"] == {"taste": 3}


def _policy_crossfold(
        tmp_path, label, *, capacity, target_prefix="target", bank="bank",
        target_arm_id="target", big_job="70b", readout="readout",
        binary_readout=None,
        mae_margin=0.02, functional_rho_floor=0.7,
        source_schema="policy_isomorphism_experiment/v4",
        crossfold_schema="crossfold_policy_isomorphism_fibers/v4"):
    references = []
    fold_rows = []
    for partition in ("residual_prompt_selection", "residual_unit_certification"):
        source = {
            "schema": source_schema,
            "partition": partition,
            "arm_bank_sha256": bank,
            "config": {
                "small_job": label,
                "big_job": big_job,
                "target_arm_id": target_arm_id,
                "mae_margin": mae_margin,
                "rho_margin": 0.05,
                "flip_margin": 0.02,
                "bias_margin": 0.02,
                "functional_rho_floor": functional_rho_floor,
            },
            "cells": [{
                "cell_id": "cell", "domain": "humor", "gi": 49,
                "construct": "wordplay", "target_job": big_job,
                "small_job": label, "n_items": 20,
                "target_shards": [f"{target_prefix}-{partition}"],
                "executor_prompt_bank_validation": {"valid": True},
                "target_readout_template_sha256": readout,
                "small_readout_template_sha256": readout,
                "rows": [{"arm_id": "definition", "components": ["definition"],
                           "certificate": {
                    "small_sparse_point": {"candidate_robust": {
                        "spearman": 0.3, "mae_tvd": 0.4,
                    }},
                    "point": {"target_self_robust": {
                        "spearman": 0.95, "mae_tvd": 0.1,
                    }},
                }}],
            }],
        }
        if binary_readout is not None:
            source["cells"][0]["target_binary_readout"] = binary_readout
            source["cells"][0]["small_binary_readout"] = binary_readout
        source_path = tmp_path / f"{label}-{partition}.json"
        source_path.write_text(json.dumps(source))
        references.append({
            "path": str(source_path),
            "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "partition": partition,
        })
        fold_rows.append({
            "partition": partition,
            "adverse_rho_point": capacity,
            "adverse_rho_CI": [capacity - 0.1, capacity + 0.1],
            "adverse_mae_tvd": 0.2,
            "mae_gain_over_small_sparse": 0.2,
            "observed_max_rho_floor": capacity,
            "certified_max_rho_floor": capacity - 0.1,
            "observed_base_gates_pass": True,
            "certified_base_gates_pass": True,
        })
    observed = ["definition"] if capacity >= functional_rho_floor else []
    crossfold = {
        "schema": crossfold_schema,
        "arm_bank_sha256": bank,
        "functional_rho_floor": functional_rho_floor,
        "reports": references,
        "cells": [{
            "cell_id": "cell", "domain": "humor", "construct": "wordplay",
            "common_arms": ["definition"],
            "functional_capacity_by_arm": [{
                "arm_id": "definition", "components": ["definition"],
                "stable_observed_max_rho_floor": capacity,
                "stable_certified_max_rho_floor": capacity - 0.1,
                "folds": fold_rows,
            }],
            "functional_floor_profile": [{
                "rho_floor": functional_rho_floor,
                "observed_functional_members": observed,
                "observed_component_minimal_members": observed,
            }],
        }],
    }
    path = tmp_path / f"{label}-crossfold.json"
    path.write_text(json.dumps(crossfold))
    return path


def _rewrite_source(crossfold_path, partition, mutate):
    crossfold = json.loads(crossfold_path.read_text())
    reference = next(
        row for row in crossfold["reports"] if row["partition"] == partition
    )
    source_path = type(crossfold_path)(reference["path"])
    source = json.loads(source_path.read_text())
    mutate(source)
    source_path.write_text(json.dumps(source))
    reference["sha256"] = hashlib.sha256(source_path.read_bytes()).hexdigest()
    crossfold_path.write_text(json.dumps(crossfold))


def _rewrite_crossfold(crossfold_path, mutate):
    crossfold = json.loads(crossfold_path.read_text())
    mutate(crossfold)
    crossfold_path.write_text(json.dumps(crossfold))


def _expand_policy_crossfold_to_three_levels(crossfold_path):
    crossfold = json.loads(crossfold_path.read_text())
    for reference in crossfold["reports"]:
        source_path = type(crossfold_path)(reference["path"])
        source = json.loads(source_path.read_text())
        base = source["cells"][0]
        source["cells"] = []
        for level in ("R1", "R2", "R3"):
            cell = copy.deepcopy(base)
            cell.update({
                "cell_id": f"TB::humor::{level}::node",
                "task": "humor",
                "level": level,
                "bucket": "general",
                "node_id": f"humor::{level}::node",
                "metric_id": f"humor::{level}::node",
                "gi": 49,
                "construct": f"wordplay {level}",
            })
            source["cells"].append(cell)
        source_path.write_text(json.dumps(source))
        reference["sha256"] = hashlib.sha256(source_path.read_bytes()).hexdigest()
    base = crossfold["cells"][0]
    crossfold["cells"] = []
    for level in ("R1", "R2", "R3"):
        cell = copy.deepcopy(base)
        cell.update({
            "cell_id": f"TB::humor::{level}::node",
            "task": "humor",
            "level": level,
            "bucket": "general",
            "node_id": f"humor::{level}::node",
            "metric_id": f"humor::{level}::node",
            "gi": 49,
            "construct": f"wordplay {level}",
        })
        crossfold["cells"].append(cell)
    crossfold_path.write_text(json.dumps(crossfold))


def test_policy_executor_ladder_requires_and_summarizes_fixed_target(tmp_path):
    one = _policy_crossfold(tmp_path, "1b", capacity=0.3)
    eight = _policy_crossfold(tmp_path, "8b", capacity=0.75)
    result = build_policy_executor_ladder([one, eight], labels=["1B", "8B"])
    assert result["schema"] == "fixed_target_policy_executor_ladder/v4"
    assert result["validation"]["same_target_shards"]
    assert result["validation"]["source_reports_bound_to_crossfold_bank"]
    assert result["validation"]["target_arm_id"] == "target"
    cell = result["cells"][0]
    assert cell["executors"]["1B"][
        "best_observed_worst_fold_functional_rank_capacity"] == 0.3
    assert cell["executors"]["8B"]["observed_functional_members_at_floor"] == [
        "definition"]
    assert cell["per_arm"][0]["executors"]["8B"][
        "descriptive_observed_worst_fold_mae_gain_over_name"] == 0.2
    assert "confidence-bound" in result["gap_closure_scope"]


def test_policy_executor_ladder_preserves_three_levels_with_same_legacy_gi(tmp_path):
    one = _policy_crossfold(tmp_path, "1b", capacity=0.3)
    eight = _policy_crossfold(tmp_path, "8b", capacity=0.75)
    _expand_policy_crossfold_to_three_levels(one)
    _expand_policy_crossfold_to_three_levels(eight)

    result = build_policy_executor_ladder([one, eight], labels=["1B", "8B"])

    assert result["validation"]["n_cells"] == 3
    assert {cell["level"] for cell in result["cells"]} == {"R1", "R2", "R3"}
    assert {cell["gi"] for cell in result["cells"]} == {49}
    assert len({cell["cell_id"] for cell in result["cells"]}) == 3
    assert len({cell["node_id"] for cell in result["cells"]}) == 3


def test_policy_ladder_readers_accept_v5_isomorphism_reports(tmp_path):
    crossfolds = [
        _policy_crossfold(
            tmp_path, label, capacity=capacity,
            source_schema="policy_isomorphism_experiment/v5",
            crossfold_schema="crossfold_policy_isomorphism_fibers/v5",
        )
        for label, capacity in (("1b", 0.3), ("8b", 0.75))
    ]
    assert build_policy_executor_ladder(crossfolds)["validation"][
        "same_target_shards"]

    folds = [
        _policy_fold(
            tmp_path, label, sparse_rho=sparse_rho, sparse_mae=0.5,
            candidate_rho=candidate_rho, candidate_mae=0.4,
            schema="policy_isomorphism_experiment/v5",
        )
        for label, sparse_rho, candidate_rho in (
            ("1b-response", 0.2, 0.3),
            ("8b-response", 0.6, 0.7),
        )
    ]
    assert build_policy_executor_response_surface(folds)["validation"][
        "executor_order_is_caller_declared"]


def test_policy_executor_ladder_rejects_target_shard_drift(tmp_path):
    one = _policy_crossfold(tmp_path, "1b", capacity=0.3)
    eight = _policy_crossfold(
        tmp_path, "8b", capacity=0.75, target_prefix="different-target")
    with pytest.raises(ValueError, match="target shard identity"):
        build_policy_executor_ladder([one, eight])


def test_policy_executor_ladder_binds_each_source_to_crossfold_bank(tmp_path):
    one = _policy_crossfold(tmp_path, "1b", capacity=0.3)
    eight = _policy_crossfold(tmp_path, "8b", capacity=0.75)
    _rewrite_source(
        eight,
        "residual_unit_certification",
        lambda source: source.update(arm_bank_sha256="different-bank"),
    )
    with pytest.raises(ValueError, match="source/crossfold arm bank identity"):
        build_policy_executor_ladder([one, eight])


def test_policy_executor_ladder_rejects_crossfold_bank_drift(tmp_path):
    one = _policy_crossfold(tmp_path, "1b", capacity=0.3, bank="bank-one")
    eight = _policy_crossfold(tmp_path, "8b", capacity=0.75, bank="bank-eight")
    with pytest.raises(ValueError, match="arm banks differ"):
        build_policy_executor_ladder([one, eight])


def test_policy_executor_ladder_rejects_target_arm_drift_within_entry(tmp_path):
    one = _policy_crossfold(tmp_path, "1b", capacity=0.3)
    eight = _policy_crossfold(tmp_path, "8b", capacity=0.75)
    _rewrite_source(
        eight,
        "residual_unit_certification",
        lambda source: source["config"].update(target_arm_id="name"),
    )
    with pytest.raises(ValueError, match="mixes config 'target_arm_id'"):
        build_policy_executor_ladder([one, eight])


def test_policy_executor_ladder_rejects_target_arm_drift_between_entries(tmp_path):
    one = _policy_crossfold(tmp_path, "1b", capacity=0.3, target_arm_id="name")
    eight = _policy_crossfold(tmp_path, "8b", capacity=0.75, target_arm_id="target")
    with pytest.raises(ValueError, match="config 'target_arm_id' differs"):
        build_policy_executor_ladder([one, eight])


@pytest.mark.parametrize("kwargs", [
    {"big_job": "other-target"},
    {"mae_margin": 0.03},
    {"functional_rho_floor": 0.75},
])
def test_policy_executor_ladder_rejects_fixed_config_drift_between_entries(
        tmp_path, kwargs):
    one = _policy_crossfold(tmp_path, "1b", capacity=0.3)
    eight = _policy_crossfold(tmp_path, "8b", capacity=0.75, **kwargs)
    changed_key = next(iter(kwargs))
    with pytest.raises(ValueError, match=rf"config '{changed_key}' differs"):
        build_policy_executor_ladder([one, eight])


@pytest.mark.parametrize("key", [
    "big_job", "mae_margin", "rho_margin", "flip_margin", "bias_margin",
    "functional_rho_floor",
])
def test_policy_executor_ladder_rejects_fixed_config_drift(tmp_path, key):
    one = _policy_crossfold(tmp_path, "1b", capacity=0.3)
    eight = _policy_crossfold(tmp_path, "8b", capacity=0.75)
    replacement = "other-target" if key == "big_job" else 0.123
    _rewrite_source(
        eight,
        "residual_unit_certification",
        lambda source: source["config"].update({key: replacement}),
    )
    match = "functional floor identity" if key == "functional_rho_floor" else "mixes config"
    with pytest.raises(ValueError, match=match):
        build_policy_executor_ladder([one, eight])


def test_policy_executor_ladder_rejects_small_job_drift_within_entry(tmp_path):
    one = _policy_crossfold(tmp_path, "1b", capacity=0.3)
    eight = _policy_crossfold(tmp_path, "8b", capacity=0.75)
    _rewrite_source(
        eight,
        "residual_unit_certification",
        lambda source: source["config"].update(small_job="other-executor"),
    )
    with pytest.raises(ValueError, match="mixes executor small_job"):
        build_policy_executor_ladder([one, eight])


def test_policy_executor_ladder_rejects_reference_partition_drift(tmp_path):
    one = _policy_crossfold(tmp_path, "1b", capacity=0.3)
    eight = _policy_crossfold(tmp_path, "8b", capacity=0.75)
    _rewrite_crossfold(
        eight,
        lambda report: report["reports"][0].update(partition="wrong-partition"),
    )
    with pytest.raises(ValueError, match="reference/source partition identity"):
        build_policy_executor_ladder([one, eight])


def test_policy_executor_ladder_rejects_crossfold_source_cell_drift(tmp_path):
    one = _policy_crossfold(tmp_path, "1b", capacity=0.3)
    eight = _policy_crossfold(tmp_path, "8b", capacity=0.75)
    _rewrite_source(
        eight,
        "residual_unit_certification",
        lambda source: source["cells"][0].update(construct="different construct"),
    )
    with pytest.raises(ValueError, match="source cell construct identity"):
        build_policy_executor_ladder([one, eight])


def test_policy_executor_ladder_rejects_readout_drift(tmp_path):
    one = _policy_crossfold(tmp_path, "1b", capacity=0.3, readout="readout-one")
    eight = _policy_crossfold(tmp_path, "8b", capacity=0.75, readout="readout-eight")
    with pytest.raises(ValueError, match="readout templates differ"):
        build_policy_executor_ladder([one, eight])


def test_policy_executor_ladder_rejects_binary_protocol_drift(tmp_path):
    one = _policy_crossfold(
        tmp_path, "1b", capacity=0.3,
        binary_readout="legacy_top20_normalized")
    eight = _policy_crossfold(
        tmp_path, "8b", capacity=0.75,
        binary_readout="teacher_forced_declared_labels")
    with pytest.raises(ValueError, match="binary readout protocols differ"):
        build_policy_executor_ladder([one, eight])


def test_policy_executor_ladder_requires_source_schema(tmp_path):
    one = _policy_crossfold(tmp_path, "1b", capacity=0.3)
    eight = _policy_crossfold(tmp_path, "8b", capacity=0.75)
    _rewrite_source(
        eight,
        "residual_unit_certification",
        lambda source: source.update(schema="unrelated/v1"),
    )
    with pytest.raises(ValueError, match="unexpected source policy schema"):
        build_policy_executor_ladder([one, eight])


def _policy_fold(tmp_path, label, *, sparse_rho, sparse_mae, candidate_rho,
                 candidate_mae, target_shard="fixed-target",
                 schema="policy_isomorphism_experiment/v4"):
    def robust(rho, mae):
        return {
            "spearman": rho, "mae_tvd": mae, "all_positive_polarity": True,
            "binary_flip_rate": 0.2, "absolute_bias": 0.1, "n_forms": 3,
        }
    report = {
        "schema": schema,
        "partition": "residual_prompt_selection",
        "arm_bank_sha256": "fixed-bank",
        "config": {
            "small_job": label, "big_job": "70b", "target_arm_id": "target",
            "mae_margin": 0.02, "rho_margin": 0.05, "flip_margin": 0.02,
            "bias_margin": 0.02, "functional_rho_floor": 0.7,
        },
        "cells": [{
            "cell_id": "cell", "domain": "humor", "gi": 49,
            "construct": "wordplay", "target_job": "70b", "small_job": label,
            "n_items": 200,
            "target_shards": [target_shard],
            "executor_prompt_bank_validation": {"valid": True},
            "target_readout_template_sha256": "readout",
            "small_readout_template_sha256": "readout",
            "rows": [{
                "arm_id": "explanation_rubric",
                "components": ["explanation", "rubric"],
                "certificate": {
                    "small_sparse_point": {
                        "candidate_robust": robust(sparse_rho, sparse_mae),
                        "candidate_forms": {
                            "canonical": robust(sparse_rho, sparse_mae)},
                        "quotient": robust(sparse_rho, sparse_mae),
                    },
                    "point": {
                        "candidate_robust": robust(candidate_rho, candidate_mae),
                        "candidate_forms": {
                            "canonical": robust(candidate_rho, candidate_mae)},
                        "quotient": robust(candidate_rho, candidate_mae),
                        "target_self_robust": robust(0.97, 0.09),
                        "target_self_forms": {
                            "canonical": robust(0.97, 0.09)},
                    },
                },
            }],
        }],
    }
    path = tmp_path / f"{label}-fold.json"
    path.write_text(json.dumps(report))
    return path


def test_policy_executor_response_surface_identifies_point_scale_substitution(tmp_path):
    one = _policy_fold(
        tmp_path, "1b", sparse_rho=0.1, sparse_mae=0.52,
        candidate_rho=0.3, candidate_mae=0.46)
    three = _policy_fold(
        tmp_path, "3b", sparse_rho=0.57, sparse_mae=0.54,
        candidate_rho=0.66, candidate_mae=0.39)
    eight = _policy_fold(
        tmp_path, "8b", sparse_rho=0.63, sparse_mae=0.42,
        candidate_rho=0.75, candidate_mae=0.25)
    result = build_policy_executor_response_surface(
        [one, three, eight], labels=["1B", "3B", "8B"])
    assert result["validation"]["same_prompt_hash_bank"]
    assert result["validation"]["executor_order_is_caller_declared"]
    cell = result["cells"][0]
    three_to_eight = cell["adjacent_steps"][1]
    arm = three_to_eight["per_arm"][0]
    assert arm["adverse_envelope_point_gates"]["rank_mae_point_dominance"]
    assert arm["adverse_envelope_point_gates"]["rank_mae_margin_match"]
    assert arm["matched_form_rank_mae_point_gates"]["all_forms_point_dominance"]
    assert arm["descriptive_step_closure"]["rho"] == pytest.approx(1.5)
    assert arm["descriptive_step_closure"]["mae_tvd"] == pytest.approx(1.25)
    assert three_to_eight["summary"][
        "n_adverse_envelope_rank_mae_point_dominance"] == 1
    assert three_to_eight["summary"]["n_functional_target_reconstructions"] == 0
    assert "point estimates" in result["claim_boundary"]


def test_policy_executor_response_surface_rejects_target_shard_drift(tmp_path):
    one = _policy_fold(
        tmp_path, "1b", sparse_rho=0.1, sparse_mae=0.52,
        candidate_rho=0.3, candidate_mae=0.46)
    three = _policy_fold(
        tmp_path, "3b", sparse_rho=0.57, sparse_mae=0.54,
        candidate_rho=0.66, candidate_mae=0.39, target_shard="changed-target")
    with pytest.raises(ValueError, match="target shards differ"):
        build_policy_executor_response_surface([one, three])
