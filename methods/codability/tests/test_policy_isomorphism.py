"""Direct policy transplantation and articulation-fiber tests."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest

from methods.codability.experiments import run_policy_isomorphism as policy_runner

from methods.codability.experiments.policy_isomorphism import (
    _bootstrap_orbit,
    _bootstrap_samples,
    articulation_distance,
    certify_pairwise_policy_fidelity,
    certify_policy_isomorphism,
    certify_scale_step_substitution,
    compare_articulation_to_matched_control,
    oracle_mean_shift_diagnostic,
    summarize_isomorphism_fiber,
)
from methods.codability.experiments.policy_data import (
    load_partition_source_groups,
    validate_executor_prompt_arms,
)
from methods.codability.experiments.build_fresh_item_partitions import (
    sha256_bytes,
    sha256_file,
    text_sha256,
)
from methods.codability.grid_auc_report import _rank, spearman
from methods.codability.experiments.run_policy_isomorphism import (
    _BREADTH_BINARY_OUTCOMES,
    _BREADTH_COORDINATES,
    _bootstrap_metric_estimates,
    _breadth_cell_decomposition,
    _content_specific_joint_fiber,
    _content_specific_scale_memberships,
    _identity_payload,
    _resolve_scoring_cell_ids,
    _validate_scored_breadth_identity,
    _validate_frozen_runner_invocation,
    _validate_frozen_scored_arm_panel,
    _validate_frozen_score_bundle,
    SOURCE_HIERARCHY_PROVENANCES,
    build_policy_articulation_selection,
    pool_crossfold_policy_reports,
    run,
    summarize_breadth_decomposition,
    summarize_crossfold_fibers,
    write_calibration_release_artifact,
    write_policy_articulation_selection,
)
from methods.codability.experiments.common_target_ladder import (
    validate_policy_cell_panel,
)


def _arm(arm_id, channel, text):
    return {"id": arm_id, "channel": channel, "provenance": "source_telling",
            "components": [arm_id],
            "forms": [{"id": "canonical", "prompt": text}],
            "semantic_content_word_count": len(text.split())}


def test_oracle_mean_shift_is_crossfit_one_scalar_diagnostic_only():
    q = np.linspace(0.15, 0.85, 40)
    target = {
        "canonical": q,
        "question": np.clip(q + 0.01, 0.0, 1.0),
        "boilerplate": np.clip(q - 0.01, 0.0, 1.0),
    }
    name = {form: np.clip(values - 0.20, 0.0, 1.0)
            for form, values in target.items()}
    articulation = {form: np.clip(values + 0.03 * np.sin(np.arange(40)), 0.0, 1.0)
                    for form, values in target.items()}
    report = oracle_mean_shift_diagnostic(
        target,
        {"name": name, "definition": articulation},
        item_hashes=[f"{index:064x}" for index in range(40)],
        n_boot=100,
        seed=17,
        confidence=0.95,
    )

    assert report["claim_eligible_as_unsupervised_reconstruction"] is False
    assert report["split"]["n_calibration"] + report["split"]["n_evaluation"] == 40
    assert report["oracle"]["calibration_level_absolute_error"] < 1e-12
    assert report["oracle"]["kind"] == "one_scalar_bounded_additive_mean_matching"
    rows = {row["arm_id"]: row for row in report["rows"]}
    assert set(rows) == {"name", "definition", "name_plus_crossfit_mean_shift"}
    assert rows["definition"]["improvement_over_oracle_mean_shift"]["spearman"][
        "positive_means_candidate_improves"] is True


def test_scoring_cli_omission_expands_exact_frozen_cell_panel_before_run(
        monkeypatch, tmp_path, capsys):
    manifest_path = tmp_path / "execution.json"
    manifest_path.write_text(json.dumps({
        "schema": "fresh_name_execution_manifest/v2",
        "status": "frozen-before-test-model-outcomes",
        "analysis": {"runner": {"cell_ids": ["cell-z", "cell-a", "cell-m"]}},
    }))
    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return {"summary": {"synthetic": True}}

    monkeypatch.setattr(policy_runner, "run", fake_run)
    monkeypatch.setattr(sys, "argv", [
        "run_policy_isomorphism.py",
        "--executor-shard-root", str(tmp_path / "scores"),
        "--arm-bank", str(tmp_path / "bank.json"),
        "--partition", "synthetic-partition",
        "--execution-manifest", str(manifest_path),
        "--out", str(tmp_path / "report.json"),
    ])

    policy_runner.main()

    assert captured["cell_ids"] == ("cell-z", "cell-a", "cell-m")
    assert json.loads(capsys.readouterr().out)["synthetic"] is True


def test_scoring_cli_cell_resolution_preserves_explicit_and_legacy_modes(tmp_path):
    absent_manifest = tmp_path / "absent.json"
    assert _resolve_scoring_cell_ids(
        ["explicit-b", "explicit-a"], execution_manifest_path=absent_manifest
    ) == ("explicit-b", "explicit-a")
    assert _resolve_scoring_cell_ids(
        None, execution_manifest_path=None
    ) is None


@pytest.mark.parametrize("cell_ids", [[], "cell-a", [""], ["cell-a", "cell-a"]])
def test_scoring_cli_rejects_invalid_manifest_cell_panel(tmp_path, cell_ids):
    manifest_path = tmp_path / "execution.json"
    manifest_path.write_text(json.dumps({
        "schema": "fresh_name_execution_manifest/v2",
        "status": "frozen-before-test-model-outcomes",
        "analysis": {"runner": {"cell_ids": cell_ids}},
    }))

    with pytest.raises(
            ValueError, match="analysis.runner.cell_ids.*nonempty unique string list"):
        _resolve_scoring_cell_ids(None, execution_manifest_path=manifest_path)


def test_breadth_runner_admits_every_compiled_source_hierarchy_provenance():
    assert {
        "source_hierarchy_definition",
        "source_hierarchy_immediate_children",
        "source_hierarchy_leaf_signals",
        "source_definition_plus_children",
        "source_definition_children_and_leaf_signals",
        "source_address_prefix",
        "source_address_prefix_full",
    } <= SOURCE_HIERARCHY_PROVENANCES


def test_frozen_all_arm_search_rejects_missing_and_extra_executor_arms():
    cell = {
        "id": "cell",
        "arms": [
            {"id": "name", "forms": []},
            {"id": "source", "forms": []},
        ],
    }
    job = {"arm_policy": "all"}
    with pytest.raises(ValueError, match=r"missing=\['source'\]"):
        _validate_frozen_scored_arm_panel(
            observed_arm_ids={"name"}, cell=cell, model_job=job,
            label="executor",
        )
    with pytest.raises(ValueError, match=r"extra=\['extra'\]"):
        _validate_frozen_scored_arm_panel(
            observed_arm_ids={"name", "source", "extra"}, cell=cell,
            model_job=job, label="executor",
        )


def test_frozen_name_target_requires_exact_arm_and_prompt_hash():
    cell = {
        "id": "cell",
        "arms": [{
            "id": "name",
            "forms": [{"id": "canonical", "prompt_sha256": "expected"}],
        }],
    }
    closure = _validate_frozen_scored_arm_panel(
        observed_arm_ids={"name"}, cell=cell,
        model_job={"arm_policy": "name_only"}, label="target",
    )
    assert closure["valid"]
    with pytest.raises(ValueError, match="prompt hash mismatch"):
        validate_executor_prompt_arms(
            [{
                "cell_id": "cell", "arm_id": "name", "form": "canonical",
                "prompt_sha256": "changed",
            }],
            cell,
            arm_ids={"name"},
        )


def test_report_identity_payload_preserves_breadth_sampling_design():
    cell = {
        "id": "TB::demo::R2::node",
        "domain": "demo",
        "task": "demo",
        "level": "R2",
        "bucket": "general",
        "node_id": "demo::R2::node",
        "metric_id": "demo::R2::node",
        "gi": 4,
        "construct": "demo construct",
        "breadth_stratum": "medium",
        "stratum_population_n": 20,
        "stratum_selected_n": 5,
        "stratum_coverage_fraction": 0.25,
        "nominal_poststratification_weight": 4.0,
        "dependency_component_id": "component::1",
        "dependency_component_size": 3,
        "dependency_degree": 2,
        "source_assignment_multiplicity_max": 2,
        "provenance_component_id": "provenance::1",
        "provenance_component_size": 4,
        "provenance_overlap_degree": 3,
        "provenance_assignment_multiplicity_max": 3,
        "task_raw_provenance_component_id": "task-provenance::1",
        "task_raw_provenance_component_size": 5,
        "task_raw_provenance_overlap_degree": 4,
        "task_raw_provenance_assignment_multiplicity_max": 4,
        "source_kind": "merged_group",
        "source_index": 4,
        "source_path": "outputs/demo.json",
        "source_sha256": "source-sha",
        "leaf_support_count": 7,
        "leaf_support_sha256": "leaf-sha",
    }

    payload = _identity_payload(cell, context="sampling-design test")

    for key in (
            "breadth_stratum", "stratum_population_n", "stratum_selected_n",
            "stratum_coverage_fraction", "nominal_poststratification_weight",
            "dependency_component_id",
            "dependency_component_size", "dependency_degree",
            "source_assignment_multiplicity_max", "source_kind",
            "provenance_component_id", "provenance_component_size",
            "provenance_overlap_degree",
            "provenance_assignment_multiplicity_max", "source_index",
            "task_raw_provenance_component_id",
            "task_raw_provenance_component_size",
            "task_raw_provenance_overlap_degree",
            "task_raw_provenance_assignment_multiplicity_max",
            "source_path", "source_sha256", "leaf_support_count",
            "leaf_support_sha256"):
        assert payload[key] == cell[key]


def _synthetic_breadth_decomposition_inputs():
    tasks = [f"task-{index:02d}" for index in range(11)]
    levels = ["R1", "R2", "R3"]
    panel_cells = []
    report_cells = []

    def policy_certificate(rho, quotient_rho, mae):
        return {
            "point": {
                "candidate_robust": {
                    "spearman": rho,
                    "mae_tvd": mae,
                    "binary_flip_rate": 0.10,
                    "absolute_bias": 0.03,
                },
                "target_self_robust": {
                    "spearman": 0.90,
                    "mae_tvd": 0.05,
                    "binary_flip_rate": 0.02,
                    "absolute_bias": 0.01,
                },
            },
            "small_sparse_point": {
                "candidate_robust": {
                    "spearman": 0.50,
                    "mae_tvd": 0.30,
                    "binary_flip_rate": 0.20,
                    "absolute_bias": 0.10,
                },
            },
            "differences": {
                "mae_gain_over_small_sparse": {"point": 0.30 - mae},
            },
            "functional": {
                "gates": {
                    "target_identity_valid": True,
                    "positive_polarity": True,
                    "mae_point_improves_over_small_sparse": True,
                    "mae_CI_improves_over_small_sparse": True,
                },
                "adverse_rho_point": rho,
                "adverse_rho_CI": [rho - 0.02, rho + 0.02],
                "quotient_rho_point": quotient_rho,
                "quotient_rho_CI": [quotient_rho - 0.02, quotient_rho + 0.02],
                "small_sparse_adverse_rho_CI": [0.48, 0.52],
            },
        }

    def scale_certificate(rho, mae, *, observed=False, certified=False):
        rank_closure = (rho - 0.50) / (0.75 - 0.50)
        mae_closure = (0.30 - mae) / (0.30 - 0.12)
        def evidence(passing):
            return {
                "joint_fixed_target_and_endpoint_functional_isomorphic_"
                "scale_substitution": passing,
                "joint_fixed_target_and_endpoint_functional_equivalent_"
                "scale_substitution": passing,
            }
        return {
            "point": {
                "small_sparse": {
                    "spearman": 0.50, "mae_tvd": 0.30,
                    "binary_flip_rate": 0.20, "absolute_bias": 0.10,
                },
                "candidate": {
                    "spearman": rho, "mae_tvd": mae,
                    "binary_flip_rate": 0.10, "absolute_bias": 0.03,
                },
                "larger_sparse": {
                    "spearman": 0.75, "mae_tvd": 0.12,
                    "binary_flip_rate": 0.08, "absolute_bias": 0.025,
                },
            },
            "descriptive_step_closure": {
                "rank": {"chi_articulation_gain_over_native_gap": rank_closure},
                "mae": {"chi_articulation_gain_over_native_gap": mae_closure},
            },
            "evidence": {
                "observed": evidence(observed),
                "certified": evidence(certified),
            },
        }

    cell_index = 0
    for task in tasks:
        for level in levels:
            for local_index in range(30):
                node_id = f"{task}::{level}::node-{local_index:02d}"
                cell_id = f"TB::{node_id}"
                positive_observed = cell_index % 2 == 0
                positive_certified = cell_index % 4 == 0
                positive_simultaneous = cell_index % 8 == 0
                source_kind = (
                    ("merged_tree" if local_index % 2 == 0 else "parented_tree")
                    if level == "R1" else
                    ("merged_group" if local_index % 2 == 0 else "grandparent")
                )
                design = {
                    "breadth_stratum": ("small", "medium", "large")[local_index % 3],
                    "stratum_population_n": 60,
                    "stratum_selected_n": 10,
                    "stratum_coverage_fraction": 1 / 6,
                    "nominal_poststratification_weight": 6.0,
                    "dependency_component_id": f"dep::{local_index // 2}",
                    "dependency_component_size": 2,
                    "dependency_degree": 1 if local_index % 2 else 0,
                    "source_assignment_multiplicity_max": 2,
                    "provenance_component_id": f"prov::{local_index // 3}",
                    "provenance_component_size": 3,
                    "provenance_overlap_degree": 2 if local_index % 3 else 0,
                    "provenance_assignment_multiplicity_max": 3,
                    "task_raw_provenance_component_id": (
                        f"task-prov::{local_index // 3}"),
                    "task_raw_provenance_component_size": 9,
                    "task_raw_provenance_overlap_degree": 8,
                    "task_raw_provenance_assignment_multiplicity_max": 9,
                    "source_kind": source_kind,
                    "source_index": local_index,
                    "source_path": f"outputs/{task}-{level}.json",
                    "source_sha256": f"source-sha-{task}-{level}",
                    "leaf_support_count": local_index + 1,
                    "leaf_support_sha256": f"leaf-sha-{task}-{level}-{local_index}",
                }
                panel_cell = {
                    "id": cell_id,
                    "domain": task,
                    "task": task,
                    "level": level,
                    "bucket": "general",
                    "node_id": node_id,
                    "metric_id": node_id,
                    "construct": f"construct {cell_index}",
                    **design,
                }
                panel_cells.append(panel_cell)

                candidates = []
                for arm_index, (arm_id, rho, quotient_rho, mae) in enumerate((
                        ("address_1", 0.72, 0.71, 0.15),
                        ("address_2", 0.73, 0.72, 0.14))):
                    candidates.append({
                        "arm_id": arm_id,
                        "channel": "address_dose",
                        "provenance": "source_address_prefix",
                        "control_for": None,
                        "components": [f"component-{arm_index}"],
                        "composition_degree": 1,
                        "n_address_units": arm_index + 1,
                        "added_content_word_count": 10 * (arm_index + 1),
                        "semantic_content_word_count": 12 + 10 * arm_index,
                        "certificate": policy_certificate(rho, quotient_rho, mae),
                        "scale_step_certificate": scale_certificate(
                            rho, mae,
                            observed=positive_observed,
                            certified=positive_certified,
                        ),
                        "scale_step_simultaneous_certificate": scale_certificate(
                            rho, mae, certified=positive_simultaneous),
                        "scale_step_specificity_simultaneous_certificate": (
                            scale_certificate(
                                rho, mae, certified=positive_simultaneous)),
                    })
                controls = []
                matched_controls = []
                for candidate in candidates:
                    for label, provenance in (
                            ("inert", "inert_length_control"),
                            ("wrong", "wrong_construct_control")):
                        control_id = f"{candidate['arm_id']}::{label}"
                        controls.append({
                            "arm_id": control_id,
                            "channel": candidate["channel"],
                            "provenance": provenance,
                            "control_for": candidate["arm_id"],
                            "components": [],
                            "n_address_units": candidate["n_address_units"],
                            "added_content_word_count": candidate[
                                "added_content_word_count"],
                            "semantic_content_word_count": candidate[
                                "semantic_content_word_count"],
                        })
                        matched_controls.append({
                            "source_arm_id": candidate["arm_id"],
                            "control_arm_id": control_id,
                            "control_provenance": provenance,
                            "certificate": {"gates": {
                                "source_rank_better_point": positive_observed,
                                "source_mae_better_point": positive_observed,
                                "source_rank_better_CI": positive_certified,
                                "source_mae_better_CI": positive_certified,
                            }},
                            "simultaneous_certificate": {"gates": {
                                "source_rank_better_CI": positive_simultaneous,
                                "source_mae_better_CI": positive_simultaneous,
                            }},
                            "specificity_simultaneous_certificate": {"gates": {
                                "source_rank_better_CI": positive_simultaneous,
                                "source_mae_better_CI": positive_simultaneous,
                            }},
                        })
                content_specific = _content_specific_scale_memberships(
                    [*candidates, *controls], matched_controls)
                pair = [{"left": "address_1", "right": "address_2"}]
                pair_certificate = {
                    "left": "address_1",
                    "right": "address_2",
                    "structural_basis": "declared_component_topology",
                    "component_minimal": True,
                    "components_incomparable": True,
                    "both_frozen_atomic_routes": False,
                    "channels_distinct": False,
                    "articulation_surface_distance": 0.75,
                    "distinctness_floor": 0.35,
                    "structural_gate": True,
                    "mutual_policy_certificate": {"gates": {
                        "point_at_least_primary_floor": True,
                        "lower_CI_at_least_primary_floor": True,
                        "point_at_least_sensitivity_floor": True,
                        "lower_CI_at_least_sensitivity_floor": True,
                        "point_vector_equivalent": True,
                        "certified_vector_equivalent": True,
                    }},
                    "grades": {},
                }
                for grade, passing in (
                        ("observed", positive_observed),
                        ("certified", positive_certified),
                        ("simultaneous_certified", positive_simultaneous)):
                    pair_certificate["grades"][grade] = {
                        "both_members_pass_content_specific_H_J": passing,
                        "both_members_also_pass_H_J_eq": passing,
                        "mutual_rank_gate": True,
                        "mutual_rank_sensitivity_gate": True,
                        "mutual_quotient_vector_equivalence_gate": True,
                        "H_fiber": passing,
                        "H_fiber_sensitivity": passing,
                        "H_fiber_eq": passing,
                        "H_fiber_vec": passing,
                        "H_fiber_vec_eq": passing,
                    }
                report_cells.append({
                    "cell_id": cell_id,
                    "domain": task,
                    "task": task,
                    "level": level,
                    "bucket": "general",
                    "node_id": node_id,
                    "metric_id": node_id,
                    "construct": f"construct {cell_index}",
                    **design,
                    "rows": [*candidates, *controls],
                    "scored_arm_panel_validation": {
                        "small": {"valid": True, "arm_policy": "frozen_selection"},
                        "target": {"valid": True, "arm_policy": "name_only"},
                    },
                    "score_cell_identity_validation": {
                        "small": {"valid": True}, "target": {"valid": True},
                    },
                    "executor_prompt_bank_validation": {"valid": True},
                    "target_prompt_bank_validation": {"valid": True},
                    "source_group_validation": {"valid": True},
                    "scale_comparator_validation": {"valid": True},
                    "score_provenance_validation": {
                        "small": {"valid": True, "fake_backend": False},
                        "target": {"valid": True, "fake_backend": False},
                    },
                    "matched_control_certificates": matched_controls,
                    "content_specific_scale_step": content_specific,
                    "content_specific_joint_fiber": {
                        "candidate_arm_ids": ["address_1", "address_2"],
                        "pair_certificates": [pair_certificate],
                        "observed_H_fiber_pairs": pair if positive_observed else [],
                        "certified_H_fiber_pairs": pair if positive_certified else [],
                        "simultaneous_certified_H_fiber_pairs": (
                            pair if positive_simultaneous else []),
                        "simultaneous_certified_H_fiber_sensitivity_pairs": (
                            pair if positive_simultaneous else []),
                        "simultaneous_certified_H_fiber_eq_pairs": (
                            pair if positive_simultaneous else []),
                        "observed_H_fiber_vec_pairs": pair if positive_observed else [],
                        "certified_H_fiber_vec_pairs": pair if positive_certified else [],
                        "simultaneous_certified_H_fiber_vec_pairs": (
                            pair if positive_simultaneous else []),
                        "simultaneous_certified_H_fiber_vec_eq_pairs": (
                            pair if positive_simultaneous else []),
                    },
                })
                cell_index += 1
    terminal = []
    for task in tasks:
        terminal.append({
            "task": task, "level": "R1", "available": False,
            "reason": "R1 terminal frame unavailable",
        })
        terminal.append({
            "task": task, "level": "R2", "available": True,
            "n_frontier_nodes": 100, "n_eligible_nodes": 90,
            "global_partition_claim": False,
        })
        r3 = {
            "task": task, "level": "R3", "available": True,
            "n_frontier_nodes": 60, "n_eligible_nodes": 50,
            "global_partition_claim": False,
        }
        if task == tasks[0]:
            r3["frontier_node_ids"] = [
                f"{task}::R3::node-00",
                *(f"{task}::R3::unscored-carry-{index:02d}" for index in range(59)),
            ]
        terminal.append(r3)
    panel = {
        "schema": "tacit_breadth_metric_panel/v3",
        "tasks": tasks,
        "levels": levels,
        "n_per_task_level": 30,
        "n_cells": len(panel_cells),
        "terminal_frontier_sensitivities": terminal,
        "cells": panel_cells,
    }
    panel["panel_content_sha256"] = sha256_bytes(json.dumps(
        panel, sort_keys=True, separators=(",", ":")).encode())
    runner = {
        "small_job": "llama31_8b_executor",
        "big_job": "llama31_70b_name_target",
        "target_arm_id": "name",
        "scale_comparator_job": None,
        "scale_comparator_arm_id": "name",
        "scale_comparator_use_target": True,
        "n_boot": 10000,
        "seed": 1207,
        "mae_margin": 0.02,
        "rho_margin": 0.05,
        "flip_margin": 0.02,
        "bias_margin": 0.02,
        "functional_rho_floor": 0.70,
        "confidence": 0.95,
        "fiber_mutual_rho_floor": 0.90,
        "fiber_mutual_rho_sensitivity_floor": 0.85,
        "fiber_min_rank_valid_fraction": 0.99,
        "fiber_distinctness_floor": 0.35,
        "include_controls": True,
        "crossfit_only": False,
        "cell_ids": [cell["id"] for cell in panel_cells],
        "source_group_inference": True,
        "allow_fake_inputs": False,
    }
    report = {
        "schema": "policy_isomorphism_experiment/v5",
        "partition": "tacit_breadth_validation",
        "arm_bank_sha256": "bank-sha",
        "partition_authorization": {
            "phase": "lockbox",
            "sealed_partition_authorized": True,
            "lockbox_release_validation": {"valid": True},
        },
        "frozen_invocation_validation": {"valid": True, "runner": runner},
        "cell_panel_identity_validation": {"valid": True},
        "source_group_inference": {"enabled": True},
        "scale_comparator": {
            "enabled": True, "use_fixed_target_orbit": True,
        },
        "config": {
            key: value for key, value in runner.items()
            if key not in {"source_group_inference", "allow_fake_inputs"}
        },
        "cells": report_cells,
    }
    return report, panel


def test_breadth_decomposition_summarizes_complete_panel_with_block_sensitivities():
    report, panel = _synthetic_breadth_decomposition_inputs()
    result = summarize_breadth_decomposition(
        report, panel, n_boot=20, seed=19, confidence=0.90)

    assert result["schema"] == "tacit_breadth_decomposition_report/v3"
    assert result["panel_validation"]["n_cells"] == 990
    assert len(result["task_level"]) == 33
    aggregate = result["aggregate"]
    inference = aggregate["inference"]["balanced_selected_action_node_panel"]
    independent = inference["bootstrap_designs"]["independent_cell"]
    dependency = inference["bootstrap_designs"]["immediate_dependency_component"]
    provenance = inference["bootstrap_designs"][
        "inherited_raw_provenance_component"]
    task_provenance = inference["bootstrap_designs"][
        "task_raw_provenance_component"]
    outcome = "observed_content_specific_joint_functional_substitution"
    assert independent["prevalence"][outcome]["point"] == pytest.approx(0.5)
    assert independent["n_blocks"] == 990
    assert dependency["n_blocks"] == 495
    assert dependency["largest_observed_block_n"] == 2
    assert provenance["n_blocks"] == 330
    assert provenance["largest_observed_block_n"] == 3
    assert task_provenance["n_blocks"] == 110
    assert task_provenance["largest_observed_block_n"] == 9
    assert aggregate[
        "address_dose_minimum_passing_tested_dose_distribution"]["observed"][
        "median_units"] == 1
    assert aggregate[
        "address_dose_minimum_passing_tested_dose_distribution"]["observed"][
            "n_cells_with_authenticated_complete_panel_onset"] == 0
    assert aggregate["fiber_pair_totals"]["observed"] == 495
    assert aggregate["fiber_topology_summary"]["observed"][
        "channel_pair_counts"] == {"address_dose <-> address_dose": 495}
    first_cell = result["cells"][0]
    assert first_cell["coordinates"][
        "best_raw_joint_adverse_quotient_rho_floor"] == pytest.approx(0.72)
    assert first_cell["coordinate_frontiers"][
        "best_raw_joint_adverse_quotient_rho_floor"]["arm_id"] == "address_2"
    terminal = result["sensitivities"]["terminal_frontier"]
    exact = next(row for row in terminal
                 if row["task"] == "task-00" and row["level"] == "R3")
    assert exact["n_exactly_matched_scored_primary_nodes"] == 1
    assert exact["exact_primary_node_coverage"] == pytest.approx(1 / 60)
    hashed_only = next(row for row in terminal
                       if row["task"] == "task-00" and row["level"] == "R2")
    assert hashed_only["outcome_status"] == "not_identifiable_from_frozen_panel"
    assert hashed_only["estimates"] is None


def test_breadth_decomposition_rejects_reordered_or_incomplete_panels():
    report, panel = _synthetic_breadth_decomposition_inputs()
    report["cells"][0], report["cells"][1] = report["cells"][1], report["cells"][0]
    with pytest.raises(ValueError, match="identities and order"):
        summarize_breadth_decomposition(report, panel, n_boot=2)

    report, panel = _synthetic_breadth_decomposition_inputs()
    panel["cells"].pop()
    core = {key: value for key, value in panel.items()
            if key != "panel_content_sha256"}
    panel["panel_content_sha256"] = sha256_bytes(json.dumps(
        core, sort_keys=True, separators=(",", ":")).encode())
    with pytest.raises(ValueError, match="complete 990-cell"):
        summarize_breadth_decomposition(report, panel, n_boot=2)


def _rehash_breadth_panel(panel):
    core = {key: value for key, value in panel.items()
            if key != "panel_content_sha256"}
    panel["panel_content_sha256"] = sha256_bytes(json.dumps(
        core, sort_keys=True, separators=(",", ":")).encode())


def _set_synthetic_fiber_grades(report_cell, passing_by_grade):
    certificate = report_cell["content_specific_joint_fiber"][
        "pair_certificates"][0]
    fiber = report_cell["content_specific_joint_fiber"]
    pair = [{"left": "address_1", "right": "address_2"}]
    for grade, passing in passing_by_grade.items():
        row = certificate["grades"][grade]
        row.update({
            "both_members_pass_content_specific_H_J": passing,
            "both_members_also_pass_H_J_eq": passing,
            "H_fiber": passing,
            "H_fiber_sensitivity": passing,
            "H_fiber_eq": passing,
            "H_fiber_vec": passing,
            "H_fiber_vec_eq": passing,
        })
    list_specs = {
        "observed_H_fiber_pairs": "observed",
        "certified_H_fiber_pairs": "certified",
        "simultaneous_certified_H_fiber_pairs": "simultaneous_certified",
        "simultaneous_certified_H_fiber_sensitivity_pairs": (
            "simultaneous_certified"),
        "simultaneous_certified_H_fiber_eq_pairs": "simultaneous_certified",
        "observed_H_fiber_vec_pairs": "observed",
        "certified_H_fiber_vec_pairs": "certified",
        "simultaneous_certified_H_fiber_vec_pairs": "simultaneous_certified",
        "simultaneous_certified_H_fiber_vec_eq_pairs": "simultaneous_certified",
    }
    for key, grade in list_specs.items():
        fiber[key] = pair if passing_by_grade[grade] else []


def test_breadth_readout_replays_scale_and_fiber_memberships():
    report, panel = _synthetic_breadth_decomposition_inputs()
    report["cells"][1]["content_specific_scale_step"][
        "observed_joint_fixed_target_endpoint_isomorphic_members"] = ["address_1"]
    with pytest.raises(ValueError, match="scale memberships fail deterministic replay"):
        summarize_breadth_decomposition(report, panel, n_boot=1)

    report, panel = _synthetic_breadth_decomposition_inputs()
    report["cells"][1]["content_specific_joint_fiber"][
        "observed_H_fiber_pairs"] = [{"left": "address_1", "right": "address_2"}]
    with pytest.raises(ValueError, match="fiber membership list .* fails grade replay"):
        summarize_breadth_decomposition(report, panel, n_boot=1)


def test_breadth_content_route_requires_functional_substitution_not_only_controls():
    report, panel = _synthetic_breadth_decomposition_inputs()
    report_cell = report["cells"][0]
    for row in report_cell["rows"]:
        if row.get("control_for") is not None:
            continue
        for evidence in (
                row["scale_step_certificate"]["evidence"].values(),
                row["scale_step_specificity_simultaneous_certificate"][
                    "evidence"].values()):
            for grade in evidence:
                for key in grade:
                    grade[key] = False
    report_cell["content_specific_scale_step"] = (
        _content_specific_scale_memberships(
            report_cell["rows"], report_cell["matched_control_certificates"])
    )
    _set_synthetic_fiber_grades(report_cell, {
        "observed": False, "certified": False, "simultaneous_certified": False,
    })

    record = _breadth_cell_decomposition(panel["cells"][0], report_cell)

    assert record["outcomes"]["observed_matched_control_improvement"] is True
    assert record["outcomes"][
        "observed_content_specific_joint_functional_substitution"] is False
    assert record["best_content_specific_route"] is None


def test_breadth_readout_never_counts_unknown_control_like_row_as_content():
    report, panel = _synthetic_breadth_decomposition_inputs()
    report["cells"][0]["rows"][0]["provenance"] = "wrong_construct_control_typo"
    with pytest.raises(ValueError, match="candidate panel contains a name, control, or non-source"):
        _breadth_cell_decomposition(panel["cells"][0], report["cells"][0])


def test_breadth_block_bootstrap_preserves_unequal_block_mass_and_null_denominators():
    records = []
    for index in range(10):
        outcomes = {name: False for name in _BREADTH_BINARY_OUTCOMES}
        outcomes["observed_matched_control_improvement"] = index > 0
        coordinates = {name: None for name in _BREADTH_COORDINATES}
        if index == 0:
            coordinates["best_native_scale_step_rank_closure"] = 0.25
        records.append({"outcomes": outcomes, "coordinates": coordinates})
    groups = ["singleton", *("large" for _ in range(9))]

    balanced = _bootstrap_metric_estimates(
        records, weights=np.ones(10), group_ids=groups,
        n_boot=200, seed=17, confidence=0.90,
    )
    repeated = _bootstrap_metric_estimates(
        records, weights=np.ones(10), group_ids=groups,
        n_boot=200, seed=17, confidence=0.90,
    )
    weighted = _bootstrap_metric_estimates(
        records, weights=np.asarray([9.0, *(1.0 for _ in range(9))]),
        group_ids=groups, n_boot=200, seed=17, confidence=0.90,
    )

    prevalence = balanced["prevalence"]["observed_matched_control_improvement"]
    weighted_prevalence = weighted["prevalence"][
        "observed_matched_control_improvement"]
    sparse_coordinate = balanced["coordinate_means"][
        "best_native_scale_step_rank_closure"]
    assert balanced == repeated
    assert prevalence["point"] == pytest.approx(0.9)
    assert prevalence["n_defined_blocks"] == 2
    assert weighted_prevalence["point"] == pytest.approx(0.5)
    assert sparse_coordinate["point"] == pytest.approx(0.25)
    assert sparse_coordinate["n_defined_cells"] == 1
    assert sparse_coordinate["n_defined_blocks"] == 1
    assert sparse_coordinate["CI"] is None
    assert sparse_coordinate["CI_status"] == "insufficient_defined_blocks"
    assert sparse_coordinate["n_valid_bootstrap_draws"] == 0


def test_breadth_design_and_terminal_sensitivity_are_scope_bound():
    report, panel = _synthetic_breadth_decomposition_inputs()
    panel["cells"][0]["nominal_poststratification_weight"] = 5.0
    report["cells"][0]["nominal_poststratification_weight"] = 5.0
    _rehash_breadth_panel(panel)
    with pytest.raises(ValueError, match="nominal post-stratification factor"):
        summarize_breadth_decomposition(report, panel, n_boot=1)

    report, panel = _synthetic_breadth_decomposition_inputs()
    other_task_audit = next(
        row for row in panel["terminal_frontier_sensitivities"]
        if row["task"] == "task-01" and row["level"] == "R3"
    )
    other_task_audit["frontier_node_ids"] = [
        "task-00::R3::node-00",
        *(f"task-01::R3::unscored-carry-{index:02d}" for index in range(59)),
    ]
    _rehash_breadth_panel(panel)
    result = summarize_breadth_decomposition(report, panel, n_boot=1)
    scoped = next(
        row for row in result["sensitivities"]["terminal_frontier"]
        if row["task"] == "task-01" and row["level"] == "R3"
    )
    assert scoped["n_exactly_matched_scored_primary_nodes"] == 0
    assert scoped["outcome_status"] == "no_exactly_matched_scored_primary_nodes"


def test_two_manifest_calibration_release_writer_binds_search_report_and_lockbox(
        tmp_path):
    calibration_manifest_path = tmp_path / "search-manifest.json"
    report_path = tmp_path / "search-report.json"
    selection_path = tmp_path / "selection.json"
    lockbox_manifest_path = tmp_path / "lockbox-manifest.json"
    release_path = tmp_path / "release.json"
    calibration_manifest = {
        "schema": "fresh_name_execution_manifest/v2",
        "phases": {"calibration": ["tacit_breadth_search"]},
        "arm_bank_sha256": "bank-sha",
        "analysis": {"runner": {"cell_ids": ["cell"]}},
        "implementation": {"analysis": {"files": []}},
    }
    calibration_manifest_path.write_text(json.dumps(calibration_manifest))
    calibration_manifest_sha = sha256_file(calibration_manifest_path)
    report = {
        "schema": "policy_isomorphism_experiment/v5",
        "partition": "tacit_breadth_search",
        "arm_bank_sha256": "bank-sha",
        "partition_authorization": {
            "phase": "calibration",
            "execution_manifest_sha256": calibration_manifest_sha,
        },
        "frozen_invocation_validation": {"valid": True},
        "analysis_implementation": {"files": []},
        "cells": [{
            "cell_id": "cell",
            "score_provenance_validation": {
                role: {
                    "valid": True,
                    "fake_backend": False,
                    "backend_class": "OfflineVLLM",
                }
                for role in ("small", "target")
            },
        }],
    }
    report_path.write_text(json.dumps(report))
    selection = {
        "schema": "policy_articulation_selection/v1",
        "search_execution_manifest_path": str(calibration_manifest_path),
        "search_execution_manifest_sha256": calibration_manifest_sha,
        "search_report_path": str(report_path),
        "search_report_sha256": sha256_file(report_path),
    }
    selection_path.write_text(json.dumps(selection))
    selection_sha = sha256_file(selection_path)
    lockbox_manifest = {
        "schema": "fresh_name_execution_manifest/v2",
        "phases": {"lockbox": ["tacit_breadth_validation"]},
        "phase_access": {"lockbox": "sealed_confirmation"},
        "selection_artifact_path": str(selection_path),
        "selection_artifact_sha256": selection_sha,
        "arm_bank_sha256": "bank-sha",
        "execution_environment": {"production_backend_class": "OfflineVLLM"},
        "lockbox_release": {
            "required": True,
            "schema": "policy_isomorphism_calibration_release/v1",
            "artifact_path": str(release_path),
            "calibration_report_path": str(report_path),
            "calibration_report_schema": "policy_isomorphism_experiment/v5",
            "calibration_partition": "tacit_breadth_search",
            "lockbox_partition": "tacit_breadth_validation",
        },
    }
    lockbox_manifest_path.write_text(json.dumps(lockbox_manifest))

    result = write_calibration_release_artifact(
        report,
        report_path=report_path,
        execution_manifest_path=lockbox_manifest_path,
        selection_artifact_path=selection_path,
    )

    release = json.loads(release_path.read_text())
    assert result["valid"]
    assert release["execution_manifest_sha256"] == sha256_file(lockbox_manifest_path)
    assert release["calibration_execution_manifest_sha256"] == calibration_manifest_sha
    assert release["selection_artifact_sha256"] == selection_sha


def _selection_generator_fixture(tmp_path):
    def forms(prefix, words):
        return [{
            "id": "canonical", "prompt_sha256": f"{prefix}-prompt",
            "total_word_count": words,
        }]

    def source_arm(arm_id, channel, components, words=12):
        return {
            "id": arm_id, "channel": channel,
            "provenance": "source_hierarchy_definition",
            "control_for": None, "components": components,
            "semantic_content_word_count": words,
            "added_content_word_count": words - 2,
            "forms": forms(arm_id, words),
        }

    def controls(source):
        return [{
            "id": f"control_{label}_{source['id']}",
            "channel": source["channel"], "provenance": provenance,
            "control_for": source["id"], "components": [],
            "semantic_content_word_count": source["semantic_content_word_count"],
            "added_content_word_count": source["added_content_word_count"],
            "forms": forms(
                f"control-{label}-{source['id']}",
                source["semantic_content_word_count"],
            ),
        } for label, provenance in (
            ("wrong", "wrong_construct_control"),
            ("inert", "inert_length_control"),
        )]

    def cell(cell_id, candidates):
        arms = [{
            "id": "name", "channel": "sparse", "provenance": "construct_name",
            "control_for": None, "components": [],
            "semantic_content_word_count": 2, "added_content_word_count": 0,
            "forms": forms("name", 2),
        }]
        for candidate in candidates:
            arms.extend([candidate, *controls(candidate)])
        return {
            "id": cell_id, "domain": "demo", "task": "demo", "level": "R1",
            "bucket": "general", "node_id": f"node::{cell_id}",
            "metric_id": f"metric::{cell_id}", "construct": "demo construct",
            "arms": arms,
        }

    a = source_arm("source_a", "declarative", ["component-a"])
    b = source_arm("source_b", "procedural", ["component-b"], 13)
    c = source_arm("source_c", "composed", ["component-a", "component-b"], 14)
    d = source_arm("source_d", "address_dose", ["component-d"], 15)
    d["n_address_units"] = 1
    null = source_arm("source_null", "declarative", ["component-null"])
    bank = {"cells": [cell("cell-rich", [a, b, c, d]), cell("cell-null", [null])]}
    bank_path = tmp_path / "bank.json"
    bank_path.write_text(json.dumps(bank))
    panel_path = tmp_path / "panel.json"
    panel_path.write_text(json.dumps({"cells": ["cell-rich", "cell-null"]}))
    model_path = tmp_path / "model-template.json"
    model_path.write_text("{}")
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps({
        "partitions": [
            {"id": "search-items", "domains": ["demo"]},
            {"id": "validation-items", "domains": ["demo"]},
        ],
    }))
    packet_path = tmp_path / "packet.json"
    packet_path.write_text(json.dumps({
        "domains": [{
            "domain": "demo",
            "partitions": [{"id": "search-items"}, {"id": "validation-items"}],
        }],
    }))
    policy = {
        "schema": "tacit_breadth_selection_policy/v2",
        "maximum_candidates_per_cell": 4,
        "minimum_candidates_per_cell": 1,
        "rank_diversity_tolerance": 0.10,
        "roles_in_order": [
            "best_functional_rank", "best_vector_identity",
            "best_component_distinct_route_within_rank_tolerance",
            "best_address_dose",
        ],
        "primary_order": "frozen primary order",
        "vector_order": "frozen vector order",
        "diversity_rule": "frozen diversity rule",
        "dose_rule": "frozen dose rule",
        "null_cell_rule": "frozen null rule",
        "control_rule": "frozen control rule",
    }
    implementation_files = [{"path": "analysis.py", "sha256": "analysis-sha"}]
    manifest = {
        "schema": "fresh_name_execution_manifest/v2",
        "status": "frozen-before-tacit-breadth-search-model-outcomes",
        "domains": ["demo"],
        "phases": {"calibration": ["search-items"]},
        "protocol_manifest_path": str(protocol_path),
        "protocol_manifest_sha256": sha256_file(protocol_path),
        "packet_manifest_path": str(packet_path),
        "packet_manifest_sha256": sha256_file(packet_path),
        "arm_bank_path": str(bank_path),
        "arm_bank_sha256": sha256_file(bank_path),
        "additional_artifacts": [
            {"role": "metric_panel", "path": str(panel_path),
             "sha256": sha256_file(panel_path)},
            {"role": "model_environment_template", "path": str(model_path),
             "sha256": sha256_file(model_path)},
        ],
        "selection_policy": policy,
        "analysis": {"runner": {
            "cell_ids": ["cell-rich", "cell-null"], "include_controls": True,
        }},
        "implementation": {"analysis": {"files": implementation_files}},
    }
    manifest_path = tmp_path / "search-manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    def certificate(rank, *, functional, vector_excess):
        margins = {"mae": 0.02, "rho": 0.05, "flip": 0.02, "bias": 0.02}
        return {
            "functional": {
                "adverse_rho_point": rank,
                "quotient_rho_point": rank + 0.01,
                "observed_functional_policy_substitution": functional,
            },
            "point": {"candidate_robust": {
                "mae_tvd": 0.10 + vector_excess / 100,
                "binary_flip_rate": 0.10 + vector_excess / 100,
                "absolute_bias": 0.03 + vector_excess / 100,
            }},
            "differences": {
                "mae_excess_over_target_self": {
                    "point": margins["mae"] * (1 + vector_excess)},
                "rho_minus_target_self": {
                    "point": -margins["rho"] * (1 + vector_excess)},
                "flip_excess_over_target_self": {
                    "point": margins["flip"] * (1 + vector_excess)},
                "bias_excess_over_target_self": {
                    "point": margins["bias"] * (1 + vector_excess)},
            },
            "margins": margins,
        }

    outcomes = {
        "source_a": (0.80, True, 1.0, True),
        "source_b": (0.75, True, 0.5, True),
        "source_c": (0.78, False, 0.0, True),
        "source_d": (0.70, False, 2.0, False),
        "source_null": (0.20, False, 3.0, False),
    }
    report_cells = []
    for bank_cell in bank["cells"]:
        rows, grade_rows = [], []
        for arm in bank_cell["arms"]:
            if arm["id"] == "name":
                continue
            row = {
                "arm_id": arm["id"], "channel": arm["channel"],
                "provenance": arm["provenance"], "control_for": arm["control_for"],
                "components": arm["components"],
                "semantic_content_word_count": arm["semantic_content_word_count"],
            }
            if arm["control_for"] is None:
                rank, functional, excess, control_superiority = outcomes[arm["id"]]
                row["certificate"] = certificate(
                    rank, functional=functional, vector_excess=excess)
                grade_rows.append({
                    "arm_id": arm["id"],
                    "grades": {"observed": {
                        "better_than_every_required_control_on_rank_and_mae": (
                            control_superiority),
                    }},
                })
            rows.append(row)
        report_cells.append({
            "cell_id": bank_cell["id"], "rows": rows,
            "content_specific_scale_step": {"arm_grades": grade_rows},
            "executor_prompt_bank_validation": {"valid": True},
            "target_prompt_bank_validation": {"valid": True},
            "scored_arm_panel_validation": {
                "small": {"valid": True}, "target": {"valid": True},
            },
            "score_provenance_validation": {
                "small": {"valid": True, "fake_backend": False},
                "target": {"valid": True, "fake_backend": False},
            },
        })
    report = {
        "schema": "policy_isomorphism_experiment/v5",
        "partition": "search-items",
        "arm_bank_sha256": sha256_file(bank_path),
        "partition_authorization": {
            "phase": "calibration",
            "execution_manifest_sha256": sha256_file(manifest_path),
            "selection_artifact_sha256": None,
        },
        "source_group_inference": {"packet_manifest_sha256": sha256_file(packet_path)},
        "frozen_invocation_validation": {"valid": True},
        "additional_artifact_validation": {
            "valid": True,
            "files": [
                {"path": str(panel_path), "sha256": sha256_file(panel_path)},
                {"path": str(model_path), "sha256": sha256_file(model_path)},
            ],
        },
        "analysis_implementation": {"files": implementation_files},
        "config": {"include_controls": True},
        "cell_panel_identity_validation": {"valid": True},
        "cells": report_cells,
    }
    report_path = tmp_path / "search-report.json"
    report_path.write_text(json.dumps(report))
    return {
        "manifest_path": manifest_path, "report_path": report_path,
        "bank_path": bank_path, "packet_path": packet_path,
        "panel_path": panel_path, "model_path": model_path,
    }


def test_breadth_selection_generator_applies_roles_and_keeps_null_cells(tmp_path):
    fixture = _selection_generator_fixture(tmp_path)
    kwargs = {
        "search_execution_manifest_path": fixture["manifest_path"],
        "search_report_path": fixture["report_path"],
        "arm_bank_path": fixture["bank_path"],
        "packet_manifest_path": fixture["packet_path"],
        "metric_panel_path": fixture["panel_path"],
        "additional_artifact_paths": (fixture["model_path"],),
        "selected_phase": "validation",
        "selected_partition": "validation-items",
    }

    selection = build_policy_articulation_selection(**kwargs)

    cells = {cell["cell_id"]: cell for cell in selection["cells"]}
    assert cells["cell-rich"]["candidate_arm_ids"] == [
        "source_a", "source_c", "source_b", "source_d"]
    assert cells["cell-null"]["candidate_arm_ids"] == ["source_null"]
    assignments = {
        row["role"]: row["arm_id"]
        for row in cells["cell-rich"]["role_assignments"]
    }
    assert assignments["best_functional_rank"] == "source_a"
    assert assignments[
        "best_component_distinct_route_within_rank_tolerance"] == "source_b"
    assert selection["selection_policy"]["schema"] == (
        "tacit_breadth_selection_policy/v2")
    assert all(
        len(cell["control_ids"]) == 2 * len(cell["candidate_arm_ids"])
        for cell in cells.values()
    )
    assert selection["search_execution_manifest_sha256"] == sha256_file(
        fixture["manifest_path"])
    assert selection["search_report_sha256"] == sha256_file(fixture["report_path"])
    assert selection["metric_panel_sha256"] == sha256_file(fixture["panel_path"])
    assert selection["additional_artifacts"][0]["sha256"] == sha256_file(
        fixture["model_path"])

    first_path = tmp_path / "selection-a.json"
    second_path = tmp_path / "selection-b.json"
    first = write_policy_articulation_selection(out_path=first_path, **kwargs)
    second = write_policy_articulation_selection(out_path=second_path, **kwargs)
    assert first["sha256"] == second["sha256"]
    assert first_path.read_bytes() == second_path.read_bytes()


def test_breadth_primary_selection_is_rank_first_even_without_control_superiority():
    common = {
        "observed_functional_policy_substitution": False,
        "adverse_mae_tvd": 0.2,
        "binary_flip_rate": 0.1,
        "absolute_bias": 0.05,
        "semantic_content_word_count": 20,
    }
    higher_rank = {
        **common,
        "arm_id": "higher-rank",
        "rank_floor": 0.86,
        "content_specific_point_superiority": False,
    }
    lower_rank_specific = {
        **common,
        "arm_id": "lower-rank-specific",
        "rank_floor": 0.74,
        "content_specific_point_superiority": True,
        "observed_functional_policy_substitution": True,
    }

    assert policy_runner._primary_selection_key(higher_rank) < (
        policy_runner._primary_selection_key(lower_rank_specific))


def test_breadth_selection_generator_rejects_missing_scientific_report_fields(tmp_path):
    fixture = _selection_generator_fixture(tmp_path)
    report = json.loads(fixture["report_path"].read_text())
    report["cells"][0]["rows"][0]["certificate"]["differences"].pop(
        "bias_excess_over_target_self")
    fixture["report_path"].write_text(json.dumps(report))

    with pytest.raises(ValueError, match="search report shape.*bias_excess"):
        build_policy_articulation_selection(
            search_execution_manifest_path=fixture["manifest_path"],
            search_report_path=fixture["report_path"],
            arm_bank_path=fixture["bank_path"],
            packet_manifest_path=fixture["packet_path"],
            metric_panel_path=fixture["panel_path"],
            additional_artifact_paths=(fixture["model_path"],),
            selected_phase="validation",
            selected_partition="validation-items",
        )


def test_breadth_selection_generator_rejects_unbound_partition_and_report(tmp_path):
    fixture = _selection_generator_fixture(tmp_path)
    common = {
        "search_execution_manifest_path": fixture["manifest_path"],
        "search_report_path": fixture["report_path"],
        "arm_bank_path": fixture["bank_path"],
        "packet_manifest_path": fixture["packet_path"],
        "metric_panel_path": fixture["panel_path"],
        "additional_artifact_paths": (fixture["model_path"],),
        "selected_phase": "validation",
    }
    with pytest.raises(ValueError, match="absent from the search-bound protocol"):
        build_policy_articulation_selection(
            **common, selected_partition="unbound-items")

    report = json.loads(fixture["report_path"].read_text())
    report["cells"].append(dict(report["cells"][0]))
    fixture["report_path"].write_text(json.dumps(report))
    with pytest.raises(ValueError, match="different cell panels"):
        build_policy_articulation_selection(
            **common, selected_partition="validation-items")

    report["cells"].pop()
    report["partition_authorization"]["execution_manifest_sha256"] = "0" * 64
    fixture["report_path"].write_text(json.dumps(report))
    with pytest.raises(ValueError, match="not bound to the exact frozen search execution"):
        build_policy_articulation_selection(
            **common, selected_partition="validation-items")


def test_breadth_identity_treats_same_gi_at_three_levels_as_three_cells():
    cells = [{
        "id": f"TB::demo::{level}::node",
        "domain": "demo",
        "task": "demo",
        "level": level,
        "bucket": "general",
        "node_id": f"demo::{level}::node",
        "metric_id": f"demo::{level}::node",
        "gi": 7,
        "construct": f"construct {level}",
    } for level in ("R1", "R2", "R3")]

    validation = validate_policy_cell_panel(cells, context="test breadth panel")

    assert validation["n_cells"] == 3
    assert validation["n_breadth_cells"] == 3
    for cell in cells:
        meta = [{**cell, "cell_id": cell["id"]}]
        result = _validate_scored_breadth_identity(
            meta, cell, label="executor"
        )
        assert result["valid"]
        assert result["level"] == cell["level"]


def test_breadth_score_identity_rejects_level_drift_even_when_gi_matches():
    cell = {
        "id": "TB::demo::R1::node",
        "domain": "demo",
        "task": "demo",
        "level": "R1",
        "bucket": "general",
        "node_id": "demo::R1::node",
        "metric_id": "demo::R1::node",
        "gi": 7,
        "construct": "construct",
    }
    meta = [{**cell, "cell_id": cell["id"], "level": "R2"}]
    with pytest.raises(ValueError, match="score/bank breadth identity mismatch"):
        _validate_scored_breadth_identity(meta, cell, label="executor")


def test_mutual_policy_fidelity_has_a_real_interval_gate():
    q = np.linspace(0.01, 0.99, 300)
    rng = np.random.default_rng(107)
    left = {"canonical": q, "question": np.clip(q + 0.002, 0, 1)}
    right_values = np.clip(q + rng.normal(0, 0.015, len(q)), 0, 1)
    right = {"canonical": right_values, "question": right_values}
    result = certify_pairwise_policy_fidelity(
        left, right, n_boot=300, seed=11, rho_floor=0.90)
    assert result["point"]["quotient_spearman"] > 0.99
    assert result["quotient_spearman_CI"][0] > 0.90
    assert result["gates"]["lower_CI_at_least_primary_floor"]
    assert result["schema"] == "pairwise_policy_fidelity/v3_interval_vector"


def test_mutual_rank_point_pass_does_not_replace_interval_pass():
    q = np.linspace(0.05, 0.95, 12)
    noisy = np.clip(q + np.random.default_rng(3).normal(0, 0.03, len(q)), 0, 1)
    result = certify_pairwise_policy_fidelity(
        {"canonical": q}, {"canonical": noisy},
        n_boot=300, seed=7, rho_floor=0.90, min_rank_valid_fraction=0.90,
    )
    assert result["gates"]["point_at_least_primary_floor"] is True
    assert result["gates"]["lower_CI_at_least_primary_floor"] is False


def test_mutual_rank_rejects_an_inverted_sensitivity_hierarchy():
    q = np.linspace(0.05, 0.95, 20)
    with pytest.raises(ValueError, match="cannot exceed"):
        certify_pairwise_policy_fidelity(
            {"canonical": q}, {"canonical": q},
            rho_floor=0.85, rho_sensitivity_floor=0.90,
        )


def test_content_specific_H_fiber_intersects_HJ_and_mutual_interval():
    q = np.linspace(0.01, 0.99, 250)
    rng = np.random.default_rng(19)
    arms = {
        "definition": _arm(
            "definition", "declarative",
            "precise lexical transformation with ambiguity and a rewarding reinterpretation"),
        "rubric": _arm(
            "rubric", "procedural",
            "inspect phonetic pivots then verify clarity restraint timing and payoff"),
    }
    orbits = {
        "definition": {"canonical": q, "question": q},
        "rubric": {
            "canonical": np.clip(q + rng.normal(0, 0.012, len(q)), 0, 1),
            "question": np.clip(q + rng.normal(0, 0.012, len(q)), 0, 1),
        },
    }
    membership = {}
    for grade in ("observed", "certified", "simultaneous_certified"):
        membership[
            f"{grade}_joint_fixed_target_endpoint_isomorphic_members"
        ] = ["definition", "rubric"]
        membership[
            f"{grade}_joint_fixed_target_endpoint_equivalent_members"
        ] = []
    result = _content_specific_joint_fiber(
        candidate_arm_ids=["definition", "rubric"],
        content_specific_membership=membership,
        arm_specs=arms,
        arm_orbits=orbits,
        bootstrap_clusters=None,
        n_boot=300,
        seed=31,
        confidence=0.95,
        mutual_rho_floor=0.90,
        mutual_rho_sensitivity_floor=0.85,
        min_rank_valid_fraction=0.99,
        mutual_mae_margin=0.02,
        mutual_flip_margin=0.02,
        mutual_bias_margin=0.02,
        distinctness_floor=0.35,
    )
    assert result["simultaneous_certified_H_fiber_pairs"] == [
        {"left": "definition", "right": "rubric"}]
    assert not result["simultaneous_certified_H_fiber_eq_pairs"]
    certificate = result["pair_certificates"][0]["mutual_policy_certificate"]
    assert certificate["gates"]["lower_CI_at_least_primary_floor"]

    strict = _content_specific_joint_fiber(
        candidate_arm_ids=["definition", "rubric"],
        content_specific_membership=membership,
        arm_specs=arms,
        arm_orbits={
            "definition": {"canonical": q, "question": q},
            "rubric": {"canonical": q, "question": q},
        },
        bootstrap_clusters=None,
        n_boot=100,
        seed=32,
        confidence=0.95,
        mutual_rho_floor=0.90,
        mutual_rho_sensitivity_floor=0.85,
        min_rank_valid_fraction=0.99,
        mutual_mae_margin=0.02,
        mutual_flip_margin=0.02,
        mutual_bias_margin=0.02,
        distinctness_floor=0.35,
    )
    assert strict["simultaneous_certified_H_fiber_vec_pairs"] == [
        {"left": "definition", "right": "rubric"}]


def test_H_fiber_pair_family_and_nested_component_gate_are_explicit():
    q = np.linspace(0.02, 0.98, 80)
    arms = {
        "a": _arm("a", "declarative", "alpha content"),
        "ab": _arm("ab", "procedural", "alpha content plus beta"),
        "c": _arm("c", "ostensive", "contrastive examples and boundaries"),
    }
    arms["ab"]["components"] = ["a", "b"]
    membership = {}
    for grade in ("observed", "certified", "simultaneous_certified"):
        membership[f"{grade}_joint_fixed_target_endpoint_isomorphic_members"] = list(arms)
        membership[f"{grade}_joint_fixed_target_endpoint_equivalent_members"] = []
    result = _content_specific_joint_fiber(
        candidate_arm_ids=list(arms),
        content_specific_membership=membership,
        arm_specs=arms,
        arm_orbits={arm_id: {"canonical": q} for arm_id in arms},
        bootstrap_clusters=None,
        n_boot=50,
        seed=15,
        confidence=0.95,
        mutual_rho_floor=0.90,
        mutual_rho_sensitivity_floor=0.85,
        min_rank_valid_fraction=0.99,
        mutual_mae_margin=0.02,
        mutual_flip_margin=0.02,
        mutual_bias_margin=0.02,
        distinctness_floor=0.0,
    )
    assert result["pairwise_multiplicity"]["family_size"] == 3
    assert result["pairwise_multiplicity"][
        "per_comparison_central_interval_confidence"] == pytest.approx(
            1.0 - 0.05 / 3)
    nested = next(
        row for row in result["pair_certificates"]
        if {row["left"], row["right"]} == {"a", "ab"}
    )
    assert nested["components_incomparable"] is False
    assert nested["structural_gate"] is False


def test_mutual_rank_interval_fails_closed_when_bootstrap_ranks_degenerate():
    constant = {"canonical": np.full(30, 0.5), "question": np.full(30, 0.5)}
    result = certify_pairwise_policy_fidelity(
        constant,
        constant,
        n_boot=100,
        min_rank_valid_fraction=0.99,
    )
    assert result["bootstrap"]["rank_valid_fraction"] == 0.0
    assert result["bootstrap"]["rank_validity_pass"] is False
    assert result["gates"]["lower_CI_at_least_primary_floor"] is False


def test_mutual_rank_requires_matching_forms_and_rank_is_not_numeric_equality():
    q = np.linspace(0.05, 0.75, 120)
    shifted = np.clip(q + 0.20, 0, 1)
    result = certify_pairwise_policy_fidelity(
        {"canonical": q}, {"canonical": shifted}, n_boot=100, seed=4)
    assert result["point"]["quotient_spearman"] == pytest.approx(1.0)
    assert result["point"]["quotient_mae_tvd"] > 0.15
    assert result["gates"]["point_vector_equivalent"] is False
    with pytest.raises(ValueError, match="identical form-id sets"):
        certify_pairwise_policy_fidelity(
            {"canonical": q}, {"question": shifted}, n_boot=10)


def test_mutual_quotient_vector_equivalence_is_a_strict_nested_pair_grade():
    q = np.linspace(0.05, 0.95, 120)
    result = certify_pairwise_policy_fidelity(
        {"canonical": q, "question": q},
        {"canonical": q, "question": q},
        n_boot=100,
        seed=8,
    )
    assert result["gates"]["lower_CI_at_least_primary_floor"] is True
    assert result["gates"]["certified_vector_equivalent"] is True
    assert result["quotient_mae_tvd_CI"] == [0.0, 0.0]
    assert result["quotient_binary_flip_rate_CI"] == [0.0, 0.0]
    assert result["quotient_absolute_bias_CI"] == [0.0, 0.0]


def test_production_sentinel_routes_are_atomic_not_component_minimal():
    root = Path(__file__).parents[3]
    bank = json.loads((root / (
        "notebooks/data/two_faces_20260702/fresh_name_arm_bank_v1.json"
    )).read_text())
    cell = next(row for row in bank["cells"] if row["id"] == "N_humor_49")
    arms = {row["id"]: row for row in cell["arms"]}
    assert not arms["source_definition"].get("components")
    assert not arms["source_full_rubric"].get("components")
    assert arms["source_definition"]["channel"] == "declarative"
    assert arms["source_full_rubric"]["channel"] == "procedural"


def test_frozen_runner_invocation_and_repetitions_fail_closed():
    expected = {"small_job": "small", "n_boot": 10, "cell_ids": ["cell"]}
    manifest = {
        "analysis": {"runner": expected},
        "binary_readout": "teacher_forced_declared_labels",
        "readout_template_sha256": "readout",
        "execution_environment": {"production_backend_class": "OfflineVLLM"},
        "model_jobs": [{
            "id": "small", "required_repetitions": [0, 1],
            "role": "small_executor",
        }],
    }
    assert _validate_frozen_runner_invocation(manifest, expected)["valid"]
    with pytest.raises(ValueError, match="differs from frozen"):
        _validate_frozen_runner_invocation(
            manifest, {**expected, "n_boot": 11})

    bundle = {
        "repetitions": [0, 1],
        "execution_manifest_sha256": "execution",
        "arm_bank_sha256": "bank",
        "packet_manifest_sha256": "packet",
        "binary_readout": "teacher_forced_declared_labels",
        "readout_template_sha256": "readout",
        "role": "small_executor",
        "backend_class": "OfflineVLLM",
        "fake_backend": False,
    }
    assert _validate_frozen_score_bundle(
        bundle,
        label="small executor",
        job_id="small",
        manifest=manifest,
        execution_manifest_sha256="execution",
        arm_bank_sha256="bank",
        packet_manifest_sha256="packet",
        allow_fake_inputs=False,
    )["valid"]
    with pytest.raises(ValueError, match="repetitions differ"):
        _validate_frozen_score_bundle(
            {**bundle, "repetitions": [0]},
            label="small executor",
            job_id="small",
            manifest=manifest,
            execution_manifest_sha256="execution",
            arm_bank_sha256="bank",
            packet_manifest_sha256="packet",
            allow_fake_inputs=False,
        )
    with pytest.raises(ValueError, match="role differs"):
        _validate_frozen_score_bundle(
            {**bundle, "role": "fixed_target"},
            label="small executor",
            job_id="small",
            manifest=manifest,
            execution_manifest_sha256="execution",
            arm_bank_sha256="bank",
            packet_manifest_sha256="packet",
            allow_fake_inputs=False,
        )


def test_exact_target_policy_is_isomorphic_and_rescues_noisy_sparse():
    rng = np.random.default_rng(9)
    q = np.tile(np.linspace(0.02, 0.98, 80), 3)
    target = {"canonical": q, "question": np.clip(q + 0.005, 0, 1)}
    sparse = {"canonical": np.clip(q + rng.normal(0, 0.22, len(q)), 0, 1),
              "question": np.clip(q + rng.normal(0, 0.22, len(q)), 0, 1)}
    candidate = {"canonical": q.copy(), "question": np.clip(q + 0.005, 0, 1)}
    result = certify_policy_isomorphism(
        target, candidate, sparse_orbit=sparse, n_boot=300, seed=2,
        mae_margin=0.02, rho_margin=0.05, flip_margin=0.02, bias_margin=0.02)
    assert result["policy_isomorphic"]
    assert not result["small_sparse_isomorphic"]
    assert result["gates"]["mae_improves_over_small_sparse"]
    assert result["articulation_rescue"]


def test_equal_but_different_fiber_prefers_isomorphism_before_diversity():
    q = np.tile(np.linspace(0.03, 0.97, 100), 2)
    target = {"canonical": q, "question": np.clip(q + 0.004, 0, 1)}
    sparse_values = np.clip(
        q + np.random.default_rng(6).normal(0, 0.50, len(q)), 0, 1
    )
    sparse = {"canonical": sparse_values, "question": sparse_values}
    orbits = {
        "definition": {"canonical": q, "question": np.clip(q + 0.004, 0, 1)},
        "examples": {"canonical": np.clip(q + 0.002, 0, 1),
                     "question": np.clip(q + 0.006, 0, 1)},
        "diverse_but_bad": {"canonical": 1.0 - q, "question": 1.0 - q},
    }
    specs = {
        "definition": _arm("definition", "declarative", "A compact constitutive definition."),
        "examples": _arm("examples", "ostensive", "Several positive and negative boundary cases."),
        "diverse_but_bad": _arm("diverse_but_bad", "formative", "Unrelated elaborate material."),
    }
    rows = []
    for index, (arm_id, orbit) in enumerate(orbits.items()):
        rows.append({"arm_id": arm_id,
                     "certificate": certify_policy_isomorphism(
                         target, orbit, sparse_orbit=sparse, n_boot=250, seed=index)})
    fiber = summarize_isomorphism_fiber(rows, specs, orbits, performance_slack=0.01)
    assert set(fiber["members"]) == {"definition", "examples"}
    assert "diverse_but_bad" not in fiber["selected_diverse"]
    assert fiber["n_equal_but_different_pairs"] == 1
    assert fiber["equal_but_different_pairs"][0]["pairwise_gate_grade"] == "point_only"
    assert fiber["n_observed_functional_equal_but_different_pairs"] == 1
    assert fiber["n_certified_functional_equal_but_different_pairs"] == 1
    assert fiber["observed_functional_component_minimal_members"] == [
        "definition", "examples"]
    assert articulation_distance(specs["definition"], specs["examples"]) > 0.35


def test_fiber_diversity_pool_uses_best_member_band_and_excludes_controls():
    values = np.linspace(0.05, 0.95, 20)
    orbits = {
        arm_id: {"canonical": values, "question": values}
        for arm_id in ("best", "near_best", "distant_member", "control")
    }
    specs = {
        "best": _arm("best", "declarative", "A short direct definition."),
        "near_best": _arm(
            "near_best", "ostensive", "Many contrasting boundary examples."
        ),
        "distant_member": _arm(
            "distant_member", "formative", "A lengthy procedural teaching sequence."
        ),
        "control": {
            **_arm("control", "declarative", "Matched but irrelevant content."),
            "provenance": "wrong_construct_control",
            "control_for": "best",
        },
    }
    rows = []
    for arm_id, mae in (
        ("best", 0.10),
        ("near_best", 0.105),
        ("distant_member", 0.30),
        ("control", 0.01),
    ):
        rows.append({
            "arm_id": arm_id,
            "provenance": specs[arm_id]["provenance"],
            "control_for": specs[arm_id].get("control_for"),
            "certificate": {
                "point": {"candidate_robust": {"mae_tvd": mae}},
                "policy_isomorphic": True,
                "functional": {
                    "observed_functional_policy_substitution": False,
                    "certified_functional_policy_substitution": False,
                },
            },
        })
    fiber = summarize_isomorphism_fiber(
        rows, specs, orbits, performance_slack=0.01
    )
    assert fiber["n_tested"] == 3
    assert fiber["controls_excluded_from_fiber"] == ["control"]
    assert fiber["best_adverse_mae_tvd"] == pytest.approx(0.10)
    assert set(fiber["members"]) == {"best", "near_best", "distant_member"}
    assert set(fiber["diversity_pool"]) == {"best", "near_best"}
    assert set(fiber["selected_diverse"]) == {"best", "near_best"}


def test_pairwise_behavior_gate_and_threshold_sensitivity_are_explicit():
    left = np.linspace(0.05, 0.95, 10)
    right = left.copy()
    right[[2, 5]] = right[[5, 2]]
    assert 0.85 <= spearman(left, right) < 0.90
    orbits = {
        "definition": {"canonical": left},
        "examples": {"canonical": right},
    }
    specs = {
        "definition": _arm(
            "definition", "declarative", "A compact constitutive definition."
        ),
        "examples": _arm(
            "examples", "ostensive", "Many positive and negative boundary cases."
        ),
    }
    rows = [{
        "arm_id": arm_id,
        "certificate": {
            "point": {"candidate_robust": {"mae_tvd": 0.10}},
            "policy_isomorphic": True,
            "functional": {
                "observed_functional_policy_substitution": True,
                "certified_functional_policy_substitution": True,
            },
        },
    } for arm_id in orbits]
    fiber = summarize_isomorphism_fiber(rows, specs, orbits)
    assert fiber["n_equal_but_different_pairs"] == 0
    assert fiber["n_observed_functional_equal_but_different_pairs"] == 0
    profile = {
        row["rho_floor"]: row
        for row in fiber["pairwise_behavior_threshold_sensitivity"]
    }
    assert profile[0.85]["near_identity"]["n_pairs"] == 1
    assert profile[0.85]["observed_functional"]["n_pairs"] == 1
    assert profile[0.90]["near_identity"]["n_pairs"] == 0
    assert profile[0.85]["pairwise_gate_grade"] == "point_only"


def test_empty_fiber_has_complete_summary_shape():
    fiber = summarize_isomorphism_fiber([], {}, {})
    assert fiber["fiber_status"] == "no_eligible_arms"
    assert fiber["n_equal_but_different_pairs"] == 0
    assert fiber["equal_but_different_pairs"] == []
    assert fiber["n_observed_functional_equal_but_different_pairs"] == 0


def test_direct_runner_rejects_undeclared_partition_before_reading_shards():
    with pytest.raises(ValueError, match="does not authorize partition"):
        run(
            executor_shard_root="missing",
            arm_bank_path="missing",
            partition="gestalt_lockbox",
        )


def test_partition_source_groups_are_authenticated_and_hash_aligned(tmp_path):
    partition = "residual_prompt_selection"
    item_dir = tmp_path / "packet" / "humor" / "items"
    item_dir.mkdir(parents=True)
    rows = []
    for index, (text, group) in enumerate((
        ("first item", "thread-a"),
        ("second item", "thread-a"),
        ("third item", "thread-b"),
    )):
        rows.append({
            "item_id": str(index),
            "text": text,
            "text_sha256": text_sha256(text),
            "source_group": group,
            "source_split": None,
        })
    item_path = item_dir / f"{partition}.jsonl"
    item_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    ordered_hash = sha256_bytes(
        "\n".join(row["text_sha256"] for row in rows).encode()
    )
    manifest = {
        "schema": "fresh_item_partitions/v1",
        "domains": [{
            "domain": "humor",
            "source_group_method": "test-thread",
            "holdout_grade": "source-group-disjoint",
            "partitions": [{
                "id": partition,
                "n": len(rows),
                "items_sha256": sha256_file(item_path),
                "ordered_item_set_sha256": ordered_hash,
                "n_source_groups": 2,
            }],
        }],
    }
    manifest_path = tmp_path / "packet" / "packet_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    reversed_hashes = [row["text_sha256"] for row in reversed(rows)]
    loaded = load_partition_source_groups(
        tmp_path / "packet",
        manifest_path,
        domain="humor",
        partition=partition,
        item_hashes=reversed_hashes,
    )
    assert loaded["source_groups"] == ["thread-b", "thread-a", "thread-a"]
    assert loaded["validation"]["n_source_groups"] == 2
    assert loaded["validation"]["items_sha256"] == sha256_file(item_path)

    item_path.write_text(item_path.read_text() + "\n")
    with pytest.raises(ValueError, match="file SHA-256 mismatch"):
        load_partition_source_groups(
            tmp_path / "packet",
            manifest_path,
            domain="humor",
            partition=partition,
            item_hashes=reversed_hashes,
        )


def test_direct_runner_rejects_executor_target_readout_mismatch(tmp_path, monkeypatch):
    from methods.codability.experiments import run_policy_isomorphism as runner

    bank_path = tmp_path / "bank.json"
    bank_path.write_text(json.dumps({
        "cells": [{"id": "cell", "domain": "humor", "arms": []}],
    }))
    small = {"readout_template_sha256": "small-readout"}
    big = {"readout_template_sha256": "target-readout"}
    index = {("small", "humor"): small, ("big", "humor"): big}
    monkeypatch.setattr(runner, "load_public_index", lambda *_args: index)
    monkeypatch.setattr(runner, "_average_repetitions", lambda value: value)
    with pytest.raises(ValueError, match="readout identity mismatch"):
        run(
            executor_shard_root="executor",
            arm_bank_path=str(bank_path),
            partition="residual_prompt_selection",
            small_job="small",
            big_job="big",
        )


def _patch_minimal_frozen_runner(monkeypatch, runner, *, phase):
    monkeypatch.setattr(
        runner,
        "authorize_policy_partition",
        lambda *_args, **_kwargs: {
            "phase": phase,
            "execution_manifest_sha256": "execution-sha",
        },
    )
    monkeypatch.setattr(runner, "validate_frozen_implementation", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(runner, "_validate_frozen_runner_invocation", lambda *_args: {})
    monkeypatch.setattr(runner, "_validate_frozen_score_bundle", lambda *_args, **_kwargs: {})
    small = {"readout_template_sha256": "small-readout"}
    big = {"readout_template_sha256": "target-readout"}
    index = {("small", "humor"): small, ("big", "humor"): big}
    monkeypatch.setattr(runner, "load_public_index", lambda *_args: index)
    monkeypatch.setattr(runner, "_average_repetitions", lambda value: value)


def test_frozen_runner_open_search_does_not_require_selection(tmp_path, monkeypatch):
    from methods.codability.experiments import run_policy_isomorphism as runner

    bank_path = tmp_path / "bank.json"
    bank_path.write_text(json.dumps({
        "cells": [{"id": "cell", "domain": "humor", "arms": []}],
    }))
    packet_path = tmp_path / "packet.json"
    packet_path.write_text("{}")
    manifest_path = tmp_path / "execution.json"
    manifest_path.write_text(json.dumps({
        "phases": {"search": ["tacit_breadth_search"]},
        "selection_required_phases": [],
        "model_jobs": [{"id": "small"}, {"id": "big"}],
        "arm_bank_sha256": sha256_file(bank_path),
        "packet_manifest_sha256": sha256_file(packet_path),
    }))
    _patch_minimal_frozen_runner(monkeypatch, runner, phase="search")
    monkeypatch.setattr(
        runner,
        "load_lockbox_selection",
        lambda *_args, **_kwargs: pytest.fail("all-arm search loaded a selection"),
    )

    with pytest.raises(ValueError, match="readout identity mismatch"):
        run(
            executor_shard_root="executor",
            arm_bank_path=str(bank_path),
            packet_root=str(tmp_path),
            packet_manifest_path=str(packet_path),
            partition="tacit_breadth_search",
            small_job="small",
            big_job="big",
            execution_manifest_path=str(manifest_path),
            allow_fake_inputs=True,
        )


def test_frozen_runner_loads_selection_for_current_validation_phase(
        tmp_path, monkeypatch):
    from methods.codability.experiments import run_policy_isomorphism as runner

    bank_path = tmp_path / "bank.json"
    bank_path.write_text(json.dumps({
        "cells": [{"id": "cell", "domain": "humor", "arms": []}],
    }))
    packet_path = tmp_path / "packet.json"
    packet_path.write_text("{}")
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps({
        "schema": "policy_articulation_selection/v1",
    }))
    manifest_path = tmp_path / "execution.json"
    manifest_path.write_text(json.dumps({
        "phases": {"validation": ["tacit_breadth_validation"]},
        "selection_required_phases": ["validation"],
        "model_jobs": [{"id": "small"}, {"id": "big"}],
        "arm_bank_sha256": sha256_file(bank_path),
        "packet_manifest_sha256": sha256_file(packet_path),
    }))
    _patch_minimal_frozen_runner(monkeypatch, runner, phase="validation")
    observed = {}

    def load_selection(*_args, **kwargs):
        observed.update(kwargs)
        return {"cell": set()}

    monkeypatch.setattr(runner, "load_lockbox_selection", load_selection)
    with pytest.raises(ValueError, match="readout identity mismatch"):
        run(
            executor_shard_root="executor",
            arm_bank_path=str(bank_path),
            packet_root=str(tmp_path),
            packet_manifest_path=str(packet_path),
            partition="tacit_breadth_validation",
            small_job="small",
            big_job="big",
            execution_manifest_path=str(manifest_path),
            selection_artifact_path=str(selection_path),
            allow_fake_inputs=True,
        )

    assert observed["expected_phase"] == "validation"
    assert observed["expected_partition"] == "tacit_breadth_validation"


def test_frozen_runner_authenticates_additional_artifacts(tmp_path, monkeypatch):
    from methods.codability.experiments import run_policy_isomorphism as runner

    bank_path = tmp_path / "bank.json"
    bank_path.write_text(json.dumps({"cells": []}))
    packet_path = tmp_path / "packet.json"
    packet_path.write_text("{}")
    panel_path = tmp_path / "panel.json"
    panel_path.write_text("{}")
    manifest_path = tmp_path / "execution.json"
    manifest_path.write_text(json.dumps({
        "phases": {"search": ["tacit_breadth_search"]},
        "selection_required_phases": [],
        "arm_bank_sha256": sha256_file(bank_path),
        "packet_manifest_sha256": sha256_file(packet_path),
        "additional_artifacts": [{"path": str(panel_path), "sha256": "changed"}],
    }))
    monkeypatch.setattr(
        runner,
        "authorize_policy_partition",
        lambda *_args, **_kwargs: {"phase": "search"},
    )

    with pytest.raises(ValueError, match="additional artifact changed"):
        run(
            executor_shard_root="executor",
            arm_bank_path=str(bank_path),
            packet_root=str(tmp_path),
            packet_manifest_path=str(packet_path),
            partition="tacit_breadth_search",
            execution_manifest_path=str(manifest_path),
            allow_fake_inputs=True,
        )


def test_direct_runner_integrates_scale_comparator_and_excludes_controls(
        tmp_path, monkeypatch):
    from methods.codability.experiments import run_policy_isomorphism as runner

    n_items = 120
    q = np.linspace(0.02, 0.98, n_items)
    rng = np.random.default_rng(81)
    small_values = np.clip(q + rng.normal(0, 0.32, n_items), 0, 1)
    candidate_values = np.clip(q + rng.normal(0, 0.07, n_items), 0, 1)
    control_values = np.clip(q + rng.normal(0, 0.28, n_items), 0, 1)
    larger_values = np.clip(q + rng.normal(0, 0.12, n_items), 0, 1)
    prompt_hashes = {"name": "name-sha", "candidate": "candidate-sha",
                     "control": "control-sha"}
    arms = [{
        "id": arm_id,
        "channel": "control" if arm_id == "control" else "declarative",
        "provenance": (
            "inert_length_control" if arm_id == "control" else "source_telling"),
        "control_for": "candidate" if arm_id == "control" else None,
        "components": [arm_id],
        "semantic_content_word_count": 3,
        "forms": [{
            "id": "canonical",
            "prompt": f"prompt for {arm_id}",
            "prompt_sha256": prompt_hashes[arm_id],
        }],
    } for arm_id in ("name", "candidate", "control")]
    for arm in arms:
        arm["added_content_word_count"] = 2
    next(arm for arm in arms if arm["id"] == "candidate")["n_address_units"] = 2
    next(arm for arm in arms if arm["id"] == "control")["n_address_units"] = 2
    bank_path = tmp_path / "bank.json"
    bank_path.write_text(json.dumps({
        "cells": [{
            "id": "cell", "domain": "humor", "gi": 1,
            "construct": "synthetic", "arms": arms,
        }],
    }))
    hashes = [f"item-{index}" for index in range(n_items)]

    def data(arm_values, *, shard):
        arm_ids = list(arm_values)
        return {
            "scores": np.stack([arm_values[arm_id] for arm_id in arm_ids]),
            "meta": [{
                "cell_id": "cell", "arm_id": arm_id, "form": "canonical",
                "prompt_sha256": prompt_hashes.get(arm_id, "target-sha"),
            } for arm_id in arm_ids],
            "hashes": hashes,
            "shard_sha256": [shard],
            "readout_template_sha256": "same-readout",
        }

    indexes = {
        "small-root": {("small", "humor"): data({
            "name": small_values,
            "candidate": candidate_values,
            "control": control_values,
        }, shard="small")},
        "target-root": {("target", "humor"): data({"target": q}, shard="target")},
        "large-root": {("large", "humor"): data({
            "name": larger_values,
        }, shard="large")},
    }
    monkeypatch.setattr(runner, "load_public_index", lambda root, _partition: indexes[root])
    monkeypatch.setattr(runner, "_average_repetitions", lambda value: value)

    report = run(
        executor_shard_root="small-root",
        target_shard_root="target-root",
        scale_comparator_shard_root="large-root",
        scale_comparator_job="large",
        arm_bank_path=str(bank_path),
        partition="residual_prompt_selection",
        small_job="small",
        big_job="target",
        target_arm_id="target",
        include_controls=True,
        n_boot=80,
        seed=4,
    )
    rows = {row["arm_id"]: row for row in report["cells"][0]["rows"]}
    assert report["scale_comparator"]["enabled"]
    assert report["schema"] == "policy_isomorphism_experiment/v5"
    assert report["cells"][0]["scale_comparator_validation"]["valid"]
    assert rows["candidate"]["scale_step_certificate"] is not None
    assert rows["candidate"]["n_address_units"] == 2
    assert rows["candidate"]["added_content_word_count"] == 2
    assert rows["control"]["n_address_units"] == 2
    assert rows["candidate"]["scale_step_certificate"][
        "direct_endpoint_isomorphism"]["observed"]["functional_ordinal_fidelity"]
    assert rows["candidate"]["scale_step_multiplicity"]["family_size"] == 1
    assert rows["control"]["scale_step_certificate"] is None
    assert "control" in report["cells"][0]["fiber"]["controls_excluded_from_fiber"]
    assert report["summary"][
        "n_observed_functional_endpoint_isomorphic_scale_substitutions"] == 1
    specificity = report["cells"][0]["content_specific_scale_step"]
    assert specificity["multiplicity"]["family_size"] == 2
    assert rows["candidate"]["scale_step_specificity_simultaneous_certificate"][
        "bootstrap"]["confidence"] == pytest.approx(0.975)
    assert report["cells"][0]["matched_control_certificates"][0][
        "specificity_simultaneous_certificate"]["bootstrap"][
            "confidence"] == pytest.approx(0.975)
    assert not specificity[
        "observed_joint_fixed_target_endpoint_isomorphic_members"]
    candidate_grade = next(
        row for row in specificity["arm_grades"] if row["arm_id"] == "candidate")
    assert candidate_grade["missing_control_provenances"] == [
        "wrong_construct_control"]
    assert report["summary"][
        "n_observed_content_specific_joint_fixed_target_endpoint_isomorphic_"
        "scale_substitutions"] == 0


def test_direct_runner_requires_complete_explicit_scale_comparator_config():
    with pytest.raises(ValueError, match="requires both shard root and job"):
        run(
            executor_shard_root="missing",
            target_shard_root="missing",
            scale_comparator_shard_root="larger",
            arm_bank_path="missing",
            partition="residual_prompt_selection",
        )


def test_functional_rank_tier_does_not_overwrite_near_identity_tier():
    q = np.linspace(0.02, 0.98, 400)
    rng = np.random.default_rng(3)
    candidate_values = np.clip(q + rng.normal(0, 0.20, len(q)), 0, 1)
    sparse_values = np.clip(q + rng.normal(0, 0.38, len(q)), 0, 1)
    target = {"canonical": q, "question": q}
    candidate = {"canonical": candidate_values, "question": candidate_values}
    sparse = {"canonical": sparse_values, "question": sparse_values}
    result = certify_policy_isomorphism(
        target,
        candidate,
        sparse_orbit=sparse,
        functional_rho_floor=0.70,
        n_boot=400,
        seed=7,
    )
    assert result["functional"]["observed_functional_ordinal_isomorphism"]
    assert result["functional"]["certified_functional_ordinal_isomorphism"]
    assert result["functional"]["certified_functional_policy_substitution"]
    assert not result["policy_isomorphic"]


def test_functional_rank_requires_form_quotient_as_well_as_each_adverse_form():
    raw_left = np.array([
        2.923444732362179,
        -0.6467807241308587,
        0.3945317780735431,
        3.853369883566377,
        7.166268829497236,
        4.74386525403164,
    ])
    raw_right = np.array([
        2.1312492694089107,
        -3.367704663419877,
        3.010899375014976,
        6.313307626325339,
        4.405923712207413,
        3.4113848255685406,
    ])
    low = min(raw_left.min(), raw_right.min(), 0.0)
    high = max(raw_left.max(), raw_right.max(), 5.0)

    def probability(values):
        return 0.05 + 0.90 * (np.asarray(values) - low) / (high - low)

    q = probability(np.arange(6, dtype=float))
    result = certify_policy_isomorphism(
        {"canonical": q, "question": q},
        {"canonical": probability(raw_left), "question": probability(raw_right)},
        functional_rho_floor=0.70,
        n_boot=200,
        seed=12,
    )
    gates = result["functional"]["gates"]
    assert gates["adverse_rank_point_at_least_floor"]
    assert not gates["quotient_rank_point_at_least_floor"]
    assert result["functional"]["quotient_rho_point"] == pytest.approx(
        0.6571428571428573
    )
    assert result["functional"]["quotient_rho_CI"] is not None
    assert not result["functional"]["observed_functional_ordinal_isomorphism"]


def test_functional_substitution_requires_a_small_sparse_rank_gap():
    q = np.linspace(0.02, 0.98, 400)
    rng = np.random.default_rng(17)
    target = {"canonical": q, "question": q}
    sparse_values = np.clip(q + rng.normal(0, 0.10, len(q)), 0, 1)
    candidate_values = np.clip(q + rng.normal(0, 0.07, len(q)), 0, 1)
    sparse = {"canonical": sparse_values, "question": sparse_values}
    candidate = {"canonical": candidate_values, "question": candidate_values}
    result = certify_policy_isomorphism(
        target,
        candidate,
        sparse_orbit=sparse,
        functional_rho_floor=0.70,
        n_boot=400,
        seed=9,
    )
    assert result["functional"]["certified_functional_ordinal_isomorphism"]
    assert result["functional"]["gates"]["mae_CI_improves_over_small_sparse"]
    assert not result["functional"]["gates"][
        "small_sparse_point_below_functional_floor"
    ]
    assert not result["functional"]["gates"][
        "small_sparse_upper_CI_below_functional_floor"
    ]
    assert not result["functional"]["observed_functional_policy_substitution"]
    assert not result["functional"]["certified_functional_policy_substitution"]


def test_stratified_bootstrap_preserves_declared_fold_structure():
    q = np.linspace(0.02, 0.98, 200)
    target = {"canonical": q, "question": q}
    candidate = {"canonical": q, "question": q}
    result = certify_policy_isomorphism(
        target,
        candidate,
        bootstrap_strata=["fold_a"] * 80 + ["fold_b"] * 120,
        n_boot=100,
        seed=4,
    )
    assert result["bootstrap"]["n_strata"] == 2
    assert result["bootstrap"]["n_requested"] == 100
    assert result["bootstrap"]["rank_draw_counts"]["candidate_valid"] == 100
    assert "stratified" in result["bootstrap"]["sampling"]
    assert "recomputed within each resample" in result["bootstrap"]["rank_method"]
    with pytest.raises(ValueError, match="one label per item"):
        certify_policy_isomorphism(
            target,
            candidate,
            bootstrap_strata=["too_short"],
            n_boot=10,
        )


def test_singleton_source_groups_exactly_reproduce_item_bootstrap_draws():
    q = np.linspace(0.02, 0.98, 120)
    rng = np.random.default_rng(31)
    candidate_values = np.clip(q + rng.normal(0, 0.16, len(q)), 0, 1)
    target = {"canonical": q, "question": q}
    candidate = {"canonical": candidate_values, "question": candidate_values}
    plain = certify_policy_isomorphism(target, candidate, n_boot=250, seed=14)
    clustered = certify_policy_isomorphism(
        target,
        candidate,
        bootstrap_clusters=[f"item-{index}" for index in range(len(q))],
        n_boot=250,
        seed=14,
    )
    assert clustered["point"] == plain["point"]
    assert clustered["differences"] == plain["differences"]
    assert clustered["functional"]["adverse_rho_CI"] == plain["functional"][
        "adverse_rho_CI"
    ]
    assert clustered["bootstrap"]["all_source_groups_singleton"]
    assert "exactly reproduce" in clustered["bootstrap"]["sampling"]


@pytest.mark.parametrize("sparse_noise", [0.005, 0.30])
def test_nonrecursive_sparse_identity_matches_standalone_certificate(sparse_noise):
    q = np.tile(np.linspace(0.02, 0.98, 90), 2)
    sparse_values = np.clip(
        q + np.random.default_rng(61).normal(0, sparse_noise, len(q)), 0, 1
    )
    target = {"canonical": q, "question": np.clip(q + 0.004, 0, 1)}
    sparse = {"canonical": sparse_values, "question": sparse_values}
    candidate = {"canonical": q, "question": q}
    clusters = [f"source-{index // 3}" for index in range(len(q))]
    kwargs = {
        "bootstrap_clusters": clusters,
        "n_boot": 150,
        "seed": 27,
    }
    combined = certify_policy_isomorphism(
        target, candidate, sparse_orbit=sparse, **kwargs)
    standalone = certify_policy_isomorphism(target, sparse, **kwargs)
    assert combined["small_sparse_isomorphic"] == standalone["policy_isomorphic"]


def test_cluster_draws_retain_all_group_members_with_fold_stratification():
    samples, design = _bootstrap_samples(
        rng=np.random.default_rng(3),
        n_boot=30,
        n_items=10,
        strata=["a"] * 5 + ["b"] * 5,
        clusters=["x", "x", "y", "y", "y", "x", "x", "z", "z", "z"],
    )
    assert isinstance(samples, list)
    assert design["n_strata"] == 2
    assert design["n_source_groups"] == 4
    assert design["resampling_unit"] == "source_group_with_all_member_items_retained"
    for draw in samples:
        counts = np.bincount(draw, minlength=10)
        assert counts[0] == counts[1]
        assert counts[2] == counts[3] == counts[4]
        assert counts[5] == counts[6]
        assert counts[7] == counts[8] == counts[9]
        # Two cluster occurrences are sampled independently inside each fold.
        assert counts[0] + counts[2] == 2
        assert counts[5] + counts[7] == 2


def test_bootstrap_spearman_reranks_ties_inside_each_paired_resample():
    q = np.array([0.05, 0.12, 0.21, 0.38, 0.55, 0.66, 0.81, 0.95])
    values = np.array([0.11, 0.30, 0.18, 0.61, 0.42, 0.72, 0.59, 0.90])
    samples = np.array([
        [0, 0, 1, 3, 3, 4, 7, 7],
        [0, 2, 2, 2, 5, 6, 6, 7],
        [1, 1, 1, 3, 4, 5, 5, 6],
    ])
    draws = _bootstrap_orbit(
        {"canonical": q}, {"canonical": values}, samples
    )["candidate"]["spearman"]
    expected = np.array([spearman(q[index], values[index]) for index in samples])
    np.testing.assert_allclose(draws, expected)

    # This is the old, incorrect statistic: Pearson correlation after resampling
    # midranks computed once on the full panel. It differs when items repeat.
    full_q_rank = _rank(q)
    full_value_rank = _rank(values)
    fixed_rank_draws = []
    for index in samples:
        q_rank = full_q_rank[index] - np.mean(full_q_rank[index])
        value_rank = full_value_rank[index] - np.mean(full_value_rank[index])
        fixed_rank_draws.append(
            np.sum(q_rank * value_rank)
            / np.sqrt(np.sum(q_rank ** 2) * np.sum(value_rank ** 2))
        )
    assert not np.allclose(draws, fixed_rank_draws)


def test_bootstrap_reports_invalid_rank_draws_and_stratification_can_prevent_them():
    q = np.array([0.1, 0.1, 0.9, 0.9])
    target = {"canonical": q}
    candidate = {"canonical": q}
    iid = certify_policy_isomorphism(
        target, candidate, n_boot=200, seed=8
    )
    iid_counts = iid["bootstrap"]["rank_draw_counts"]
    assert iid["bootstrap"]["n_requested"] == 200
    assert 0 < iid_counts["candidate_valid"] < 200
    assert iid_counts["candidate_target_self_paired_valid"] == iid_counts[
        "candidate_valid"
    ]

    stratified = certify_policy_isomorphism(
        target,
        candidate,
        bootstrap_strata=["low", "low", "high", "high"],
        n_boot=200,
        seed=8,
    )
    assert stratified["bootstrap"]["rank_draw_counts"] == {
        "candidate_valid": 200,
        "candidate_quotient_valid": 200,
        "target_self_valid": 200,
        "candidate_target_self_paired_valid": 200,
    }


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_boot": 0}, "positive integer"),
        ({"n_boot": 2.5}, "positive integer"),
        ({"confidence": 0.0}, "strictly between"),
        ({"confidence": 1.0}, "strictly between"),
        ({"confidence": float("nan")}, "strictly between"),
    ],
)
def test_bootstrap_arguments_are_validated(kwargs, message):
    q = np.linspace(0.1, 0.9, 8)
    with pytest.raises(ValueError, match=message):
        certify_policy_isomorphism({"canonical": q}, {"canonical": q}, **kwargs)


def test_matched_control_comparison_uses_paired_rank_and_mae_differences():
    q = np.linspace(0.02, 0.98, 300)
    rng = np.random.default_rng(22)
    target = {"canonical": q, "question": q}
    source_values = np.clip(q + rng.normal(0, 0.08, len(q)), 0, 1)
    control_values = np.clip(q + rng.normal(0, 0.30, len(q)), 0, 1)
    source = {"canonical": source_values, "question": source_values}
    control = {"canonical": control_values, "question": control_values}
    result = compare_articulation_to_matched_control(
        target, source, control, n_boot=300, seed=5)
    assert result["bootstrap"]["n_requested"] == 300
    assert result["bootstrap"]["n_paired_rank_valid"] == 300
    assert "recomputed within each resample" in result["bootstrap"]["rank_method"]
    assert result["gates"]["source_rank_better_CI"]
    assert result["gates"]["source_mae_better_CI"]
    clustered = compare_articulation_to_matched_control(
        target,
        source,
        control,
        bootstrap_clusters=[f"group-{index // 3}" for index in range(len(q))],
        n_boot=200,
        seed=5,
    )
    assert clustered["bootstrap"]["n_source_groups"] == 100
    assert "cluster bootstrap" in clustered["bootstrap"]["sampling"]
    assert clustered["gates"]["source_rank_better_CI"]
    assert clustered["gates"]["source_mae_better_CI"]


def test_scale_step_certificate_separates_local_recovery_from_target_fidelity():
    q = np.tile(np.linspace(0.02, 0.98, 300), 2)
    rng = np.random.default_rng(812)
    small_values = np.clip(q + rng.normal(0, 0.34, len(q)), 0, 1)
    larger_values = np.clip(q + rng.normal(0, 0.12, len(q)), 0, 1)
    target = {"canonical": q, "question": q}
    small = {"canonical": small_values, "question": small_values}
    larger = {"canonical": larger_values, "question": larger_values}
    candidate = {"canonical": larger_values.copy(), "question": larger_values.copy()}

    result = certify_scale_step_substitution(
        target,
        small,
        candidate,
        larger,
        functional_rho_floor=0.99,
        n_boot=300,
        seed=19,
    )
    observed = result["evidence"]["observed"]
    certified = result["evidence"]["certified"]
    assert observed["native_scale_gap"]
    assert certified["native_scale_gap"]
    assert certified["articulation_gain"]
    assert certified["endpoint_noninferior_primary"]
    assert certified["endpoint_noninferior_vector"]
    assert certified["local_primary_scale_substitution"]
    assert certified["local_vector_scale_substitution"]
    assert not certified["functional_target_scale_substitution"]
    assert result["target_fidelity"]["certified"]["tier"] == (
        "below_functional_ordinal"
    )
    assert result["bootstrap"]["n_joint_rank_valid"] == 300
    assert "one shared paired" in result["bootstrap"]["joint_draw_contract"]


def test_scale_step_direct_endpoint_certificate_uses_shared_draws_and_baseline_gap():
    q = np.tile(np.linspace(0.02, 0.98, 350), 2)
    rng = np.random.default_rng(981)
    small_values = np.clip(q + rng.normal(0, 0.42, len(q)), 0, 1)
    larger_values = np.clip(q + rng.normal(0, 0.07, len(q)), 0, 1)
    target = {"canonical": q, "question": q}
    small = {"canonical": small_values, "question": small_values}
    larger = {"canonical": larger_values, "question": larger_values}

    result = certify_scale_step_substitution(
        target,
        small,
        larger,
        larger,
        n_boot=250,
        seed=51,
    )
    direct = result["direct_endpoint_isomorphism"]
    assert result["schema"] == "scale_step_policy_substitution/v2"
    assert direct["schema"] == "direct_larger_sparse_endpoint_isomorphism/v1"
    assert direct["bootstrap"]["shared_with_scale_step"]
    assert direct["bootstrap"]["rank_draw_counts"][
        "all_direct_rank_coordinates_jointly_valid"] == 250
    assert direct["observed"]["functional_ordinal_fidelity"]
    assert direct["certified"]["functional_ordinal_fidelity"]
    assert direct["certified"]["direct_mae_improvement_over_small_sparse"]
    assert direct["certified"]["small_sparse_outside_functional_region"]
    assert direct["certified"]["small_sparse_outside_near_identity_region"]
    assert direct["certified"]["target_self_band_near_identity"]
    assert direct["certified"]["near_identity_policy_substitution"]
    certified = result["evidence"]["certified"]
    assert certified["endpoint_two_sided_equivalent_primary"]
    assert certified["local_functional_endpoint_isomorphic_scale_substitution"]
    assert certified["local_functional_endpoint_equivalent_scale_substitution"]
    assert certified["local_near_identity_isomorphic_scale_substitution"]


def test_scale_step_overshoot_noninferiority_is_not_endpoint_isomorphism():
    q = np.linspace(0.02, 0.98, 800)
    rng = np.random.default_rng(1701)
    small_values = np.clip(q + rng.normal(0, 0.52, len(q)), 0, 1)
    larger_values = np.clip(q + rng.normal(0, 0.30, len(q)), 0, 1)
    target = {"canonical": q}
    small = {"canonical": small_values}
    larger = {"canonical": larger_values}
    overshooting_candidate = {"canonical": q.copy()}

    result = certify_scale_step_substitution(
        target,
        small,
        overshooting_candidate,
        larger,
        n_boot=200,
        seed=63,
    )
    observed = result["evidence"]["observed"]
    direct = result["direct_endpoint_isomorphism"]["observed"]
    # Candidate is superior to the larger endpoint against the third fixed target, so every
    # one-sided endpoint gate passes.  That superiority is not endpoint-policy equality.
    assert observed["local_primary_one_sided_noninferiority_recovery"]
    assert observed["local_primary_scale_substitution"]  # backward-audit alias
    assert not observed["endpoint_two_sided_equivalent_primary"]
    assert not direct["functional_ordinal_fidelity"]
    assert not observed["local_functional_endpoint_isomorphic_scale_substitution"]
    assert not observed["local_functional_endpoint_equivalent_scale_substitution"]
    assert "overshooting" in result["claim_boundary"]


def test_two_sided_equal_target_loss_does_not_imply_endpoint_policy_isomorphism():
    q = np.linspace(0.02, 0.98, 3000)
    rng = np.random.default_rng(777)
    small_values = np.clip(q + rng.normal(0, 0.50, len(q)), 0, 1)
    larger_values = np.clip(q + rng.normal(0, 0.30, len(q)), 0, 1)
    # Independent errors yield nearly equal rank/MAE loss against q while disagreeing over which
    # individual items receive that loss.  Endpoint-loss equivalence is therefore not direct
    # endpoint-policy isomorphism.
    equal_loss_candidate = np.clip(q + rng.normal(0, 0.30, len(q)), 0, 1)
    result = certify_scale_step_substitution(
        {"canonical": q},
        {"canonical": small_values},
        {"canonical": equal_loss_candidate},
        {"canonical": larger_values},
        n_boot=120,
        seed=79,
    )
    observed = result["evidence"]["observed"]
    direct = result["direct_endpoint_isomorphism"]
    assert observed["endpoint_two_sided_equivalent_primary"]
    assert observed["local_primary_two_sided_equivalence_recovery"]
    assert direct["point"]["candidate"]["spearman"] < 0.50
    assert not direct["observed"]["functional_ordinal_fidelity"]
    assert not observed["local_functional_endpoint_equivalent_scale_substitution"]


def test_scale_step_certificate_requires_each_link_and_supports_clusters():
    q = np.tile(np.linspace(0.02, 0.98, 200), 2)
    rng = np.random.default_rng(190)
    small_values = np.clip(q + rng.normal(0, 0.34, len(q)), 0, 1)
    larger_values = np.clip(q + rng.normal(0, 0.08, len(q)), 0, 1)
    # Better than the small endpoint but intentionally short of the larger endpoint.
    candidate_values = np.clip(q + rng.normal(0, 0.22, len(q)), 0, 1)
    target = {"canonical": q, "question": q}
    small = {"canonical": small_values, "question": small_values}
    larger = {"canonical": larger_values, "question": larger_values}
    candidate = {"canonical": candidate_values, "question": candidate_values}
    clusters = [f"source-{index // 4}" for index in range(len(q))]

    result = certify_scale_step_substitution(
        target,
        small,
        candidate,
        larger,
        bootstrap_clusters=clusters,
        endpoint_mae_margin=0.01,
        endpoint_rho_margin=0.01,
        n_boot=250,
        seed=23,
    )
    assert result["evidence"]["certified"]["native_scale_gap"]
    assert result["evidence"]["certified"]["articulation_gain"]
    assert not result["evidence"]["observed"]["endpoint_noninferior_primary"]
    assert not result["evidence"]["observed"]["local_primary_scale_substitution"]
    assert result["bootstrap"]["n_source_groups"] == 100
    assert "cluster bootstrap" in result["bootstrap"]["sampling"]


def test_scale_step_certificate_rejects_a_step_with_no_native_advantage():
    q = np.linspace(0.02, 0.98, 300)
    rng = np.random.default_rng(43)
    sparse_values = np.clip(q + rng.normal(0, 0.18, len(q)), 0, 1)
    target = {"canonical": q}
    sparse = {"canonical": sparse_values}
    result = certify_scale_step_substitution(
        target,
        sparse,
        {"canonical": q},
        sparse,
        n_boot=200,
        seed=4,
    )
    assert not result["evidence"]["observed"]["native_scale_gap"]
    assert not result["evidence"]["certified"]["native_scale_gap"]
    assert not result["evidence"]["observed"]["local_primary_scale_substitution"]
    assert result["target_fidelity"]["certified"]["functional_ordinal"]


def test_scale_step_certificate_keeps_observed_and_certified_grades_distinct():
    q = np.linspace(0.02, 0.98, 80)
    rng = np.random.default_rng(2)
    small_values = np.clip(q + rng.normal(0, 0.23, len(q)), 0, 1)
    larger_values = np.clip(q + rng.normal(0, 0.21, len(q)), 0, 1)
    target = {"canonical": q}
    small = {"canonical": small_values}
    larger = {"canonical": larger_values}
    result = certify_scale_step_substitution(
        target,
        small,
        larger,
        larger,
        n_boot=200,
        seed=7,
    )
    assert result["evidence"]["observed"]["local_primary_scale_substitution"]
    assert not result["evidence"]["certified"]["local_primary_scale_substitution"]
    assert result["differences"]["native_rho_larger_minus_small"]["point"] > 0
    assert result["differences"]["native_rho_larger_minus_small"]["CI"][0] < 0


def test_functional_target_scale_substitution_requires_floor_crossing():
    q = np.linspace(0.02, 0.98, 500)
    rng = np.random.default_rng(91)
    target = {"canonical": q, "question": q}
    small_values = np.clip(q + rng.normal(0, 0.09, len(q)), 0, 1)
    larger_values = np.clip(q + rng.normal(0, 0.06, len(q)), 0, 1)
    candidate_values = np.clip(q + rng.normal(0, 0.03, len(q)), 0, 1)
    result = certify_scale_step_substitution(
        target,
        {"canonical": small_values, "question": small_values},
        {"canonical": candidate_values, "question": candidate_values},
        {"canonical": larger_values, "question": larger_values},
        n_boot=250,
        seed=12,
    )
    assert result["target_fidelity"]["observed"]["functional_ordinal"]
    assert result["target_fidelity"]["observed_gates"][
        "small_sparse_adverse_rank_below_functional_floor"] is False
    assert not result["evidence"]["observed"][
        "functional_target_scale_substitution"]
    assert result["descriptive_step_closure"]["rank"]["defined"]


def test_near_identity_target_scale_substitution_requires_baseline_outside_floor():
    q = np.linspace(0.02, 0.98, 700)
    rng = np.random.default_rng(118)
    target = {"canonical": q, "question": q}
    small_values = np.clip(q + rng.normal(0, 0.045, len(q)), 0, 1)
    larger_values = np.clip(q + rng.normal(0, 0.025, len(q)), 0, 1)
    candidate_values = q.copy()
    result = certify_scale_step_substitution(
        target,
        {"canonical": small_values, "question": small_values},
        {"canonical": candidate_values, "question": candidate_values},
        {"canonical": larger_values, "question": larger_values},
        n_boot=200,
        seed=31,
    )
    observed = result["evidence"]["observed"]
    assert observed["native_scale_gap"]
    assert observed["articulation_gain"]
    assert result["target_fidelity"]["observed"][
        "target_self_band_near_identity"]
    assert not result["target_fidelity"]["observed_gates"][
        "small_sparse_adverse_rank_below_functional_floor"]
    assert not observed["near_identity_target_scale_substitution"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"endpoint_mae_margin": -0.01}, "finite and nonnegative"),
        ({"min_rho_gain": float("nan")}, "finite and nonnegative"),
        ({"functional_rho_floor": 1.1}, "functional_rho_floor"),
    ],
)
def test_scale_step_certificate_validates_thresholds(kwargs, message):
    q = np.linspace(0.1, 0.9, 12)
    orbit = {"canonical": q}
    with pytest.raises(ValueError, match=message):
        certify_scale_step_substitution(
            orbit, orbit, orbit, orbit, n_boot=10, **kwargs)


def _synthetic_scale_certificate(*, passes=True):
    results = {
        "joint_fixed_target_and_endpoint_functional_isomorphic_"
        "scale_substitution": passes,
        "joint_fixed_target_and_endpoint_functional_equivalent_"
        "scale_substitution": passes,
    }
    return {"evidence": {"observed": results, "certified": results}}


def _synthetic_control_contrast(provenance, *, point=True, ci=True,
                                simultaneous=True):
    def certificate(value):
        return {"gates": {
            "source_rank_better_point": point,
            "source_mae_better_point": point,
            "source_rank_better_CI": value,
            "source_mae_better_CI": value,
        }}

    return {
        "source_arm_id": "definition",
        "control_arm_id": f"control_{provenance}",
        "control_provenance": provenance,
        "certificate": certificate(ci),
        "specificity_simultaneous_certificate": certificate(simultaneous),
    }


def test_content_specific_joint_grade_requires_and_beats_both_control_types():
    scale_certificate = _synthetic_scale_certificate()
    rows = [{
        "arm_id": "definition",
        "provenance": "source_telling",
        "control_for": None,
        "scale_step_certificate": scale_certificate,
        "scale_step_specificity_simultaneous_certificate": scale_certificate,
    }, {
        "arm_id": "control_inert_length_control",
        "provenance": "inert_length_control",
        "control_for": "definition",
    }]
    contrasts = [
        _synthetic_control_contrast("inert_length_control"),
        _synthetic_control_contrast("wrong_construct_control"),
    ]
    result = _content_specific_scale_memberships(rows, contrasts)
    for grade in ("observed", "certified", "simultaneous_certified"):
        assert result[
            f"{grade}_joint_fixed_target_endpoint_isomorphic_members"
        ] == ["definition"]
        assert result[
            f"{grade}_joint_fixed_target_endpoint_equivalent_members"
        ] == ["definition"]
    assert [row["arm_id"] for row in result["arm_grades"]] == ["definition"]
    assert result["arm_grades"][0]["control_coverage_complete"]
    assert not result["arm_grades"][0]["failure_reasons"]


@pytest.mark.parametrize("failure", ["missing_wrong_construct", "failed_inert"])
def test_content_specific_joint_grade_fails_closed_on_control_gap(failure):
    scale_certificate = _synthetic_scale_certificate()
    rows = [{
        "arm_id": "definition",
        "provenance": "source_telling",
        "scale_step_certificate": scale_certificate,
        "scale_step_specificity_simultaneous_certificate": scale_certificate,
    }]
    contrasts = [_synthetic_control_contrast(
        "inert_length_control", point=failure != "failed_inert",
        ci=failure != "failed_inert", simultaneous=failure != "failed_inert")]
    if failure != "missing_wrong_construct":
        contrasts.append(_synthetic_control_contrast("wrong_construct_control"))
    result = _content_specific_scale_memberships(rows, contrasts)
    for grade in ("observed", "certified", "simultaneous_certified"):
        assert not result[
            f"{grade}_joint_fixed_target_endpoint_isomorphic_members"
        ]
        assert not result[
            f"{grade}_joint_fixed_target_endpoint_equivalent_members"
        ]
    arm = result["arm_grades"][0]
    if failure == "missing_wrong_construct":
        assert not arm["control_coverage_complete"]
        assert arm["failure_reasons"] == [
            "missing matched control provenance: wrong_construct_control"]
    else:
        assert arm["control_coverage_complete"]
        assert not arm["grades"]["observed"][
            "better_than_every_required_control_on_rank_and_mae"]


def test_crossfold_fiber_join_intersects_members_and_pairs(tmp_path):
    paths = []
    for partition in ("residual_prompt_selection", "residual_unit_certification"):
        report = {
            "schema": "policy_isomorphism_experiment/v4",
            "partition": partition,
            "arm_bank_sha256": "bank",
            "config": {
                "small_job": "small", "big_job": "big", "target_arm_id": "target",
                "mae_margin": 0.02, "rho_margin": 0.05, "flip_margin": 0.02,
                "bias_margin": 0.02, "functional_rho_floor": 0.7,
                "confidence": 0.95, "n_boot": 500,
            },
            "cells": [{
                "cell_id": "cell",
                "domain": "humor",
                "construct": "construct",
                "executor_prompt_bank_validation": {"valid": True},
                "small_readout_template_sha256": "readout",
                "target_readout_template_sha256": "readout",
                "rows": [{
                    "arm_id": arm_id,
                    "components": (
                        ["definition", "rubric"]
                        if arm_id == "definition_rubric" else [arm_id]
                    ),
                    "certificate": {
                        "policy_isomorphic": False,
                        "small_sparse_point": {
                            "candidate_robust": {"spearman": 0.50, "mae_tvd": 0.30},
                        },
                            "functional": {
                                "adverse_rho_point": {
                                    "definition": 0.74,
                                    "rubric": 0.71,
                                    "definition_rubric": 0.705,
                                }[arm_id],
                                "quotient_rho_point": {
                                    "definition": 0.74,
                                    "rubric": 0.71,
                                    "definition_rubric": 0.705,
                                }[arm_id],
                                "adverse_rho_CI": (
                                    [0.68, 0.80] if arm_id == "definition" else [0.64, 0.78]
                                ),
                                "quotient_rho_CI": (
                                    [0.68, 0.80] if arm_id == "definition" else [0.64, 0.78]
                                ),
                            "small_sparse_adverse_rho_CI": [0.45, 0.55],
                            "gates": {
                                "target_identity_valid": True,
                                "positive_polarity": True,
                                "mae_point_improves_over_small_sparse": True,
                                "mae_CI_improves_over_small_sparse": True,
                            },
                            "observed_functional_policy_substitution": True,
                            "certified_functional_policy_substitution": False,
                        },
                    },
                } for arm_id in ("definition", "rubric", "definition_rubric")],
                "fiber": {
                    "equal_but_different_pairs": [],
                    "certified_functional_equal_but_different_pairs": [],
                    "observed_functional_equal_but_different_pairs": [{
                        "left": "definition",
                        "right": "rubric",
                        "articulation_surface_distance": 0.6,
                        "behavior_rho_floor": 0.9,
                        "behavior": {"quotient_spearman": 0.97},
                    }],
                },
                "matched_control_certificates": [{
                    "source_arm_id": "definition",
                    "control_arm_id": control_id,
                    "control_provenance": provenance,
                    "certificate": {"gates": {
                        "source_rank_better_point": True,
                        "source_rank_better_CI": True,
                        "source_mae_better_point": True,
                        "source_mae_better_CI": True,
                    }},
                } for control_id, provenance in (
                    ("control_inert_definition", "inert_length_control"),
                    ("control_wrong_definition", "wrong_construct_control"),
                )],
            }],
        }
        for row in report["cells"][0]["rows"]:
            direct_match = row["arm_id"] == "definition"
            scale_results = {
                "local_primary_scale_substitution": direct_match,
                "local_primary_two_sided_equivalence_recovery": direct_match,
                "functional_target_scale_substitution": direct_match,
                "local_functional_endpoint_isomorphic_scale_substitution": direct_match,
                "local_functional_endpoint_equivalent_scale_substitution": direct_match,
                "local_near_identity_isomorphic_scale_substitution": direct_match,
                "joint_fixed_target_and_endpoint_functional_isomorphic_"
                "scale_substitution": direct_match,
                "joint_fixed_target_and_endpoint_functional_equivalent_"
                "scale_substitution": direct_match,
            }
            row["scale_step_certificate"] = {
                "evidence": {
                    "observed": scale_results,
                    "certified": scale_results,
                },
                "target_fidelity": {},
                "direct_endpoint_isomorphism": {},
                "descriptive_step_closure": {},
            }
            row["scale_step_simultaneous_certificate"] = None
        path = tmp_path / f"{partition}.json"
        path.write_text(json.dumps(report))
        paths.append(str(path))
    result = summarize_crossfold_fibers(paths)
    assert result["summary"]["n_observed_functional_members"] == 3
    assert result["summary"]["n_observed_functional_equal_but_different_pairs"] == 1
    assert result["summary"]["n_certified_functional_members"] == 0
    assert result["schema"] == "crossfold_policy_isomorphism_fibers/v5"
    assert result["summary"][
        "n_observed_functional_endpoint_isomorphic_scale_step_members"] == 1
    assert result["summary"][
        "n_certified_functional_endpoint_equivalent_scale_step_members"] == 1
    assert result["summary"][
        "n_observed_near_identity_endpoint_isomorphic_scale_step_members"] == 1
    assert result["summary"][
        "n_observed_two_sided_equivalence_recovery_members"] == 1
    assert result["summary"][
        "n_observed_joint_fixed_target_endpoint_isomorphic_scale_step_members"] == 1
    assert result["summary"][
        "n_observed_content_specific_joint_fixed_target_endpoint_isomorphic_"
        "scale_step_members"] == 1
    assert result["summary"][
        "n_certified_content_specific_joint_fixed_target_endpoint_equivalent_"
        "scale_step_members"] == 1
    assert result["summary"][
        "n_simultaneous_certified_content_specific_joint_fixed_target_endpoint_"
        "isomorphic_scale_step_members"] == 0
    cell = result["cells"][0]
    profile = {row["rho_floor"]: row for row in cell["functional_floor_profile"]}
    assert profile[0.7]["observed_functional_members"] == [
        "definition", "definition_rubric", "rubric"]
    assert profile[0.75]["observed_functional_members"] == []
    assert profile[0.65]["certified_functional_members"] == ["definition"]
    assert profile[0.7]["observed_component_minimal_members"] == [
        "definition", "rubric"]
    capacity = {row["arm_id"]: row for row in cell["functional_capacity_by_arm"]}
    assert capacity["definition"]["stable_observed_max_rho_floor"] == 0.74
    assert capacity["rubric"]["stable_certified_max_rho_floor"] == 0.64
    topology = {
        row["component_minimal_arm_id"]: row
        for row in cell["component_topology_at_reported_floor"]
    }
    assert set(topology) == {"definition", "rubric"}
    assert topology["definition"][
        "all_strict_supersets_nonimproving_on_rank_across_folds"]
    for matched in cell["matched_control_certificates"]:
        assert not matched["simultaneous_inference_available_all_folds"]
        assert not matched["source_better_rank_simultaneous_CI_all_folds"]
        assert not matched["source_better_mae_simultaneous_CI_all_folds"]


@pytest.mark.parametrize(("key", "left", "right"), [
    ("confidence", 0.95, 0.99),
    ("n_boot", 500, 1000),
])
def test_crossfold_fiber_join_rejects_inference_config_drift(
        tmp_path, key, left, right):
    paths = []
    for index, value in enumerate((left, right)):
        report = {
            "schema": "policy_isomorphism_experiment/v4",
            "partition": (
                "residual_prompt_selection" if index == 0
                else "residual_unit_certification"
            ),
            "arm_bank_sha256": "bank",
            "config": {
                "small_job": "small", "big_job": "big", "target_arm_id": "target",
                "mae_margin": 0.02, "rho_margin": 0.05, "flip_margin": 0.02,
                "bias_margin": 0.02, "functional_rho_floor": 0.7,
                "confidence": 0.95, "n_boot": 500,
            },
            "cells": [],
        }
        report["config"][key] = value
        path = tmp_path / f"fold-{index}.json"
        path.write_text(json.dumps(report))
        paths.append(str(path))
    with pytest.raises(ValueError, match=rf"disagree on config '{key}'"):
        summarize_crossfold_fibers(paths)


def test_pooled_crossfold_report_uses_disjoint_stratified_items(tmp_path, monkeypatch):
    from methods.codability.experiments import run_policy_isomorphism as runner

    indexes = {}
    paths = []
    partitions = ("residual_prompt_selection", "residual_unit_certification")
    for fold_index, partition in enumerate(partitions):
        n_items = 60
        hashes = [f"fold{fold_index}-item{index}" for index in range(n_items)]
        q = np.linspace(0.02, 0.98, n_items)
        sparse = np.clip(
            q + np.random.default_rng(fold_index).normal(0, 0.5, n_items), 0, 1
        )
        executor = {
            "scores": np.stack([sparse, sparse, q, q]),
            "meta": [
                {"cell_id": "cell", "arm_id": "name", "form": "canonical",
                 "domain": "humor"},
                {"cell_id": "cell", "arm_id": "name", "form": "question",
                 "domain": "humor"},
                {"cell_id": "cell", "arm_id": "definition", "form": "canonical",
                 "domain": "humor"},
                {"cell_id": "cell", "arm_id": "definition", "form": "question",
                 "domain": "humor"},
            ],
            "hashes": hashes,
            "shard_sha256": [f"small-{fold_index}"],
            "readout_template_sha256": "readout",
        }
        target = {
            "scores": np.stack([q, q]),
            "meta": [
                {"cell_id": "cell", "arm_id": "target", "form": "canonical",
                 "domain": "humor"},
                {"cell_id": "cell", "arm_id": "target", "form": "question",
                 "domain": "humor"},
            ],
            "hashes": hashes,
            "shard_sha256": [f"target-{fold_index}"],
            "readout_template_sha256": "readout",
        }
        indexes[(f"executor-{fold_index}", partition)] = {("small", "humor"): executor}
        indexes[(f"target-{fold_index}", partition)] = {("big", "humor"): target}
        report = {
            "schema": "policy_isomorphism_experiment/v4",
            "partition": partition,
            "arm_bank_sha256": "bank",
            "executor_shard_root": f"executor-{fold_index}",
            "target_shard_root": f"target-{fold_index}",
            "config": {
                "small_job": "small", "big_job": "big", "target_arm_id": "target",
                    "mae_margin": 0.02, "rho_margin": 0.05, "flip_margin": 0.02,
                    "bias_margin": 0.02, "functional_rho_floor": 0.7,
                    "confidence": 0.95, "n_boot": 500,
            },
            "cells": [{
                "cell_id": "cell", "domain": "humor", "construct": "humor",
                "n_items": n_items,
                "small_shards": [f"small-{fold_index}"],
                "target_shards": [f"target-{fold_index}"],
                "executor_prompt_bank_validation": {"valid": True},
                "small_readout_template_sha256": "readout",
                "target_readout_template_sha256": "readout",
                "rows": [{"arm_id": "definition"}],
            }],
        }
        path = tmp_path / f"pool-{fold_index}.json"
        path.write_text(json.dumps(report))
        paths.append(str(path))

    monkeypatch.setattr(
        runner, "load_public_index", lambda root, partition: indexes[(root, partition)])
    monkeypatch.setattr(runner, "_average_repetitions", lambda value: value)
    result = pool_crossfold_policy_reports(paths, n_boot=200, seed=8)
    assert result["cells"][0]["n_items"] == 120
    row = result["cells"][0]["rows"][0]
    assert row["nominal_certificate"]["bootstrap"]["n_strata"] == 2
    assert row["nominal_certificate"]["functional"][
        "certified_functional_policy_substitution"]
    assert row["simultaneous_certificate"]["functional"][
        "certified_functional_policy_substitution"]

    def source_groups(_root, _manifest, *, domain, partition, item_hashes):
        assert domain == "humor"
        assert partition in partitions
        return {
            "source_groups": [f"group-{index // 2}" for index in range(len(item_hashes))],
            "validation": {
                "valid": True,
                "domain": domain,
                "partition": partition,
                "n_items": len(item_hashes),
                "n_source_groups": len(item_hashes) // 2,
            },
        }

    monkeypatch.setattr(runner, "load_partition_source_groups", source_groups)
    clustered = pool_crossfold_policy_reports(
        paths,
        n_boot=100,
        seed=8,
        packet_root="local-packet",
        packet_manifest_path="local-manifest",
    )
    clustered_certificate = clustered["cells"][0]["rows"][0]["nominal_certificate"]
    assert clustered_certificate["bootstrap"]["n_strata"] == 2
    assert clustered_certificate["bootstrap"]["n_source_groups"] == 60
    assert "cluster bootstrap" in clustered["bootstrap"]["sampling"]
