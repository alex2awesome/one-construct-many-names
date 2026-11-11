"""Source-only arm-bank compilation and matched-control checks."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from methods.codability.experiments import compile_fresh_name_arm_bank as compiler
from methods.codability.experiments import run_policy_isomorphism as policy_runner
from methods.codability.experiments.build_fresh_item_partitions import (
    build as build_item_packet,
    sha256_bytes,
    sha256_file,
)
from methods.codability.experiments.compile_fresh_name_arm_bank import (
    BREADTH_LEVELS,
    BREADTH_RUNTIME_ENVIRONMENT_OVERRIDES,
    BREADTH_SELECTION_LEVEL_ORDER,
    BREADTH_TEACHER_FORCED_ROW_BATCH_SIZE,
    DEFAULT_BREADTH_TASKS,
    compile_bank,
    compile_breadth_bank,
    compile_breadth_execution_manifest,
    compile_concluding_confirmation_manifest,
    compile_metric_panel,
    validate_bank,
    validate_metric_panel,
)
from methods.codability.experiments.policy_data import (
    authorize_policy_partition,
    validate_frozen_implementation,
    validate_policy_articulation_selection_provenance,
)
from methods.codability.experiments.run_policy_isomorphism import (
    build_policy_articulation_selection,
)
from methods.codability.experiments.validate_fresh_item_partitions import (
    validate_packet,
)


def test_real_source_bank_is_closed_and_controls_are_length_matched():
    bank = compile_bank()

    assert not validate_bank(bank)
    assert len(bank["cells"]) == 4
    assert all(len(cell["arms"]) == 16 for cell in bank["cells"])
    humor_49 = next(cell for cell in bank["cells"] if cell["id"] == "N_humor_49")
    assert set(humor_49["target_model_jobs"]) == {"llama70_n_target", "gemma31_target"}
    for cell in bank["cells"]:
        by_id = {arm["id"]: arm for arm in cell["arms"]}
        for arm in cell["arms"]:
            if arm["control_for"]:
                assert arm["semantic_content_word_count"] == \
                    by_id[arm["control_for"]]["semantic_content_word_count"]


def test_legacy_bank_compiler_accepts_a_batched_canonical_domain_panel(tmp_path):
    target_manifest = json.loads(Path(
        "methods/codability/experiments/fresh_llama70_name_target_manifest_v1.json"
    ).read_text())
    target_manifest["cells"] = [
        cell for cell in target_manifest["cells"]
        if cell["id"] == "N_humor_23"
    ]
    manifest_path = tmp_path / "target.json"
    manifest_path.write_text(json.dumps(target_manifest))

    bank = compile_bank(
        source_files={"humor": compiler.SOURCE_FILES["humor"]},
        target_manifest_path=manifest_path,
        cell_targets={("humor", 23): ["llama31_70b_name_target"]},
        domain_tasks={"humor": "humor"},
    )

    assert not validate_bank(bank)
    assert [cell["id"] for cell in bank["cells"]] == ["N_humor_23"]
    assert bank["cells"][0]["task"] == "humor"
    assert bank["cells"][0]["target_model_jobs"] == ["llama31_70b_name_target"]


def test_frozen_concluding_batch_is_exact_hash_bound_and_sealed():
    root = Path(__file__).parents[3]
    exp = root / "methods/codability/experiments"
    data = root / "notebooks/data/two_faces_20260702"
    # v2 supersedes v1 (2026-07-15): identical frozen design; only the analysis-closure
    # bookkeeping fix and the v2 output directory differ.  v1's lockbox was never opened.
    manifest_path = exp / "concluding_policy_execution_manifest_v2.json"
    manifest = json.loads(manifest_path.read_text())

    rebuilt = compile_concluding_confirmation_manifest(
        template_manifest_path=data / (
            "tacit_breadth_confirmation_v3/search_execution_manifest_v2.json"),
        construct_panel_path=exp / "concluding_policy_construct_panel_v1.json",
        arm_bank_path=data / "concluding_policy_arm_bank_v1.json",
        target_manifest_path=exp / "concluding_policy_target_manifest_v1.json",
        selection_artifact_path=exp / "concluding_policy_selection_v1.json",
    )
    assert rebuilt == manifest
    assert manifest["analysis"]["runner"]["cell_ids"] == [
        "N_humor_23", "N_humor_11", "N_press-releases_35",
    ]
    assert manifest["resource_policy"]["permitted_physical_gpu_indices"] == [5, 6, 7]
    assert manifest["resource_policy"]["forbidden_physical_gpu_indices"] == [0, 1, 2, 3, 4]
    for section in ("scoring", "analysis", "compilation"):
        assert validate_frozen_implementation(
            manifest, manifest_path=manifest_path, section=section,
        )["valid"] is True

    selection_path = exp / "concluding_policy_selection_v1.json"
    calibration = authorize_policy_partition(
        partition="tacit_breadth_search",
        operation="calibration",
        execution_manifest_path=manifest_path,
        selection_artifact_path=selection_path,
    )
    assert calibration["sealed_partition_authorized"] is False
    with pytest.raises(ValueError, match="requires a calibration-release artifact"):
        authorize_policy_partition(
            partition="tacit_breadth_validation",
            operation="lockbox",
            execution_manifest_path=manifest_path,
            selection_artifact_path=selection_path,
        )


def test_real_breadth_panel_meets_every_task_level_quota_without_duplicates():
    panel = compile_metric_panel()

    assert panel["hierarchy_frame"]["generation"] == \
        "legacy-expanded-source-action-node-dag-v1"
    assert panel["schema"] == "tacit_breadth_metric_panel/v3"
    assert panel["selection_level_order"] == list(BREADTH_SELECTION_LEVEL_ORDER)
    assert panel["hierarchy_frame"]["is_partition"] is False
    assert panel["hierarchy_frame"]["not_the_rebuilt_lexicon_partition"] is True
    assert panel["n_cells"] == len(DEFAULT_BREADTH_TASKS) * len(BREADTH_LEVELS) * 30
    assert len({cell["id"] for cell in panel["cells"]}) == panel["n_cells"]
    assert len({cell["node_id"] for cell in panel["cells"]}) == panel["n_cells"]
    assert len({
        (cell["task"], " ".join(cell["construct"].casefold().split()))
        for cell in panel["cells"]
    }) == panel["n_cells"]
    assert not validate_metric_panel(panel)
    tampered = deepcopy(panel)
    tampered["cells"][0]["task_raw_provenance_component_size"] += 1
    assert any(
        "task raw-provenance metadata changed" in error
        for error in validate_metric_panel(tampered)
    )
    for task in DEFAULT_BREADTH_TASKS:
        task_rows = [cell for cell in panel["cells"] if cell["task"] == task]
        task_components = {
            cell["task_raw_provenance_component_id"] for cell in task_rows
        }
        assert len(task_components) >= 40
        assert max(
            cell["task_raw_provenance_component_size"] for cell in task_rows
        ) <= 15
        for level in BREADTH_LEVELS:
            rows = [
                cell for cell in panel["cells"]
                if cell["task"] == task and cell["level"] == level
            ]
            assert len(rows) == 30
            assert all(
                cell["description"] and cell["components"] and cell["children"]
                for cell in rows
            )
            assert all(cell["dependency_component_id"] for cell in rows)
            assert all(cell["dependency_component_size"] >= 1 for cell in rows)
            assert all(cell["task_raw_provenance_component_id"] for cell in rows)
            assert all(
                cell["task_raw_provenance_component_size"] >= 1 for cell in rows
            )
            sampling_frame = next(
                item["n_sampling_frame_nodes"] for item in panel["inventory"]
                if item["task"] == task and item["level"] == level
            )
            assert abs(sum(
                cell["nominal_poststratification_weight"] for cell in rows
            ) - sampling_frame) < 1e-9
            assert all(0 < cell["stratum_coverage_fraction"] <= 1 for cell in rows)


def test_real_breadth_panel_is_byte_stable_before_outcomes():
    first = compile_metric_panel()
    second = compile_metric_panel()

    assert first["panel_content_sha256"] == second["panel_content_sha256"]
    assert [cell["id"] for cell in first["cells"]] == [
        cell["id"] for cell in second["cells"]]


def test_breadth_panel_sampler_prioritizes_distinct_dependence_components():
    nodes = []
    for index, component in enumerate(("a", "a", "a", "a", "b", "c")):
        nodes.append({
            "node_id": f"node-{index}",
            "merged_name": f"construct {index}",
            "merged_description": (
                "one two three four five six seven eight source words"
            ),
            "all_leaves": [{"name": f"leaf {index}", "key": f"leaf-{index}"}],
            "source_kind": "merged_group",
            "total_leaf_rubrics": 1,
            "dependency_component_id": f"dependency-{component}",
            "dependency_component_size": 4 if component == "a" else 1,
            "provenance_component_id": f"provenance-{component}",
            "provenance_component_size": 4 if component == "a" else 1,
        })

    selected = compiler._stable_stratified_panel(
        nodes, n=3, salt="outcome-blind-test-salt")

    assert len({row["dependency_component_id"] for row in selected}) == 3
    assert len({row["provenance_component_id"] for row in selected}) == 3
    assert [row["node_id"] for row in selected] == [
        row["node_id"] for row in compiler._stable_stratified_panel(
            nodes, n=3, salt="outcome-blind-test-salt")
    ]
    assert all(row["stratum_coverage_fraction"] == 0.5 for row in selected)
    assert all(row["nominal_poststratification_weight"] == 2.0 for row in selected)


def test_breadth_sampler_avoids_cross_round_raw_provenance_collisions():
    def node(index, leaf_key):
        return {
            "node_id": f"node-{index}",
            "merged_name": f"construct {index}",
            "merged_description": "one two three four five six seven eight source words",
            "all_leaves": [{"name": f"leaf {leaf_key}", "key": leaf_key}],
            "source_kind": "merged_group",
            "total_leaf_rubrics": 1,
            "dependency_component_id": f"dependency-{index}",
            "dependency_component_size": 1,
            "provenance_component_id": f"provenance-{index}",
            "provenance_component_size": 1,
        }

    graph = compiler._TaskRawProvenanceGraph(task="demo", bucket="general")
    first = compiler._stable_stratified_panel(
        [node("a", "shared"), node("b", "first-only")],
        n=2,
        salt="cross-round-first",
        task_provenance_graph=graph,
    )
    second = compiler._stable_stratified_panel(
        [
            node("c", "shared"),
            node("d", "second-new-a"),
            node("e", "second-new-b"),
        ],
        n=2,
        salt="cross-round-second",
        task_provenance_graph=graph,
    )

    assert len(first) == len(second) == 2
    assert {row["node_id"] for row in second} == {"node-d", "node-e"}
    annotations = graph.annotations()
    assert len({
        row["task_raw_provenance_component_id"]
        for row in annotations.values()
    }) == 4


def test_breadth_bank_unifies_full_text_units_and_exact_matched_controls():
    panel = compile_metric_panel()
    bank = compile_breadth_bank(panel=panel)

    assert bank["schema"] == "tacit_breadth_arm_bank/v3"
    assert not validate_bank(bank)
    assert len(bank["cells"]) == panel["n_cells"]
    for cell in bank["cells"]:
        by_id = {arm["id"]: arm for arm in cell["arms"]}
        assert {"name", "source_definition", "source_rules",
                "source_leaf_inventory", "source_definition_rules", "source_dossier",
                "source_units_full"} <= set(by_id)
        assert by_id["source_definition"]["components"]
        assert by_id["source_rules"]["components"]
        assert by_id["source_leaf_inventory"]["components"]
        assert by_id["source_definition"]["channel"] != by_id["source_rules"]["channel"]
        combined = set(by_id["source_definition_rules"]["components"])
        assert combined & set(by_id["source_definition"]["components"])
        assert combined & set(by_id["source_rules"]["components"])
        dossier = set(by_id["source_dossier"]["components"])
        assert dossier & set(by_id["source_definition"]["components"])
        assert dossier & set(by_id["source_rules"]["components"])
        assert dossier & set(by_id["source_leaf_inventory"]["components"])
        for arm in cell["arms"]:
            assert len(arm["forms"]) == 3
            assert arm["forms"][0]["prompt"].startswith(cell["construct"])
            if arm["control_for"]:
                source = by_id[arm["control_for"]]
                assert arm["semantic_content_word_count"] == \
                    source["semantic_content_word_count"]
                assert arm["added_content_word_count"] == source["added_content_word_count"]
                assert arm["components"] == []


def test_breadth_bank_is_stable_and_contains_no_external_targets():
    panel = compile_metric_panel()
    first = compile_breadth_bank(panel=panel)
    second = compile_breadth_bank(panel=panel)

    assert first["bank_content_sha256"] == second["bank_content_sha256"]
    assert "practice_target" not in str(first)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=1) + "\n")


def _certificate(*, rank: float, vector_excess: float) -> dict:
    margins = {"mae": 0.02, "rho": 0.05, "flip": 0.02, "bias": 0.02}
    return {
        "functional": {
            "adverse_rho_point": rank,
            "quotient_rho_point": rank + 0.01,
            "observed_functional_policy_substitution": rank >= 0.70,
        },
        "point": {"candidate_robust": {
            "mae_tvd": 0.10 + vector_excess / 100.0,
            "binary_flip_rate": 0.08 + vector_excess / 100.0,
            "absolute_bias": 0.03 + vector_excess / 100.0,
        }},
        "differences": {
            "mae_excess_over_target_self": {
                "point": margins["mae"] * (1.0 + vector_excess)},
            "rho_minus_target_self": {
                "point": -margins["rho"] * (1.0 + vector_excess)},
            "flip_excess_over_target_self": {
                "point": margins["flip"] * (1.0 + vector_excess)},
            "bias_excess_over_target_self": {
                "point": margins["bias"] * (1.0 + vector_excess)},
        },
        "margins": margins,
    }


def _production_score_provenance(
        manifest: dict, manifest_sha256: str, *, job_id: str) -> dict:
    jobs = {job["id"]: job for job in manifest["model_jobs"]}
    job = jobs[job_id]
    return {
        "valid": True,
        "job_id": job_id,
        "repetitions": job["required_repetitions"],
        "execution_manifest_sha256": manifest_sha256,
        "arm_bank_sha256": manifest["arm_bank_sha256"],
        "packet_manifest_sha256": manifest["packet_manifest_sha256"],
        "binary_readout": manifest["binary_readout"],
        "readout_template_sha256": manifest["readout_template_sha256"],
        "role": job["role"],
        "backend_class": manifest["execution_environment"][
            "production_backend_class"],
        "fake_backend": False,
    }


def _authenticated_search_report(
        *, search: dict, search_path: Path, bank: dict) -> dict:
    """Construct synthetic test statistics with the full production provenance shape.

    These values are deliberately not experimental outcomes.  They exercise the frozen DAG
    using exact report certificates that a production analysis must emit.
    """
    search_sha = sha256_file(search_path)
    runner = search["analysis"]["runner"]
    cells = []
    for bank_cell in bank["cells"]:
        rows = []
        grades = []
        candidate_index = 0
        for arm in bank_cell["arms"]:
            row = {
                "arm_id": arm["id"],
                "channel": arm["channel"],
                "provenance": arm["provenance"],
                "control_for": arm["control_for"],
                "components": arm["components"],
                "semantic_content_word_count": arm[
                    "semantic_content_word_count"],
            }
            if arm["id"] != "name" and arm["control_for"] is None:
                # Deterministic variation ensures the real selector exercises ranking,
                # vector, component-diversity, and address-dose roles.
                rank = 0.82 - 0.004 * candidate_index
                vector_excess = float(candidate_index % 4) / 4.0
                row["certificate"] = _certificate(
                    rank=rank, vector_excess=vector_excess)
                grades.append({
                    "arm_id": arm["id"],
                    "grades": {"observed": {
                        "better_than_every_required_control_on_rank_and_mae": True,
                    }},
                })
                candidate_index += 1
            rows.append(row)
        cells.append({
            "cell_id": bank_cell["id"],
            "rows": rows,
            "content_specific_scale_step": {"arm_grades": grades},
            "executor_prompt_bank_validation": {"valid": True},
            "target_prompt_bank_validation": {"valid": True},
            "scored_arm_panel_validation": {
                "small": {"valid": True}, "target": {"valid": True},
            },
            "score_cell_identity_validation": {
                "small": {"valid": True}, "target": {"valid": True},
            },
            "source_group_validation": {"valid": True},
            "score_provenance_validation": {
                "small": _production_score_provenance(
                    search, search_sha, job_id=runner["small_job"]),
                "target": _production_score_provenance(
                    search, search_sha, job_id=runner["big_job"]),
            },
        })
    return {
        "schema": "policy_isomorphism_experiment/v5",
        "partition": "tacit_breadth_search",
        "arm_bank_sha256": search["arm_bank_sha256"],
        "partition_authorization": {
            "phase": "calibration",
            "execution_manifest_sha256": search_sha,
            "selection_artifact_sha256": None,
        },
        "source_group_inference": {
            "enabled": True,
            "packet_manifest_sha256": search["packet_manifest_sha256"],
        },
        "frozen_invocation_validation": {
            "valid": True,
            "runner": runner,
        },
        "additional_artifact_validation": {
            "valid": True,
            "files": [
                {"path": row["path"], "sha256": row["sha256"]}
                for row in search["additional_artifacts"]
            ],
        },
        "analysis_implementation": search["implementation"]["analysis"],
        "config": {"include_controls": True},
        "cell_panel_identity_validation": {"valid": True},
        "cells": cells,
    }


def _canonical_breadth_dag_fixture(monkeypatch, tmp_path) -> dict:
    monkeypatch.setattr(compiler, "DEFAULT_BREADTH_TASKS", ("humor",))
    monkeypatch.setattr(compiler, "BREADTH_LEVELS", ("R1",))
    monkeypatch.setattr(compiler, "BREADTH_SELECTION_LEVEL_ORDER", ("R1",))
    monkeypatch.setattr(
        compiler, "BREADTH_CALIBRATION_REPORT", tmp_path / "search_report.json"
    )
    monkeypatch.setattr(
        compiler, "BREADTH_LOCKBOX_RELEASE", tmp_path / "calibration_release.json"
    )
    panel = compile_metric_panel(tasks=("humor",), n_per_task_level=30)
    bank = compile_breadth_bank(panel=panel)
    panel_path = tmp_path / "panel.json"
    bank_path = tmp_path / "bank.json"
    _write_json(panel_path, panel)
    _write_json(bank_path, bank)

    protocol = {
        "schema": "fresh_item_partition_protocol/v2_tacit_breadth",
        "status": "test-fixture-only",
        "salt": "canonical-breadth-dag-test",
        "emit_practice_targets": False,
        "anchor_policy": "unsupervised model-to-model reconstruction only",
        "domains": {
            "humor": {
                "task": "humor",
                "source_group": {"strategy": "item_hash"},
                "holdout_grade": "deduplicated-item-disjoint",
                "label": "no dataset label is retained, emitted, selected on, or used",
            },
        },
        "partitions": [
            {
                "id": "tacit_breadth_search", "n": 1,
                "domains": ["humor"], "visibility": "open test search",
            },
            {
                "id": "tacit_breadth_validation", "n": 1,
                "domains": ["humor"], "visibility": "held-out test validation",
            },
        ],
        "legacy_exclusion": {"enabled": False, "sampling_seed": 7},
        "exclude_packet_manifests": [],
    }
    protocol_path = tmp_path / "protocol.json"
    _write_json(protocol_path, protocol)
    packet_dir = tmp_path / "packet"
    build_item_packet(
        domains=["humor"], out_dir=packet_dir, manifest_path=protocol_path)
    packet_path = packet_dir / "packet_manifest.json"
    integrity_path = tmp_path / "integrity.json"
    integrity = validate_packet(
        packet_path,
        protocol_path=protocol_path,
        domains={"humor"},
        partitions={"tacit_breadth_search", "tacit_breadth_validation"},
    )
    assert integrity["valid"], integrity["errors"]
    _write_json(integrity_path, integrity)

    search = compile_breadth_execution_manifest(
        stage="search",
        metric_panel_path=panel_path,
        arm_bank_path=bank_path,
        protocol_manifest_path=protocol_path,
        packet_manifest_path=packet_path,
        partition_integrity_path=integrity_path,
    )
    search_path = tmp_path / "search_execution.json"
    _write_json(search_path, search)
    report = _authenticated_search_report(
        search=search, search_path=search_path, bank=bank)
    report_path = tmp_path / "search_report.json"
    _write_json(report_path, report)
    selection = build_policy_articulation_selection(
        search_execution_manifest_path=search_path,
        search_report_path=report_path,
        arm_bank_path=bank_path,
        packet_manifest_path=packet_path,
        metric_panel_path=panel_path,
        additional_artifact_paths=(compiler.SAME_VERSION_MODEL_TEMPLATE,),
        selected_phase="lockbox",
        selected_partition="tacit_breadth_validation",
    )
    selection_path = tmp_path / "selection.json"
    _write_json(selection_path, selection)
    return {
        "panel": panel,
        "bank": bank,
        "panel_path": panel_path,
        "bank_path": bank_path,
        "protocol_path": protocol_path,
        "packet_path": packet_path,
        "integrity_path": integrity_path,
        "search": search,
        "search_path": search_path,
        "report": report,
        "report_path": report_path,
        "selection": selection,
        "selection_path": selection_path,
        "release_path": tmp_path / "calibration_release.json",
    }


def _refresh_selection_content_hash(selection: dict) -> None:
    selection.pop("selection_content_sha256", None)
    selection["selection_content_sha256"] = sha256_bytes(json.dumps(
        selection, sort_keys=True, separators=(",", ":")).encode())


def test_breadth_execution_manifests_form_an_authenticated_canonical_dag(
        monkeypatch, tmp_path):
    fixture = _canonical_breadth_dag_fixture(monkeypatch, tmp_path)
    search = fixture["search"]
    panel = fixture["panel"]
    assert search["phases"] == {"calibration": ["tacit_breadth_search"]}
    assert search["phase_access"] == {"calibration": "open_development"}
    assert search["selection_required_phases"] == []
    assert "selection_artifact_sha256" not in search
    assert search["analysis"]["runner"]["cell_ids"] == [
        cell["id"] for cell in panel["cells"]]
    assert search["analysis"]["runner"]["n_boot"] == 2000
    assert search["teacher_forced_row_batch_size"] == \
        BREADTH_TEACHER_FORCED_ROW_BATCH_SIZE
    assert search["teacher_forced_batching_audit"]["status"] == \
        "eight-row-batching-rejected-before-model-outcomes"
    assert search["teacher_forced_batching_audit"][
        "scalar_vs_explicit_row_batch_one"]["n_exact"] == 72
    assert search["execution_environment"][
        "runtime_environment_overrides"] == BREADTH_RUNTIME_ENVIRONMENT_OVERRIDES
    assert search["execution_environment"]["runtime_environment_overrides"][
        "CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert {job["tensor_parallel_size"] for job in search["model_jobs"]} == {1}
    assert search["resource_policy"]["maximum_gpus_for_any_job"] == 1
    assert search["resource_policy"]["maximum_total_gpus"] == 1
    assert search["resource_policy"]["permitted_physical_gpu_indices"] == [0]
    assert set(search["resource_policy"]["forbidden_physical_gpu_indices"]) >= {
        1, 2, 3, 4}
    assert "methods/codability/experiments/run_tacit_breadth_search_sk3.sh" in {
        row["path"] for row in search["implementation"]["scoring"]["files"]
    }
    assert search["selection_policy"]["schema"] == (
        "tacit_breadth_selection_policy/v2")

    validation = compile_breadth_execution_manifest(
        stage="validation",
        metric_panel_path=fixture["panel_path"],
        arm_bank_path=fixture["bank_path"],
        protocol_manifest_path=fixture["protocol_path"],
        packet_manifest_path=fixture["packet_path"],
        partition_integrity_path=fixture["integrity_path"],
        selection_artifact_path=fixture["selection_path"],
    )
    validation_path = tmp_path / "validation_execution.json"
    _write_json(validation_path, validation)
    assert validation["phases"] == {
        "lockbox": ["tacit_breadth_validation"]}
    assert validation["phase_access"] == {"lockbox": "sealed_confirmation"}
    assert validation["selection_required_phases"] == ["lockbox"]
    assert validation["analysis"]["runner"]["n_boot"] == 10000
    assert validation["teacher_forced_row_batch_size"] == \
        BREADTH_TEACHER_FORCED_ROW_BATCH_SIZE
    assert validation["teacher_forced_batching_audit"] == \
        search["teacher_forced_batching_audit"]
    assert [job["required_repetitions"] for job in search["model_jobs"]] == [
        [0], [0]]
    assert [job["required_repetitions"] for job in validation["model_jobs"]] == [
        [0, 1], [0, 1]]
    freeze = validation["selection_provenance_validation_at_freeze"]
    assert freeze == {
        "valid": True,
        "search_execution_manifest_sha256": sha256_file(fixture["search_path"]),
        "search_report_sha256": sha256_file(fixture["report_path"]),
        "n_cells": 30,
    }
    provenance = validate_policy_articulation_selection_provenance(
        fixture["selection"],
        selection_path=fixture["selection_path"],
        execution_manifest=validation,
        execution_manifest_path=validation_path,
    )
    transition = provenance["manifest_transition_validation"]
    assert transition["valid"] is True
    assert transition["search_phase"] == "calibration"
    assert transition["validation_phase"] == "lockbox"
    assert transition["normalized_runner"] == {
        **search["analysis"]["runner"],
        "n_boot": "search=2000;validation=10000",
    }

    forged_validation = deepcopy(validation)
    forged_validation["selection_provenance_validation_at_freeze"]["n_cells"] = 29
    with pytest.raises(ValueError, match="derived selection provenance freeze certificate"):
        validate_policy_articulation_selection_provenance(
            fixture["selection"],
            selection_path=fixture["selection_path"],
            execution_manifest=forged_validation,
            execution_manifest_path=validation_path,
        )

    # A sophisticated forgery can update both the report hash and selection self-hash.  It
    # still cannot change the authenticated runner after search and compile a validation DAG.
    tampered_report = deepcopy(fixture["report"])
    tampered_report["frozen_invocation_validation"]["runner"][
        "functional_rho_floor"] = 0.95
    _write_json(fixture["report_path"], tampered_report)
    tampered_selection = deepcopy(fixture["selection"])
    tampered_selection["search_report_sha256"] = sha256_file(
        fixture["report_path"])
    _refresh_selection_content_hash(tampered_selection)
    _write_json(fixture["selection_path"], tampered_selection)
    with pytest.raises(ValueError, match="runner differs from source analysis.runner"):
        compile_breadth_execution_manifest(
            stage="validation",
            metric_panel_path=fixture["panel_path"],
            arm_bank_path=fixture["bank_path"],
            protocol_manifest_path=fixture["protocol_path"],
            packet_manifest_path=fixture["packet_path"],
            partition_integrity_path=fixture["integrity_path"],
            selection_artifact_path=fixture["selection_path"],
        )


def test_breadth_launcher_is_hard_restricted_to_sequential_physical_gpu_zero():
    root = Path(__file__).parents[3]
    launcher = (root / (
        "methods/codability/experiments/run_tacit_breadth_search_sk3.sh"
    )).read_text()

    assert "ALLOWED_DEVICE=0" in launcher
    assert "MAX_ACCOUNT_GPUS=4" in launcher
    assert 'ROOT=${ROOT:-/remote/anon/norm-research}' in launcher
    assert 'PY=${PY:-/remote/anon/miniconda3/bin/python}' in launcher
    assert 'MANIFEST=$CONF/search_execution_manifest_v2.json' in launcher
    assert '${TARGET_DEVICE:-0}' in launcher
    assert '${EXECUTOR_A_DEVICE:-0}' in launcher
    assert '${EXECUTOR_B_DEVICE:-0}' in launcher
    assert '${EXECUTOR_C_DEVICE:-0}' in launcher
    assert "physical GPU $ALLOWED_DEVICE only" in launcher
    assert "starting three domain-disjoint executor shards sequentially on GPU 0" in launcher
    assert "TARGET_DEVICES" not in launcher
    assert "0,2" not in launcher
    assert "EXECUTOR_B_DEVICE:-2" not in launcher
    assert "EXECUTOR_C_DEVICE:-4" not in launcher

def _release_only_dag_fixture(monkeypatch, tmp_path) -> dict:
    fixture = _canonical_breadth_dag_fixture(monkeypatch, tmp_path)
    validation = compile_breadth_execution_manifest(
        stage="validation",
        metric_panel_path=fixture["panel_path"],
        arm_bank_path=fixture["bank_path"],
        protocol_manifest_path=fixture["protocol_path"],
        packet_manifest_path=fixture["packet_path"],
        partition_integrity_path=fixture["integrity_path"],
        selection_artifact_path=fixture["selection_path"],
    )
    validation_path = tmp_path / "validation_execution.json"
    _write_json(validation_path, validation)
    return {
        **fixture,
        "validation": validation,
        "validation_path": validation_path,
    }


def test_release_only_cli_emits_authenticated_capability_without_validation_analysis(
        monkeypatch, tmp_path, capsys):
    fixture = _release_only_dag_fixture(monkeypatch, tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("release-only mode entered validation scoring/report analysis")

    monkeypatch.setattr(policy_runner, "run", forbidden)
    monkeypatch.setattr(policy_runner, "summarize_breadth_decomposition", forbidden)
    monkeypatch.setattr(sys, "argv", [
        "run_policy_isomorphism.py",
        "--release-only",
        "--search-execution-manifest", str(fixture["search_path"]),
        "--search-report", str(fixture["report_path"]),
        "--execution-manifest", str(fixture["validation_path"]),
        "--selection-artifact", str(fixture["selection_path"]),
        "--out", str(fixture["release_path"]),
    ])

    policy_runner.main()

    result = json.loads(capsys.readouterr().out)
    release = json.loads(fixture["release_path"].read_text())
    assert result["valid"] is True
    assert result["mode"] == "release_only"
    assert result["search_selection_provenance"]["valid"] is True
    assert result["lockbox_authorization"]["sealed_partition_authorized"] is True
    assert result["lockbox_authorization"]["lockbox_release_validation"][
        "valid"] is True
    assert release["fake_inputs"] is False
    assert release["selection_artifact_sha256"] == sha256_file(
        fixture["selection_path"])


def test_release_only_rejects_fake_nonproduction_and_forged_invocation(
        monkeypatch, tmp_path):
    fixture = _release_only_dag_fixture(monkeypatch, tmp_path)

    def bind_forged_report(report: dict) -> None:
        _write_json(fixture["report_path"], report)
        selection = deepcopy(fixture["selection"])
        selection["search_report_sha256"] = sha256_file(fixture["report_path"])
        _refresh_selection_content_hash(selection)
        _write_json(fixture["selection_path"], selection)
        validation = deepcopy(fixture["validation"])
        validation["selection_artifact_sha256"] = sha256_file(
            fixture["selection_path"])
        validation["selection_provenance_validation_at_freeze"][
            "search_report_sha256"] = sha256_file(fixture["report_path"])
        _write_json(fixture["validation_path"], validation)

    def attempt() -> None:
        policy_runner.write_two_manifest_lockbox_release(
            search_execution_manifest_path=fixture["search_path"],
            search_report_path=fixture["report_path"],
            selection_artifact_path=fixture["selection_path"],
            validation_execution_manifest_path=fixture["validation_path"],
            release_artifact_path=fixture["release_path"],
        )

    fake_report = deepcopy(fixture["report"])
    fake_report["cells"][0]["score_provenance_validation"]["small"][
        "fake_backend"] = True
    bind_forged_report(fake_report)
    with pytest.raises(ValueError, match="production score provenance"):
        attempt()
    assert not fixture["release_path"].exists()

    nonproduction_report = deepcopy(fixture["report"])
    nonproduction_report["cells"][0]["score_provenance_validation"]["target"][
        "backend_class"] = "FakeVLLM"
    bind_forged_report(nonproduction_report)
    with pytest.raises(ValueError, match="production score provenance"):
        attempt()
    assert not fixture["release_path"].exists()

    forged_invocation_report = deepcopy(fixture["report"])
    forged_invocation_report["frozen_invocation_validation"]["runner"][
        "functional_rho_floor"] = 0.95
    bind_forged_report(forged_invocation_report)
    with pytest.raises(ValueError, match="runner differs from source analysis.runner"):
        attempt()
    assert not fixture["release_path"].exists()
