"""Shared policy-experiment data invariants."""

import json
import numpy as np
import pytest
from pathlib import Path

from methods.codability.experiments.policy_data import (
    align_orbit,
    authorize_policy_partition,
    score_orbits,
    selection_required_for_phase,
    validate_additional_artifacts,
    validate_executor_prompt_bank,
    validate_policy_articulation_selection_provenance,
)
from methods.codability.experiments.build_fresh_item_partitions import (
    sha256_bytes,
    sha256_file,
)


def test_align_orbit_rejects_duplicate_source_and_target_hashes():
    orbit = {"canonical": np.array([0.1, 0.2])}
    with pytest.raises(ValueError, match="source hashes contain duplicates"):
        align_orbit(orbit, ["a", "a"], ["a"])
    with pytest.raises(ValueError, match="target hashes contain duplicates"):
        align_orbit(orbit, ["a", "b"], ["a", "a"])


def test_score_orbits_rejects_silent_duplicate_arm_form_rows():
    scores = np.array([[0.1, 0.2], [0.3, 0.4]])
    meta = [
        {"cell_id": "cell", "arm_id": "arm", "form": "canonical"},
        {"cell_id": "cell", "arm_id": "arm", "form": "canonical"},
    ]
    with pytest.raises(ValueError, match="duplicate orbit row"):
        score_orbits(scores, meta, cell_id="cell")


def test_executor_prompt_bank_requires_exact_arm_form_hash_identity():
    cell = {
        "id": "cell",
        "arms": [{
            "id": "definition",
            "forms": [{"id": "canonical", "prompt_sha256": "expected"}],
        }],
    }
    meta = [{
        "cell_id": "cell", "arm_id": "definition", "form": "canonical",
        "prompt_sha256": "expected",
    }]
    assert validate_executor_prompt_bank(meta, cell)["valid"]
    changed = [dict(meta[0], prompt_sha256="changed")]
    with pytest.raises(ValueError, match="prompt hash mismatch"):
        validate_executor_prompt_bank(changed, cell)


def test_same_version_lockbox_needs_hash_bound_authorization_and_release(tmp_path):
    with pytest.raises(ValueError, match="does not authorize partition"):
        authorize_policy_partition(
            "same_version_upper_lockbox",
            operation="test analysis",
        )

    root = Path(__file__).parents[3]
    execution = root / (
        "methods/codability/experiments/same_version_upper_execution_manifest_v1.json")
    selection = root / (
        "methods/codability/experiments/same_version_upper_selection_v1.json")
    with pytest.raises(ValueError, match="calibration-release"):
        authorize_policy_partition(
            "same_version_upper_lockbox",
            operation="test analysis",
            execution_manifest_path=execution,
            selection_artifact_path=selection,
        )

    manifest = json.loads(execution.read_text())
    report_path = tmp_path / "calibration_report.json"
    release_path = tmp_path / "calibration_release.json"
    manifest["lockbox_release"]["calibration_report_path"] = str(report_path)
    manifest["lockbox_release"]["artifact_path"] = str(release_path)
    temporary_execution = tmp_path / "execution.json"
    temporary_execution.write_text(json.dumps(manifest))
    execution_sha256 = sha256_file(temporary_execution)
    report = {
        "schema": manifest["lockbox_release"]["calibration_report_schema"],
        "partition": "same_version_upper_calibration",
        "arm_bank_sha256": manifest["arm_bank_sha256"],
        "partition_authorization": {
            "phase": "calibration",
            "execution_manifest_sha256": execution_sha256,
        },
        "frozen_invocation_validation": {"valid": True},
        "analysis_implementation": manifest["implementation"]["analysis"],
        "cells": [{
            "cell_id": "N_humor_49",
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
    release_path.write_text(json.dumps({
        "schema": manifest["lockbox_release"]["schema"],
        "status": "calibration-complete-production-only-lockbox-release",
        "execution_manifest_sha256": execution_sha256,
        "selection_artifact_sha256": sha256_file(selection),
        "calibration_partition": "same_version_upper_calibration",
        "lockbox_partition": "same_version_upper_lockbox",
        "calibration_report_path": str(report_path),
        "calibration_report_sha256": sha256_file(report_path),
        "fake_inputs": False,
    }))
    result = authorize_policy_partition(
        "same_version_upper_lockbox",
        operation="test analysis",
        execution_manifest_path=temporary_execution,
        selection_artifact_path=selection,
        lockbox_release_artifact_path=release_path,
    )
    assert result["sealed_partition_authorized"]
    assert result["phase"] == "lockbox"


def test_same_version_calibration_authenticates_same_frozen_selection():
    root = Path(__file__).parents[3]
    result = authorize_policy_partition(
        "same_version_upper_calibration",
        operation="test analysis",
        execution_manifest_path=root / (
            "methods/codability/experiments/same_version_upper_execution_manifest_v1.json"),
        selection_artifact_path=root / (
            "methods/codability/experiments/same_version_upper_selection_v1.json"),
    )
    assert not result["sealed_partition_authorized"]
    assert result["phase"] == "calibration"


def test_frozen_manifest_can_declare_a_new_open_calibration_partition(tmp_path):
    root = Path(__file__).parents[3]
    source_execution = root / (
        "methods/codability/experiments/same_version_upper_execution_manifest_v1.json")
    manifest = json.loads(source_execution.read_text())
    manifest["phases"] = {"calibration": ["breadth_calibration"]}
    manifest["phase_access"] = {"calibration": "open_development"}
    manifest["selection_required_phases"] = []

    source_integrity = root / manifest["partition_integrity_path"]
    integrity = json.loads(source_integrity.read_text())
    integrity["validated_partitions"] = ["breadth_calibration"]
    integrity_path = tmp_path / "integrity.json"
    integrity_path.write_text(json.dumps(integrity))
    manifest["partition_integrity_path"] = str(integrity_path)
    manifest["partition_integrity_sha256"] = sha256_file(integrity_path)
    execution_path = tmp_path / "execution.json"
    execution_path.write_text(json.dumps(manifest))

    result = authorize_policy_partition(
        "breadth_calibration",
        operation="breadth calibration test",
        execution_manifest_path=execution_path,
    )

    assert result["phase"] == "calibration"
    assert result["authorization"] == "manifest_declared_hash_bound_phase_access"
    assert not result["sealed_partition_authorized"]


def _generic_selection_dag(tmp_path):
    root = Path(__file__).parents[3]
    source_execution = root / (
        "methods/codability/experiments/same_version_upper_execution_manifest_v1.json")
    manifest = json.loads(source_execution.read_text())
    manifest.pop("selection_artifact_path", None)
    manifest.pop("selection_artifact_sha256", None)
    manifest["status"] = "frozen-before-tacit-breadth-validation-model-outcomes"
    manifest["phases"] = {"validation": ["tacit_breadth_validation"]}
    manifest["phase_access"] = {"validation": "open_development"}
    manifest["selection_required_phases"] = ["validation"]
    manifest["domain_tasks"] = {"humor": "humor"}
    manifest["item_text_max_chars_by_task"] = {"humor": 4096}
    for job in manifest["model_jobs"]:
        job["required_repetitions"] = [0, 1]
    manifest["analysis"]["n_boot"] = 10000
    manifest["analysis"]["runner"]["n_boot"] = 10000
    manifest["analysis"]["runner"]["cell_ids"] = ["N_humor_49"]

    source_integrity = root / manifest["partition_integrity_path"]
    integrity = json.loads(source_integrity.read_text())
    integrity["validated_partitions"] = [
        "tacit_breadth_search", "tacit_breadth_validation",
    ]
    integrity_path = tmp_path / "integrity.json"
    integrity_path.write_text(json.dumps(integrity))
    manifest["partition_integrity_path"] = str(integrity_path)
    manifest["partition_integrity_sha256"] = sha256_file(integrity_path)

    panel_path = tmp_path / "metric_panel.json"
    panel_path.write_text('{"panel": "frozen"}')
    model_template_path = tmp_path / "model_template.json"
    model_template_path.write_text('{"models": "frozen"}')
    manifest["additional_artifacts"] = [
        {
            "role": "metric_panel", "path": str(panel_path),
            "sha256": sha256_file(panel_path),
        },
        {
            "role": "model_environment_template", "path": str(model_template_path),
            "sha256": sha256_file(model_template_path),
        },
    ]
    bank = {
        "schema": "fresh_name_arm_bank/v3",
        "cells": [{
            "id": "N_humor_49",
            "arms": [
                {
                    "id": "name", "channel": "sparse",
                    "provenance": "construct_name", "control_for": None,
                    "components": [], "n_address_units": None,
                },
                {
                    "id": "source_definition", "channel": "declarative",
                    "provenance": "source_telling", "control_for": None,
                    "components": ["definition"], "n_address_units": None,
                },
                {
                    "id": "control_wrong_definition", "channel": "declarative",
                    "provenance": "wrong_construct_control",
                    "control_for": "source_definition", "components": [],
                    "n_address_units": None,
                },
                {
                    "id": "control_inert_definition", "channel": "declarative",
                    "provenance": "inert_length_control",
                    "control_for": "source_definition", "components": [],
                    "n_address_units": None,
                },
            ],
        }],
    }
    bank_path = tmp_path / "arm_bank.json"
    bank_path.write_text(json.dumps(bank))
    manifest["arm_bank_path"] = str(bank_path)
    manifest["arm_bank_sha256"] = sha256_file(bank_path)
    roles = [
        "best_functional_rank",
        "best_vector_identity",
        "best_component_distinct_route_within_rank_tolerance",
        "best_address_dose",
    ]
    policy = {
        "schema": "tacit_breadth_selection_policy/v1",
        "minimum_candidates_per_cell": 1,
        "maximum_candidates_per_cell": 4,
        "roles_in_order": roles,
        "frozen": True,
    }
    manifest["selection_policy"] = policy
    search_manifest = json.loads(json.dumps(manifest))
    search_manifest["status"] = "frozen-before-tacit-breadth-search-model-outcomes"
    search_manifest["phases"] = {"search": ["tacit_breadth_search"]}
    search_manifest["phase_access"] = {"search": "open_development"}
    search_manifest["selection_required_phases"] = []
    for job in search_manifest["model_jobs"]:
        job["required_repetitions"] = [0]
    search_manifest["analysis"]["n_boot"] = 2000
    search_manifest["analysis"]["runner"]["n_boot"] = 2000
    search_manifest_path = tmp_path / "search-execution.json"
    search_manifest_path.write_text(json.dumps(search_manifest))
    bank_ids = [cell["id"] for cell in bank["cells"]]
    search_sha = sha256_file(search_manifest_path)
    jobs = {row["id"]: row for row in search_manifest["model_jobs"]}
    backend = search_manifest["execution_environment"]["production_backend_class"]

    def score_provenance(job_id):
        job = jobs[job_id]
        return {
            "valid": True,
            "job_id": job_id,
            "repetitions": [0],
            "execution_manifest_sha256": search_sha,
            "arm_bank_sha256": manifest["arm_bank_sha256"],
            "packet_manifest_sha256": manifest["packet_manifest_sha256"],
            "binary_readout": manifest["binary_readout"],
            "readout_template_sha256": manifest["readout_template_sha256"],
            "role": job["role"],
            "backend_class": backend,
            "fake_backend": False,
        }

    search_report_path = tmp_path / "search-report.json"
    report = {
        "schema": "policy_isomorphism_experiment/v5",
        "partition": "tacit_breadth_search",
        "arm_bank_sha256": manifest["arm_bank_sha256"],
        "partition_authorization": {
            "phase": "search",
            "execution_manifest_sha256": search_sha,
            "selection_artifact_sha256": None,
        },
        "source_group_inference": {
            "enabled": True,
            "packet_manifest_sha256": manifest["packet_manifest_sha256"],
        },
        "additional_artifact_validation": {
            "valid": True,
            "files": [
                {"path": str(panel_path), "sha256": sha256_file(panel_path)},
                {"path": str(model_template_path),
                 "sha256": sha256_file(model_template_path)},
            ],
        },
        "frozen_invocation_validation": {
            "valid": True,
            "runner": search_manifest["analysis"]["runner"],
        },
        "cell_panel_identity_validation": {"valid": True},
        "analysis_implementation": search_manifest["implementation"]["analysis"],
        "cells": [{
            "cell_id": cell_id,
            "score_provenance_validation": {
                "small": score_provenance(
                    search_manifest["analysis"]["runner"]["small_job"]),
                "target": score_provenance(
                    search_manifest["analysis"]["runner"]["big_job"]),
            },
            "executor_prompt_bank_validation": {"valid": True},
            "target_prompt_bank_validation": {"valid": True},
            "scored_arm_panel_validation": {
                "small": {"valid": True}, "target": {"valid": True},
            },
            "score_cell_identity_validation": {
                "small": {"valid": True}, "target": {"valid": True},
            },
            "source_group_validation": {"valid": True},
            "rows": [
                {"arm_id": "source_definition"},
                {"arm_id": "control_wrong_definition"},
                {"arm_id": "control_inert_definition"},
            ],
        } for cell_id in bank_ids],
    }
    search_report_path.write_text(json.dumps(report))
    policy_sha = sha256_bytes(json.dumps(
        policy, sort_keys=True, separators=(",", ":")).encode())
    assigned_reason = "best frozen primary-order explicit articulation"
    role_assignments = [{
        "role": role,
        "status": "assigned" if index == 0 else "not_available",
        "arm_id": "source_definition" if index == 0 else None,
        "role_rank": 1 if index == 0 else None,
        "selection_reason": (
            assigned_reason if index == 0 else f"no arm satisfies the frozen {role} rule"
        ),
    } for index, role in enumerate(roles)]
    selected_cell = {
        "cell_id": "N_humor_49",
        "allowed_arm_ids": [
            "name", "source_definition", "control_wrong_definition",
            "control_inert_definition",
        ],
        "candidate_arm_ids": ["source_definition"],
        "control_ids": ["control_wrong_definition", "control_inert_definition"],
        "required_control_provenances": [
            "wrong_construct_control", "inert_length_control"],
        "candidate_selections": [{
            "arm_id": "source_definition",
            "roles": [roles[0]],
            "role_ranks": {roles[0]: 1},
            "selection_reasons": [assigned_reason],
            "primary_rank": 1,
            "vector_rank": 1,
            "matched_control_ids": [
                "control_wrong_definition", "control_inert_definition"],
            "selection_features": {
                "arm_id": "source_definition", "channel": "declarative",
                "components": ["definition"], "n_address_units": None,
            },
        }],
        "role_assignments": role_assignments,
        "selection_reason": "deterministic frozen role policy",
    }
    selection_path = tmp_path / "selection.json"
    selection = {
        "schema": "policy_articulation_selection/v1",
        "status": "frozen-after-search-before-validation-scoring",
        "search_phase": "search",
        "search_partition": "tacit_breadth_search",
        "selected_phase": "validation",
        "selected_partition": "tacit_breadth_validation",
        "search_execution_manifest_path": str(search_manifest_path),
        "search_execution_manifest_sha256": sha256_file(search_manifest_path),
        "search_report_path": str(search_report_path),
        "search_report_sha256": sha256_file(search_report_path),
        "arm_bank_path": str(bank_path),
        "arm_bank_sha256": manifest["arm_bank_sha256"],
        "packet_manifest_path": manifest["packet_manifest_path"],
        "packet_manifest_sha256": manifest["packet_manifest_sha256"],
        "metric_panel_path": str(panel_path),
        "metric_panel_sha256": sha256_file(panel_path),
        "additional_artifacts": [{
            "role": "model_environment_template", "path": str(model_template_path),
            "sha256": sha256_file(model_template_path),
        }],
        "selection_policy": policy,
        "selection_policy_sha256": policy_sha,
        "n_cells": len(bank_ids),
        "candidate_count_range": [1, 1],
        "cells": [selected_cell],
    }
    selection["selection_content_sha256"] = sha256_bytes(json.dumps(
        selection, sort_keys=True, separators=(",", ":")).encode())
    selection_path.write_text(json.dumps(selection))
    manifest["selection_artifact_path"] = str(selection_path)
    manifest["selection_artifact_sha256"] = sha256_file(selection_path)
    execution_path = tmp_path / "execution.json"
    execution_path.write_text(json.dumps(manifest))
    return {
        "manifest": manifest,
        "manifest_path": execution_path,
        "search_manifest": search_manifest,
        "search_manifest_path": search_manifest_path,
        "report": report,
        "report_path": search_report_path,
        "selection": selection,
        "selection_path": selection_path,
        "panel_path": panel_path,
    }


def _refresh_selection_self_hash(selection):
    selection.pop("selection_content_sha256", None)
    selection["selection_content_sha256"] = sha256_bytes(json.dumps(
        selection, sort_keys=True, separators=(",", ":")).encode())


def test_open_search_needs_no_selection_but_open_validation_uses_generic_selection(
        tmp_path):
    fixture = _generic_selection_dag(tmp_path)
    manifest = fixture["manifest"]
    execution_path = fixture["manifest_path"]
    search_manifest_path = fixture["search_manifest_path"]
    selection_path = fixture["selection_path"]
    panel_path = fixture["panel_path"]

    assert not selection_required_for_phase(fixture["search_manifest"], "search")
    assert selection_required_for_phase(manifest, "validation")
    search = authorize_policy_partition(
        "tacit_breadth_search",
        operation="all-arm search test",
        execution_manifest_path=search_manifest_path,
    )
    assert search["phase"] == "search"
    assert search["selection_artifact_path"] is None

    validation = authorize_policy_partition(
        "tacit_breadth_validation",
        operation="selected-arm validation test",
        execution_manifest_path=execution_path,
        selection_artifact_path=selection_path,
    )
    assert validation["phase"] == "validation"
    assert not validation["sealed_partition_authorized"]
    assert validation["lockbox_release_validation"] is None
    assert validation["selection_provenance_validation"]["valid"] is True
    assert validation["additional_artifact_validation"]["files"][0] == {
        "path": str(panel_path), "sha256": sha256_file(panel_path),
    }
    assert len(validation["additional_artifact_validation"]["files"]) == 2

    frozen_selection = json.loads(selection_path.read_text())
    wrong_hash = dict(frozen_selection, search_execution_manifest_sha256="0" * 64)
    _refresh_selection_self_hash(wrong_hash)
    with pytest.raises(ValueError, match="source search hashes"):
        validate_policy_articulation_selection_provenance(
            wrong_hash,
            selection_path=selection_path,
            execution_manifest=manifest,
            execution_manifest_path=execution_path,
        )
    empty = dict(frozen_selection, cells=[], n_cells=0, candidate_count_range=[])
    _refresh_selection_self_hash(empty)
    with pytest.raises(ValueError, match="nonemptily cover every bank/report cell"):
        validate_policy_articulation_selection_provenance(
            empty,
            selection_path=selection_path,
            execution_manifest=manifest,
            execution_manifest_path=execution_path,
        )

    panel_path.write_text('{"panel": "changed"}')
    with pytest.raises(ValueError, match="additional artifact changed"):
        validate_additional_artifacts(manifest, manifest_path=execution_path)


@pytest.mark.parametrize("drift", ["implementation", "gate", "model"])
def test_policy_articulation_dag_rejects_post_search_code_gate_and_model_drift(
        tmp_path, drift):
    fixture = _generic_selection_dag(tmp_path)
    manifest = json.loads(json.dumps(fixture["manifest"]))
    if drift == "implementation":
        manifest["implementation"]["analysis"]["files"][0]["sha256"] = "0" * 64
    elif drift == "gate":
        manifest["analysis"]["runner"]["functional_rho_floor"] = 0.95
    else:
        manifest["model_jobs"][0]["provider_model"] = "drifted/model"

    with pytest.raises(ValueError, match="immutable search/validation fields differ"):
        validate_policy_articulation_selection_provenance(
            fixture["selection"],
            selection_path=fixture["selection_path"],
            execution_manifest=manifest,
            execution_manifest_path=fixture["manifest_path"],
        )


def test_policy_articulation_dag_rejects_selection_self_hash(tmp_path):
    fixture = _generic_selection_dag(tmp_path)
    forged_selection = json.loads(json.dumps(fixture["selection"]))
    forged_selection["candidate_count_range"] = [1, 4]
    with pytest.raises(ValueError, match="content self-hash"):
        validate_policy_articulation_selection_provenance(
            forged_selection,
            selection_path=fixture["selection_path"],
            execution_manifest=fixture["manifest"],
            execution_manifest_path=fixture["manifest_path"],
        )


@pytest.mark.parametrize(
    ("forgery", "error"),
    [
        ("runner", "runner differs from source analysis.runner"),
        ("analysis", "analysis implementation is unauthenticated"),
        ("prompt", "lacks prompt/arm/cell closure"),
        ("arm", "lacks prompt/arm/cell closure"),
        ("cell", "lacks prompt/arm/cell closure"),
        ("provenance", "lacks complete production score provenance"),
    ],
)
def test_policy_articulation_dag_rejects_forged_search_report(
        tmp_path, forgery, error):
    fixture = _generic_selection_dag(tmp_path)
    report = json.loads(json.dumps(fixture["report"]))
    if forgery == "runner":
        report["frozen_invocation_validation"]["runner"][
            "functional_rho_floor"] = 0.95
    elif forgery == "analysis":
        report["analysis_implementation"]["files"][0]["sha256"] = "0" * 64
    elif forgery == "prompt":
        report["cells"][0]["executor_prompt_bank_validation"]["valid"] = False
    elif forgery == "arm":
        report["cells"][0]["scored_arm_panel_validation"]["small"]["valid"] = False
    elif forgery == "cell":
        report["cells"][0]["score_cell_identity_validation"]["target"]["valid"] = False
    else:
        report["cells"][0]["score_provenance_validation"]["small"].pop("job_id")
    fixture["report_path"].write_text(json.dumps(report))
    selection = json.loads(json.dumps(fixture["selection"]))
    selection["search_report_sha256"] = sha256_file(fixture["report_path"])
    _refresh_selection_self_hash(selection)
    fixture["selection_path"].write_text(json.dumps(selection))
    manifest = json.loads(json.dumps(fixture["manifest"]))
    manifest["selection_artifact_sha256"] = sha256_file(fixture["selection_path"])
    with pytest.raises(ValueError, match=error):
        validate_policy_articulation_selection_provenance(
            selection,
            selection_path=fixture["selection_path"],
            execution_manifest=manifest,
            execution_manifest_path=fixture["manifest_path"],
        )


@pytest.mark.parametrize(
    ("malformation", "error"),
    [
        ("count_range", "candidate_count_range"),
        ("duplicate_artifact", "duplicate role bindings"),
        ("roles", "metadata is invalid"),
        ("controls", "invalid control metadata"),
    ],
)
def test_policy_articulation_dag_rejects_malformed_selection_metadata(
        tmp_path, malformation, error):
    fixture = _generic_selection_dag(tmp_path)
    selection = json.loads(json.dumps(fixture["selection"]))
    if malformation == "count_range":
        selection["candidate_count_range"] = [1, 4]
    elif malformation == "duplicate_artifact":
        selection["additional_artifacts"].append(
            dict(selection["additional_artifacts"][0]))
    elif malformation == "roles":
        selection["cells"][0]["candidate_selections"][0]["roles"] = []
    else:
        selection["cells"][0]["control_ids"].reverse()
    _refresh_selection_self_hash(selection)
    fixture["selection_path"].write_text(json.dumps(selection))
    manifest = json.loads(json.dumps(fixture["manifest"]))
    manifest["selection_artifact_sha256"] = sha256_file(fixture["selection_path"])
    with pytest.raises(ValueError, match=error):
        validate_policy_articulation_selection_provenance(
            selection,
            selection_path=fixture["selection_path"],
            execution_manifest=manifest,
            execution_manifest_path=fixture["manifest_path"],
        )


def test_selection_required_phases_must_name_declared_phases():
    manifest = {
        "phases": {"search": ["items"]},
        "selection_required_phases": ["validation"],
    }
    with pytest.raises(ValueError, match="undeclared phases"):
        selection_required_for_phase(manifest, "search")
