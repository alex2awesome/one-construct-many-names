"""Development/lockbox filtering and arm-policy tests."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from methods.codability.experiments import score_fresh_name_arms as scorer_module
from methods.codability.experiments.build_fresh_item_partitions import sha256_file
from methods.codability.experiments.policy_data import validate_frozen_implementation
from methods.codability.experiments.score_fresh_name_arms import (
    filter_items,
    load_lockbox_selection,
    run as run_scoring,
    score_domain,
    selection_scope_for_manifest,
    selected_arms,
)
from methods.codability.experiments.run_policy_isomorphism import (
    _analysis_implementation,
)


def test_filter_items_and_name_only_policy():
    items = {
        "rows": [{"i": 0}, {"i": 1}], "texts": ["a", "b"], "hashes": ["h0", "h1"],
        "partitions": ["dev", "lockbox"], "partition_files": [
            {"path": "/x/dev.jsonl", "sha256": "a"},
            {"path": "/x/lockbox.jsonl", "sha256": "b"}],
    }
    filtered = filter_items(items, ["dev"])
    assert filtered["texts"] == ["a"]
    cell = {"arms": [{"id": "name"}, {"id": "source_definition"}]}
    assert [row["id"] for row in selected_arms(cell, "name_only")] == ["name"]
    assert len(selected_arms(cell, "all")) == 2


@pytest.mark.parametrize("prior", [None, b"authenticated-old-artifact"])
def test_atomic_score_writer_never_publishes_a_partial_destination(
        tmp_path, monkeypatch, prior):
    destination = tmp_path / "scores.npz"
    if prior is not None:
        destination.write_bytes(prior)

    def fail_after_partial_write(path, **_arrays):
        Path(path).write_bytes(b"partial")
        raise RuntimeError("simulated interrupted compression")

    monkeypatch.setattr(scorer_module.np, "savez_compressed", fail_after_partial_write)
    with pytest.raises(RuntimeError, match="interrupted compression"):
        scorer_module.atomic_savez_compressed(
            destination, scores=np.asarray([[0.2, 0.8]]))
    if prior is None:
        assert not destination.exists()
        assert list(tmp_path.iterdir()) == []
    else:
        assert destination.read_bytes() == prior
        assert list(tmp_path.iterdir()) == [destination]


def test_generic_score_domain_uses_declared_task_and_level_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "methods.metric_implementer.experiments.alpha_probe.signature",
        lambda *_args, **_kwargs: np.asarray([0.2, 0.8]),
    )
    cell = {
        "id": "TB::demo::R2::node",
        "domain": "math-stackexchange",
        "task": "math-stackexchange",
        "level": "R2",
        "bucket": "general",
        "metric_id": "demo::R2::node",
        "node_id": "demo::R2::node",
        "gi": 7,
        "construct": "Clear proof",
        "arms": [{
            "id": "name", "channel": "sparse", "provenance": "construct_name",
            "control_for": None, "semantic_content_word_count": 2,
            "forms": [{"id": "canonical", "prompt": "Clear proof",
                       "prompt_sha256": "prompt-hash"}],
        }],
    }
    items = {
        "texts": ["a", "b"], "hashes": ["ha", "hb"],
        "partitions": ["cal", "cal"], "item_set_sha256": "set-hash",
        "partition_files": [],
    }
    score_kwargs = {
        "backend": object(),
        "model_job": {
            "id": "small", "model": "org/model", "role": "executor",
            "arm_policy": "all",
        },
        "domain": "math-stackexchange",
        "cells": [cell],
        "items": items,
        "readout_template": "{rubric}\n{text}",
        "arm_bank_sha256": "bank",
        "packet_manifest_sha256": "packet",
        "execution_manifest_sha256": "execution",
        "phase": "calibration",
        "out_dir": tmp_path,
        "repetition": 0,
    }
    result = score_domain(**score_kwargs)

    with np.load(result["path"], allow_pickle=True) as artifact:
        meta = json.loads(str(artifact["meta"][0]))
    assert meta["task"] == "math-stackexchange"
    assert meta["level"] == "R2"
    assert meta["metric_id"] == "demo::R2::node"
    artifact_path = Path(result["path"])
    sidecar_path = artifact_path.with_suffix(".json")
    sidecar = json.loads(sidecar_path.read_text())
    assert sidecar["score_artifact_sha256"] == sha256_file(artifact_path)

    monkeypatch.setattr(
        "methods.metric_implementer.experiments.alpha_probe.signature",
        lambda *_args, **_kwargs: pytest.fail("authenticated resume rescored a domain"),
    )
    resumed = score_domain(**score_kwargs)
    assert resumed["status"] == "already_complete"

    sidecar["score_artifact_sha256"] = "0" * 64
    sidecar_path.write_text(json.dumps(sidecar))
    with pytest.raises(ValueError, match="stale output sidecar"):
        score_domain(**score_kwargs)


def _teacher_forced_batch_fixture():
    forms = [
        {
            "id": form_id,
            "prompt": f"{{construct}} guidance in {form_id} form",
            "prompt_sha256": f"prompt-{form_id}",
        }
        for form_id in ("canonical", "question", "boilerplate")
    ]
    cells = []
    for cell_index in range(2):
        arms = []
        for arm_index in range(4):
            arm_id = "name" if arm_index == 0 else f"source_{arm_index}"
            arms.append({
                "id": arm_id,
                "channel": "sparse" if arm_index == 0 else "declarative",
                "provenance": (
                    "construct_name" if arm_index == 0 else "source_hierarchy_definition"
                ),
                "control_for": None,
                "semantic_content_word_count": 2 + arm_index,
                "added_content_word_count": arm_index,
                "n_address_units": None if arm_index == 0 else arm_index,
                "forms": [
                    {
                        **form,
                        "prompt": form["prompt"].format(
                            construct=f"cell {cell_index} arm {arm_index}"
                        ),
                        "prompt_sha256": (
                            f"cell-{cell_index}-arm-{arm_index}-{form['id']}"
                        ),
                    }
                    for form in forms
                ],
            })
        cells.append({
            "id": f"TB::humor::R1::{cell_index}",
            "domain": "humor",
            "task": "humor",
            "level": "R1",
            "bucket": "general",
            "metric_id": f"humor::R1::{cell_index}",
            "node_id": f"humor::R1::{cell_index}",
            "gi": cell_index,
            "construct": f"construct {cell_index}",
            "arms": arms,
        })
    items = {
        "texts": ["item zero", "item one", "item two", "item three"],
        "hashes": ["h0", "h1", "h2", "h3"],
        "partitions": ["search"] * 4,
        "item_set_sha256": "set-hash",
        "partition_files": [],
    }
    return cells, items


class FakeVLLM:
    """Counting equivalent of the production FakeVLLM for scorer batching tests."""

    def __init__(self):
        self.calls = []

    def score_binary(self, prompts, *, pos, neg, seed):
        seeds = list(seed) if isinstance(seed, (list, tuple)) else [seed] * len(prompts)
        assert len(seeds) == len(prompts)
        self.calls.append({
            "prompts": list(prompts), "pos": pos, "neg": neg,
            "seeds": list(seeds), "seed_was_scalar": not isinstance(seed, (list, tuple)),
        })
        return [
            round((int(hashlib.sha256((prompt + str(item_seed)).encode()).hexdigest(), 16)
                   % 1000) / 999.0, 3)
            for prompt, item_seed in zip(prompts, seeds)
        ]


def _score_teacher_forced_fixture(tmp_path, *, row_batch_size):
    cells, items = _teacher_forced_batch_fixture()
    backend = FakeVLLM()
    result = score_domain(
        backend=backend,
        model_job={
            "id": "small", "model": "org/model", "role": "executor",
            "arm_policy": "all",
        },
        domain="humor",
        cells=cells,
        items=items,
        readout_template="RUBRIC:\n{rubric}\nITEM:\n{text}",
        arm_bank_sha256="bank",
        packet_manifest_sha256="packet",
        execution_manifest_sha256="execution",
        phase="calibration",
        out_dir=tmp_path,
        repetition=2,
        binary_readout="teacher_forced_declared_labels",
        label_token_ids={"YES": 1, "NO": 2},
        teacher_forced_row_batch_size=row_batch_size,
    )
    with np.load(result["path"], allow_pickle=True) as artifact:
        payload = {key: artifact[key].copy() for key in artifact.files}
    return backend, payload


def test_teacher_forced_row_batching_is_artifact_equivalent_and_reduces_calls(tmp_path):
    legacy_backend, legacy = _score_teacher_forced_fixture(
        tmp_path / "legacy", row_batch_size=None)
    one_backend, row_one = _score_teacher_forced_fixture(
        tmp_path / "row-one", row_batch_size=1)
    eight_backend, row_eight = _score_teacher_forced_fixture(
        tmp_path / "row-eight", row_batch_size=8)

    assert len(legacy_backend.calls) == 24
    assert len(one_backend.calls) == 24
    assert len(eight_backend.calls) == 3
    assert all(call["seed_was_scalar"] for call in legacy_backend.calls)
    assert not any(call["seed_was_scalar"] for call in one_backend.calls)
    assert not any(call["seed_was_scalar"] for call in eight_backend.calls)
    # Every frozen row seed is repeated over its four item prompts, in unchanged row-major order.
    first_seed = 2 * 1_000_003 + 20260713
    assert eight_backend.calls[0]["seeds"][:8] == [first_seed] * 4 + [
        first_seed + 1009] * 4

    assert legacy.keys() == row_one.keys() == row_eight.keys()
    for key in legacy:
        assert np.array_equal(legacy[key], row_one[key]), key
        assert np.array_equal(row_one[key], row_eight[key]), key
    metadata = [json.loads(str(value)) for value in row_eight["meta"]]
    assert metadata[0]["added_content_word_count"] == 0
    assert metadata[0]["n_address_units"] is None
    assert metadata[4]["added_content_word_count"] == 1
    assert metadata[4]["n_address_units"] == 1
    assert metadata[4]["prompt_sha256"] == "cell-0-arm-1-question"


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "8"])
def test_teacher_forced_row_batching_rejects_invalid_batch_size(tmp_path, value):
    cells, items = _teacher_forced_batch_fixture()
    with pytest.raises(ValueError, match="positive integer"):
        score_domain(
            backend=FakeVLLM(),
            model_job={
                "id": "small", "model": "org/model", "role": "executor",
                "arm_policy": "all",
            },
            domain="humor", cells=cells, items=items,
            readout_template="{rubric}\n{text}",
            arm_bank_sha256="bank", packet_manifest_sha256="packet",
            execution_manifest_sha256="execution", phase="calibration",
            out_dir=tmp_path / str(value), repetition=0,
            binary_readout="teacher_forced_declared_labels",
            teacher_forced_row_batch_size=value,
        )


def test_scoring_resolves_manifest_relative_inputs_after_cwd_change(tmp_path, monkeypatch):
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text("{}")
    packet_path = tmp_path / "packet.json"
    packet_path.write_text("{}")
    packet_root = tmp_path / "packet"
    packet_root.mkdir()
    bank_path = tmp_path / "bank.json"
    bank_path.write_text(json.dumps({
        "cells": [{
            "id": "cell", "domain": "humor", "task": "humor",
            "construct": "funny", "arms": [],
        }],
    }))
    target_path = tmp_path / "target.json"
    target_path.write_text(json.dumps({"readout_template": "{rubric}\n{text}"}))
    manifest_path = tmp_path / "execution.json"
    manifest_path.write_text(json.dumps({
        "schema": "fresh_name_execution_manifest/v1",
        "phases": {"development": ["dev"]},
        "protocol_manifest_path": "protocol.json",
        "protocol_manifest_sha256": sha256_file(protocol_path),
        "packet_manifest_sha256": sha256_file(packet_path),
        "arm_bank_sha256": sha256_file(bank_path),
        "domains": ["humor"],
        "model_jobs": [{
            "id": "small", "model": "fake-model", "role": "executor",
            "arm_policy": "all", "required_repetitions": [0],
        }],
        "resource_policy": {"maximum_gpus_for_any_job": 4},
    }))
    observed = {}

    def validate_packet(_packet, **kwargs):
        observed["protocol_path"] = kwargs["protocol_path"]
        return {"valid": True, "errors": []}

    monkeypatch.setattr(
        "methods.codability.experiments.score_fresh_name_arms.validate_packet",
        validate_packet,
    )
    monkeypatch.setattr(
        "methods.codability.experiments.score_fresh_name_arms.make_judge_backend",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("backend-stop")),
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    with pytest.raises(RuntimeError, match="backend-stop"):
        run_scoring(
            model_job_id="small",
            phase="development",
            arm_bank_path="bank.json",
            packet_root="packet",
            packet_manifest="packet.json",
            target_manifest_path="target.json",
            out_dir=str(tmp_path / "out"),
            execution_manifest_path=manifest_path,
            fake=True,
        )

    assert Path(observed["protocol_path"]).resolve() == protocol_path.resolve()


_SHARD_DOMAIN_TASKS = {
    "humor": "humor",
    "math": "math-stackexchange",
    "pr": "press-releases",
}


def _write_sharded_scoring_fixture(
        tmp_path: Path, *, bank: dict | None = None,
        phase: str = "development", partition: str = "dev",
        selection: dict | None = None,
        manifest_updates: dict | None = None) -> dict:
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text("{}")
    packet_path = tmp_path / "packet.json"
    packet_path.write_text("{}")
    packet_root = tmp_path / "packet"
    packet_root.mkdir()
    bank = bank or {
        "cells": [
            {
                "id": f"{domain}-cell",
                "domain": domain,
                "task": task,
                "construct": f"{domain} construct",
                "arms": [],
            }
            for domain, task in _SHARD_DOMAIN_TASKS.items()
        ],
    }
    bank_path = tmp_path / "bank.json"
    bank_path.write_text(json.dumps(bank))
    target_path = tmp_path / "target.json"
    target_path.write_text(json.dumps({"readout_template": "{rubric}\n{text}"}))
    selection_path = None
    if selection is not None:
        selection = dict(selection)
        selection["arm_bank_sha256"] = sha256_file(bank_path)
        selection["packet_manifest_sha256"] = sha256_file(packet_path)
        selection_path = tmp_path / "selection.json"
        selection_path.write_text(json.dumps(selection))
    manifest = {
        "schema": "fresh_name_execution_manifest/v1",
        "phases": {phase: [partition]},
        "protocol_manifest_path": "protocol.json",
        "protocol_manifest_sha256": sha256_file(protocol_path),
        "packet_manifest_sha256": sha256_file(packet_path),
        "arm_bank_sha256": sha256_file(bank_path),
        "domains": list(_SHARD_DOMAIN_TASKS),
        "domain_tasks": _SHARD_DOMAIN_TASKS,
        "model_jobs": [{
            "id": "small", "model": "fake-model", "role": "executor",
            "arm_policy": "all", "required_repetitions": [0],
        }],
        "resource_policy": {"maximum_gpus_for_any_job": 4},
    }
    if selection_path is not None:
        manifest.update({
            "selection_required_phases": [phase],
            "selection_artifact_path": "selection.json",
            "selection_artifact_sha256": sha256_file(selection_path),
        })
    if manifest_updates:
        manifest.update(manifest_updates)
    manifest_path = tmp_path / "execution.json"
    manifest_path.write_text(json.dumps(manifest))
    return {
        "manifest": manifest_path,
        "bank": bank_path,
        "packet": packet_path,
        "packet_root": packet_root,
        "target": target_path,
        "selection": selection_path,
        "out": tmp_path / "out",
    }


def _patch_sharded_scoring(monkeypatch, observed: dict) -> None:
    def validate_packet(_packet, **kwargs):
        observed.setdefault("packet_validations", []).append(kwargs)
        return {"valid": True, "errors": []}

    def load_items(_root, domain, *, partitions):
        observed.setdefault("loaded_domains", []).append(domain)
        return {
            "texts": [f"{domain} item"],
            "hashes": [f"{domain}-hash"],
            "partitions": list(partitions),
            "partition_files": [],
            "item_set_sha256": f"{domain}-set",
        }

    def make_backend(*_args, **_kwargs):
        observed["backend_calls"] = observed.get("backend_calls", 0) + 1
        return object()

    def score(**kwargs):
        domain = kwargs["domain"]
        observed.setdefault("scored_domains", []).append(domain)
        observed.setdefault("scored_cell_domains", []).append(
            {cell["domain"] for cell in kwargs["cells"]})
        observed.setdefault("row_batch_sizes", []).append(
            kwargs["teacher_forced_row_batch_size"])
        return {"domain": domain, "status": "complete"}

    monkeypatch.setattr(
        "methods.codability.experiments.score_fresh_name_arms.validate_packet",
        validate_packet,
    )
    monkeypatch.setattr(
        "methods.codability.experiments.score_fresh_name_arms.load_domain_items",
        load_items,
    )
    monkeypatch.setattr(
        "methods.codability.experiments.score_fresh_name_arms.make_judge_backend",
        make_backend,
    )
    monkeypatch.setattr(
        "methods.codability.experiments.score_fresh_name_arms.score_domain",
        score,
    )


def _run_sharded_fixture(paths: dict, **kwargs) -> dict:
    return run_scoring(
        model_job_id="small",
        phase=kwargs.pop("phase", "development"),
        arm_bank_path=str(paths["bank"]),
        packet_root=str(paths["packet_root"]),
        packet_manifest=str(paths["packet"]),
        target_manifest_path=str(paths["target"]),
        out_dir=str(paths["out"]),
        execution_manifest_path=paths["manifest"],
        selection_artifact=(
            str(paths["selection"]) if paths["selection"] is not None else None
        ),
        fake=True,
        **kwargs,
    )


def test_domain_subset_scores_only_requested_domains_and_records_canonical_order(
        tmp_path, monkeypatch):
    paths = _write_sharded_scoring_fixture(tmp_path)
    observed = {}
    _patch_sharded_scoring(monkeypatch, observed)

    result = _run_sharded_fixture(paths, domains=("pr", "humor"))

    # Execution order is the frozen manifest order, not caller-dependent shard order.
    assert result["requested_domains"] == ["humor", "pr"]
    assert [row["domain"] for row in result["results"]] == ["humor", "pr"]
    assert observed["loaded_domains"] == ["humor", "pr"]
    assert observed["scored_domains"] == ["humor", "pr"]
    assert observed["scored_cell_domains"] == [{"humor"}, {"pr"}]
    assert observed["row_batch_sizes"] == [None, None]
    assert observed["backend_calls"] == 1
    assert observed["packet_validations"][0]["domains"] == {"humor", "pr"}
    assert observed["packet_validations"][0]["partitions"] == {"dev"}


def test_manifest_frozen_teacher_forced_row_batch_size_reaches_each_domain(
        tmp_path, monkeypatch):
    paths = _write_sharded_scoring_fixture(
        tmp_path,
        manifest_updates={
            "binary_readout": "teacher_forced_declared_labels",
            "teacher_forced_row_batch_size": 8,
            "label_support": ["YES", "NO"],
            "teacher_forced_label_validation": {"YES_token_id": 1, "NO_token_id": 2},
        },
    )
    observed = {}
    _patch_sharded_scoring(monkeypatch, observed)

    _run_sharded_fixture(paths, domains=("humor", "math"))

    assert observed["row_batch_sizes"] == [8, 8]


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "8"])
def test_manifest_rejects_invalid_teacher_forced_row_batch_size_before_backend(
        tmp_path, monkeypatch, value):
    paths = _write_sharded_scoring_fixture(
        tmp_path,
        manifest_updates={
            "binary_readout": "teacher_forced_declared_labels",
            "teacher_forced_row_batch_size": value,
        },
    )
    observed = {}
    _patch_sharded_scoring(monkeypatch, observed)

    with pytest.raises(ValueError, match="positive integer"):
        _run_sharded_fixture(paths, domains=("humor",))

    assert observed.get("backend_calls", 0) == 0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "contains no cells for declared domain 'pr'"),
        ("registry", "domain/task registry differs"),
    ],
)
def test_domain_subset_still_validates_complete_declared_bank_registry(
        tmp_path, monkeypatch, mutation, message):
    cells = [
        {
            "id": f"{domain}-cell", "domain": domain, "task": task,
            "construct": f"{domain} construct", "arms": [],
        }
        for domain, task in _SHARD_DOMAIN_TASKS.items()
    ]
    if mutation == "missing":
        cells = [cell for cell in cells if cell["domain"] != "pr"]
    else:
        next(cell for cell in cells if cell["domain"] == "pr")["task"] = "humor"
    paths = _write_sharded_scoring_fixture(tmp_path, bank={"cells": cells})
    observed = {}
    _patch_sharded_scoring(monkeypatch, observed)

    with pytest.raises(ValueError, match=message):
        _run_sharded_fixture(paths, domains=("humor",))

    # The defect is in an unrequested domain, but it is still fatal before model loading.
    assert observed.get("loaded_domains", []) == []
    assert observed.get("backend_calls", 0) == 0


@pytest.mark.parametrize(
    "domains",
    [
        pytest.param((), id="empty"),
        pytest.param(("humor", "humor"), id="duplicate"),
        pytest.param(("unknown",), id="unknown"),
    ],
)
def test_domain_subset_rejects_empty_duplicate_or_unknown_values(
        tmp_path, monkeypatch, domains):
    paths = _write_sharded_scoring_fixture(tmp_path)
    observed = {}
    _patch_sharded_scoring(monkeypatch, observed)

    with pytest.raises(ValueError, match="nonempty unique subset"):
        _run_sharded_fixture(paths, domains=domains)

    assert observed.get("packet_validations", []) == []
    assert observed.get("backend_calls", 0) == 0


def test_sealed_validation_release_precedes_all_packet_io_and_valid_path_runs(
        tmp_path, monkeypatch):
    """A denied release must fail before even resolving caller-supplied packet paths."""
    paths = _write_sharded_scoring_fixture(
        tmp_path,
        phase="lockbox",
        partition="tacit_breadth_validation",
        manifest_updates={
            "schema": "fresh_name_execution_manifest/v2",
            "status": "frozen-before-validation-scoring",
            "phase_access": {"lockbox": "sealed_confirmation"},
            # Selection authentication is covered independently.  This fixture isolates the
            # release/packet-I/O ordering barrier itself.
            "selection_required_phases": [],
        },
    )
    release_path = tmp_path / "validation-release.json"
    release_path.write_text("denied")
    authorized = {"value": False}
    events = []

    real_resolve = scorer_module._resolve_declared_path
    guarded_caller_paths = {
        str(paths["bank"]), str(paths["packet"]), str(paths["packet_root"]),
        str(paths["target"]),
    }

    def guarded_resolve(value, *, manifest_path):
        if str(value) in guarded_caller_paths:
            assert authorized["value"], (
                f"caller packet-related path resolved before release: {value}"
            )
            events.append("resolve")
        return real_resolve(value, manifest_path=manifest_path)

    def authorize(partition, **kwargs):
        events.append("authorize")
        assert partition == "tacit_breadth_validation"
        assert kwargs["lockbox_release_artifact_path"] == str(release_path)
        if release_path.read_text() != "released":
            raise ValueError("validation release denied")
        authorized["value"] = True
        return {
            "partition": partition,
            "phase": "lockbox",
            "sealed_partition_authorized": True,
            "lockbox_release_validation": {"valid": True},
        }

    def packet_access(label):
        def guarded(*_args, **_kwargs):
            assert authorized["value"], f"{label} ran before release authentication"
            events.append(label)
            if label == "validate_packet":
                return {"valid": True, "errors": []}
            if label == "load_domain_items":
                return {
                    "texts": ["held-out item"],
                    "hashes": ["held-out-hash"],
                    "partitions": ["tacit_breadth_validation"],
                    "partition_files": [],
                    "item_set_sha256": "held-out-set",
                }
            if label == "load_partition_source_groups":
                return {"validation": {"valid": True}}
            raise AssertionError(label)
        return guarded

    monkeypatch.setattr(scorer_module, "_resolve_declared_path", guarded_resolve)
    monkeypatch.setattr(scorer_module, "authorize_policy_partition", authorize)
    monkeypatch.setattr(
        scorer_module, "validate_frozen_implementation", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(scorer_module, "validate_packet", packet_access("validate_packet"))
    monkeypatch.setattr(
        scorer_module, "load_domain_items", packet_access("load_domain_items"))
    monkeypatch.setattr(
        scorer_module,
        "load_partition_source_groups",
        packet_access("load_partition_source_groups"),
    )
    monkeypatch.setattr(scorer_module, "make_judge_backend", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        scorer_module,
        "score_domain",
        lambda **kwargs: {"domain": kwargs["domain"], "status": "complete"},
    )

    with pytest.raises(ValueError, match="validation release denied"):
        _run_sharded_fixture(
            paths,
            phase="lockbox",
            domains=("humor",),
            lockbox_release_artifact=str(release_path),
        )
    assert events == ["authorize"]

    events.clear()
    release_path.write_text("released")
    result = _run_sharded_fixture(
        paths,
        phase="lockbox",
        domains=("humor",),
        lockbox_release_artifact=str(release_path),
    )

    assert result["results"] == [{"domain": "humor", "status": "complete"}]
    assert events[0] == "authorize"
    assert events.index("authorize") < events.index("validate_packet")
    assert events.index("validate_packet") < events.index("load_domain_items")
    assert events.index("load_domain_items") < events.index(
        "load_partition_source_groups")


def test_malformed_validation_selection_fails_before_backend_construction(
        tmp_path, monkeypatch):
    _path, selection, bank = _breadth_selection_fixture(tmp_path, n_candidates=1)
    cell = bank["cells"][0]
    cell.update({
        "domain": "humor", "task": "humor", "construct": "funny",
    })
    bank["cells"].extend({
        "id": f"{domain}-cell", "domain": domain, "task": task,
        "construct": f"{domain} construct", "arms": [],
    } for domain, task in _SHARD_DOMAIN_TASKS.items() if domain != "humor")
    # This passes superficial hashes/panel binding but fails deep matched-control validation.
    bank["cells"][0]["arms"][2]["added_content_word_count"] += 1
    paths = _write_sharded_scoring_fixture(
        tmp_path,
        bank=bank,
        phase="validation",
        partition="tacit_breadth_validation",
        selection=selection,
    )
    observed = {}
    _patch_sharded_scoring(monkeypatch, observed)

    with pytest.raises(ValueError, match="added-content-word-count matched"):
        _run_sharded_fixture(paths, phase="validation", domains=("humor",))

    assert observed.get("loaded_domains", []) == []
    assert observed.get("backend_calls", 0) == 0


def test_lockbox_selection_requires_controls_and_hashes(tmp_path):
    path = tmp_path / "selection.json"
    selection = {
        "schema": "fresh_name_arm_selection/v1", "arm_bank_sha256": "a",
        "packet_manifest_sha256": "p", "cells": [{
            "cell_id": "N_humor_49", "selected_arm_id": "source_full_rubric",
            "matched_control_ids": ["control_wrong_full_rubric",
                                    "control_inert_full_rubric"],
            "cuf_status": "not-yet-certified",
        }],
    }
    path.write_text(json.dumps(selection))
    allowed = load_lockbox_selection(path, arm_bank_sha256="a", packet_manifest_sha256="p")
    assert allowed["N_humor_49"] == {"name", "source_full_rubric",
                                      "control_wrong_full_rubric",
                                      "control_inert_full_rubric"}
    selection["cells"][0]["matched_control_ids"] = []
    path.write_text(json.dumps(selection))
    with pytest.raises(ValueError, match="matched controls"):
        load_lockbox_selection(path, arm_bank_sha256="a", packet_manifest_sha256="p")


def test_policy_isomorphism_lockbox_selection_is_exact_and_frozen(tmp_path):
    path = tmp_path / "selection.json"
    selection = {
        "schema": "policy_isomorphism_lockbox_selection/v1",
        "status": "frozen-before-residual-lockbox-target-or-executor-scoring",
        "arm_bank_sha256": "a", "packet_manifest_sha256": "p",
        "lockbox_partition": "residual_lockbox",
        "cells": [{"cell_id": "N_humor_49",
                   "allowed_arm_ids": ["name", "incumbent_source", "confirm_rule"],
                   "control_ids": ["name", "incumbent_source"]}],
    }
    path.write_text(json.dumps(selection))
    allowed = load_lockbox_selection(path, arm_bank_sha256="a", packet_manifest_sha256="p")
    assert allowed["N_humor_49"] == {"name", "incumbent_source", "confirm_rule"}
    selection["status"] = "draft"
    path.write_text(json.dumps(selection))
    with pytest.raises(ValueError, match="not frozen"):
        load_lockbox_selection(path, arm_bank_sha256="a", packet_manifest_sha256="p")


def test_v2_lockbox_selection_binds_partition_and_both_control_types(tmp_path):
    path = tmp_path / "selection.json"
    selection = {
        "schema": "policy_isomorphism_lockbox_selection/v2",
        "status": "frozen-before-declared-lockbox-target-or-executor-scoring",
        "arm_bank_sha256": "a",
        "packet_manifest_sha256": "p",
        "lockbox_partition": "same_version_upper_lockbox",
        "cells": [{
            "cell_id": "N_humor_49",
            "allowed_arm_ids": [
                "name", "source_definition", "control_wrong_definition",
                "control_inert_definition",
            ],
            "candidate_arm_ids": ["source_definition"],
            "control_ids": ["control_wrong_definition", "control_inert_definition"],
            "required_control_provenances": [
                "wrong_construct_control", "inert_length_control",
            ],
        }],
    }
    path.write_text(json.dumps(selection))

    allowed = load_lockbox_selection(
        path,
        arm_bank_sha256="a",
        packet_manifest_sha256="p",
        expected_partition="same_version_upper_lockbox",
    )

    assert allowed["N_humor_49"] == set(selection["cells"][0]["allowed_arm_ids"])
    with pytest.raises(ValueError, match="partition mismatch"):
        load_lockbox_selection(
            path,
            arm_bank_sha256="a",
            packet_manifest_sha256="p",
            expected_partition="another_lockbox",
        )
    selection["cells"][0]["required_control_provenances"] = ["inert_length_control"]
    path.write_text(json.dumps(selection))
    with pytest.raises(ValueError, match="required controls/provenances"):
        load_lockbox_selection(
            path,
            arm_bank_sha256="a",
            packet_manifest_sha256="p",
            expected_partition="same_version_upper_lockbox",
        )


def test_selection_scope_preserves_h49_calibration_lockbox_binding():
    root = Path(__file__).parents[3]
    manifest = json.loads((root / (
        "methods/codability/experiments/same_version_upper_execution_manifest_v1.json"
    )).read_text())
    selection_path = root / (
        "methods/codability/experiments/same_version_upper_selection_v1.json")

    assert selection_scope_for_manifest(
        manifest, phase="calibration", selection_path=selection_path
    ) == ("same_version_upper_lockbox", None)


def _breadth_selection_fixture(tmp_path: Path, *, n_candidates: int = 3):
    form_ids = ("canonical", "question", "boilerplate")

    def forms(prefix: str):
        return [
            {
                "id": form_id,
                "prompt_sha256": f"{prefix}-{form_id}",
                "total_word_count": 20 + index,
            }
            for index, form_id in enumerate(form_ids)
        ]

    arms = [{
        "id": "name", "channel": "sparse", "provenance": "construct_name",
        "control_for": None, "semantic_content_word_count": 2,
        "added_content_word_count": 0, "forms": forms("name"),
    }]
    candidate_ids = []
    control_ids = []
    for index in range(n_candidates):
        candidate_id = f"source_route_{index}"
        candidate_ids.append(candidate_id)
        arms.append({
            "id": candidate_id, "channel": "declarative",
            "provenance": "source_hierarchy_definition", "control_for": None,
            "semantic_content_word_count": 12 + index,
            "added_content_word_count": 10 + index,
            "forms": forms(candidate_id),
        })
        for provenance, label in (
                ("wrong_construct_control", "wrong"),
                ("inert_length_control", "inert")):
            control_id = f"control_{label}_route_{index}"
            control_ids.append(control_id)
            arms.append({
                "id": control_id, "channel": "declarative",
                "provenance": provenance, "control_for": candidate_id,
                "semantic_content_word_count": 12 + index,
                "added_content_word_count": 10 + index,
                "forms": forms(control_id),
            })
    bank = {"cells": [{"id": "breadth-cell", "arms": arms}]}
    selection = {
        "schema": "policy_articulation_selection/v1",
        "status": "frozen-after-search-before-validation-scoring",
        "selected_phase": "validation",
        "selected_partition": "tacit_breadth_validation",
        "arm_bank_sha256": "bank-sha",
        "packet_manifest_sha256": "packet-sha",
        "cells": [{
            "cell_id": "breadth-cell",
            "allowed_arm_ids": ["name", *candidate_ids, *control_ids],
            "candidate_arm_ids": candidate_ids,
            "control_ids": control_ids,
            "required_control_provenances": [
                "wrong_construct_control", "inert_length_control",
            ],
        }],
    }
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(selection))
    return path, selection, bank


def test_generic_articulation_selection_accepts_any_positive_candidate_count(tmp_path):
    path, selection, bank = _breadth_selection_fixture(tmp_path, n_candidates=3)

    allowed = load_lockbox_selection(
        path,
        arm_bank_sha256="bank-sha",
        packet_manifest_sha256="packet-sha",
        expected_phase="validation",
        expected_partition="tacit_breadth_validation",
        arm_bank=bank,
    )

    assert allowed["breadth-cell"] == set(
        selection["cells"][0]["allowed_arm_ids"])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("added_count", "added-content-word-count matched"),
        ("total_count", "prompt-word-count matched"),
        ("form", "changes the form orbit"),
        ("missing_control", "exact matched controls/provenances"),
    ],
)
def test_generic_articulation_selection_rejects_inexact_controls(
        tmp_path, mutation, message):
    path, selection, bank = _breadth_selection_fixture(tmp_path, n_candidates=2)
    control = bank["cells"][0]["arms"][2]
    if mutation == "added_count":
        control["added_content_word_count"] += 1
    elif mutation == "total_count":
        control["forms"][0]["total_word_count"] += 1
    elif mutation == "form":
        control["forms"][0]["id"] = "changed"
    else:
        selection["cells"][0]["control_ids"].pop()
        selection["cells"][0]["allowed_arm_ids"].pop()
        path.write_text(json.dumps(selection))

    with pytest.raises(ValueError, match=message):
        load_lockbox_selection(
            path,
            arm_bank_sha256="bank-sha",
            packet_manifest_sha256="packet-sha",
            expected_phase="validation",
            expected_partition="tacit_breadth_validation",
            arm_bank=bank,
        )


def test_generic_articulation_selection_binds_current_phase_and_partition(tmp_path):
    path, _selection, bank = _breadth_selection_fixture(tmp_path, n_candidates=1)

    with pytest.raises(ValueError, match="selected phase/partition mismatch"):
        load_lockbox_selection(
            path,
            arm_bank_sha256="bank-sha",
            packet_manifest_sha256="packet-sha",
            expected_phase="calibration",
            expected_partition="tacit_breadth_search",
            arm_bank=bank,
        )


def test_generic_articulation_selection_cannot_drop_hard_cells(tmp_path):
    path, _selection, bank = _breadth_selection_fixture(tmp_path, n_candidates=1)
    bank["cells"].append({"id": "unselected-hard-cell", "arms": []})

    with pytest.raises(ValueError, match="cell panel differs from the arm bank"):
        load_lockbox_selection(
            path,
            arm_bank_sha256="bank-sha",
            packet_manifest_sha256="packet-sha",
            expected_phase="validation",
            expected_partition="tacit_breadth_validation",
            arm_bank=bank,
        )


def test_same_version_upper_manifests_bind_unsupervised_two_arm_experiment():
    root = Path(__file__).parents[3]
    execution_path = root / (
        "methods/codability/experiments/same_version_upper_execution_manifest_v1.json")
    selection_path = root / (
        "methods/codability/experiments/same_version_upper_selection_v1.json")
    execution = json.loads(execution_path.read_text())
    selection = json.loads(selection_path.read_text())
    arm_bank = json.loads((root / execution["arm_bank_path"]).read_text())
    integrity_path = root / execution["partition_integrity_path"]
    integrity = json.loads(integrity_path.read_text())
    archive_root = root / (
        "notebooks/data/two_faces_20260702/same_version_upper_confirmation_v1/"
        "frozen_implementation_v1"
    )

    assert execution["anchor_policy"].startswith("unsupervised model-to-model")
    assert execution["binary_readout"] == "teacher_forced_declared_labels"
    assert execution["selection_artifact_sha256"] == sha256_file(selection_path)
    assert execution["packet_manifest_sha256"] == sha256_file(
        root / execution["packet_manifest_path"])
    assert execution["arm_bank_sha256"] == sha256_file(root / execution["arm_bank_path"])
    assert execution["protocol_manifest_sha256"] == sha256_file(
        root / execution["protocol_manifest_path"])
    assert execution["target_prompt_manifest_sha256"] == sha256_file(
        root / execution["target_prompt_manifest_path"])
    assert execution["partition_integrity_sha256"] == sha256_file(integrity_path)
    assert integrity["valid"] is True
    assert integrity["n_items"] == 1900
    assert set(integrity["validated_partitions"]) == {
        "same_version_upper_calibration", "same_version_upper_lockbox"}
    assert all(integrity["joint_checks"].values())
    archived = set()
    for section in ("scoring", "analysis"):
        for row in execution["implementation"][section]["files"]:
            current = root / row["path"]
            if current.is_file() and sha256_file(current) == row["sha256"]:
                continue
            archived_path = archive_root / row["path"]
            assert archived_path.is_file()
            assert sha256_file(archived_path) == row["sha256"]
            archived.add(row["path"])
    assert archived == {
        "methods/codability/experiments/score_fresh_name_arms.py",
        "methods/codability/experiments/score_fresh_target_views.py",
        "methods/codability/experiments/validate_fresh_item_partitions.py",
        "methods/metric_implementer/vllm_backend.py",
        # Drifted by the 2026-07-15 analysis-closure unification; frozen bytes archived.
        "methods/codability/experiments/policy_data.py",
        "methods/codability/experiments/run_policy_isomorphism.py",
    }
    # This sealed manifest predates the 2026-07-15 closure fix that added the three package
    # __init__.py files to the recorder.  Its lockbox is already opened and complete, so the
    # relationship is documented rather than re-frozen: current recorder = frozen list + inits.
    current_paths = [row["path"] for row in _analysis_implementation()["files"]]
    manifest_paths = [
        row["path"] for row in execution["implementation"]["analysis"]["files"]
    ]
    init_additions = {
        "methods/codability/__init__.py",
        "methods/codability/experiments/__init__.py",
        "methods/metric_implementer/__init__.py",
    }
    assert [p for p in current_paths if p not in init_additions] == manifest_paths
    assert set(current_paths) - set(manifest_paths) == init_additions
    assert execution["phases"]["lockbox"] == [selection["lockbox_partition"]]
    assert set(execution["selection_required_phases"]) == {"calibration", "lockbox"}
    jobs = {row["id"]: row for row in execution["model_jobs"]}
    assert jobs["llama31_8b_executor"]["nominal_parameters_b"] < jobs[
        "llama31_70b_name_target"]["nominal_parameters_b"]
    assert {tuple(row["provider_model"].split("/")[1].split("-")[1:3])
            for row in jobs.values()} == {("3.1", "8B"), ("3.1", "70B")}
    assert len({row["tokenizer_config_sha256"] for row in jobs.values()}) == 1
    assert max(row["tensor_parallel_size"] for row in jobs.values()) == 1
    assert execution["resource_policy"]["launch_status"].startswith(
        "authorized by the user")
    assert execution["execution_environment"]["runtime_environment_overrides"][
        "VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"
    assert execution["execution_environment"]["runtime_environment_overrides"][
        "CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert execution["teacher_forced_label_validation"][
        "both_labels_single_token"] is True
    assert execution["teacher_forced_label_validation"][
        "contextual_continuation_ids_match_isolated_ids"] is True
    assert execution["execution_environment"]["hostname"] == "gpuhost.stanford.edu"
    assert execution["execution_environment"][
        "production_backend_class"] == "OfflineVLLM"
    assert set(execution["execution_environment"][
        "runtime_environment_overrides"]) == {
            "VLLM_GPU_MEM_UTIL", "VLLM_BLOCK_SIZE", "VLLM_ENFORCE_EAGER",
            "VLLM_WORKER_MULTIPROC_METHOD", "CUDA_DEVICE_ORDER",
            "FLASHINFER_CUDA_ARCHS"}
    assert execution["lockbox_release"]["required"] is True
    assert execution["lockbox_release"]["artifact_path"].endswith(
        "calibration_release.json")
    runner = execution["analysis"]["runner"]
    assert runner["cell_ids"] == ["N_humor_49"]
    assert runner["small_job"] == "llama31_8b_executor"
    assert runner["big_job"] == "llama31_70b_name_target"
    assert runner["n_boot"] == 10000
    assert runner["seed"] == 1207
    assert runner["functional_rho_floor"] == 0.7
    assert runner["fiber_mutual_rho_floor"] == 0.9
    assert runner["fiber_min_rank_valid_fraction"] == 0.99
    assert selection["decision_parameters"][
        "mutual_quotient_vector_equivalence_margins"] == {
            "mae_tvd": 0.02,
            "binary_flip_rate": 0.02,
            "absolute_bias": 0.02,
        }
    assert "H_fiber^vec" in execution["analysis"]["strict_vector_fiber"]
    assert runner["include_controls"] is True
    assert runner["scale_comparator_use_target"] is True
    assert runner["source_group_inference"] is True
    assert runner["allow_fake_inputs"] is False
    cell = selection["cells"][0]
    assert len(cell["candidate_arm_ids"]) == 2
    assert len(cell["control_ids"]) == 4
    assert set(cell["required_control_provenances"]) == {
        "wrong_construct_control", "inert_length_control"}
    allowed = load_lockbox_selection(
        selection_path,
        arm_bank_sha256=execution["arm_bank_sha256"],
        packet_manifest_sha256=execution["packet_manifest_sha256"],
        expected_partition=selection["lockbox_partition"],
        arm_bank=arm_bank,
    )
    assert allowed[cell["cell_id"]] == set(cell["allowed_arm_ids"])
