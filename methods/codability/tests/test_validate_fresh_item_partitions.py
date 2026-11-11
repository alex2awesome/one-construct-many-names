"""Small helper tests for fresh-partition integrity validation."""

import json
import math
from types import SimpleNamespace

import pytest

from methods.codability.experiments import validate_fresh_item_partitions as validator
from methods.codability.experiments.validate_fresh_item_partitions import (
    _target_valid,
    validate_packet,
)


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _packet_fixture(tmp_path, monkeypatch, *, n, n_by_domain=None,
                    emit_practice_targets=False):
    domain = "grant-funding"
    partition_id = "search"
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text("authenticated source corpus")
    entry = SimpleNamespace(task=domain, path=str(dataset_path))
    monkeypatch.setattr(
        validator, "full_manifest",
        lambda: SimpleNamespace(datasets=[entry]),
    )
    monkeypatch.setattr(
        validator, "reconstruct_legacy_exclusions", lambda _entry, _protocol: set())
    monkeypatch.setattr(
        validator, "load_prior_packet_exclusions",
        lambda _protocol, *, domain: {"hashes": set(), "groups": set()},
    )

    partition_spec = {
        "id": partition_id, "n": n, "domains": [domain],
    }
    if n_by_domain is not None:
        partition_spec["n_by_domain"] = {domain: n_by_domain}
    protocol = {
        "domains": {domain: {"task": domain}},
        "partitions": [partition_spec],
    }
    if emit_practice_targets is not None:
        protocol["emit_practice_targets"] = emit_practice_targets
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol))

    effective_n = n_by_domain if n_by_domain is not None else n
    items = []
    targets = []
    for index in range(effective_n):
        text = f"grant application {index}"
        item_hash = validator.text_sha256(text)
        items.append({
            "item_id": str(index), "text": text, "text_sha256": item_hash,
            "source_group": f"applicant:{index}", "source_split": None,
        })
        targets.append({"text_sha256": item_hash, "practice_target": index % 2})
    items_path = tmp_path / "items.jsonl"
    _write_jsonl(items_path, items)
    effective_emit_targets = (
        True if emit_practice_targets is None else emit_practice_targets
    )
    targets_path = tmp_path / "targets.jsonl"
    if effective_emit_targets:
        _write_jsonl(targets_path, targets)

    empty_hash = validator.sha256_bytes(b"")
    partition = {
        "id": partition_id,
        "n": effective_n,
        "items_path": str(items_path),
        "items_sha256": validator.sha256_file(items_path),
        "targets_path": str(targets_path) if effective_emit_targets else None,
        "targets_sha256": (
            validator.sha256_file(targets_path) if effective_emit_targets else None
        ),
        "ordered_item_set_sha256": validator.sha256_bytes(
            "\n".join(row["text_sha256"] for row in items).encode()),
    }
    packet = {
        "protocol_manifest_sha256": validator.sha256_file(protocol_path),
        "domains": [{
            "domain": domain,
            "dataset_sha256": validator.sha256_file(dataset_path),
            "legacy_exclusion_set_sha256": empty_hash,
            "prior_packet_exclusion_count": 0,
            "prior_packet_source_group_exclusion_count": 0,
            "prior_packet_exclusion_set_sha256": empty_hash,
            "prior_packet_source_group_set_sha256": empty_hash,
            "partitions": [partition],
        }],
    }
    packet_path = tmp_path / "packet.json"
    packet_path.write_text(json.dumps(packet))
    return packet_path, protocol_path, packet


def test_practice_target_validation():
    assert _target_valid(0) and _target_valid(1.0) and _target_valid(True)
    assert not _target_valid(None)
    assert not _target_valid(float("nan"))
    assert not _target_valid(math.inf)
    assert not _target_valid("")


def test_validator_honors_per_domain_size_and_certifies_partition(tmp_path, monkeypatch):
    packet_path, protocol_path, _ = _packet_fixture(
        tmp_path, monkeypatch, n=200, n_by_domain=100)

    report = validate_packet(
        packet_path, protocol_path=protocol_path,
        domains={"grant-funding"}, partitions={"search"},
    )

    assert report["valid"], report["errors"]
    assert report["n_items"] == 100
    assert report["validated_partitions"] == ["search"]
    assert report["domains"][0]["n_items"] == 100


def test_structural_validation_can_use_an_external_authenticated_source_certificate(
        tmp_path, monkeypatch):
    packet_path, protocol_path, packet = _packet_fixture(
        tmp_path, monkeypatch, n=3)
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.unlink()

    strict = validate_packet(
        packet_path, protocol_path=protocol_path,
        domains={"grant-funding"}, partitions={"search"},
    )
    structural = validate_packet(
        packet_path, protocol_path=protocol_path,
        domains={"grant-funding"}, partitions={"search"},
        verify_dataset_files=False,
    )

    assert not strict["valid"]
    assert "dataset file is missing" in strict["errors"]
    assert structural["valid"], structural["errors"]
    assert structural["dataset_files_verified"] is False
    assert structural["validated_partitions"] == ["search"]
    assert packet["domains"][0]["dataset_sha256"]
    with pytest.raises(ValueError, match="cannot be recomputed"):
        validate_packet(
            packet_path, protocol_path=protocol_path,
            verify_dataset_files=False, verify_source_membership=True,
        )


def test_validator_rejects_requested_packet_members_that_are_missing(
        tmp_path, monkeypatch):
    packet_path, protocol_path, packet = _packet_fixture(
        tmp_path, monkeypatch, n=3)
    packet["domains"][0]["partitions"] = []
    packet_path.write_text(json.dumps(packet))

    report = validate_packet(
        packet_path, protocol_path=protocol_path,
        domains={"grant-funding"}, partitions={"search"},
    )

    assert not report["valid"]
    assert report["validated_partitions"] == []
    assert "requested partition 'search' is absent from packet" in report["errors"]
    assert any(
        "grant-funding/search: requested partition is absent from packet" == error
        for error in report["errors"]
    )


def test_validator_rejects_requests_absent_from_both_protocol_and_packet(
        tmp_path, monkeypatch):
    packet_path, protocol_path, _ = _packet_fixture(
        tmp_path, monkeypatch, n=3)

    report = validate_packet(
        packet_path, protocol_path=protocol_path,
        domains={"missing-domain"}, partitions={"missing-partition"},
    )

    assert not report["valid"]
    assert report["validated_partitions"] == []
    assert "requested domain 'missing-domain' is absent from protocol" in report["errors"]
    assert "requested domain 'missing-domain' is absent from packet" in report["errors"]
    assert "requested partition 'missing-partition' is absent from protocol" in (
        report["errors"])
    assert "requested partition 'missing-partition' is absent from packet" in (
        report["errors"])


def test_legacy_protocol_without_n_by_domain_or_target_flag_still_validates(
        tmp_path, monkeypatch):
    packet_path, protocol_path, _ = _packet_fixture(
        tmp_path, monkeypatch, n=3, emit_practice_targets=None)

    report = validate_packet(
        packet_path, protocol_path=protocol_path,
        domains={"grant-funding"}, partitions={"search"},
    )

    assert report["valid"], report["errors"]
    assert report["n_items"] == 3
    assert report["validated_partitions"] == ["search"]


def test_validator_binds_packet_to_protocol_allocation_strategy(tmp_path, monkeypatch):
    packet_path, protocol_path, packet = _packet_fixture(
        tmp_path, monkeypatch, n=3)
    protocol = json.loads(protocol_path.read_text())
    protocol["allocation_strategy"] = "breadth_first_group_round_robin_v2"
    protocol_path.write_text(json.dumps(protocol))
    packet["protocol_manifest_sha256"] = validator.sha256_file(protocol_path)
    packet_path.write_text(json.dumps(packet))

    report = validate_packet(
        packet_path, protocol_path=protocol_path,
        domains={"grant-funding"}, partitions={"search"},
    )

    assert not report["valid"]
    assert "allocation strategy differs from protocol" in report["errors"]
    assert "domain allocation strategy differs from protocol" in report["errors"]


def _source_membership_fixture(tmp_path, monkeypatch):
    domain = "peer-review"
    strategy = {
        "strategy": "columns",
        "columns": ["paper_id"],
        "tag": "paper",
        "required": True,
    }
    dataset_path = tmp_path / "peer.csv"
    dataset_path.write_text(
        "id,text,paper_id,outcome\n"
        "1,careful review one,paper-a,accept\n"
        "2,careful review two,paper-b,reject\n"
        "3,careful review three,paper-c,accept\n"
    )
    entry = SimpleNamespace(
        task=domain,
        path=dataset_path,
        text_column="text",
        id_column="id",
        label_column="outcome",
    )
    monkeypatch.setattr(
        validator,
        "full_manifest",
        lambda: SimpleNamespace(datasets=[entry]),
    )
    monkeypatch.setattr(
        validator, "reconstruct_legacy_exclusions", lambda _entry, _protocol: set()
    )
    monkeypatch.setattr(
        validator,
        "load_prior_packet_exclusions",
        lambda _protocol, *, domain: {"hashes": set(), "groups": set()},
    )
    allocation_strategy = "breadth_first_group_round_robin_v2"
    partition_spec = {"id": "search", "n": 2, "domains": [domain]}
    protocol = {
        "allocation_strategy": allocation_strategy,
        "emit_practice_targets": False,
        "domains": {
            domain: {"task": domain, "source_group": strategy},
        },
        "partitions": [partition_spec],
    }
    protocol_path = tmp_path / "protocol-source-membership.json"
    protocol_path.write_text(json.dumps(protocol))
    rows = []
    for item_id, text, paper_id in (
        ("1", "careful review one", "paper-a"),
        ("2", "careful review two", "paper-b"),
    ):
        item_hash = validator.text_sha256(text)
        rows.append(
            {
                "item_id": item_id,
                "text": text,
                "text_sha256": item_hash,
                "source_group": validator.source_group(
                    domain,
                    text,
                    {"paper_id": paper_id},
                    item_hash,
                    strategy=strategy,
                ),
                "source_split": None,
            }
        )
    items_path = tmp_path / "source-membership-items.jsonl"
    _write_jsonl(items_path, rows)
    empty_hash = validator.sha256_bytes(b"")
    packet = {
        "protocol_manifest_sha256": validator.sha256_file(protocol_path),
        "allocation_strategy": allocation_strategy,
        "domains": [
            {
                "domain": domain,
                "allocation_strategy": allocation_strategy,
                "dataset_sha256": validator.sha256_file(dataset_path),
                "legacy_exclusion_set_sha256": empty_hash,
                "prior_packet_exclusion_count": 0,
                "prior_packet_source_group_exclusion_count": 0,
                "prior_packet_exclusion_set_sha256": empty_hash,
                "prior_packet_source_group_set_sha256": empty_hash,
                "source_io_projection": {
                    "enabled": True,
                    "loaded_columns": ["text", "id", "paper_id"],
                    "projection_grade": "parser_column_projection",
                    "declared_outcome_column": "outcome",
                    "outcome_column_retained": False,
                },
                "partitions": [
                    {
                        "id": "search",
                        "n": 2,
                        "items_path": str(items_path),
                        "items_sha256": validator.sha256_file(items_path),
                        "targets_path": None,
                        "targets_sha256": None,
                        "ordered_item_set_sha256": validator.sha256_bytes(
                            "\n".join(row["text_sha256"] for row in rows).encode()
                        ),
                    }
                ],
            }
        ],
    }
    packet_path = tmp_path / "source-membership-packet.json"
    packet_path.write_text(json.dumps(packet))
    return packet_path, protocol_path, packet, items_path, rows


def test_validator_recomputes_source_membership_without_loading_outcomes(
    tmp_path, monkeypatch
):
    packet_path, protocol_path, _, _, _ = _source_membership_fixture(
        tmp_path, monkeypatch
    )

    report = validate_packet(
        packet_path,
        protocol_path=protocol_path,
        verify_source_membership=True,
    )

    assert report["valid"], report["errors"]
    assert report["source_membership_verified"] is True
    certificate = report["source_membership"][0]
    assert certificate["valid"] is True
    assert certificate["n_matched_items"] == 2
    assert certificate["projected_columns"] == ["text", "id", "paper_id"]
    assert certificate["projection_grade"] == "parser_column_projection"
    assert certificate["declared_outcome_column"] == "outcome"
    assert certificate["outcome_column_retained"] is False
    assert certificate["source_group_identity_recomputed"] is True
    assert certificate["allocation_replay_verified"] is False


def test_source_membership_recompute_rejects_forged_source_group(
    tmp_path, monkeypatch
):
    packet_path, protocol_path, packet, items_path, rows = _source_membership_fixture(
        tmp_path, monkeypatch
    )
    rows[0]["source_group"] = "paper:forged"
    _write_jsonl(items_path, rows)
    packet["domains"][0]["partitions"][0]["items_sha256"] = validator.sha256_file(
        items_path
    )
    packet_path.write_text(json.dumps(packet))

    report = validate_packet(
        packet_path,
        protocol_path=protocol_path,
        verify_source_membership=True,
    )

    assert not report["valid"]
    assert report["source_membership_verified"] is False
    certificate = report["source_membership"][0]
    assert certificate["valid"] is False
    assert certificate["errors"] == ["source_group: 1 row(s)"]
    assert "source membership/group identity verification failed" in report[
        "domains"
    ][0]["errors"]
