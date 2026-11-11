"""Leakage and reproducibility tests for fresh confirmatory item partitions."""

import json
from copy import deepcopy
from pathlib import Path

from methods.codability.experiments.build_fresh_item_partitions import (
    BREADTH_FIRST_ALLOCATION_STRATEGY,
    LEGACY_ALLOCATION_STRATEGY,
    _allocate_item_hash_partitions_from_dataset,
    _allocate_item_hash_partitions_from_frame,
    allocate_partitions,
    load_manifest,
    load_projected_source_frame,
    load_prior_packet_exclusions,
    projected_source_columns,
    records_from_frame,
    reconstruct_legacy_exclusions,
    sha256_file,
    source_group,
    source_projection_grade,
    validate_protocol,
)

import pandas as pd
import pytest


def test_source_group_allocation_is_exact_disjoint_and_deterministic():
    frame = pd.DataFrame([
        {"id": i, "text": f"Question: q{i // 3}\n\nAnswer: a{i}", "judgement": i % 2}
        for i in range(60)
    ])
    records = records_from_frame("math", frame, text_column="text", id_column="id",
                                 label_column="judgement")
    specs = [{"id": "dev", "n": 11}, {"id": "lockbox", "n": 17}]

    first = allocate_partitions(records, specs, domain="math", salt="frozen",
                                excluded_hashes={records[0]["text_sha256"]},
                                allocation_strategy=BREADTH_FIRST_ALLOCATION_STRATEGY)
    second = allocate_partitions(records, specs, domain="math", salt="frozen",
                                 excluded_hashes={records[0]["text_sha256"]},
                                 allocation_strategy=BREADTH_FIRST_ALLOCATION_STRATEGY)

    assert {key: [row["text_sha256"] for row in value] for key, value in first.items()} == {
        key: [row["text_sha256"] for row in value] for key, value in second.items()}
    assert len(first["dev"]) == 11 and len(first["lockbox"]) == 17
    assert {row["source_group"] for row in first["dev"]}.isdisjoint(
        {row["source_group"] for row in first["lockbox"]})
    assert records[0]["text_sha256"] not in {
        row["text_sha256"] for rows in first.values() for row in rows}


def test_allocation_strategy_is_explicit_and_legacy_default_still_drains_groups():
    records = [
        {
            "text_sha256": f"h-{group}-{item}",
            "source_group": f"g-{group}",
            "source_split": None,
        }
        for group in range(6)
        for item in range(5)
    ]
    specs = [{"id": "search", "n": 4}]

    implicit_legacy = allocate_partitions(
        records, specs, domain="task", salt="frozen", excluded_hashes=set())
    explicit_legacy = allocate_partitions(
        records, specs, domain="task", salt="frozen", excluded_hashes=set(),
        allocation_strategy=LEGACY_ALLOCATION_STRATEGY)
    breadth = allocate_partitions(
        records, specs, domain="task", salt="frozen", excluded_hashes=set(),
        allocation_strategy=BREADTH_FIRST_ALLOCATION_STRATEGY)

    assert implicit_legacy == explicit_legacy
    assert len({row["source_group"] for row in implicit_legacy["search"]}) == 1
    assert len({row["source_group"] for row in breadth["search"]}) == 4


def test_breadth_first_round_robin_visits_each_touched_group_before_a_second():
    records = [
        {
            "text_sha256": f"h-{group}-{item}",
            "source_group": f"g-{group}",
            "source_split": None,
        }
        for group in range(3)
        for item in range(3)
    ]

    result = allocate_partitions(
        records, [{"id": "search", "n": 5}], domain="task", salt="frozen",
        excluded_hashes=set(),
        allocation_strategy=BREADTH_FIRST_ALLOCATION_STRATEGY)

    assert len({row["source_group"] for row in result["search"][:3]}) == 3
    assert len({row["source_group"] for row in result["search"]}) == 3


def test_breadth_first_lookahead_preserves_capacity_for_later_partition():
    records = [
        {
            "text_sha256": f"h-{group}-{item}",
            "source_group": f"g-{group}",
            "source_split": None,
        }
        for group, size in enumerate([1, 1, 1, 2, 2])
        for item in range(size)
    ]

    result = allocate_partitions(
        records,
        [{"id": "search", "n": 3}, {"id": "validation", "n": 4}],
        domain="task", salt="frozen", excluded_hashes=set(),
        allocation_strategy=BREADTH_FIRST_ALLOCATION_STRATEGY)

    assert len(result["search"]) == 3
    assert len(result["validation"]) == 4
    assert {row["source_group"] for row in result["search"]}.isdisjoint(
        {row["source_group"] for row in result["validation"]})
    assert len({row["source_group"] for row in result["search"]}) == 3
    assert len({row["source_group"] for row in result["validation"]}) == 2


def test_breadth_first_impossible_group_disjoint_request_fails_closed():
    records = [
        {
            "text_sha256": f"h-{item}",
            "source_group": "shared-group",
            "source_split": None,
        }
        for item in range(2)
    ]

    with pytest.raises(ValueError, match="source-group-disjoint future capacity"):
        allocate_partitions(
            records,
            [{"id": "search", "n": 1}, {"id": "validation", "n": 1}],
            domain="task", salt="frozen", excluded_hashes=set(),
            allocation_strategy=BREADTH_FIRST_ALLOCATION_STRATEGY)


def test_memory_bounded_item_hash_path_matches_generic_round_robin(tmp_path):
    frame = pd.DataFrame([
        {"id": index, "text": f"text-{index}", "split": None}
        for index in range(20)
    ] + [{"id": 20, "text": "text-0", "split": None}])
    specs = [{"id": "search", "n": 6}, {"id": "validation", "n": 7}]
    records = records_from_frame(
        "task", frame, text_column="text", id_column="id", label_column=None,
        source_group_strategy={"strategy": "item_hash"})
    excluded_hashes = {records[0]["text_sha256"]}

    generic = allocate_partitions(
        records, specs, domain="task", salt="frozen",
        excluded_hashes=excluded_hashes,
        allocation_strategy=BREADTH_FIRST_ALLOCATION_STRATEGY)
    compact, n_unique = _allocate_item_hash_partitions_from_frame(
        frame, specs, domain="task", text_column="text", id_column="id",
        salt="frozen", excluded_hashes=excluded_hashes, excluded_groups=set())
    path = tmp_path / "items.csv.gz"
    frame.assign(outcome_label=1).to_csv(path, index=False, compression="gzip")
    streamed, n_rows, streamed_unique = _allocate_item_hash_partitions_from_dataset(
        path, specs, domain="task", text_column="text", id_column="id",
        salt="frozen", excluded_hashes=excluded_hashes, excluded_groups=set())

    assert n_unique == 20
    assert n_rows == 21
    assert streamed_unique == 20
    assert {
        key: [row["text_sha256"] for row in rows] for key, rows in compact.items()
    } == {
        key: [row["text_sha256"] for row in rows] for key, rows in generic.items()
    }
    assert {
        key: [row["text_sha256"] for row in rows] for key, rows in streamed.items()
    } == {
        key: [row["text_sha256"] for row in rows] for key, rows in generic.items()
    }


def test_projected_source_loader_physically_excludes_outcome_column(tmp_path):
    path = tmp_path / "source.csv.gz"
    pd.DataFrame([
        {"id": 1, "text": "one", "paper_id": "p1", "outcome": 1},
        {"id": 2, "text": "two", "paper_id": "p2", "outcome": 0},
    ]).to_csv(path, index=False, compression="gzip")
    columns = projected_source_columns(
        text_column="text", id_column="id",
        source_group_strategy={
            "strategy": "columns", "columns": ["paper_id"], "tag": "paper"},
        partition_specs=[{"id": "search", "n": 1}], domain="peer-review")

    projected = load_projected_source_frame(path, columns=columns)

    assert columns == ["text", "id", "paper_id"]
    assert set(projected.columns) == set(columns)
    assert "outcome" not in projected.columns
    assert source_projection_grade(path) == "parser_column_projection"
    assert source_projection_grade(tmp_path / "source.jsonl") == (
        "post_decode_key_projection"
    )


def test_allocation_excludes_prior_source_groups_even_when_item_hash_is_new():
    frame = pd.DataFrame([
        {"id": i, "text": f"Question: q{i // 2}\n\nAnswer: a{i}", "judgement": i % 2}
        for i in range(20)
    ])
    records = records_from_frame(
        "math", frame, text_column="text", id_column="id", label_column="judgement")
    excluded_group = records[0]["source_group"]

    result = allocate_partitions(
        records,
        [{"id": "final", "n": 4}],
        domain="math",
        salt="new-final",
        excluded_hashes=set(),
        excluded_groups={excluded_group},
    )

    assert all(row["source_group"] != excluded_group for row in result["final"])


def test_prior_packet_exclusions_authenticate_and_retain_only_identities(tmp_path):
    items = tmp_path / "prior.jsonl"
    rows = [
        {"text": "sealed text one", "text_sha256": "hash-1", "source_group": "group-1"},
        {"text": "sealed text two", "text_sha256": "hash-2", "source_group": "group-2"},
    ]
    items.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = tmp_path / "packet.json"
    manifest.write_text(json.dumps({
        "schema": "fresh_item_partitions/v1",
        "domains": [{
            "domain": "humor",
            "partitions": [{
                "id": "old_lockbox",
                "n": 2,
                "items_path": str(items),
                "items_sha256": sha256_file(items),
            }],
        }],
    }))
    protocol = {"exclude_packet_manifests": [{
        "path": str(manifest), "sha256": sha256_file(manifest),
    }]}

    result = load_prior_packet_exclusions(protocol, domain="humor")

    assert result["hashes"] == {"hash-1", "hash-2"}
    assert result["groups"] == {"group-1", "group-2"}
    assert "text" not in result
    assert result["packets"][0]["n_new_item_hashes"] == 2


def test_prior_packet_exclusion_can_be_scoped_to_declared_domains(tmp_path):
    protocol = {"exclude_packet_manifests": [{
        "path": str(tmp_path / "does-not-exist.json"),
        "sha256": "unused",
        "domains": ["humor"],
    }]}

    result = load_prior_packet_exclusions(protocol, domain="peer-review")

    assert result == {"hashes": set(), "groups": set(), "packets": []}


def test_prior_packet_domain_alias_is_authenticated(tmp_path):
    items = tmp_path / "prior.jsonl"
    items.write_text(json.dumps({
        "text_sha256": "hash-1", "source_group": "group-1"}) + "\n")
    manifest = tmp_path / "packet.json"
    manifest.write_text(json.dumps({
        "domains": [{"domain": "cw", "partitions": [{
            "id": "old", "n": 1, "items_path": str(items),
            "items_sha256": sha256_file(items),
        }]}],
    }))
    protocol = {"exclude_packet_manifests": [{
        "path": str(manifest), "sha256": sha256_file(manifest),
        "domains": ["creative-writing"],
        "domain_aliases": {"creative-writing": "cw"},
    }]}

    result = load_prior_packet_exclusions(protocol, domain="creative-writing")

    assert result["hashes"] == {"hash-1"}


def test_press_release_native_split_is_respected():
    frame = pd.DataFrame([
        {"id": i, "text": f"release {i}", "group": f"g{i}",
         "split": "train" if i < 10 else "test", "judgement": i % 2}
        for i in range(20)
    ])
    records = records_from_frame("pr", frame, text_column="text", id_column="id",
                                 label_column="judgement")
    specs = [
        {"id": "dev", "n": 5, "source_split": {"pr": ["train"]}},
        {"id": "lockbox", "n": 5, "source_split": {"pr": ["test"]}},
    ]

    result = allocate_partitions(records, specs, domain="pr", salt="frozen",
                                 excluded_hashes=set())

    assert {row["source_split"] for row in result["dev"]} == {"train"}
    assert {row["source_split"] for row in result["lockbox"]} == {"test"}


def test_source_group_strategy_is_manifest_driven_for_arbitrary_tasks():
    content_hash = "content"
    assert source_group(
        "peer-review", "paper", {"paper_id": "p1"}, content_hash,
        strategy={"strategy": "columns", "columns": ["paper_id"], "tag": "paper"},
    ).startswith("paper:")
    assert source_group(
        "math-stackexchange", "Q\n\nAnswer:A", {}, content_hash,
        strategy={"strategy": "text_prefix", "separator": "\n\nAnswer:",
                  "tag": "question"},
    ).startswith("question:")
    assert source_group(
        "humor", "joke", {}, content_hash, strategy={"strategy": "item_hash"},
    ) == "item:content"


def test_partition_sizes_can_be_frozen_per_domain():
    records = [
        {"text_sha256": f"h{i}", "source_group": f"g{i}", "source_split": None}
        for i in range(20)
    ]
    specs = [{"id": "calibration", "n": 10, "n_by_domain": {"grant": 7}}]

    result = allocate_partitions(
        records, specs, domain="grant", salt="frozen", excluded_hashes=set())

    assert len(result["calibration"]) == 7


def test_records_can_omit_practice_labels_entirely():
    frame = pd.DataFrame([{"id": 1, "text": "item", "judgement": 1}])
    records = records_from_frame(
        "humor", frame, text_column="text", id_column="id", label_column=None,
        source_group_strategy={"strategy": "item_hash"})

    assert records[0]["practice_target"] is None


def test_calibration_protocol_can_explicitly_disable_legacy_window_exclusion():
    assert reconstruct_legacy_exclusions(
        object(), {"legacy_exclusion": {"enabled": False}}) == set()


def test_tacit_breadth_protocol_is_complete_label_free_and_statically_valid():
    path = (Path(__file__).parents[1] / "experiments"
            / "tacit_breadth_item_partition_manifest_v1.json")
    protocol = load_manifest(path)

    assert validate_protocol(protocol) == []
    assert protocol["emit_practice_targets"] is False
    assert len(protocol["domains"]) == 11
    assert {row["id"] for row in protocol["partitions"]} == {
        "tacit_breadth_search", "tacit_breadth_validation"}
    for partition in protocol["partitions"]:
        assert set(partition["domains"]) == set(protocol["domains"])
        assert partition["n"] == 200
        assert partition["n_by_domain"] == {"grant-funding": 100}


def test_tacit_breadth_protocol_rejects_practice_target_emission():
    path = (Path(__file__).parents[1] / "experiments"
            / "tacit_breadth_item_partition_manifest_v1.json")
    protocol = deepcopy(load_manifest(path))
    protocol["emit_practice_targets"] = True

    assert "tacit breadth protocol must explicitly disable practice targets" in (
        validate_protocol(protocol))


def test_tacit_breadth_v2_declares_group_round_robin_and_400_item_folds():
    path = (Path(__file__).parents[1] / "experiments"
            / "tacit_breadth_item_partition_manifest_v2.json")
    protocol = load_manifest(path)

    assert validate_protocol(protocol) == []
    assert protocol["allocation_strategy"] == BREADTH_FIRST_ALLOCATION_STRATEGY
    assert protocol["emit_practice_targets"] is False
    assert protocol["status"] == "frozen-before-tacit-breadth-model-scoring"
    assert {
        row["label"] for row in protocol["domains"].values()
    } == {"no dataset label is retained, emitted, selected on, or used"}
    for partition in protocol["partitions"]:
        assert partition["n"] == 400
        assert partition["n_by_domain"] == {"grant-funding": 100}
