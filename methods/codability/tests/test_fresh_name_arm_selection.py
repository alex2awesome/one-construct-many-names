"""Frozen development-arm selection rule."""

import pytest

from methods.codability.experiments.fresh_name_arm_selection import choose_candidate, select


def test_selection_uses_lower_bound_then_shorter_tie():
    rows = [
        {"arm_id": "long", "selection_eligible": True, "score_CI": [0.50, 0.7],
         "semantic_content_word_count": 100},
        {"arm_id": "short", "selection_eligible": True, "score_CI": [0.495, 0.7],
         "semantic_content_word_count": 20},
        {"arm_id": "ineligible", "selection_eligible": False, "score_CI": [0.9, 1.0],
         "semantic_content_word_count": 1},
    ]
    decision = choose_candidate(rows, tie_width=0.01)
    assert decision["chosen"]["arm_id"] == "short"
    assert decision["had_eligible_candidate"]


def test_selector_rejects_every_nonpublic_partition_before_reading_shards():
    with pytest.raises(ValueError, match="does not authorize partition"):
        select(
            target_shard_root="missing",
            executor_shard_root="missing",
            arm_bank_path="missing",
            packet_manifest_path="missing",
            partition="residual_lockbox",
        )
