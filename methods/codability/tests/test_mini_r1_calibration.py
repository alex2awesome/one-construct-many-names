import json

import pytest

from methods.codability.lexicon.mini_r1_calibration import (
    _parse_batch,
    _round_robin_sample,
)


def test_parse_batch_requires_exact_ids_but_not_model_order():
    raw = json.dumps(
        {
            "decisions": [
                {"calibration_id": "b", "reasoning": "second", "score": 2},
                {"calibration_id": "a", "reasoning": "first", "score": 1},
            ]
        }
    )
    rows = _parse_batch(raw, ["a", "b"])
    assert [row["calibration_id"] for row in rows] == ["a", "b"]


@pytest.mark.parametrize(
    "decisions",
    [
        [{"calibration_id": "a", "reasoning": "x", "score": 2}],
        [
            {"calibration_id": "a", "reasoning": "x", "score": 2},
            {"calibration_id": "a", "reasoning": "y", "score": 1},
        ],
        [
            {"calibration_id": "a", "reasoning": "x", "score": True},
            {"calibration_id": "b", "reasoning": "y", "score": 1},
        ],
    ],
)
def test_parse_batch_fails_closed(decisions):
    with pytest.raises(ValueError):
        _parse_batch(json.dumps({"decisions": decisions}), ["a", "b"])


def test_round_robin_sample_spreads_tasks_deterministically():
    pools = {
        "a": [{"pair_id": f"a{i}"} for i in range(4)],
        "b": [{"pair_id": f"b{i}"} for i in range(4)],
        "c": [{"pair_id": f"c{i}"} for i in range(4)],
    }
    first = _round_robin_sample(pools, n=8, seed=17, score=2)
    second = _round_robin_sample(pools, n=8, seed=17, score=2)
    assert first == second
    counts = {prefix: sum(row["pair_id"].startswith(prefix) for row in first) for prefix in pools}
    assert sorted(counts.values()) == [2, 3, 3]
