from __future__ import annotations

import pytest
import json

from methods.codability.lexicon_distill.train_gemma4_similarity_lora import (
    collate,
    label_token_ids,
    length_bucketed_indices,
    nonfinite_window_limit,
    promote_trainable_parameters_to_fp32,
    read_rows,
)


class FakeTokenizer:
    def __call__(self, text: str, add_special_tokens: bool = False):
        # Treat a final digit as one dedicated next token.
        if text and text[-1] in "012":
            return {"input_ids": [ord(char) for char in text[:-1]] + [1000 + int(text[-1])]}
        return {"input_ids": [ord(char) for char in text]}


def test_label_tokens_are_unique_stable_next_tokens() -> None:
    assert label_token_ids(FakeTokenizer(), "prompt:") == (1000, 1001, 1002)


class BadBoundaryTokenizer(FakeTokenizer):
    def __call__(self, text: str, add_special_tokens: bool = False):
        if text and text[-1] in "012":
            return {"input_ids": [999]}
        return super().__call__(text, add_special_tokens=add_special_tokens)


def test_label_boundary_drift_fails_closed() -> None:
    with pytest.raises(ValueError):
        label_token_ids(BadBoundaryTokenizer(), "prompt:")


def test_auxiliary_scope_uses_auxiliary_distribution(tmp_path) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text(
        json.dumps(
            {
                "level": "R1", "task": "humor", "split": "train",
                "target_probs": [0.0, 0.0, 1.0], "example_weight": 1.0,
                "family_distributions": {"sonnet": [0.0, 0.0, 1.0], "opus": [1.0, 0.0, 0.0]},
            }
        ) + "\n",
        encoding="utf-8",
    )
    rows = read_rows(path, level="R1", task=None, primary_only=False, auxiliary_only=True)
    assert rows[0]["target_probs"] == [1.0, 0.0, 0.0]
    assert rows[0]["example_weight"] == 0.25


def test_collate_left_pads_so_last_position_is_always_prompt_token() -> None:
    batch = collate(
        [
            {"input_ids": [4, 5], "target_probs": [1, 0, 0], "weight": 1},
            {"input_ids": [6], "target_probs": [0, 1, 0], "weight": 1},
        ],
        pad_token_id=0,
    )
    assert batch["input_ids"].tolist() == [[4, 5], [0, 6]]
    assert batch["attention_mask"].tolist() == [[1, 1], [0, 1]]
    assert batch["position_ids"].tolist() == [[0, 1], [0, 0]]


def test_length_bucketed_order_is_complete_deterministic_and_local() -> None:
    torch = pytest.importorskip("torch")
    encoded = [{"input_ids": list(range(length))} for length in range(1, 33)]
    first_generator = torch.Generator(device="cpu").manual_seed(17)
    second_generator = torch.Generator(device="cpu").manual_seed(17)

    first = length_bucketed_indices(
        encoded, batch_size=4, bucket_batches=8, generator=first_generator,
    )
    second = length_bucketed_indices(
        encoded, batch_size=4, bucket_batches=8, generator=second_generator,
    )

    assert first == second
    assert sorted(first) == list(range(len(encoded)))
    for start in range(0, len(first), 4):
        lengths = [len(encoded[index]["input_ids"]) for index in first[start : start + 4]]
        assert max(lengths) - min(lengths) <= 3


def test_nonfinite_window_ceiling_scales_with_fit_size() -> None:
    assert nonfinite_window_limit(9, 0.01) == 1
    assert nonfinite_window_limit(488, 0.01) == 5
    assert nonfinite_window_limit(1092, 0.01) == 11
    assert nonfinite_window_limit(13008, 0.01) == 131


def test_only_trainable_parameters_are_promoted_to_fp32() -> None:
    torch = pytest.importorskip("torch")
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)).to(torch.bfloat16)
    for parameter in model[0].parameters():
        parameter.requires_grad = False

    promoted = promote_trainable_parameters_to_fp32(model)

    assert promoted == 2
    assert {parameter.dtype for parameter in model[0].parameters()} == {torch.bfloat16}
    assert {parameter.dtype for parameter in model[1].parameters()} == {torch.float32}
