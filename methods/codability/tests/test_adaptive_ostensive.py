"""Cross-fitted adaptive ostension invariants."""

import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from methods.codability.experiments.compile_adaptive_ostensive_bank import (
    retrieve_assignments,
    similarity_matrices,
)
from methods.codability.experiments.score_adaptive_ostensive_orbits import (
    adaptive_content,
    score_declared_binary,
    score_prompt,
)


def test_similarity_is_training_vocabulary_retrieval_with_expected_neighbors():
    train = ["banana peel pun", "legal contract clause", "orange citrus joke"]
    test = ["banana wordplay", "contract wording"]
    observed = similarity_matrices(train, test)
    assert set(observed) == {"word", "char", "hybrid"}
    assert all(matrix.shape == (2, 3) for matrix in observed.values())
    assert int(np.argmax(observed["hybrid"][0])) == 0
    assert int(np.argmax(observed["hybrid"][1])) == 1


def test_retrieval_is_balanced_and_uses_directional_residual_priority():
    train_hashes = ["p_weak", "p_miss", "n_weak", "n_miss"]
    test_hashes = ["test"]
    similarity = np.full((1, 4), 0.3)
    target = np.asarray([0.6, 0.9, 0.4, 0.1])
    sparse = np.asarray([0.6, 0.1, 0.4, 0.9])
    observed = retrieve_assignments(
        train_hashes=train_hashes, test_hashes=test_hashes, similarity=similarity,
        target=target, sparse=sparse, pool="residual", k=1)
    assert observed["test"] == {"positive": ["p_miss"], "negative": ["n_miss"]}


def test_dynamic_content_uses_only_frozen_assignment_and_marks_policy_not_truth():
    evaluation = {
        "retrievals": [{"id": "hybrid_residual_k1", "pool": "residual",
                        "assignments": {"test": {
                            "positive": ["p"], "negative": ["n"]}}}],
        "teaching_examples": {
            "p": {"text": "positive example", "target_score": 0.85,
                  "small_name_score": 0.2},
            "n": {"text": "negative example", "target_score": 0.15,
                  "small_name_score": 0.8},
        },
        "parent_texts": {"self_contrastive": "explicit decision rule"},
    }
    arm = {"retrieval_id": "hybrid_residual_k1", "parent_id": "self_contrastive"}
    content = adaptive_content(evaluation, arm, "test")
    assert "explicit decision rule" in content
    assert "target-policy score 0.85; smaller name-only score 0.20" in content
    assert "not external ground truth" in content
    prompt = score_prompt(content, "new item", form="canonical")
    assert "NEW ITEM:\nnew item" in prompt
    assert prompt.endswith("Answer exactly YES or NO.")


def test_declared_binary_has_a_fake_backend_path_for_dry_runs():
    class FakeVLLM:
        def score_binary(self, prompts, pos, neg, seed):
            assert (pos, neg, seed) == ("YES", "NO", 7)
            return [0.25] * len(prompts)

    assert np.allclose(score_declared_binary(FakeVLLM(), ["one", "two"], seed=7),
                       [0.25, 0.25])


def test_declared_binary_fake_path_preserves_per_prompt_seeds_and_checks_shape():
    class FakeVLLM:
        def __init__(self):
            self.observed = None

        def score_binary(self, prompts, pos, neg, seed):
            self.observed = (list(prompts), pos, neg, list(seed))
            return [value / 100 for value in seed]

    backend = FakeVLLM()
    observed = score_declared_binary(
        backend, ["one", "two", "three"], seed=[11, 22, 33])

    assert np.array_equal(observed, [0.11, 0.22, 0.33])
    assert backend.observed == (
        ["one", "two", "three"], "YES", "NO", [11, 22, 33])
    with pytest.raises(ValueError, match="2 seeds for 3 prompts"):
        score_declared_binary(backend, ["one", "two", "three"], seed=[1, 2])

    class FakeVLLM:  # noqa: F811 - the exact class name selects the dry-run backend path.
        @staticmethod
        def score_binary(_prompts, **_kwargs):
            return [[0.5]]

    with pytest.raises(ValueError, match="invalid output shape"):
        score_declared_binary(FakeVLLM(), ["one"])


def test_declared_binary_expands_each_prompt_seed_across_yes_no_continuations(
        monkeypatch):
    class Tokenizer:
        @staticmethod
        def encode(label, add_special_tokens=False):
            assert add_special_tokens is False
            return {"YES": [101], "NO": [202]}[label]

        @staticmethod
        def apply_chat_template(messages, **_kwargs):
            return f"CHAT:{messages[0]['content']}::"

    class Engine:
        def __init__(self):
            self.texts = None
            self.params = None

        @staticmethod
        def get_tokenizer():
            return Tokenizer()

        def generate(self, texts, params):
            self.texts = list(texts)
            self.params = params
            return [
                SimpleNamespace(
                    prompt_token_ids=[token_id],
                    prompt_logprobs=[{
                        token_id: SimpleNamespace(logprob=logprob),
                    }],
                )
                for token_id, logprob in (
                    (101, -0.2), (202, -1.2),
                    (101, -1.4), (202, -0.4),
                )
            ]

    engine = Engine()

    class Backend:
        model = "model"
        cfg = object()
        stats = SimpleNamespace(n_calls=0, n_prompts=0)

        @staticmethod
        def _engine(_model, _cfg):
            return engine

    monkeypatch.setitem(
        sys.modules,
        "vllm",
        types.SimpleNamespace(SamplingParams=lambda **kwargs: kwargs),
    )
    observed = score_declared_binary(
        Backend(), ["one", "two"], seed=[17, 29],
        expected_token_ids={"YES": 101, "NO": 202},
    )

    assert engine.texts == [
        "CHAT:one::YES", "CHAT:one::NO", "CHAT:two::YES", "CHAT:two::NO",
    ]
    assert [row["seed"] for row in engine.params] == [17, 17, 29, 29]
    assert np.allclose(observed, [1 / (1 + np.exp(-1.0)), 1 / (1 + np.exp(1.0))])
    assert Backend.stats.n_calls == 1
    assert Backend.stats.n_prompts == 4


def test_declared_binary_binds_frozen_token_ids_and_fails_on_missing_actual_logprob(
        monkeypatch):
    class Tokenizer:
        def encode(self, label, add_special_tokens=False):
            assert add_special_tokens is False
            return {"YES": [14331], "NO": [9173]}[label]

        def apply_chat_template(self, messages, **_kwargs):
            return messages[0]["content"]

    class Engine:
        def get_tokenizer(self):
            return Tokenizer()

        def generate(self, _texts, _params):
            missing = {999: SimpleNamespace(logprob=-2.0)}
            return [
                SimpleNamespace(
                    prompt_token_ids=[token_id],
                    prompt_logprobs=[missing],
                )
                for token_id in (14331, 9173)
            ]

    class Backend:
        model = "model"
        cfg = object()
        stats = SimpleNamespace(n_calls=0, n_prompts=0)

        @staticmethod
        def _engine(_model, _cfg):
            return Engine()

    monkeypatch.setitem(
        sys.modules,
        "vllm",
        types.SimpleNamespace(SamplingParams=lambda **kwargs: kwargs),
    )
    with pytest.raises(ValueError, match="differ from the frozen manifest"):
        score_declared_binary(
            Backend(), ["prompt"], expected_token_ids={"YES": 1, "NO": 2})
    with pytest.raises(ValueError, match="omits actual continuation token"):
        score_declared_binary(
            Backend(), ["prompt"],
            expected_token_ids={"YES": 14331, "NO": 9173},
        )
