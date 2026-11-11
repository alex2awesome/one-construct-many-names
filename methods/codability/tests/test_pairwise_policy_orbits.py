"""Comparative policy-elicitation graph and recovery tests."""

import json

import numpy as np
import pytest

from methods.codability.experiments.score_pairwise_policy_orbits import (
    circulant_edges,
    pair_prompt,
    recover_pairwise_scores,
    run,
    score_ab,
)


def test_circulant_graph_is_regular_simple_and_deterministic():
    first = circulant_edges(40, degree=12, seed=17)
    replay = circulant_edges(40, degree=12, seed=17)
    other = circulant_edges(40, degree=12, seed=18)
    assert np.array_equal(first, replay)
    assert not np.array_equal(first, other)
    assert len(first) == 40 * 12 // 2
    assert len({tuple(edge) for edge in first}) == len(first)
    assert np.all(first[:, 0] < first[:, 1])
    assert np.all(np.bincount(first.ravel(), minlength=40) == 12)


def test_pairwise_recovery_recovers_latent_order_and_cancels_order_bias():
    n_items = 36
    edges = circulant_edges(n_items, degree=10, seed=3)
    latent = np.linspace(-0.7, 0.7, n_items)
    probability = 1.0 / (1.0 + np.exp(-(latent[edges[:, 0]] - latent[edges[:, 1]])))
    # Add equal-and-opposite presentation effects.  Forward P(A) and reverse P(B)
    # disagree, but their mean is the intended probability that the original left wins.
    forward_left = probability + 0.08
    reverse_left = probability - 0.08
    forward = np.column_stack([forward_left, 1.0 - forward_left])
    reverse = np.column_stack([1.0 - reverse_left, reverse_left])
    recovered = recover_pairwise_scores(edges, forward, reverse, n_items=n_items)
    borda_rank = np.argsort(np.argsort(recovered["borda"]))
    latent_rank = np.argsort(np.argsort(latent))
    assert np.corrcoef(borda_rank, latent_rank)[0, 1] > 0.97
    assert np.array_equal(np.argsort(recovered["bradley_terry"]), np.argsort(latent))
    assert np.allclose(recovered["edge_left_probability"], probability)
    assert recovered["mean_order_disagreement"] > 0.1


def test_pair_prompt_preserves_declared_criterion_and_literal_choices():
    for form in ("canonical", "question", "boilerplate"):
        prompt = pair_prompt("prefer a clean semantic reversal", "first", "second", form=form)
        assert "prefer a clean semantic reversal" in prompt
        assert "Item A:\nfirst" in prompt
        assert "Item B:\nsecond" in prompt
        assert prompt.endswith("exactly one letter: A or B.")


def test_binary_backend_compatibility_adapter_returns_two_choice_rows():
    class BinaryOnly:
        def score_binary(self, prompts, pos, neg, seed):
            assert (pos, neg, seed) == ("A", "B", 19)
            return [0.2, 0.75][:len(prompts)]

    observed = score_ab(BinaryOnly(), ["one", "two"], seed=19)
    assert np.allclose(observed, [[0.2, 0.8], [0.75, 0.25]])


def test_pairwise_runner_rejects_nonpublic_partition_before_backend_start(tmp_path):
    bank = tmp_path / "bank.json"
    bank.write_text(json.dumps({
        "status": "frozen-before-pairwise-small-executor-scoring",
        "cell": {"arms": []},
    }))
    with pytest.raises(ValueError, match="does not authorize partition"):
        run(
            bank_path=str(bank),
            packet_root="missing",
            partitions=["residual_lockbox"],
            out_dir=str(tmp_path / "out"),
            fake=True,
        )
