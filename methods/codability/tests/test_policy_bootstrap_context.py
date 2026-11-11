import json

import numpy as np
import pytest

from methods.codability.experiments.policy_isomorphism import (
    PolicyBootstrapContext,
    _bootstrap_samples,
    certify_pairwise_policy_fidelity,
    certify_policy_isomorphism,
    certify_scale_step_substitution,
    compare_articulation_to_matched_control,
)


def _orbits(n_items: int = 24) -> dict[str, dict[str, np.ndarray]]:
    x = np.linspace(-2.7, 2.7, n_items)
    target_mean = 1.0 / (1.0 + np.exp(-x))

    def orbit(base: np.ndarray, offsets: tuple[float, float, float]) -> dict[str, np.ndarray]:
        return {
            form: np.clip(base + offset + 0.008 * np.sin(x * frequency), 0.001, 0.999)
            for form, offset, frequency in zip(
                ("canonical", "paraphrase", "compressed"), offsets, (1.0, 1.7, 2.3)
            )
        }

    return {
        "target": orbit(target_mean, (0.0, 0.012, -0.009)),
        "small": orbit(0.50 + 0.28 * (target_mean - 0.50), (0.018, -0.014, 0.010)),
        "candidate": orbit(0.50 + 0.82 * (target_mean - 0.50), (0.008, -0.006, 0.004)),
        "larger": orbit(0.50 + 0.78 * (target_mean - 0.50), (0.006, -0.004, 0.003)),
        "control": orbit(0.50 + 0.39 * (target_mean - 0.50), (-0.012, 0.016, -0.009)),
        "peer": orbit(0.50 + 0.80 * (target_mean - 0.50), (-0.005, 0.007, -0.003)),
    }


def _unequal_clusters(n_items: int = 24) -> list[str]:
    sizes = (1, 2, 3, 4, 6, 8)
    assert sum(sizes) == n_items
    return [f"source-{group}" for group, size in enumerate(sizes) for _ in range(size)]


def _serialized(value: dict) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


@pytest.mark.parametrize("clustered", [False, True])
def test_context_is_bit_exact_for_every_certificate_and_confidence(clustered):
    orbits = _orbits()
    clusters = _unequal_clusters() if clustered else None
    kwargs = {
        "bootstrap_clusters": clusters,
        "n_boot": 80,
        "seed": 271,
    }
    context = PolicyBootstrapContext(
        n_items=24,
        n_boot=kwargs["n_boot"],
        seed=kwargs["seed"],
        bootstrap_clusters=clusters,
    )
    if clustered:
        assert isinstance(context._samples, tuple)
        assert len({len(row) for row in context._samples}) > 1

    for confidence in (0.95, 0.9875):
        uncached = certify_policy_isomorphism(
            orbits["target"],
            orbits["candidate"],
            sparse_orbit=orbits["small"],
            confidence=confidence,
            **kwargs,
        )
        cached = certify_policy_isomorphism(
            orbits["target"],
            orbits["candidate"],
            sparse_orbit=orbits["small"],
            confidence=confidence,
            bootstrap_context=context,
            **kwargs,
        )
        assert _serialized(cached) == _serialized(uncached)

        uncached = certify_scale_step_substitution(
            orbits["target"],
            orbits["small"],
            orbits["candidate"],
            orbits["larger"],
            confidence=confidence,
            **kwargs,
        )
        cached = certify_scale_step_substitution(
            orbits["target"],
            orbits["small"],
            orbits["candidate"],
            orbits["larger"],
            confidence=confidence,
            bootstrap_context=context,
            **kwargs,
        )
        assert _serialized(cached) == _serialized(uncached)

        uncached = compare_articulation_to_matched_control(
            orbits["target"],
            orbits["candidate"],
            orbits["control"],
            confidence=confidence,
            **kwargs,
        )
        cached = compare_articulation_to_matched_control(
            orbits["target"],
            orbits["candidate"],
            orbits["control"],
            confidence=confidence,
            bootstrap_context=context,
            **kwargs,
        )
        assert _serialized(cached) == _serialized(uncached)

        uncached = certify_pairwise_policy_fidelity(
            orbits["candidate"],
            orbits["peer"],
            confidence=confidence,
            **kwargs,
        )
        cached = certify_pairwise_policy_fidelity(
            orbits["candidate"],
            orbits["peer"],
            confidence=confidence,
            bootstrap_context=context,
            **kwargs,
        )
        assert _serialized(cached) == _serialized(uncached)

    info = context.cache_info()
    assert info["orbit_bundle_hits"] > info["orbit_bundle_misses"]
    assert info["pairwise_draw_hits"] == 1
    assert info["pairwise_draw_misses"] == 1
    assert info["array_storage_bytes"] > 0


@pytest.mark.parametrize("clustered", [False, True])
def test_context_samples_exactly_match_fresh_resampling_design(clustered):
    clusters = _unequal_clusters() if clustered else None
    context = PolicyBootstrapContext(
        n_items=24, n_boot=100, seed=811, bootstrap_clusters=clusters
    )
    samples, design = _bootstrap_samples(
        rng=np.random.default_rng(811),
        n_boot=100,
        n_items=24,
        clusters=clusters,
    )
    context_rows = (
        list(context._samples)
        if isinstance(context._samples, tuple)
        else [row for row in context._samples]
    )
    fresh_rows = list(samples) if not isinstance(samples, np.ndarray) else [row for row in samples]
    assert len(context_rows) == len(fresh_rows)
    assert all(np.array_equal(left, right) for left, right in zip(context_rows, fresh_rows))
    assert context._bootstrap_design == design
    assert all(not row.flags.writeable for row in context_rows)


@pytest.mark.parametrize(
    ("context_kwargs", "call_kwargs", "message"),
    [
        ({"n_boot": 40}, {"n_boot": 41}, "n_boot mismatch"),
        ({"seed": 7}, {"seed": 8}, "seed mismatch"),
        (
            {"bootstrap_strata": ["a"] * 12 + ["b"] * 12},
            {"bootstrap_strata": ["a"] * 11 + ["b"] * 13},
            "bootstrap_strata mismatch",
        ),
        (
            {"bootstrap_clusters": _unequal_clusters()},
            {"bootstrap_clusters": [f"item-{i}" for i in range(24)]},
            "bootstrap_clusters mismatch",
        ),
    ],
)
def test_context_rejects_mismatched_draw_parameters(context_kwargs, call_kwargs, message):
    orbits = _orbits()
    base = {"n_boot": 40, "seed": 7, **context_kwargs}
    context = PolicyBootstrapContext(n_items=24, **base)
    call = {**base, **call_kwargs}
    with pytest.raises(ValueError, match=message):
        certify_policy_isomorphism(
            orbits["target"],
            orbits["candidate"],
            bootstrap_context=context,
            **call,
        )


def test_context_rejects_item_count_and_context_type_mismatches():
    context = PolicyBootstrapContext(n_items=24, n_boot=20, seed=3)
    longer = _orbits(25)
    with pytest.raises(ValueError, match="n_items mismatch"):
        certify_policy_isomorphism(
            longer["target"],
            longer["candidate"],
            n_boot=20,
            seed=3,
            bootstrap_context=context,
        )
    with pytest.raises(TypeError, match="PolicyBootstrapContext"):
        certify_policy_isomorphism(
            longer["target"],
            longer["candidate"],
            n_boot=20,
            seed=3,
            bootstrap_context=object(),
        )


@pytest.mark.parametrize("mutation", ["vector", "mapping"])
def test_context_fails_closed_on_in_place_orbit_mutation(mutation):
    orbits = _orbits()
    context = PolicyBootstrapContext(n_items=24, n_boot=30, seed=17)
    certify_policy_isomorphism(
        orbits["target"],
        orbits["candidate"],
        n_boot=30,
        seed=17,
        bootstrap_context=context,
    )
    if mutation == "vector":
        orbits["candidate"]["canonical"][0] += 0.001
    else:
        orbits["candidate"]["canonical"] = orbits["candidate"]["canonical"].copy()
        orbits["candidate"]["canonical"][0] += 0.001
    with pytest.raises(ValueError, match="mutated after binding"):
        certify_policy_isomorphism(
            orbits["target"],
            orbits["candidate"],
            n_boot=30,
            seed=17,
            bootstrap_context=context,
        )


def test_cached_arrays_and_configuration_are_immutable_and_result_mutation_is_isolated():
    orbits = _orbits()
    context = PolicyBootstrapContext(n_items=24, n_boot=30, seed=19)
    baseline = certify_policy_isomorphism(
        orbits["target"], orbits["candidate"], n_boot=30, seed=19
    )
    cached = certify_policy_isomorphism(
        orbits["target"],
        orbits["candidate"],
        n_boot=30,
        seed=19,
        bootstrap_context=context,
    )
    cached["point"]["candidate_robust"]["mae_tvd"] = 999.0
    repeated = certify_policy_isomorphism(
        orbits["target"],
        orbits["candidate"],
        n_boot=30,
        seed=19,
        bootstrap_context=context,
    )
    assert _serialized(repeated) == _serialized(baseline)

    first_array = next(
        values
        for bundle in context._orbit_bundles.values()
        for orbit in bundle.values()
        for values in orbit.values()
    )
    with pytest.raises(ValueError, match="read-only"):
        first_array[0] = 0.0
    with pytest.raises((AttributeError, TypeError)):
        context.n_boot = 99


def test_context_accepts_equal_fresh_orbit_objects_and_reuses_content_cache():
    orbits = _orbits()
    context = PolicyBootstrapContext(n_items=24, n_boot=30, seed=23)
    first = certify_policy_isomorphism(
        orbits["target"],
        orbits["candidate"],
        n_boot=30,
        seed=23,
        bootstrap_context=context,
    )
    target_copy = {key: values.copy() for key, values in orbits["target"].items()}
    candidate_copy = {key: values.copy() for key, values in orbits["candidate"].items()}
    second = certify_policy_isomorphism(
        target_copy,
        candidate_copy,
        n_boot=30,
        seed=23,
        confidence=0.99,
        bootstrap_context=context,
    )
    uncached_second = certify_policy_isomorphism(
        target_copy,
        candidate_copy,
        n_boot=30,
        seed=23,
        confidence=0.99,
    )
    assert first["bootstrap"]["confidence"] == 0.95
    assert _serialized(second) == _serialized(uncached_second)
    assert context.cache_info()["orbit_bundle_hits"] == 1


def test_run_path_cache_is_serialization_neutral_and_exercises_all_contexts(
        tmp_path, monkeypatch):
    from methods.codability.experiments import run_policy_isomorphism as runner

    n_items = 40
    x = np.linspace(-2.8, 2.8, n_items)
    q = 1.0 / (1.0 + np.exp(-x))
    values = {
        "name": 0.50 + 0.30 * (q - 0.50),
        "candidate_a": 0.50 + 0.82 * (q - 0.50),
        "candidate_b": 0.50 + 0.79 * (q - 0.50),
        "control": 0.50 + 0.38 * (q - 0.50),
    }
    forms = ("canonical", "paraphrase", "compressed")
    prompt_hashes = {
        arm_id: {form: f"{arm_id}-{form}-sha" for form in forms}
        for arm_id in values
    }
    arms = []
    for arm_id in values:
        is_control = arm_id == "control"
        arms.append({
            "id": arm_id,
            "channel": "control" if is_control else f"channel-{arm_id}",
            "provenance": "inert_length_control" if is_control else "source_telling",
            "control_for": "candidate_a" if is_control else None,
            "components": [] if arm_id == "name" else [arm_id],
            "semantic_content_word_count": 4,
            "added_content_word_count": 3,
            "forms": [{
                "id": form,
                "prompt": f"{arm_id} {form} prompt",
                "prompt_sha256": prompt_hashes[arm_id][form],
            } for form in forms],
        })
    bank_path = tmp_path / "bank.json"
    bank_path.write_text(json.dumps({
        "cells": [{
            "id": "cell", "domain": "humor", "gi": 1,
            "construct": "synthetic", "arms": arms,
        }],
    }))
    hashes = [f"item-{index}" for index in range(n_items)]

    def scored(arm_values, *, shard):
        rows, meta = [], []
        for arm_id, base in arm_values.items():
            for form_index, form in enumerate(forms):
                rows.append(np.clip(
                    base + (form_index - 1) * 0.006
                    + 0.004 * np.sin((form_index + 1) * x),
                    0.001,
                    0.999,
                ))
                meta.append({
                    "cell_id": "cell",
                    "arm_id": arm_id,
                    "form": form,
                    "prompt_sha256": prompt_hashes.get(
                        arm_id, {form: f"{arm_id}-{form}-sha"})[form],
                })
        return {
            "scores": np.stack(rows),
            "meta": meta,
            "hashes": hashes,
            "shard_sha256": [shard],
            "readout_template_sha256": "same-readout",
        }

    indexes = {
        "small-root": {("small", "humor"): scored(values, shard="small")},
        "target-root": {("target", "humor"): scored({"target": q}, shard="target")},
        "large-root": {("large", "humor"): scored({"name": 0.50 + 0.76 * (
            q - 0.50)}, shard="large")},
    }
    monkeypatch.setattr(
        runner, "load_public_index", lambda root, _partition: indexes[root]
    )
    monkeypatch.setattr(runner, "_average_repetitions", lambda value: value)

    created = []
    real_context = PolicyBootstrapContext

    def tracking_context(**kwargs):
        context = real_context(**kwargs)
        created.append(context)
        return context

    monkeypatch.setattr(runner, "PolicyBootstrapContext", tracking_context)
    kwargs = {
        "executor_shard_root": "small-root",
        "target_shard_root": "target-root",
        "scale_comparator_shard_root": "large-root",
        "scale_comparator_job": "large",
        "arm_bank_path": str(bank_path),
        "partition": "residual_prompt_selection",
        "small_job": "small",
        "big_job": "target",
        "target_arm_id": "target",
        "include_controls": True,
        "n_boot": 30,
        "seed": 41,
    }
    cached = runner.run(**kwargs)
    cached_contexts = list(created)
    created.clear()
    uncached = runner.run(**kwargs, use_bootstrap_cache=False)

    assert _serialized(cached) == _serialized(uncached)
    assert not created
    assert sum(context.cache_info()["orbit_bundle_hits"]
               for context in cached_contexts) > 0
    assert sum(context.cache_info()["pairwise_draw_misses"]
               for context in cached_contexts) == 1
    assert any(context.cache_info()["pairwise_draw_bundles"] == 1
               for context in cached_contexts)
