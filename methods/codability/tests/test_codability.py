"""Planted-ground-truth tests for the stratified Codability Profile (methods/codability). All CPU,
no model. The §4.3 discipline lives here: every planted control must land on its stated level, and
the genre-indexed control landing L2 (never L4) is the proof the decomposition separates
indexicality from tacitness."""
from __future__ import annotations

import numpy as np
import pytest

from methods.codability import controls as C
from methods.codability.decompose import (attenuation_correct, delta_context, mixed_model)
from methods.codability.levels import codability_map, profile_level
from methods.codability.strata import (normalize_strata, probe_balance_guard, stratified_split)
from methods.codability.transfer import block_structure, kappa, transfer_matrix


# ------------------------------- primitives -------------------------------------------------------

def test_kappa_identities():
    rng = np.random.default_rng(0)
    a = (rng.uniform(0, 1, 400) > 0.5).astype(float)
    assert kappa(a, a) == pytest.approx(1.0)
    b = (rng.uniform(0, 1, 400) > 0.5).astype(float)      # independent → κ ≈ 0
    assert abs(kappa(a, b)) < 0.15
    assert kappa(a, np.ones_like(a)) == 0.0               # degenerate marginal → 0, not NaN/1
    # perfect disagreement: κ = −pe/(1−pe) (exactly −1 only at balanced marginals)
    p = a.mean(); pe = 2 * p * (1 - p)
    assert kappa(a, 1 - a) == pytest.approx(-pe / (1 - pe))


def test_normalize_and_split_frozen_quotas():
    strata = ["Horror", " horror ", None, "ADVENTURE"] * 40
    ns = normalize_strata(strata)
    assert set(ns) == {"horror", "unknown", "adventure"}
    sp1 = stratified_split(ns, held_frac=0.5, min_train=30, min_held=30, seed=7)
    sp2 = stratified_split(ns, held_frac=0.5, min_train=30, min_held=30, seed=7)
    for g in sp1["strata"]:                                # frozen: function of (strata, seed) only
        assert np.array_equal(sp1["held_idx"][g], sp2["held_idx"][g])
        assert not set(sp1["held_idx"][g]) & set(sp1["train_idx"][g])
    assert sp1["viable"]["horror"] and not sp1["viable"]["unknown"]  # 40 < 30+30 quota


def test_probe_balance_guard_flags_near_constant_stratum():
    rng = np.random.default_rng(1)
    strata = np.asarray(["a"] * 200 + ["b"] * 200, dtype=object)
    tgt = np.concatenate([(rng.uniform(0, 1, 200) > 0.5).astype(float),
                          np.zeros(200) + 0.1])           # metric never fires on stratum b
    g = probe_balance_guard(tgt, strata)
    assert g["defined"]["a"] and not g["defined"]["b"]    # UNDEFINED there, not "low codability"


def test_transfer_matrix_diag_vs_universal_and_block():
    rng = np.random.default_rng(2)
    prof_u, _ = C.planted_universal(rng)
    tr_u = prof_u["transfer"]
    assert tr_u["diag_mean"] > 0.6
    assert abs(tr_u["diag_dominance"]) < 0.2               # universal: off-diag ≈ diag
    prof_i, _ = C.planted_indexical(rng)
    tr_i = prof_i["transfer"]
    assert tr_i["diag_mean"] > 0.6
    assert tr_i["diag_dominance"] > 0.4                    # indexical: strongly diagonal-dominant
    prof_f, _ = C.planted_fragmented(rng)
    blk = prof_f["transfer"]["block"]
    assert blk["score"] > 0.3                              # two interchangeable blocks
    assert sorted(map(sorted, blk["partition"])) == sorted(
        [sorted(["horror", "adventure"]), sorted(["romance", "mystery"])])


def test_mixed_model_recovers_planted_components_and_attenuation():
    rng = np.random.default_rng(3)
    I, G = 6, 4
    a = rng.normal(0, 0.15, I)
    b = rng.normal(0, 0.08, G)
    ab = rng.normal(0, 0.05, (I, G))
    R = 0.5 + a[:, None] + b[None, :] + ab
    mm = mixed_model(R)
    assert mm["mu"] == pytest.approx(R.mean(), abs=1e-9)
    assert np.corrcoef(mm["a_metric"], a)[0, 1] > 0.95     # subfield-adjusted codability recovered
    assert np.corrcoef(mm["b_stratum"], b)[0, 1] > 0.9
    assert mm["var_ab"] == pytest.approx(np.var(ab - ab.mean(0) - ab.mean(1)[:, None]
                                                + ab.mean()), rel=0.2)
    rho = np.full((I, G), 0.8); rho[0, 0] = 0.3            # below the reliability floor
    Rc = attenuation_correct(R, rho, floor=0.5)
    assert np.isnan(Rc[0, 0])                              # UNDEFINED, not exploded
    assert Rc[1, 1] == pytest.approx(min(R[1, 1] / 0.8, 1.0))
    assert delta_context({"a": 0.8, "b": 0.6}, 0.3) == pytest.approx(0.4)


# ------------------------------- the §4.3 planted-control discipline ------------------------------

def test_planted_controls_land_on_their_levels():
    rng = np.random.default_rng(0)
    for ctor in C.ALL_CONTROLS:
        prof, expected = ctor(rng)
        v = profile_level(prof)
        assert v["level"] == expected, f"{ctor.__name__}: {v['level']} != {expected} ({v['reasons']})"


def test_indexical_is_never_tacit_and_noise_is_never_L4():
    # the two catastrophic misreads the design exists to prevent
    rng = np.random.default_rng(4)
    v_idx = profile_level(C.planted_indexical(rng)[0])
    assert v_idx["level"] == "L2-INDEXICAL" and "TACIT" not in v_idx["level"]
    v_noise = profile_level(C.planted_noise(rng)[0])
    assert v_noise["level"] == "NO-SIGNAL"                 # T_g ≈ 0 excluded before any level


def test_L4_discipline_requires_exemplars_and_operational_saturation():
    rng = np.random.default_rng(5)
    prof, _ = C.planted_tacit(rng)
    assert profile_level(prof)["level"] == "L4-TACIT-WITHIN-FRAME"
    no_ex = dict(prof); no_ex["R_ex_g"] = None             # exemplar channel never tried
    v = profile_level(no_ex)
    assert v["level"] == "INDETERMINATE" and "exemplar-channel-not-run" in v["flags"]
    no_sat = dict(prof); no_sat["search_horizon_reached_g"] = {
        g: False for g in prof["search_horizon_reached_g"]}
    assert profile_level(no_sat)["level"] == "INDETERMINATE"


def test_withdrawn_epsilon_diagnostic_cannot_waive_undersampling():
    rng = np.random.default_rng(6)
    prof, _ = C.planted_tacit(rng)
    prof["f1_over_N_g"] = {g: 0.9 for g in prof["f1_over_N_g"]}
    prof["eps_frac_g"] = {g: 0.0 for g in prof["f1_over_N_g"]}
    assert profile_level(prof)["level"] == "UNDERSAMPLED"


def test_universal_requires_every_stratum_and_live_cross_family_check():
    prof = {"R_rules_g": {"a": .95, "b": .45}, "T_g": {"a": .8, "b": .8},
            "defined_g": {"a": True, "b": True},
            "R_global": .70, "kappa_families_g": {"a": .6, "b": .6}}
    assert profile_level(prof)["level"] != "L1-UNIVERSAL"


def test_levels_fail_closed_on_partial_evidence():
    base = {"R_rules_g": {"a": .8, "b": .8}, "T_g": {"a": .8, "b": .8},
            "R_global": .8, "defined_g": {"a": True, "b": True}}
    assert profile_level({**base, "R_rules_g": {"a": .8}})["level"] == "UNDEFINED"
    assert profile_level({**base, "kappa_families_g": {"a": .9}})["level"] != "L1-UNIVERSAL"
    tacit = {"R_rules_g": {"a": .1, "b": .1}, "T_g": {"a": .8, "b": .8},
             "R_ex_g": {"a": .2}, "defined_g": {"a": True, "b": True},
             "search_horizon_reached_g": {"a": True, "b": True}}
    assert profile_level(tacit)["level"] != "L4-TACIT-WITHIN-FRAME"


def test_fragmented_needs_categorical_evidence():
    rng = np.random.default_rng(7)
    v_with = profile_level(C.planted_fragmented(rng, with_categorical_evidence=True)[0])
    assert v_with["level"] == "FRAGMENTED"
    v_wo = profile_level(C.planted_fragmented(rng, with_categorical_evidence=False)[0])
    assert v_wo["level"] != "FRAGMENTED"                   # block structure alone never fires it
    assert any("block-structure" in f for f in v_wo["flags"])


def test_form_gate_and_L0_and_map():
    rng = np.random.default_rng(8)
    prof, _ = C.planted_universal(rng)
    assert profile_level({**prof, "form_invariant": False})["level"] == "FORM-DOMINATED"
    assert profile_level({**prof, "code_convergence": 0.9})["level"] == "L0-COMPILABLE"
    verdicts = {"m1": {"level": "L1-UNIVERSAL"}, "m2": {"level": "L1-UNIVERSAL"},
                "m3": {"level": "L4-TACIT-WITHIN-FRAME"}}
    cmap = codability_map(verdicts)
    assert cmap["fractions"]["L1-UNIVERSAL"] == pytest.approx(2 / 3)
    assert cmap["n_metrics"] == 3
