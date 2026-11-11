"""Planted controls (proposal §4.3) — MANDATORY before any real codability claim. Each control builds
a synthetic verdict-level world (strata, a target practice with a test–retest ceiling, rubric channels
with designed transfer patterns), pushes it through the REAL pipeline (stratified split → transfer
matrix → profile → ``levels.profile_level``), and states the level it must land on:

  1. universal code rule                      → L1   (fully tellable)
  2. genre-indexed code rule                  → L2, NOT L4 — the positive control for the whole
                                                decomposition: indexicality separated from tacitness
  3. exemplar-transmissible template          → L3   (showable, not tellable)
  4. shuffled-verdict noise                   → NO-SIGNAL gate (never L4: tacitness needs a practice)
  plus a fragmented two-concept cluster       → FRAGMENTED with categorical evidence, else the block
                                                flag only (never silently L1/L2)

Under the tacitness thesis every instrument weakness inflates the desired gap (direction-of-error
flip, `project_anthropological_framing`) — these controls ARE the credibility. Offline analogs here;
the live versions run the same constructions through the real executor."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .strata import stratified_split

G_DEFAULT = ("horror", "adventure", "romance", "mystery")


def _flip(rng, bits: np.ndarray, rate: float) -> np.ndarray:
    f = rng.uniform(0, 1, len(bits)) < rate
    return np.where(f, 1 - bits, bits).astype(float)


def _world(rng, *, n_per: int = 500, groups=G_DEFAULT, flip_target: float = 0.03):
    """Strata + per-stratum latent rule bits + a two-pass target realization (the test–retest
    ceiling). Returns (strata, v: {g: latent-rule verdicts over ALL items}, passes (2, n), split)."""
    strata = np.asarray([g for g in groups for _ in range(n_per)], dtype=object)
    n = len(strata)
    v = {g: (rng.uniform(0, 1, n) > 0.5).astype(float) for g in groups}
    M = np.empty(n)
    for g in groups:
        m = strata == g
        M[m] = v[g][m]
    p1, p2 = _flip(rng, M, flip_target), _flip(rng, M, flip_target)
    split = stratified_split(strata, held_frac=0.5, min_train=30, min_held=30, seed=0)
    return strata, v, M, np.stack([p1, p2]), split


def _base_profile(strata, passes, split, rubric_rules: Dict[str, np.ndarray],
                  rubric_global: np.ndarray, rubric_ex: Optional[Dict[str, np.ndarray]] = None,
                  **extra) -> dict:
    """Assemble the profile from verdicts through the real instruments (T_g = per-stratum κ of the
    two target passes; m̄ = their mean; transfer matrix on the frozen held mask)."""
    # Exercise the actual live assembler so planted controls catch split-viability/masking bugs.
    from .run_codability import assemble_profile
    return assemble_profile(passes, rubric_rules, strata, rubric_global=rubric_global,
                            rubric_ex=rubric_ex, split=split, **extra)


def planted_universal(rng) -> Tuple[dict, str]:
    """One rule, codable everywhere by the same articulation → L1."""
    groups = G_DEFAULT
    strata = np.asarray([g for g in groups for _ in range(500)], dtype=object)
    n = len(strata)
    rule = (rng.uniform(0, 1, n) > 0.5).astype(float)
    passes = np.stack([_flip(rng, rule, 0.03), _flip(rng, rule, 0.03)])
    split = stratified_split(strata, seed=0)
    rub = _flip(rng, rule, 0.05)
    rules = {g: rub for g in groups}
    return _base_profile(strata, passes, split, rules, rub,
                         kappa_families_g={g: 0.9 for g in groups}), "L1-UNIVERSAL"


def planted_indexical(rng) -> Tuple[dict, str]:
    """M(x) = f_g(x) with a DIFFERENT codable rule per genre; each r_g states its own frame's rule.
    Pooled articulation fails (Δ_context large), within-frame succeeds → L2, NOT L4. This is the
    positive control that the design separates indexicality from tacitness."""
    strata, v, M, passes, split = _world(rng)
    rules = {g: _flip(rng, v[g], 0.02) for g in split["strata"]}
    rub_global = rules[split["strata"][0]]        # the best single pooled rule = one frame's rule
    return _base_profile(strata, passes, split, rules, rub_global), "L2-INDEXICAL"


def planted_ostensive(rng) -> Tuple[dict, str]:
    """A stylistic template easy to match from exemplars, hard to state: the rules channel plateaus,
    rules+few-shot closes the gap → L3."""
    strata, v, M, passes, split = _world(rng)
    rules = {g: _flip(rng, M, 0.28) for g in split["strata"]}     # stated rules: weak everywhere
    ex = {g: _flip(rng, M, 0.05) for g in split["strata"]}        # exemplars transmit it
    return _base_profile(strata, passes, split, rules, rules[split["strata"][0]],
                         rubric_ex=ex), "L3-OSTENSIVE"


def planted_tacit(rng) -> Tuple[dict, str]:
    """A reproducible practice (T high) that no channel reaches, with the articulation process
    run to a preregistered operational horizon per stratum and exemplars tried → L4. This is a
    within-tested-horizon control, not a global codifiability certificate."""
    strata, v, M, passes, split = _world(rng)
    groups = split["strata"]
    rules = {g: _flip(rng, M, 0.35) for g in groups}
    ex = {g: _flip(rng, M, 0.32) for g in groups}
    prof = _base_profile(strata, passes, split, rules, rules[groups[0]], rubric_ex=ex,
                         search_horizon_reached_g={g: True for g in groups},
                         f1_over_N_g={g: 0.2 for g in groups})
    return prof, "L4-TACIT-WITHIN-FRAME"


def planted_noise(rng) -> Tuple[dict, str]:
    """Shuffled verdicts: the two target passes are INDEPENDENT → T_g ≈ 0 → excluded by the
    NO-SIGNAL gate, never L4 (tacitness is a property of a practice that exists)."""
    groups = G_DEFAULT
    strata = np.asarray([g for g in groups for _ in range(150)], dtype=object)
    n = len(strata)
    passes = np.stack([(rng.uniform(0, 1, n) > 0.5).astype(float),
                       (rng.uniform(0, 1, n) > 0.5).astype(float)])
    split = stratified_split(strata, seed=0)
    rub = (rng.uniform(0, 1, n) > 0.5).astype(float)
    rules = {g: rub for g in groups}
    return _base_profile(strata, passes, split, rules, rub), "NO-SIGNAL"


def planted_fragmented(rng, *, with_categorical_evidence: bool = True) -> Tuple[dict, str]:
    """Two concepts hiding in one cluster: strata {1,2} share rule A, {3,4} share rule B. The
    transfer matrix is 2-block. With the categorical evidence (per-stratum rubrics judged
    semantically DIFFERENT) → FRAGMENTED; without it → the block flag only (INDETERMINATE here,
    since within-block transfer kills diagonal dominance)."""
    groups = G_DEFAULT
    strata = np.asarray([g for g in groups for _ in range(150)], dtype=object)
    n = len(strata)
    rule_a = (rng.uniform(0, 1, n) > 0.5).astype(float)
    rule_b = (rng.uniform(0, 1, n) > 0.5).astype(float)
    block_of = {groups[0]: rule_a, groups[1]: rule_a, groups[2]: rule_b, groups[3]: rule_b}
    M = np.empty(n)
    for g in groups:
        m = strata == g
        M[m] = block_of[g][m]
    passes = np.stack([_flip(rng, M, 0.03), _flip(rng, M, 0.03)])
    split = stratified_split(strata, seed=0)
    rules = {g: _flip(rng, block_of[g], 0.05) for g in groups}
    prof = _base_profile(strata, passes, split, rules, rules[groups[0]],
                         rubrics_judged_different=with_categorical_evidence)
    return prof, ("FRAGMENTED" if with_categorical_evidence else "INDETERMINATE")


ALL_CONTROLS = (planted_universal, planted_indexical, planted_ostensive, planted_tacit,
                planted_noise, planted_fragmented)
