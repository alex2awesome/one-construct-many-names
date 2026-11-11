"""The transfer matrix — the behavioral (embedding-free) heterogeneity instrument (proposal §4.1).

``M[g→g'] = κ( exec(r_g, ·), m̄ )`` on held-out stratum g′, where ``r_g`` is the rubric induced from
stratum-g pairs ONLY. The diagonal is the conditional recovery ``R_g``; the off-diagonal structure is
the heterogeneity read: diagonal-dominant ⇒ indexical family (one concept, per-frame realizations);
block-structured (+ judge-DIFFERENT rubrics + provenance splits) ⇒ FRAGMENTED — not one concept, a
cluster-audit flag for the deferred R2/R3 re-clustering, never a codability level.

All agreement is Cohen's κ on binarized verdicts: chance-corrected, verdict-space, no geometry."""
from __future__ import annotations

from itertools import combinations
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

from .strata import normalize_strata


def kappa(a: np.ndarray, b: np.ndarray, *, thresh: float = 0.5) -> float:
    """Cohen's κ of two verdict vectors, binarized at ``thresh``. Degenerate marginals (either side
    constant ⇒ chance agreement undefined/1) return 0.0 — which correctly reads as NO evidence of
    agreement beyond chance, the conservative direction for every consumer here (T_g of a constant
    target → 0 → the NO-SIGNAL gate; R_g of a constant rubric → 0)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 2:
        return 0.0
    x = (a[m] > thresh).astype(int)
    y = (b[m] > thresh).astype(int)
    po = float((x == y).mean())
    px, py = float(x.mean()), float(y.mean())
    pe = px * py + (1 - px) * (1 - py)
    if 1.0 - pe < 1e-9:
        return 0.0
    return float((po - pe) / (1.0 - pe))


def transfer_matrix(rubric_verdicts: Mapping[str, np.ndarray], target: np.ndarray,
                    strata: Sequence, held_mask: Optional[np.ndarray] = None, *,
                    agree_fn=None) -> dict:
    """Assemble ``M[g→g']`` (rows = inducing stratum g, cols = evaluation stratum g′), evaluated on
    HELD-OUT items only (``held_mask``; default all items — pass the frozen split's mask for real
    runs). ``rubric_verdicts[g]`` = full-length verdict vector of rubric r_g executed everywhere;
    ``target`` = m̄_ω. Returns the matrix, ``R_g`` (the diagonal), diagonal-dominance summaries, and
    the 2-block structure read."""
    agree = agree_fn or kappa
    strata = normalize_strata(strata)
    target = np.asarray(target, float)
    held = np.ones(len(target), bool) if held_mask is None else np.asarray(held_mask, bool)
    groups = sorted(set(strata.tolist()) & set(str(k) for k in rubric_verdicts))
    G = len(groups)
    M = np.full((G, G), np.nan)
    for i, g in enumerate(groups):
        rv = np.asarray(rubric_verdicts[g], float)
        for j, g2 in enumerate(groups):
            m = held & (strata == g2)
            if m.sum() >= 2:
                M[i, j] = agree(rv[m], target[m])
    diag = np.array([M[i, i] for i in range(G)])
    off_max = np.array([np.nanmax(np.delete(M[i], i)) if G > 1 else np.nan for i in range(G)])
    off_mean = np.array([np.nanmean(np.delete(M[i], i)) if G > 1 else np.nan for i in range(G)])
    return {"strata": groups, "M": M,
            "R_g": {g: float(diag[i]) for i, g in enumerate(groups)},
            "diag_mean": float(np.nanmean(diag)) if G else float("nan"),
            "diag_dominance": float(np.nanmean(diag - off_max)) if G > 1 else float("nan"),
            "diag_minus_offmean": float(np.nanmean(diag - off_mean)) if G > 1 else float("nan"),
            "block": block_structure(M, groups)}


def block_structure(M: np.ndarray, labels: Sequence[str]) -> dict:
    """Best 2-partition of the strata by symmetrized OFF-DIAGONAL transfer: score = (mean within-block
    off-diag) − (mean between-block). High score = the strata fall into interchangeable blocks — the
    quantitative half of the FRAGMENTED evidence (the other half is categorical: per-stratum rubrics
    judged semantically DIFFERENT and/or leaf provenance splitting by stratum — assembled in
    ``levels.profile_level``, never decided here). Exact enumeration (G is small)."""
    M = np.asarray(M, float)
    G = len(labels)
    if G < 3:                                              # a 2-partition of G<3 has no within pair
        return {"score": float("nan"), "partition": None}
    S = (M + M.T) / 2.0
    best_score, best_part = -np.inf, None
    idx = list(range(G))
    for r in range(1, G // 2 + 1):
        for blk in combinations(idx, r):
            b1, b2 = set(blk), set(idx) - set(blk)
            within = [S[i, j] for bb in (b1, b2) for i in bb for j in bb if i < j]
            between = [S[i, j] for i in b1 for j in b2]
            if not within or not between:
                continue
            sc = float(np.nanmean(within) - np.nanmean(between))
            if np.isfinite(sc) and sc > best_score:
                best_score, best_part = sc, (sorted(labels[i] for i in b1),
                                             sorted(labels[i] for i in b2))
    if best_part is None:
        return {"score": float("nan"), "partition": None}
    return {"score": best_score, "partition": best_part}
