"""The subfield-controlled decomposition (proposal §4.1):

    Δ_context = mean_g R_g − R_global          — INDEXICALITY (codable given the frame)
    A_g       = T_g − R_g                      — the within-frame articulation gap
    R_ig      = μ + a_i + b_g + (ab)_ig        — two-way mixed model: a_i is the subfield-ADJUSTED
                                                 codability of metric i; Var[(ab)] is the
                                                 indexicality variance of the task

plus reliability attenuation correction (each R_ig divided by its per-stratum test–retest ρ_ig, with
a floor below which the corrected value is UNDEFINED rather than exploded)."""
from __future__ import annotations

from typing import Dict, Mapping, Optional, Tuple

import numpy as np


def delta_context(R_g: Mapping[str, float], R_global: float) -> float:
    """Δ_context = mean_g R_g − R_global (≥ 0 typically; large ⇒ the metric is codable given the
    frame but its pooled articulation averages over incompatible realizations)."""
    vals = np.asarray([v for v in R_g.values()], float)
    return float(np.nanmean(vals) - float(R_global))


def articulation_gaps(R_g: Mapping[str, float], T_g: Mapping[str, float]) -> Dict[str, float]:
    """A_g = T_g − R_g per stratum (the within-frame gap; same agreement units on both sides)."""
    return {g: float(T_g[g]) - float(R_g[g]) for g in R_g if g in T_g}


def mixed_model(R: np.ndarray, *, boot_items: Optional[Mapping[Tuple[int, int], np.ndarray]] = None,
                n_boot: int = 500, seed: int = 0) -> dict:
    """Two-way means decomposition of the metric × stratum recovery matrix ``R`` (I × G, NaN =
    undefined cell): R_ig = μ + a_i + b_g + (ab)_ig. NaN-safe means (adequate for the near-complete
    designs we run; a heavily unbalanced design needs iterative fitting — report, don't hide).

    ``boot_items``: optional {(i, g): per-item binary agreement vector} — the item-level indicators
    behind each R_ig; when given, items are resampled WITHIN strata (proposal §4.1) for percentile
    CIs on Var[(ab)] (the indexicality variance) and on each a_i."""
    R = np.asarray(R, float)
    mu = float(np.nanmean(R))
    a = np.nanmean(R, axis=1) - mu
    b = np.nanmean(R, axis=0) - mu
    ab = R - mu - a[:, None] - b[None, :]
    out = {"mu": mu, "a_metric": a, "b_stratum": b, "ab": ab,
           "var_a": float(np.nanvar(a)), "var_b": float(np.nanvar(b)),
           "var_ab": float(np.nanvar(ab))}
    if boot_items:
        rng = np.random.default_rng(seed)
        var_ab_bs, a_bs = [], []
        for _ in range(n_boot):
            Rb = R.copy()
            for (i, g), items in boot_items.items():
                items = np.asarray(items, float)
                if len(items):
                    Rb[i, g] = float(items[rng.integers(0, len(items), len(items))].mean())
            mub = float(np.nanmean(Rb))
            ai = np.nanmean(Rb, axis=1) - mub
            bg = np.nanmean(Rb, axis=0) - mub
            var_ab_bs.append(float(np.nanvar(Rb - mub - ai[:, None] - bg[None, :])))
            a_bs.append(ai)
        a_bs = np.asarray(a_bs)
        out["var_ab_ci"] = (float(np.percentile(var_ab_bs, 2.5)),
                            float(np.percentile(var_ab_bs, 97.5)))
        out["a_metric_ci"] = np.stack([np.percentile(a_bs, 2.5, axis=0),
                                       np.percentile(a_bs, 97.5, axis=0)], axis=1)
    return out


def attenuation_correct(R: np.ndarray, rho: np.ndarray, *, floor: float = 0.5,
                        cap: float = 1.0) -> np.ndarray:
    """Disattenuate each R_ig by its per-stratum reliability ρ_ig (test–retest of the TARGET —
    dividing by the ceiling, the standard Spearman correction). Cells with ρ < ``floor`` become NaN:
    below the floor the correction explodes noise and the honest read is UNDEFINED (the reliability
    gate), not a big number. Results are capped at ``cap``."""
    R = np.asarray(R, float)
    rho = np.asarray(rho, float)
    out = np.full_like(R, np.nan)
    ok = np.isfinite(R) & np.isfinite(rho) & (rho >= floor)
    out[ok] = np.minimum(R[ok] / rho[ok], cap)
    return out
