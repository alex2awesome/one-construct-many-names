"""The ordinal codability verdict (proposal §4.2): gates + L0–L4 router.

L1→L4 is the anthropological gradient — fully tellable → tellable-in-context → showable → only
learnable by immersion. The router consumes a per-metric PROFILE of stratified, embedding-free
quantities (all in the same agreement units, Cohen's κ, except descriptive ``eps_frac_g``):

    R_global        κ of the pooled-induced rubric on pooled held-out            (recon_channel)
    R_rules_g       {g: κ} rubric induced on stratum g, executed on held-out g   (transfer diagonal)
    R_ex_g          {g: κ} the rules+exemplars channel (ostension), optional
    T_g             {g: κ} per-stratum transmission ceiling (test–retest of m̄_ω; task-level: C_g)
    eps_frac_g      {g: ε/H(M)} withdrawn §12.6 diagnostic, optional; NEVER gates a level
    search_horizon_reached_g {g: bool} preregistered articulation-search stopping condition
    f1_over_N_g     {g: f₁/N} per-stratum singleton share, optional
    form_invariant  bool/None — §12.6.2 orbit gates
    code_convergence  κ of program vs judge (scorecard #8), optional
    kappa_families_g  {g: κ} inter-family reconstruction agreement, optional (naming agreement)
    mdl_g           reported only, never gates
    transfer        dict from ``transfer.transfer_matrix`` (diag_dominance, block), optional
    rubrics_judged_different / provenance_splits   bool/None — the categorical halves of FRAGMENTED
    defined_g       {g: bool} from split viability + ``probe_balance_guard``, required

Gate order (each is an EXCLUSION, not a level): UNDEFINED → FORM-DOMINATED → NO-SIGNAL → FRAGMENTED
→ UNDERSAMPLED. The former ε-based exception to UNDERSAMPLED was removed when the §12.6 upper-bound
claim was withdrawn: a descriptive flux estimate cannot certify that high f₁/N is harmless.

L4 discipline: TACIT-WITHIN-FRAME requires (i) the best channel INCLUDING exemplars still ≪ T_g in
every defined stratum, (ii) a preregistered operational search horizon reached in most strata, (iii)
T_g materially > 0, and (iv) the exemplar channel actually attempted. This is explicitly a
WITHIN-TESTED-HORIZON verdict, not a global optimality/codifiability certificate."""
from __future__ import annotations

from typing import Dict, List, Mapping, Optional

import numpy as np

GATES = ("UNDEFINED", "FORM-DOMINATED", "NO-SIGNAL", "FRAGMENTED", "UNDERSAMPLED")
LEVELS = ("L0-COMPILABLE", "L1-UNIVERSAL", "L2-INDEXICAL", "L3-OSTENSIVE", "L4-TACIT-WITHIN-FRAME")


def _mean(d: Optional[Mapping[str, float]], keys) -> float:
    if not d:
        return float("nan")
    v = [float(d[g]) for g in keys if g in d and np.isfinite(float(d[g]))]
    return float(np.mean(v)) if v else float("nan")


def profile_level(p: dict, *, t_signal_min: float = 0.10, rel_hi: float = 0.8,
                  tacit_rel_max: float = 0.6, ctx_frac: float = 0.15, dd_min: float = 0.05,
                  eps0_frac: float = 0.05, f1_hi: float = 0.8, code_hi: float = 0.8,
                  closure_hi: float = 0.5, block_hi: float = 0.15,
                  kappa_fam_min: float = 0.5) -> dict:
    """Route one metric's profile to a gate or level. Returns {"level": str, "reasons": [str…],
    "flags": [str…], "aggregates": {…}} — ``reasons`` justify the verdict, ``flags`` record
    pass-with-caveat conditions (missing optional instruments are flagged, never fatal)."""
    reasons: List[str] = []
    flags: List[str] = []
    R_rules: Dict[str, float] = dict(p.get("R_rules_g") or {})
    T: Dict[str, float] = dict(p.get("T_g") or {})

    # ---- restrict to strata where codability is DEFINED (probe-imbalance guard) ----------------
    defined = p.get("defined_g")
    if defined is None:
        return {"level": "UNDEFINED",
                "reasons": ["defined_g/expected-strata evidence missing; cannot detect omitted "
                            "or nonviable strata conservatively"],
                "flags": flags, "aggregates": {}}
    strata = [g for g in R_rules if g in T]
    if defined is not None:
        declared = [g for g, ok in defined.items() if ok]
        missing_declared = sorted(g for g in declared if g not in R_rules or g not in T)
        if missing_declared:
            return {"level": "UNDEFINED",
                    "reasons": [f"defined strata lack R_g/T_g measurements: {missing_declared}"],
                    "flags": flags, "aggregates": {}}
        dropped = [g for g in strata if not defined.get(g, False)]
        strata = declared
        if dropped:
            flags.append(f"undefined-strata-dropped={sorted(dropped)}")
    if not strata:
        return {"level": "UNDEFINED", "reasons": ["no stratum with defined codability "
                "(probe imbalance / missing R_g,T_g overlap)"], "flags": flags, "aggregates": {}}

    mean_T = _mean(T, strata)
    mean_R_rules = _mean(R_rules, strata)
    R_ex = p.get("R_ex_g")
    mean_R_ex = _mean(R_ex, strata)
    rel_rules_g = {g: float(R_rules[g]) / max(float(T[g]), t_signal_min) for g in strata}
    rel_ex_g = ({g: float(R_ex[g]) / max(float(T[g]), t_signal_min)
                 for g in strata if g in R_ex and np.isfinite(float(R_ex[g]))}
                if R_ex else {})
    rel_best_g = {g: max(rel_rules_g[g], rel_ex_g.get(g, -np.inf)) for g in strata}
    rel_rules = mean_R_rules / max(mean_T, t_signal_min)
    rel_best = (max(mean_R_rules, mean_R_ex) if np.isfinite(mean_R_ex) else mean_R_rules) \
        / max(mean_T, t_signal_min)
    R_global = p.get("R_global")
    d_ctx = (mean_R_rules - float(R_global)) if R_global is not None else float("nan")
    agg = {"strata": strata, "mean_T": mean_T, "mean_R_rules": mean_R_rules,
           "mean_R_ex": mean_R_ex, "rel_rules": rel_rules, "rel_best": rel_best,
           "delta_context": d_ctx, "rel_rules_g": rel_rules_g, "rel_best_g": rel_best_g}

    # ---- gates ----------------------------------------------------------------------------------
    if p.get("form_invariant") is False:
        return {"level": "FORM-DOMINATED", "reasons": ["§12.6.2 orbit gates fail — meaning unstable "
                "under rephrasing (itself a linguistic finding)"], "flags": flags, "aggregates": agg}
    low_signal = [g for g in strata if not np.isfinite(float(T[g])) or float(T[g]) < t_signal_min]
    if low_signal:
        return {"level": "NO-SIGNAL", "reasons": [f"strata {low_signal} have T_g < {t_signal_min} "
                "or non-finite — no reproducible practice there to articulate"],
                "flags": flags, "aggregates": agg}
    tr = p.get("transfer") or {}
    block = (tr.get("block") or {})
    block_score = block.get("score")
    categorical = bool(p.get("rubrics_judged_different")) or bool(p.get("provenance_splits"))
    if (block_score is not None and np.isfinite(block_score) and block_score >= block_hi):
        if categorical:
            return {"level": "FRAGMENTED",
                    "reasons": [f"transfer 2-block score={block_score:.3f} ≥ {block_hi} AND "
                                f"categorical evidence (judge-DIFFERENT rubrics / provenance split) "
                                f"— not one concept; route to the re-clustering audit"],
                    "flags": flags, "aggregates": {**agg, "block": block}}
        flags.append(f"block-structure-without-categorical-evidence(score={block_score:.3f})")
    f1 = p.get("f1_over_N_g") or {}
    undersampled = [g for g in strata if float(f1.get(g, np.nan)) >= f1_hi]
    if undersampled:
        return {"level": "UNDERSAMPLED",
                "reasons": [f"strata {sorted(undersampled)}: f₁/N ≥ {f1_hi}; the withdrawn ε "
                            f"diagnostic cannot waive this sampling gate — draw more / fix the quotient"],
                "flags": flags, "aggregates": agg}

    # ---- levels ---------------------------------------------------------------------------------
    cc = p.get("code_convergence")
    if cc is not None and np.isfinite(cc) and float(cc) >= code_hi:
        return {"level": "L0-COMPILABLE", "reasons": [f"code↔judge convergence κ={float(cc):.2f} ≥ "
                f"{code_hi} — recoverable via program"], "flags": flags, "aggregates": agg}

    kfam = p.get("kappa_families_g") or {}
    kf_vals = [float(kfam[g]) for g in strata
               if g in kfam and np.isfinite(float(kfam[g]))]
    kf_complete = len(kf_vals) == len(strata)
    if p.get("kappa_families_g") is None:
        flags.append("kappa-families-missing")
    rules_everywhere = all(v >= rel_hi for v in rel_rules_g.values())
    if rules_everywhere:
        indexical_ctx = np.isfinite(d_ctx) and d_ctx > ctx_frac * mean_T
        if (len(strata) >= 2 and not indexical_ctx and R_global is not None
                and np.isfinite(float(R_global)) and kf_complete
                and all(v >= kappa_fam_min for v in kf_vals)):
            return {"level": "L1-UNIVERSAL",
                    "reasons": [f"rules reach the ceiling in every stratum (min rel="
                                f"{min(rel_rules_g.values()):.2f}) and the "
                                f"pooled rubric matches (Δ_context={d_ctx:.3f})"],
                    "flags": flags, "aggregates": agg}
        if R_global is None:
            flags.append("R_global-missing:Δ_context-untested")
        if not kf_complete:
            flags.append("L1-withheld:kappa-families-untested")
        dd = tr.get("diag_dominance")
        if indexical_ctx and len(strata) >= 2:
            if dd is None or not np.isfinite(dd):
                flags.append("transfer-matrix-missing:diag-dominance-untested")
                dd_ok = False
            else:
                dd_ok = dd >= dd_min
            if dd_ok:
                return {"level": "L2-INDEXICAL",
                        "reasons": [f"rules reach the ceiling within-frame (rel={rel_rules:.2f}) "
                                    f"but the pooled rubric does not (Δ_context={d_ctx:.3f} > "
                                    f"{ctx_frac}·T̄) with diagonal-dominant transfer — codable "
                                    f"GIVEN the frame; the code needs a frame parameter"],
                        "flags": flags, "aggregates": {**agg, "diag_dominance": dd}}
            reasons.append("Δ_context large but transfer not diagonal-dominant "
                           f"(dd={dd if dd is None else f'{dd:.3f}'})")
    else:
        # rules plateau below the ceiling — the ostension question
        if R_ex:
            closures = []
            for g in strata:
                gap = float(T[g]) - float(R_rules.get(g, np.nan))
                if np.isfinite(gap) and gap > 0.05 and g in R_ex:
                    closures.append((float(R_ex[g]) - float(R_rules[g])) / gap)
            closure = float(np.mean(closures)) if closures else float("nan")
            agg["exemplar_closure"] = closure
            if np.isfinite(closure) and closure >= closure_hi \
                    and len(rel_ex_g) == len(strata) and all(v >= rel_hi for v in rel_ex_g.values()):
                return {"level": "L3-OSTENSIVE",
                        "reasons": [f"rules plateau (rel={rel_rules:.2f} < {rel_hi}) but exemplars "
                                    f"close {closure:.0%} of the gap — transmissible by showing, "
                                    f"not telling"], "flags": flags, "aggregates": agg}
        else:
            flags.append("exemplar-channel-not-run")
        saturated = [g for g in strata
                     if (p.get("search_horizon_reached_g") or {}).get(g) is True]
        tacit_everywhere = all(v <= tacit_rel_max for v in rel_best_g.values())
        exemplar_complete = bool(R_ex) and len(rel_ex_g) == len(strata)
        if tacit_everywhere and exemplar_complete and len(saturated) > len(strata) / 2:
            return {"level": "L4-TACIT-WITHIN-FRAME",
                    "reasons": [f"best channel incl. exemplars remains below ceiling in every "
                                f"stratum (max rel={max(rel_best_g.values()):.2f} ≤ {tacit_rel_max}); "
                                f"the preregistered search horizon was reached in "
                                f"{len(saturated)}/{len(strata)} strata and T̄={mean_T:.2f} — the "
                                f"practice resists articulation within the tested horizon"],
                    "flags": flags, "aggregates": agg}
        if tacit_everywhere:
            reasons.append("tacit-shaped gap but " + ("exemplar channel missing/incomplete" if not exemplar_complete else
                           f"operational saturation reached in only {len(saturated)}/{len(strata)} strata") +
                           " — L4 requires both and is never a global certificate")

    reasons.append(f"no level criterion met (rel_rules={rel_rules:.2f}, rel_best={rel_best:.2f}, "
                   f"Δ_context={d_ctx:.3f})")
    return {"level": "INDETERMINATE", "reasons": reasons, "flags": flags, "aggregates": agg}


def codability_map(verdicts: Mapping[str, dict]) -> dict:
    """The headline deliverable: per-task fraction of metrics at each gate/level (proposal §4.2 —
    'the codability map of the community's evaluative language')."""
    from collections import Counter
    c = Counter(v["level"] for v in verdicts.values())
    n = max(sum(c.values()), 1)
    return {"counts": dict(c), "fractions": {k: v / n for k, v in sorted(c.items())},
            "n_metrics": n}
