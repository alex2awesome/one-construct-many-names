"""Codability Profile driver.

Two modes:

  --controls        run the §4.3 planted-control discipline OFFLINE (mandatory before any real
                    claim; direction-of-error flip: under the tacitness thesis instrument weaknesses
                    inflate the desired gap, so these positive controls ARE the credibility):
                        python -m methods.codability.run_codability --controls

  --profiles F.json adjudicate LIVE per-metric profiles (assembled on sk3 from recon_channel
                    recoveries, per-stratum m̄_ω test–retest ceilings, and per-stratum
                    ``value_certificate`` runs) and print the per-task codability map:
                        python -m methods.codability.run_codability --profiles cw_profiles.json

Profile JSON schema: {metric_name: {R_global, R_rules_g: {g: κ}, R_ex_g, T_g, defined_g,
search_horizon_reached_g?, eps_frac_g?, f1_over_N_g, form_invariant, code_convergence,
kappa_families_g, transfer?, …}} — see ``levels.profile_level``. ``defined_g`` is required so missing
strata cannot disappear through map intersections. The transfer matrix can be rebuilt from raw verdict
arrays via ``assemble_profile``. Historical epsilon is retained only as a descriptive diagnostic;
it never certifies L4 or waives undersampling.

CPU-only; nothing here calls a model (stratum judging and rubric induction/execution happen in the
existing sk3 drivers — recon_channel, value_certificate)."""
from __future__ import annotations

import argparse
import json
from typing import Dict, Mapping, Optional

import numpy as np

from .decompose import articulation_gaps, delta_context, mixed_model  # noqa: F401 (re-export for drivers)
from .levels import codability_map, profile_level
from .strata import normalize_strata, probe_balance_guard, stratified_split
from .transfer import kappa, transfer_matrix


def assemble_profile(target_passes: np.ndarray, rubric_rules: Mapping[str, np.ndarray],
                     strata, *, rubric_global: Optional[np.ndarray] = None,
                     rubric_ex: Optional[Mapping[str, np.ndarray]] = None,
                     certs_g: Optional[Mapping[str, dict]] = None,
                     split: Optional[dict] = None, seed: int = 0, **extra) -> dict:
    """Build one metric's profile from raw verdict arrays (the live path): ``target_passes``
    (2, n_items) two independent executor passes of m̄_ω (their per-stratum κ = T_g; their mean =
    the target); ``rubric_rules[g]`` full-length verdicts of the stratum-g-induced rubric;
    ``certs_g[g]`` = the withdrawn §12.6 diagnostic dict per stratum (descriptive only).
    Extra keyword args (form_invariant, code_convergence, kappa_families_g, …) pass through."""
    target_passes = np.asarray(target_passes, float)
    m_bar = target_passes.mean(axis=0)
    split = split or stratified_split(strata, seed=seed)
    held = split["held_mask"]
    strata = normalize_strata(strata)
    groups = split["strata"]
    balance = probe_balance_guard(m_bar, strata)["defined"]
    defined = {g: bool(split.get("viable", {}).get(g, False) and balance.get(g, False)
                       and g in rubric_rules)
               for g in groups}
    eligible = [g for g in groups if defined[g]]
    eligible_held = held & np.isin(strata, eligible)
    T_g = {g: kappa(target_passes[0][held & (strata == g)],
                    target_passes[1][held & (strata == g)]) for g in eligible}
    tr = transfer_matrix({g: rubric_rules[g] for g in eligible}, m_bar, strata, eligible_held)
    prof = {"R_rules_g": tr["R_g"], "T_g": T_g, "transfer": tr,
            "defined_g": defined,
            "R_global": (kappa(np.asarray(rubric_global, float)[eligible_held], m_bar[eligible_held])
                         if rubric_global is not None else None),
            "R_ex_g": ({g: kappa(np.asarray(rubric_ex[g], float)[held & (strata == g)],
                                 m_bar[held & (strata == g)]) for g in eligible if g in rubric_ex}
                       if rubric_ex is not None else None)}
    if certs_g:
        # The historical epsilon is descriptive only (upper_bound_valid=False upstream). Keep it
        # for diagnostics, but operational saturation is a separate, explicit field and levels.py
        # never treats epsilon as a certificate or as permission to waive sampling requirements.
        prof["eps_frac_g"] = {g: float(c.get("eps_bits_adv", c["eps_bits"]))
                              / max(float(c.get("H_M") or 0.0), 1e-9)
                              for g, c in certs_g.items()}
        prof["f1_over_N_g"] = {g: float(c.get("f1_over_N", np.nan)) for g, c in certs_g.items()}
        # Adversarial-list dryness is not the preregistered articulation-search horizon required by
        # L4. Callers must provide search_horizon_reached_g explicitly via ``extra``.
        prof["adv_saturated_g"] = {g: c.get("adv_saturated") is True for g, c in certs_g.items()}
        prof["certificate_valid_g"] = {g: c.get("upper_bound_valid") is True
                                       for g, c in certs_g.items()}
    prof.update(extra)
    return prof


def _print_verdict(name: str, v: dict):
    a = v.get("aggregates", {})
    print(f"  {name[:44]:44s} {v['level']:22s} T̄={a.get('mean_T', float('nan')):.2f} "
          f"rel={a.get('rel_rules', float('nan')):.2f} Δctx={a.get('delta_context', float('nan')):.2f}")
    for r in v["reasons"]:
        print(f"      · {r}")
    for f in v.get("flags", []):
        print(f"      ⚑ {f}")


def run_controls() -> bool:
    from . import controls as C
    rng = np.random.default_rng(0)
    print("=" * 100)
    print("PLANTED CODABILITY CONTROLS (§4.3) — every row must land on its expected level")
    print("=" * 100)
    ok = True
    for ctor in C.ALL_CONTROLS:
        prof, expected = ctor(rng)
        v = profile_level(prof)
        good = v["level"] == expected
        ok &= good
        print(f"{'PASS' if good else 'FAIL':4s} expected={expected:22s}", end="")
        _print_verdict(ctor.__name__, v)
    print("-" * 100)
    print("controls " + ("ALL PASS — the decomposition separates indexicality from tacitness"
                         if ok else "FAILED — do NOT trust live codability claims until fixed"))
    return ok


def run_profiles(path: str, json_out: Optional[str] = None):
    profs: Dict[str, dict] = json.load(open(path))
    verdicts = {}
    print(f"=== codability profiles: {path} ({len(profs)} metrics) ===")
    for name, p in sorted(profs.items()):
        v = profile_level(p)
        verdicts[name] = v
        _print_verdict(name, v)
    cmap = codability_map(verdicts)
    print("\nCODABILITY MAP (the headline):")
    for k, f in cmap["fractions"].items():
        print(f"  {k:24s} {f * 100:5.1f}%  ({cmap['counts'][k]})")
    # the task-level mixed model over defined (metric, stratum) recoveries
    names = sorted(profs)
    groups = sorted({g for p in profs.values() for g in (p.get("R_rules_g") or {})})
    R = np.full((len(names), len(groups)), np.nan)
    for i, nm in enumerate(names):
        for j, g in enumerate(groups):
            if (profs[nm].get("defined_g") or {}).get(g, True):
                R[i, j] = (profs[nm].get("R_rules_g") or {}).get(g, np.nan)
    if np.isfinite(R).sum() >= 4:
        mm = mixed_model(R)
        print(f"\ntwo-way means decomposition: μ={mm['mu']:.3f}  Var[a_i]={mm['var_a']:.4f} "
              f"(metric) Var[b_g]={mm['var_b']:.4f} (stratum) "
              f"Var[(ab)]={mm['var_ab']:.4f} (residual metric×stratum heterogeneity; "
              f"not a fitted mixed-effects estimate)")
    if json_out:
        out = {"verdicts": {k: {kk: vv for kk, vv in v.items() if kk != "aggregates"}
                            for k, v in verdicts.items()}, "map": cmap}
        json.dump(out, open(json_out, "w"), indent=1, default=str)
        print(f"json → {json_out}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="run_codability", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--controls", action="store_true", help="run the §4.3 planted-control discipline")
    p.add_argument("--profiles", default=None, help="JSON of live per-metric profiles")
    p.add_argument("--json", default=None)
    a = p.parse_args(argv)
    if a.controls:
        raise SystemExit(0 if run_controls() else 1)
    if a.profiles:
        run_profiles(a.profiles, a.json)
        return
    p.error("pass --controls or --profiles")


if __name__ == "__main__":
    main()
