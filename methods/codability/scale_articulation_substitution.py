#!/usr/bin/env python
"""Baseline-gap-gated scale--articulation substitution analysis.

This module implements the strong estimand that the older name-sufficiency and horizontal-shift
reports only approximate.  For a fixed target and a sparse arm ``a0`` (normally ``name``), a
smaller reader counts as rescued only when:

1. the larger reader demonstrably outperforms the smaller reader at ``a0``;
2. a richer arm that explicitly articulates the same construct knowledge improves the smaller
   reader; and
3. the richer small-reader arm is non-inferior/equivalent to the larger reader at ``a0`` on
   held-out probes.

Two modes are intentionally separated:

``report``
    Re-read existing ``auc_report.json`` artifacts.  This is descriptive only because the best
    arm is chosen and evaluated on the same probes and raw signature equivalence is unavailable.

``crossfit``
    Re-read raw grid tensors.  Arms are selected on a stratified development split and evaluated
    on the untouched split.  Paired bootstrap intervals, a signature-correlation gate, and
    optional placebo arms produce the confirmatory decision record.

The target is fixed across arms: the reference M_i loaded from ``--ref-dir``.  Arm-specific
consensus targets are useful convergence diagnostics but are not valid for this substitution
estimand.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from typing import Callable, Iterable

import numpy as np

from methods.codability.grid_auc_report import auc_mw, spearman
from methods.codability.name_sufficiency import DATA, DOMAINS
from methods.codability.run_decompression_grid import _ckpts


DEFAULT_RUNGS = ["name", "definition", "explanation", "full_rubric",
                 "exemplars", "dossier", "dossier_v2"]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _finite(v):
    return v is not None and np.isfinite(v)


def _round(v, n=4):
    return None if v is None or not np.isfinite(v) else round(float(v), n)


def best_arm(rungs: dict, allowed: Iterable[str]) -> tuple[str | None, float | None]:
    """Best finite arm in predeclared order; earlier arms win exact ties."""
    candidates = [(r, rungs.get(r, {}).get("auc")) for r in allowed]
    candidates = [(r, float(v)) for r, v in candidates if _finite(v)]
    if not candidates:
        return None, None
    best = max(v for _, v in candidates)
    return next((r, v) for r, v in candidates if v == best)


def classify_point(small: dict, big: dict, *, rungs=DEFAULT_RUNGS, sparse="name",
                   delta=0.02, floor=0.55) -> dict:
    """Descriptive point classification from already-aggregated AUC cells.

    ``rescue_*`` is only true if a baseline gap exists.  This prevents trivial parity (the small
    reader already matched at the sparse arm) from being counted as articulation substitution.
    """
    s0 = small.get(sparse, {}).get("auc")
    b0 = big.get(sparse, {}).get("auc")
    sr, sbest = best_arm(small, rungs)
    br, bbest = best_arm(big, rungs)
    if not all(_finite(v) for v in (s0, b0, sbest, bbest)):
        return {"status": "missing", "small_sparse": _round(s0), "big_sparse": _round(b0)}
    if b0 < floor:
        return {"status": "big_sparse_below_floor", "small_sparse": _round(s0),
                "big_sparse": _round(b0), "small_best": _round(sbest),
                "small_best_rung": sr, "big_best": _round(bbest), "big_best_rung": br}

    gap = float(b0 - s0)
    full_gap = float(bbest - s0)
    gap_present = gap > delta
    full_gap_present = full_gap > delta
    noninferior = sbest >= b0 - delta
    equivalent = abs(sbest - b0) <= delta
    full_noninferior = sbest >= bbest - delta
    full_equivalent = abs(sbest - bbest) <= delta
    return {
        "status": "gap" if gap_present else "no_baseline_gap",
        "small_sparse": _round(s0), "big_sparse": _round(b0),
        "baseline_gap": _round(gap), "baseline_gap_present": bool(gap_present),
        "small_best": _round(sbest), "small_best_rung": sr,
        "small_gain": _round(sbest - s0),
        "big_best": _round(bbest), "big_best_rung": br,
        "big_name_sufficient_vs_best": bool(b0 >= floor and bbest - b0 <= delta),
        "small_best_noninferior_big_sparse": bool(noninferior),
        "small_best_equivalent_big_sparse": bool(equivalent),
        "rescue_big_sparse": bool(gap_present and noninferior),
        "equivalent_rescue_big_sparse": bool(gap_present and equivalent),
        "full_baseline_gap": _round(full_gap),
        "full_baseline_gap_present": bool(full_gap_present),
        "small_best_noninferior_big_best": bool(full_noninferior),
        "small_best_equivalent_big_best": bool(full_equivalent),
        "rescue_big_best": bool(full_gap_present and full_noninferior),
        "equivalent_rescue_big_best": bool(full_gap_present and full_equivalent),
    }


def bootstrap_conditional_rate(records: list[dict], denominator: Callable[[dict], bool],
                               success: Callable[[dict], bool], *, n_boot=2000,
                               seed=0) -> dict:
    eligible = [r for r in records if denominator(r)]
    if not eligible:
        return {"success": 0, "n": 0, "rate": None, "CI95": None}
    hits = int(sum(bool(success(r)) for r in eligible))
    ci = None
    if len(eligible) >= 3 and n_boot:
        rng = np.random.default_rng(seed)
        vals = np.asarray([bool(success(r)) for r in eligible], float)
        means = np.mean(vals[rng.integers(0, len(vals), (n_boot, len(vals)))], axis=1)
        ci = [_round(np.percentile(means, 2.5), 3), _round(np.percentile(means, 97.5), 3)]
    return {"success": hits, "n": len(eligible), "rate": _round(hits / len(eligible), 4),
            "CI95_metric_bootstrap": ci}


def summarize_point_records(records: list[dict], *, n_boot=2000, seed=0) -> dict:
    evaluable = [r for r in records if r.get("status") in {"gap", "no_baseline_gap"}]
    return {
        "n_records": len(records), "n_evaluable": len(evaluable),
        "n_big_sparse_below_floor": sum(r.get("status") == "big_sparse_below_floor"
                                         for r in records),
        "n_missing": sum(r.get("status") == "missing" for r in records),
        "no_baseline_gap": bootstrap_conditional_rate(
            evaluable, lambda r: True, lambda r: not r["baseline_gap_present"],
            n_boot=n_boot, seed=seed),
        "rescue_big_sparse_among_gaps": bootstrap_conditional_rate(
            evaluable, lambda r: r["baseline_gap_present"], lambda r: r["rescue_big_sparse"],
            n_boot=n_boot, seed=seed + 1),
        "equivalent_rescue_big_sparse_among_gaps": bootstrap_conditional_rate(
            evaluable, lambda r: r["baseline_gap_present"],
            lambda r: r["equivalent_rescue_big_sparse"], n_boot=n_boot, seed=seed + 2),
        "rescue_big_best_among_full_gaps": bootstrap_conditional_rate(
            evaluable, lambda r: r["full_baseline_gap_present"],
            lambda r: r["rescue_big_best"], n_boot=n_boot, seed=seed + 3),
        "big_name_sufficient_vs_own_best": bootstrap_conditional_rate(
            evaluable, lambda r: True, lambda r: r["big_name_sufficient_vs_best"],
            n_boot=n_boot, seed=seed + 4),
    }


def analyze_reports(*, data_dir: str, domains: list[str], small_reader: str, big_reader: str,
                    rungs: list[str], sparse: str, delta: float, floor: float,
                    n_boot: int, seed: int) -> dict:
    records, inputs = [], []
    by_domain = {}
    for domain in domains:
        if domain not in DOMAINS:
            raise ValueError(f"unknown domain {domain!r}; choose from {sorted(DOMAINS)}")
        gdir = DOMAINS[domain][0]
        path = os.path.join(data_dir, gdir, "auc_report.json")
        if not os.path.exists(path):
            by_domain[domain] = {"error": f"missing {path}"}
            continue
        report = json.load(open(path))
        if small_reader not in report or big_reader not in report:
            by_domain[domain] = {"error": "reader missing", "available": sorted(report)}
            continue
        inputs.append({"path": path, "sha256": sha256_file(path)})
        drows = []
        common = sorted(set(report[small_reader]) & set(report[big_reader]), key=int)
        for gi in common:
            row = {"domain": domain, "gi": int(gi),
                   **classify_point(report[small_reader][gi], report[big_reader][gi],
                                    rungs=rungs, sparse=sparse, delta=delta, floor=floor)}
            drows.append(row)
            records.append(row)
        by_domain[domain] = {"summary": summarize_point_records(drows, n_boot=n_boot,
                                                                  seed=seed),
                             "per_metric": drows}
    return {
        "schema": "scale_articulation_substitution/v1",
        "mode": "report_descriptive_same_probe_selection",
        "claim_scope": "exploratory only; best arm selected and evaluated on the same probes",
        "fixed_target_requirement": "all reader cells must use the same frozen reference M_i",
        "config": {"domains": domains, "small_reader": small_reader, "big_reader": big_reader,
                   "rungs": rungs, "sparse_rung": sparse, "delta": delta, "floor": floor,
                   "n_boot": n_boot, "seed": seed},
        "inputs": inputs, "pooled": summarize_point_records(records, n_boot=n_boot, seed=seed),
        "by_domain": by_domain,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }


def stratified_split(labels: np.ndarray, *, train_frac=0.5, seed=0) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, bool)
    if not 0 < train_frac < 1:
        raise ValueError("train_frac must be strictly between 0 and 1")
    counts = [int(np.sum(labels == value)) for value in (False, True)]
    if min(counts) < 2:
        raise ValueError(f"need at least two probes per class for held-out evaluation; got {counts}")
    rng = np.random.default_rng(seed)
    train, test = [], []
    for value in (False, True):
        idx = np.flatnonzero(labels == value)
        rng.shuffle(idx)
        cut = min(max(int(round(len(idx) * train_frac)), 1), max(len(idx) - 1, 1))
        train.extend(idx[:cut])
        test.extend(idx[cut:])
    return np.asarray(sorted(train), int), np.asarray(sorted(test), int)


def _select_auc(scores: dict[str, np.ndarray], labels: np.ndarray, idx: np.ndarray,
                allowed: list[str]) -> tuple[str, float]:
    vals = []
    for arm in allowed:
        if arm not in scores:
            continue
        v = auc_mw(np.asarray(scores[arm])[idx], labels[idx])
        if v is not None and np.isfinite(v):
            vals.append((arm, float(v)))
    if not vals:
        raise ValueError("no finite selectable arms")
    best = max(v for _, v in vals)
    return next((a, v) for a, v in vals if v == best)


def _select_small_arm(small: dict[str, np.ndarray], big: dict[str, np.ndarray],
                      labels: np.ndarray, train: np.ndarray, allowed: list[str], *,
                      sparse: str, delta: float, policy: str) -> tuple[str, float, bool | None]:
    """Select without looking at test probes; list order defines articulation cost order."""
    if policy == "best_auc":
        arm, value = _select_auc(small, labels, train, allowed)
        return arm, value, None
    if policy != "minimal_cost_noninferior":
        raise ValueError(f"unknown selection policy {policy!r}")
    target = auc_mw(np.asarray(big[sparse])[train], labels[train])
    if target is None:
        raise ValueError("big sparse arm is not measurable on the development split")
    measured = []
    for arm in allowed:
        if arm == sparse:
            continue
        value = auc_mw(np.asarray(small[arm])[train], labels[train])
        if value is None or not np.isfinite(value):
            continue
        measured.append((arm, float(value)))
        if value >= target - delta:
            return arm, float(value), True
    if not measured:
        raise ValueError("no finite richer small-reader arms")
    # Keep the metric evaluable and transparently mark failure to attain the development target.
    best = max(v for _, v in measured)
    arm, value = next((a, v) for a, v in measured if v == best)
    return arm, value, False


def paired_bootstrap_auc_diff(a: np.ndarray, b: np.ndarray, labels: np.ndarray, *,
                              n_boot=2000, seed=0) -> dict:
    """Paired AUC(a)-AUC(b), resampling positive and negative probes as shared clusters."""
    a, b, labels = np.asarray(a, float), np.asarray(b, float), np.asarray(labels, bool)
    obs_a, obs_b = auc_mw(a, labels), auc_mw(b, labels)
    if obs_a is None or obs_b is None:
        return {"diff": None, "CI95": None, "n_pos": int(labels.sum()),
                "n_neg": int((~labels).sum())}
    pos, neg = np.flatnonzero(labels), np.flatnonzero(~labels)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = np.r_[rng.choice(pos, len(pos), replace=True),
                    rng.choice(neg, len(neg), replace=True)]
        lab = labels[idx]
        diffs.append(auc_mw(a[idx], lab) - auc_mw(b[idx], lab))
    return {"diff": _round(obs_a - obs_b),
            "CI95": [_round(np.percentile(diffs, 2.5)), _round(np.percentile(diffs, 97.5))],
            "n_pos": len(pos), "n_neg": len(neg), "n_boot": n_boot}


def paired_bootstrap_spearman(a: np.ndarray, b: np.ndarray, labels: np.ndarray, *,
                              n_boot=2000, seed=0) -> dict:
    """Direct signature fidelity with a class-stratified paired probe bootstrap."""
    a, b, labels = np.asarray(a, float), np.asarray(b, float), np.asarray(labels, bool)
    obs = spearman(a, b)
    pos, neg = np.flatnonzero(labels), np.flatnonzero(~labels)
    if obs is None or not np.isfinite(obs) or not len(pos) or not len(neg) or n_boot <= 0:
        return {"rho": _round(obs), "CI95": None, "n_pos": len(pos), "n_neg": len(neg),
                "n_boot": n_boot}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = np.r_[rng.choice(pos, len(pos), replace=True),
                    rng.choice(neg, len(neg), replace=True)]
        v = spearman(a[idx], b[idx])
        if v is not None and np.isfinite(v):
            vals.append(v)
    ci = None if not vals else [_round(np.percentile(vals, 2.5)),
                                _round(np.percentile(vals, 97.5))]
    return {"rho": _round(obs), "CI95": ci, "n_pos": len(pos), "n_neg": len(neg),
            "n_boot": len(vals)}


def paired_bootstrap_spearman_gain(rich: np.ndarray, sparse: np.ndarray, target: np.ndarray,
                                   labels: np.ndarray, *, n_boot=2000, seed=0) -> dict:
    """Does articulation make the small-reader signature more like the big sparse signature?"""
    rich, sparse, target = (np.asarray(x, float) for x in (rich, sparse, target))
    labels = np.asarray(labels, bool)
    rho_rich, rho_sparse = spearman(rich, target), spearman(sparse, target)
    if rho_rich is None or rho_sparse is None:
        return {"gain": None, "CI95": None, "rho_rich": _round(rho_rich),
                "rho_sparse": _round(rho_sparse), "n_boot": 0}
    pos, neg = np.flatnonzero(labels), np.flatnonzero(~labels)
    rng = np.random.default_rng(seed)
    gains = []
    for _ in range(n_boot):
        idx = np.r_[rng.choice(pos, len(pos), replace=True),
                    rng.choice(neg, len(neg), replace=True)]
        rr, rs = spearman(rich[idx], target[idx]), spearman(sparse[idx], target[idx])
        if rr is not None and rs is not None and np.isfinite(rr) and np.isfinite(rs):
            gains.append(rr - rs)
    ci = None if not gains else [_round(np.percentile(gains, 2.5)),
                                 _round(np.percentile(gains, 97.5))]
    return {"gain": _round(rho_rich - rho_sparse), "CI95": ci,
            "rho_rich": _round(rho_rich), "rho_sparse": _round(rho_sparse),
            "n_boot": len(gains)}


def crossfit_metric(small: dict[str, np.ndarray], big: dict[str, np.ndarray], labels: np.ndarray, *,
                    rungs: list[str], sparse="name", placebo_rungs: list[str] | None = None,
                    delta=0.02, floor=0.55, train_frac=0.5, seed=0, n_boot=2000,
                    min_signature_rho=0.5, selection_policy="best_auc",
                    min_test_class_n=10) -> dict:
    """Select arms on development probes and confirm substitution on untouched probes."""
    labels = np.asarray(labels, bool)
    if sparse not in small or sparse not in big:
        raise ValueError(f"sparse arm {sparse!r} absent")
    train, test = stratified_split(labels, train_frac=train_frac, seed=seed)
    test_counts = [int(np.sum(labels[test] == value)) for value in (False, True)]
    if min(test_counts) < min_test_class_n:
        raise ValueError(f"held-out split needs >= {min_test_class_n} probes per class; "
                         f"got neg/pos={test_counts}")
    articulations = [r for r in rungs if r in small and r not in set(placebo_rungs or [])]
    s_arm, s_train, dev_target_attained = _select_small_arm(
        small, big, labels, train, articulations, sparse=sparse, delta=delta,
        policy=selection_policy)
    b_arm, b_train = _select_auc(big, labels, train, [r for r in rungs if r in big])

    def auc(scores, arm):
        return auc_mw(np.asarray(scores[arm])[test], labels[test])

    s0, b0, sbest, bbest = (auc(small, sparse), auc(big, sparse),
                             auc(small, s_arm), auc(big, b_arm))
    baseline = paired_bootstrap_auc_diff(np.asarray(big[sparse])[test],
                                         np.asarray(small[sparse])[test], labels[test],
                                         n_boot=n_boot, seed=seed + 10)
    sparse_match = paired_bootstrap_auc_diff(np.asarray(small[s_arm])[test],
                                             np.asarray(big[sparse])[test], labels[test],
                                             n_boot=n_boot, seed=seed + 11)
    improvement = paired_bootstrap_auc_diff(np.asarray(small[s_arm])[test],
                                            np.asarray(small[sparse])[test], labels[test],
                                            n_boot=n_boot, seed=seed + 12)
    full_match = paired_bootstrap_auc_diff(np.asarray(small[s_arm])[test],
                                           np.asarray(big[b_arm])[test], labels[test],
                                           n_boot=n_boot, seed=seed + 13)
    fidelity = paired_bootstrap_spearman(np.asarray(small[s_arm])[test],
                                         np.asarray(big[sparse])[test], labels[test],
                                         n_boot=n_boot, seed=seed + 15)
    fidelity_gain = paired_bootstrap_spearman_gain(
        np.asarray(small[s_arm])[test], np.asarray(small[sparse])[test],
        np.asarray(big[sparse])[test], labels[test], n_boot=n_boot, seed=seed + 16)

    baseline_point = b0 is not None and s0 is not None and b0 - s0 > delta
    baseline_confirmed = bool(baseline["CI95"] and baseline["CI95"][0] > delta)
    improved = bool(improvement["CI95"] and improvement["CI95"][0] > 0)
    noninferior = bool(sparse_match["CI95"] and sparse_match["CI95"][0] >= -delta)
    equivalent = bool(sparse_match["CI95"] and sparse_match["CI95"][0] >= -delta
                      and sparse_match["CI95"][1] <= delta)

    placebo = None
    articulation_specific = None
    if placebo_rungs:
        present = [r for r in placebo_rungs if r in small]
        if present:
            # Unit/segment grids use uk/fk matched pairs. Prefer the filler at the selected k;
            # independently selecting the best filler would no longer be length matched.
            unit_match = re.fullmatch(r"u(\d+)", s_arm)
            matched = f"f{unit_match.group(1)}" if unit_match else None
            if matched in present:
                p_arm = matched
                p_train = auc_mw(np.asarray(small[p_arm])[train], labels[train])
                selection_rule = "matched_to_selected_content_k"
            else:
                p_arm, p_train = _select_auc(small, labels, train, present)
                selection_rule = "best_placebo_on_development_split"
            placebo = {"selected_arm": p_arm, "selection_auc": _round(p_train),
                       "selection_rule": selection_rule,
                       "test_auc": _round(auc(small, p_arm)),
                       "articulation_minus_control": paired_bootstrap_auc_diff(
                           np.asarray(small[s_arm])[test], np.asarray(small[p_arm])[test],
                           labels[test], n_boot=n_boot, seed=seed + 14)}
            # Compatibility alias for existing readers of the exploratory schema.
            placebo["content_minus_placebo"] = placebo["articulation_minus_control"]
            pci = placebo["articulation_minus_control"]["CI95"]
            articulation_specific = bool(pci and pci[0] > 0)

    fidelity_ci = fidelity["CI95"]
    fidelity_gate = bool(fidelity_ci and fidelity_ci[0] >= min_signature_rho)
    fidelity_gain_ci = fidelity_gain["CI95"]
    fidelity_improved = bool(fidelity_gain_ci and fidelity_gain_ci[0] > 0)
    measurable = bool(b0 is not None and b0 >= floor)
    richer_arm = s_arm != sparse
    methodological = bool(measurable and baseline_confirmed and richer_arm and improved
                           and noninferior and fidelity_gate and fidelity_improved)
    equivalent_methodological = bool(methodological and equivalent)
    noninferior_certified = (None if articulation_specific is None else
                             bool(methodological and articulation_specific))
    # The unqualified certificate is deliberately the stronger, two-sided isomorphism claim.
    certified = (None if articulation_specific is None else
                 bool(equivalent_methodological and articulation_specific))
    return {
        "split": {"seed": seed, "train_frac": train_frac, "n_train": len(train),
                  "n_test": len(test)},
        "selection_policy": selection_policy,
        "development_target_attained": dev_target_attained,
        "selected_small_arm": s_arm, "selected_small_train_auc": _round(s_train),
        "selected_big_arm": b_arm, "selected_big_train_auc": _round(b_train),
        "test_auc": {"small_sparse": _round(s0), "big_sparse": _round(b0),
                     "small_selected": _round(sbest), "big_selected": _round(bbest)},
        "big_sparse_measurable": measurable,
        "baseline_gap_point": bool(baseline_point), "baseline_gap_test": baseline,
        "baseline_gap_confirmed": baseline_confirmed,
        "small_selected_minus_big_sparse": sparse_match,
        "small_selected_minus_small_sparse": improvement,
        "small_selected_minus_big_selected": full_match,
        "noninferior_big_sparse": noninferior, "equivalent_big_sparse": equivalent,
        "selected_arm_is_richer_than_sparse": richer_arm,
        "selected_articulation_segments": (int(unit_match.group(1)) if
                                             (unit_match := re.fullmatch(r"u(\d+)", s_arm)) else None),
        "signature_fidelity_small_selected_vs_big_sparse": fidelity,
        "signature_gate": fidelity_gate,
        "signature_fidelity_gain_over_small_sparse": fidelity_gain,
        "signature_fidelity_improved": fidelity_improved,
        "articulation_control": placebo,
        "articulation_specific": articulation_specific,
        # Compatibility aliases.
        "placebo": placebo, "content_specific": articulation_specific,
        "methodological_substitution_without_specificity_gate": methodological,
        "equivalent_methodological_substitution_without_specificity_gate":
            equivalent_methodological,
        "noninferior_certified_substitution": noninferior_certified,
        "certified_substitution": certified,
    }


def summarize_crossfit_records(rows: list[dict], *, n_boot=2000, seed=0) -> dict:
    good = [r for r in rows if "error" not in r]
    confirmed = [r for r in good if r.get("baseline_gap_confirmed")]
    point_gaps = [r for r in good if r.get("baseline_gap_point")]
    segment_costs = [r["selected_articulation_segments"] for r in confirmed
                     if r.get("certified_substitution") is True
                     and r.get("selected_articulation_segments") is not None]
    return {
        "n_records": len(rows), "n_evaluable": len(good), "n_errors": len(rows) - len(good),
        "baseline_gap_confirmed": len(confirmed), "baseline_gap_point": len(point_gaps),
        "noninferior_rescue_among_confirmed_gaps": bootstrap_conditional_rate(
            confirmed, lambda _r: True,
            lambda r: r.get("noninferior_certified_substitution") is True,
            n_boot=n_boot, seed=seed),
        "equivalent_rescue_among_confirmed_gaps": bootstrap_conditional_rate(
            confirmed, lambda _r: True, lambda r: r.get("certified_substitution") is True,
            n_boot=n_boot, seed=seed + 1),
        "equivalent_rescue_among_point_gaps": bootstrap_conditional_rate(
            point_gaps, lambda _r: True, lambda r: r.get("certified_substitution") is True,
            n_boot=n_boot, seed=seed + 2),
        "certified_segment_cost": {
            "n": len(segment_costs),
            "mean": _round(np.mean(segment_costs), 3) if segment_costs else None,
            "median": _round(np.median(segment_costs), 3) if segment_costs else None,
            "values": segment_costs,
            "interpretation": ("minimal development-selected k only when selection_policy is "
                               "minimal_cost_noninferior; test split supplies confirmation"),
        },
    }


def _load_grid_reader(grid_dir: str, tag: str, grid_glob: str = "grid_*.npz"
                      ) -> tuple[dict[int, dict[str, np.ndarray]], str]:
    matches = []
    for path in sorted(glob.glob(os.path.join(grid_dir, grid_glob))):
        z = np.load(path, allow_pickle=True)
        reader = str(z["reader"]) if "reader" in z.files else ""
        file_tag = os.path.basename(path)[5:-4]
        if tag in {file_tag, os.path.basename(reader.rstrip("/")), reader}:
            matches.append((path, z))
    if len(matches) != 1:
        raise ValueError(f"reader tag {tag!r} matched {len(matches)} grid files in {grid_dir}")
    path, z = matches[0]
    scores = np.asarray(z["scores"], float)
    meta = [json.loads(s) for s in z["meta"]]
    out = {}
    for gi in sorted({int(m["gi"]) for m in meta}):
        out[gi] = {}
        for rung in sorted({m["rung"] for m in meta if int(m["gi"]) == gi}):
            idx = [i for i, m in enumerate(meta) if int(m["gi"]) == gi and m["rung"] == rung]
            out[gi][rung] = np.nan_to_num(np.nanmean(scores[idx], axis=0), nan=0.5)
    return out, path


def analyze_raw_grid(*, grid_dir: str, ref_dir: str, small_tag: str, big_tag: str,
                     rungs: list[str], sparse: str, placebo_rungs: list[str], delta: float,
                     floor: float, train_frac: float, seed: int, n_boot: int,
                     min_signature_rho: float, grid_glob: str = "grid_*.npz",
                     messages_file: str = "messages.json",
                     selection_policy: str = "best_auc", min_test_class_n: int = 10,
                     preregistered_protocol: bool = False,
                     independent_target: bool = False,
                     target_description: str = "",
                     articulations_certified: bool = False,
                     articulation_description: str = "") -> dict:
    small, spath = _load_grid_reader(grid_dir, small_tag, grid_glob)
    big, bpath = _load_grid_reader(grid_dir, big_tag, grid_glob)
    source_protocols = {}
    grid_probe_hashes = {}
    for side, path in (("small", spath), ("big", bpath)):
        z_protocol = np.load(path, allow_pickle=True)
        source_protocols[side] = (str(z_protocol["protocol_schema"])
                                  if "protocol_schema" in z_protocol.files else None)
        grid_probe_hashes[side] = (np.asarray(z_protocol["probe_sha256"]).astype(str)
                                   if "probe_sha256" in z_protocol.files else None)
        z_protocol.close()
    both_grid_hashes = all(v is not None for v in grid_probe_hashes.values())
    if both_grid_hashes and not np.array_equal(grid_probe_hashes["small"],
                                               grid_probe_hashes["big"]):
        raise ValueError("small and big reader tensors contain different probe hashes")
    grid_alignment_verified = bool(both_grid_hashes)
    if placebo_rungs and any(v != "address_segment_grid/v2_form_matched"
                             for v in source_protocols.values()):
        raise ValueError("placebo certification requires v2 form-matched source grids; "
                         f"got {source_protocols}")
    msgs_path = os.path.join(grid_dir, messages_file)
    msgs = json.load(open(msgs_path))
    ckpts = _ckpts(ref_dir, None)
    rows = []
    ref_inputs = []
    for gi in sorted(set(small) & set(big) & set(ckpts)):
        _, ref_path = ckpts[gi]
        z = np.load(ref_path, allow_pickle=True)
        m_i = np.nan_to_num(np.asarray(z["M_i"], float), nan=0.5)
        ref_probe_hashes = (np.asarray(z["probe_sha256"]).astype(str)
                            if "probe_sha256" in z.files else None)
        ref_alignment_verified = bool(grid_alignment_verified and ref_probe_hashes is not None
                                      and np.array_equal(grid_probe_hashes["small"],
                                                         ref_probe_hashes))
        if grid_alignment_verified and ref_probe_hashes is not None and not ref_alignment_verified:
            raise ValueError(f"reference and reader tensors contain different probes for gi={gi}")
        if str(gi) not in msgs:
            continue
        mask = np.ones(len(m_i), bool)
        ex = msgs[str(gi)].get("exemplar_idx") or {}
        mask[(ex.get("pos") or []) + (ex.get("neg") or [])] = False
        if any(len(v) != len(m_i) for v in list(small[gi].values()) + list(big[gi].values())):
            raise ValueError(f"probe length mismatch for gi={gi}")
        try:
            res = crossfit_metric({k: v[mask] for k, v in small[gi].items()},
                                  {k: v[mask] for k, v in big[gi].items()}, m_i[mask] > 0.5,
                                  rungs=rungs, sparse=sparse, placebo_rungs=placebo_rungs,
                                  delta=delta, floor=floor, train_frac=train_frac,
                                  seed=seed + gi, n_boot=n_boot,
                                  min_signature_rho=min_signature_rho,
                                  selection_policy=selection_policy,
                                  min_test_class_n=min_test_class_n)
            rows.append({"gi": gi, "name": msgs[str(gi)].get("name"), **res})
        except ValueError as exc:
            rows.append({"gi": gi, "name": msgs[str(gi)].get("name"), "error": str(exc)})
        ref_inputs.append({"path": ref_path, "sha256": sha256_file(ref_path),
                           "probe_alignment_verified": ref_alignment_verified})
    summary = summarize_crossfit_records(rows, n_boot=n_boot, seed=seed)
    evaluable = [r for r in rows if "error" not in r]
    refs_alignment_verified = bool(ref_inputs and all(r["probe_alignment_verified"]
                                                       for r in ref_inputs))
    paper_grade_eligible = bool(preregistered_protocol and independent_target
                                and articulations_certified
                                and grid_alignment_verified and refs_alignment_verified)
    return {
        "schema": "scale_articulation_substitution/v1", "mode": "crossfit_raw_fixed_target",
        "analysis_status": ("confirmatory_protocol_asserted" if preregistered_protocol else
                            "exploratory_posthoc"),
        "claim_scope": ("paper-grade eligibility additionally requires a genuinely independent "
                        "target and an auditable preregistration"),
        "paper_grade_claim_eligible": paper_grade_eligible,
        "target_provenance": {"independent_of_readers_asserted": independent_target,
                              "description": target_description},
        "articulation_provenance": {
            "same_construct_knowledge_certified": articulations_certified,
            "description": articulation_description,
            "scope": ("definitions, explanations, rules, mechanisms, or ostensive contrasts that "
                      "externalize the target construct; generic prompt content is not treatment"),
        },
        "source_protocols": source_protocols,
        "probe_alignment": {
            "reader_to_reader_verified": grid_alignment_verified,
            "all_references_verified": refs_alignment_verified,
            "legacy_missing_hashes": not (grid_alignment_verified and refs_alignment_verified),
        },
        "config": {"grid_dir": grid_dir, "ref_dir": ref_dir, "grid_glob": grid_glob,
                   "messages_file": messages_file, "small_tag": small_tag,
                   "big_tag": big_tag, "rungs": rungs, "sparse_rung": sparse,
                   "placebo_rungs": placebo_rungs, "delta": delta, "floor": floor,
                   "train_frac": train_frac, "seed": seed, "n_boot": n_boot,
                   "min_signature_rho": min_signature_rho,
                   "selection_policy": selection_policy,
                   "min_test_class_n": min_test_class_n,
                   "preregistered_protocol": preregistered_protocol,
                   "independent_target": independent_target,
                   "articulations_certified": articulations_certified},
        "inputs": {"small_grid": {"path": spath, "sha256": sha256_file(spath)},
                   "big_grid": {"path": bpath, "sha256": sha256_file(bpath)},
                   "messages": {"path": msgs_path, "sha256": sha256_file(msgs_path)},
                   "references": ref_inputs},
        "n_metrics": len(rows), "summary": summary,
        "counts": {"baseline_gap_confirmed": sum(r["baseline_gap_confirmed"] for r in evaluable),
                   "methodological_substitution": sum(
                       r["methodological_substitution_without_specificity_gate"] for r in evaluable),
                   "equivalent_methodological_substitution": sum(
                       r["equivalent_methodological_substitution_without_specificity_gate"]
                       for r in evaluable),
                   "noninferior_certified_substitution": sum(
                       r["noninferior_certified_substitution"] is True for r in evaluable),
                   "certified_substitution": sum(r["certified_substitution"] is True
                                                 for r in evaluable),
                   "specificity_unavailable": sum(r["certified_substitution"] is None
                                                   for r in evaluable)},
        "per_metric": rows, "generated_utc": datetime.now(timezone.utc).isoformat(),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    rp = sub.add_parser("report", help="descriptive re-read of existing auc_report.json files")
    rp.add_argument("--data-dir", default=DATA)
    rp.add_argument("--domains", required=True)
    rp.add_argument("--small-reader", required=True)
    rp.add_argument("--big-reader", required=True)
    cp = sub.add_parser("crossfit", help="held-out analysis from raw grid tensors")
    cp.add_argument("--grid-dir", required=True)
    cp.add_argument("--ref-dir", required=True)
    cp.add_argument("--small-tag", required=True)
    cp.add_argument("--big-tag", required=True)
    cp.add_argument("--grid-glob", default="grid_*.npz",
                    help="use unitgrid_*.npz for address-segment runs")
    cp.add_argument("--messages-file", default="messages.json",
                    help="use unit_messages.json for address-segment runs")
    cp.add_argument("--placebo-rungs", default="")
    cp.add_argument("--train-frac", type=float, default=0.5)
    cp.add_argument("--min-signature-rho", type=float, default=0.5)
    cp.add_argument("--selection-policy", default="best_auc",
                    choices=["best_auc", "minimal_cost_noninferior"],
                    help="use minimal_cost_noninferior for ordered address-segment rungs")
    cp.add_argument("--min-test-class-n", type=int, default=10)
    cp.add_argument("--preregistered-protocol", action="store_true",
                    help="assert that arms, margins, split, and exclusions were frozen pre-test")
    cp.add_argument("--independent-target", action="store_true",
                    help="assert the fixed target contains none of the readers under comparison")
    cp.add_argument("--target-description", default="")
    cp.add_argument("--articulations-certified", action="store_true",
                    help="assert arms externalize the same construct knowledge and were frozen "
                         "independently of test performance")
    cp.add_argument("--articulation-description", default="")
    for p in (rp, cp):
        p.add_argument("--rungs", default=",".join(DEFAULT_RUNGS))
        p.add_argument("--sparse-rung", default="name")
        p.add_argument("--delta", type=float, default=0.02)
        p.add_argument("--floor", type=float, default=0.55)
        p.add_argument("--n-boot", type=int, default=2000)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--out", required=True)
    a = ap.parse_args()
    rungs = [r.strip() for r in a.rungs.split(",") if r.strip()]
    if a.mode == "report":
        out = analyze_reports(data_dir=a.data_dir,
                              domains=[d.strip() for d in a.domains.split(",") if d.strip()],
                              small_reader=a.small_reader, big_reader=a.big_reader, rungs=rungs,
                              sparse=a.sparse_rung, delta=a.delta, floor=a.floor,
                              n_boot=a.n_boot, seed=a.seed)
    else:
        out = analyze_raw_grid(grid_dir=a.grid_dir, ref_dir=a.ref_dir,
                               small_tag=a.small_tag, big_tag=a.big_tag, rungs=rungs,
                               sparse=a.sparse_rung,
                               placebo_rungs=[r for r in a.placebo_rungs.split(",") if r],
                               delta=a.delta, floor=a.floor, train_frac=a.train_frac,
                               seed=a.seed, n_boot=a.n_boot,
                               min_signature_rho=a.min_signature_rho,
                               grid_glob=a.grid_glob, messages_file=a.messages_file,
                               selection_policy=a.selection_policy,
                               min_test_class_n=a.min_test_class_n,
                               preregistered_protocol=a.preregistered_protocol,
                               independent_target=a.independent_target,
                               target_description=a.target_description,
                               articulations_certified=a.articulations_certified,
                               articulation_description=a.articulation_description)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
