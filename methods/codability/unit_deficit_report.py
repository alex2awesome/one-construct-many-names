#!/usr/bin/env python
"""How much more target-construct knowledge must be made explicit for a smaller reader?

Consumes unit_auc_report.json (from unit_count_grid.py) per domain. Per (reader, metric) the
u-rung curve AUC(k) is made monotone with a best-so-far envelope, then

  k*(reader, tau)          = min k with envelope(k) >= tau           (right-censored if never)
  Delta-k(small, big; tau) = k*(small) - k*(big)                     (both reached)
  local overlap shift      = mean over a tau-grid [0.52 .. min(max_small, max_big)] of
                             k_small(tau) - k_big(tau)

The local overlap shift is descriptive and is NOT a replacement rate: it excludes the portion of
the larger reader's range that the smaller reader never reaches and includes metrics with no
sparse-prompt gap. The stronger readout therefore also baseline-gates every metric and reports the
fraction of genuine big-name > small-name gaps actually rescued by a richer small-reader prompt.

Filler control (fk rungs, CUF inert filler length-matched to uk): content_minus_filler(k) =
AUC(uk) - AUC(fk). If filler alone reproduced the gains, the "units" would be prompt-length,
not content — report both. Aggregation gives each metric equal weight. A form-count mismatch marks
legacy artifacts invalid for this control and requires rescoring with address_segment_grid/v2.

Caveats carried from the line: 8B reader == executor (SELF-recovery, flagged); AUC readout
threshold-free; these are uncertified address segments, not CUF Omega units; ordinal claims only;
metric-level bootstrap CIs (probe-clustered CIs would
need per-probe scores — not retained in the AUC report).

Usage: python -m methods.codability.unit_deficit_report --domains humor,math
Writes <DATA>/unit_deficit_report.json.
"""
import argparse
import json
import os

import numpy as np

from methods.codability.name_sufficiency import DATA, DOMAINS, load_tags

READERS = {"Llama-3.2-1B-Instruct": "1B", "Llama-3.2-3B-Instruct": "3B",
           "Llama-3.1-8B-Instruct": "8B",
           "fde04ee76a27704c88f569542ef023b57d4d0362": "70B"}
PAIRS = [("1B", "3B"), ("1B", "8B"), ("3B", "8B"), ("1B", "70B"), ("3B", "70B"), ("8B", "70B")]
TAUS = [0.55, 0.60, 0.65]
SELF_READER = "8B"


def curves(per_gi):
    """gi entry -> (ks, envelope aucs, raw aucs, filler aucs by k)."""
    rungs = per_gi["rungs"]
    n = per_gi.get("n_segments", per_gi["n_units"])
    ks = list(range(0, n + 1))
    raw = [rungs.get(f"u{k}", {}).get("auc") for k in ks]
    fill = {k: rungs.get(f"f{k}", {}).get("auc") for k in range(1, n + 1)}
    env, best = [], None
    for v in raw:
        if v is not None:
            best = v if best is None else max(best, v)
        env.append(best)
    return ks, env, raw, fill


def k_star(ks, env, tau):
    for k, v in zip(ks, env):
        if v is not None and v >= tau:
            return k
    return None


def h_shift(ks_s, env_s, ks_b, env_b):
    """Mean horizontal displacement small-vs-big over the shared reachable AUC range."""
    max_s = max((v for v in env_s if v is not None), default=None)
    max_b = max((v for v in env_b if v is not None), default=None)
    if max_s is None or max_b is None:
        return None
    hi = min(max_s, max_b)
    if hi <= 0.52:
        return None
    taus = np.linspace(0.52, hi, 25)
    ds = [k_star(ks_s, env_s, t) - k_star(ks_b, env_b, t) for t in taus]
    return float(np.mean(ds))


def boot_ci(vals, n=2000, seed=0):
    vals = [v for v in vals if v is not None]
    if len(vals) < 3:
        return None
    rng = np.random.default_rng(seed)
    arr = np.array(vals, float)
    means = [float(np.mean(rng.choice(arr, len(arr)))) for _ in range(n)]
    return [round(float(np.percentile(means, 2.5)), 3), round(float(np.percentile(means, 97.5)), 3)]


def direct_rescue(curve_small, curve_big, *, delta=0.02, floor=0.55):
    """Point classification for the actual substitution claim (descriptive same-probe selection)."""
    _ks, _env_s, raw_s, _fill_s = curve_small
    _kb, _env_b, raw_b, _fill_b = curve_big
    finite_s = [v for v in raw_s if v is not None]
    finite_b = [v for v in raw_b if v is not None]
    s0 = raw_s[0] if raw_s else None
    b0 = raw_b[0] if raw_b else None
    if s0 is None or b0 is None or not finite_s or not finite_b:
        return {"status": "missing"}
    sbest, bbest = max(finite_s), max(finite_b)
    if b0 < floor:
        return {"status": "big_sparse_below_floor", "small_sparse": s0, "big_sparse": b0}
    gap = b0 - s0
    full_gap = bbest - s0
    gap_present = gap > delta
    full_gap_present = full_gap > delta
    return {
        "status": "gap" if gap_present else "no_baseline_gap",
        "small_sparse": round(float(s0), 4), "big_sparse": round(float(b0), 4),
        "small_best": round(float(sbest), 4), "big_best": round(float(bbest), 4),
        "baseline_gap": round(float(gap), 4), "baseline_gap_present": bool(gap_present),
        "small_gain": round(float(sbest - s0), 4),
        "rescue_big_sparse": bool(gap_present and sbest >= b0 - delta),
        "equivalent_rescue_big_sparse": bool(gap_present and abs(sbest - b0) <= delta),
        "full_baseline_gap": round(float(full_gap), 4),
        "full_baseline_gap_present": bool(full_gap_present),
        "rescue_big_best": bool(full_gap_present and sbest >= bbest - delta),
    }


def rate_summary(rows, denominator, success):
    eligible = [r for r in rows if denominator(r)]
    hits = sum(bool(success(r)) for r in eligible)
    vals = [float(bool(success(r))) for r in eligible]
    return {"success": hits, "n": len(eligible),
            "rate": round(hits / len(eligible), 4) if eligible else None,
            "CI95_metric_bootstrap": boot_ci(vals) if vals else None}


def summarize_direct_rescue(rows):
    evaluable = [r for r in rows if r.get("status") in {"gap", "no_baseline_gap"}]
    return {
        "n_evaluable": len(evaluable),
        "n_missing": sum(r.get("status") == "missing" for r in rows),
        "n_big_sparse_below_floor": sum(r.get("status") == "big_sparse_below_floor"
                                          for r in rows),
        "no_baseline_gap": rate_summary(evaluable, lambda _r: True,
                                         lambda r: not r["baseline_gap_present"]),
        "rescue_big_sparse_among_gaps": rate_summary(
            evaluable, lambda r: r["baseline_gap_present"], lambda r: r["rescue_big_sparse"]),
        "equivalent_rescue_big_sparse_among_gaps": rate_summary(
            evaluable, lambda r: r["baseline_gap_present"],
            lambda r: r["equivalent_rescue_big_sparse"]),
        "rescue_big_best_among_full_gaps": rate_summary(
            evaluable, lambda r: r["full_baseline_gap_present"],
            lambda r: r["rescue_big_best"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", default="humor,math")
    ap.add_argument("--delta", type=float, default=0.02)
    ap.add_argument("--floor", type=float, default=0.55)
    ap.add_argument("--out", default=os.path.join(DATA, "unit_deficit_report.json"))
    a = ap.parse_args()
    tags = load_tags()

    out = {"schema": "address_segment_deficit/v2_baseline_gated",
        "quantity": "uncertified address segments (not certified CUF Omega units)",
        "treatment": "explicit articulation of the target construct's tacit knowledge",
           "selection_scope": "descriptive: best k selected and evaluated on the same probes",
           "local_overlap_shift_warning": "not a replacement rate; ignores unreachable range and "
                                          "does not baseline-gate",
           "self_reader": SELF_READER, "taus": TAUS, "delta": a.delta, "floor": a.floor,
           "per_domain": {}}
    for short in [d.strip() for d in a.domains.split(",") if d.strip()]:
        gdir, aliases, _cls = DOMAINS[short]
        p = os.path.join(DATA, os.path.dirname(gdir), "unitgrid_v1", "unit_auc_report.json")
        if not os.path.exists(p):
            print(f"{short}: no unit_auc_report.json at {p} — skipped")
            continue
        rep = json.load(open(p))
        by_size = {}
        for tag, per in rep.items():
            size = READERS.get(tag)
            if size:
                by_size[size] = per
        gis = sorted({gi for per in by_size.values() for gi in per}, key=int)
        dom = {"n_metrics": len(gis), "sizes": sorted(by_size), "per_metric": [],
               "pairs": {}, "filler": {}}

        percurve = {}
        for size, per in by_size.items():
            for gi in gis:
                if gi in per:
                    percurve[(size, gi)] = curves(per[gi])

        for gi in gis:
            tag = None
            for al in aliases:
                tag = tag or tags.get((al, int(gi)))
            any_per = next(per[gi] for per in by_size.values() if gi in per)
            row = {"gi": int(gi), "tag": tag,
                   "n_segments": any_per.get("n_segments", any_per["n_units"]),
                   "segment_srcs": any_per.get("segment_srcs", any_per["unit_srcs"]),
                   "k_star": {}, "auc_max": {}}
            for size in by_size:
                if (size, gi) not in percurve:
                    continue
                ks, env, _raw, _fill = percurve[(size, gi)]
                row["k_star"][size] = {str(t): k_star(ks, env, t) for t in TAUS}
                row["auc_max"][size] = max((v for v in env if v is not None), default=None)
            dom["per_metric"].append(row)

        for small, big in PAIRS:
            if small not in by_size or big not in by_size:
                continue
            pk = {"self_flag": SELF_READER in (small, big)}
            for t in TAUS:
                dks, cens_small, cens_big = [], 0, 0
                for gi in gis:
                    cs, cb = percurve.get((small, gi)), percurve.get((big, gi))
                    if not cs or not cb:
                        continue
                    k_s, k_b = k_star(cs[0], cs[1], t), k_star(cb[0], cb[1], t)
                    if k_s is None and k_b is not None:
                        cens_small += 1
                    elif k_b is None and k_s is not None:
                        cens_big += 1
                    elif k_s is not None and k_b is not None:
                        dks.append(k_s - k_b)
                pk[str(t)] = {"mean_dk": round(float(np.mean(dks)), 3) if dks else None,
                              "median_dk": float(np.median(dks)) if dks else None,
                              "ci": boot_ci(dks), "n": len(dks),
                              "small_never_reaches": cens_small,
                              "big_never_reaches": cens_big}
            shifts, by_tag, direct_rows = [], {}, []
            for gi in gis:
                cs, cb = percurve.get((small, gi)), percurve.get((big, gi))
                if not cs or not cb:
                    continue
                s = h_shift(cs[0], cs[1], cb[0], cb[1])
                if s is not None:
                    shifts.append(s)
                    tg = next((r["tag"] for r in dom["per_metric"] if str(r["gi"]) == gi), None)
                    if tg:
                        by_tag.setdefault(tg, []).append(s)
                dr = direct_rescue(cs, cb, delta=a.delta, floor=a.floor)
                direct_rows.append({"gi": int(gi), **dr})
            pk["local_overlap_shift_mean"] = round(float(np.mean(shifts)), 3) if shifts else None
            pk["local_overlap_shift_ci"] = boot_ci(shifts)
            pk["local_overlap_shift_n"] = len(shifts)
            pk["local_overlap_shift_by_tag"] = {
                t: {"mean": round(float(np.mean(v)), 3), "n": len(v)}
                for t, v in sorted(by_tag.items())}
            # Legacy aliases retained so old notebooks fail softly; interpretation is superseded.
            pk["h_shift_mean_legacy"] = pk["local_overlap_shift_mean"]
            pk["direct_substitution"] = summarize_direct_rescue(direct_rows)
            pk["direct_per_metric"] = direct_rows
            dom["pairs"][f"{small}->{big}"] = pk

        for size in by_size:
            cmf, fgain, form_mismatch = [], [], []
            for gi in gis:
                c = percurve.get((size, gi))
                if not c:
                    continue
                ks, _env, raw, fill = c
                u0 = raw[0]
                cmf_gi, fgain_gi = [], []
                per_entry = by_size[size].get(gi, {})
                for k in range(1, len(ks)):
                    if raw[k] is not None and fill.get(k) is not None:
                        cmf_gi.append(raw[k] - fill[k])
                        ru = per_entry.get("rungs", {}).get(f"u{k}", {})
                        rf = per_entry.get("rungs", {}).get(f"f{k}", {})
                        if ru.get("n_forms") != rf.get("n_forms"):
                            form_mismatch.append({"gi": int(gi), "k": k,
                                                  "u_n_forms": ru.get("n_forms"),
                                                  "f_n_forms": rf.get("n_forms")})
                    if fill.get(k) is not None and u0 is not None:
                        fgain_gi.append(fill[k] - u0)
                if cmf_gi:
                    cmf.append(float(np.mean(cmf_gi)))
                if fgain_gi:
                    fgain.append(float(np.mean(fgain_gi)))
            dom["filler"][size] = {
                "articulation_minus_filler_mean": round(float(np.mean(cmf)), 4) if cmf else None,
                "content_minus_filler_mean": round(float(np.mean(cmf)), 4) if cmf else None,
                "filler_minus_name_mean": round(float(np.mean(fgain)), 4) if fgain else None,
                "n_metrics": len(cmf),
                "aggregation": "mean within metric, then equal-weight mean across metrics",
                "form_matched": not form_mismatch,
                "form_mismatch_count": len(form_mismatch),
                "form_mismatch_examples": form_mismatch[:20],
                "validity": ("valid form-matched control" if not form_mismatch else
                             "INVALID legacy control: rescore content and filler with same forms")}
        out["per_domain"][short] = dom

        print(f"\n== {short} ({len(gis)} metrics; sizes {sorted(by_size)}) ==")
        for pair, pk in dom["pairs"].items():
            t = pk[str(TAUS[0])]
            print(f"  {pair}{' [SELF]' if pk['self_flag'] else '':7s} "
                  f"dk@{TAUS[0]}: mean {t['mean_dk']} (n={t['n']}, ci {t['ci']}, "
                  f"small-censored {t['small_never_reaches']}) | "
                  f"local shift {pk['local_overlap_shift_mean']} segments "
                  f"(n={pk['local_overlap_shift_n']}, ci {pk['local_overlap_shift_ci']}) | "
                  f"gap rescue {pk['direct_substitution']['rescue_big_sparse_among_gaps']}")
        for size, f in dom["filler"].items():
            print(f"  filler[{size}]: articulation-filler "
                  f"{f['articulation_minus_filler_mean']} | "
                  f"filler-name {f['filler_minus_name_mean']} "
                  f"(metrics={f['n_metrics']}, form_matched={f['form_matched']})")

    path = a.out
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    json.dump(out, open(path, "w"), indent=1)
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
