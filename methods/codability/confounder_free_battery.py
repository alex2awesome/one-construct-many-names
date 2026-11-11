#!/usr/bin/env python
"""Free-tier confounder battery (T3, T5a, T6c, T10) — pure CPU over existing grid npz files.

T3   tie census: score-support degeneracy per reader×rung (ties attenuate AUC silently).
T5a  consensus-extension ref: probes where executor-A (Llama-8B full_rubric) and the biggest
     B reader (Qwen-7B full_rubric) AGREE (both rank-binarized at the ref base rate) — the
     shared extension. Name/def AUCs recomputed on that subset kill the "kin matches the
     executor's private quirks" explanation: an advantage that survives on the SHARED
     extension is lexicalization, not quirk-sharing.
T6c  writer-dialect proxy: per-form AUC dispersion asymmetry (does the stranger vary more
     across paraphrases of the same definition than the kin?).
T10a probe-clustered bootstrap of the tier DiDs (metrics share probes; sign-flip overstates).
T10b split-half (even/odd probes) AUC tables for cell-stability post-processing.

Usage:
  python confounder_free_battery.py --ref-dir <ckpts> --grid-a <llama grid> --grid-b <qwen grid> \
      --out <json>
"""
import argparse
import glob
import json
import os
import re

import numpy as np

ART = ["name", "definition", "explanation", "full_rubric"]


def _ckpts(ref_dir):
    out = {}
    for f in sorted(glob.glob(os.path.join(ref_dir, "*_sigs.npz"))):
        m = re.search(r"_(R[123])_metric(\d+)_sigs\.npz$", os.path.basename(f))
        if m:
            out[int(m.group(2))] = f
    return out


def _rank(a):
    uniq, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    return (np.cumsum(cnt) - (cnt - 1) / 2.0)[inv]


def auc_mw(scores, labels):
    pos = int(labels.sum())
    if pos == 0 or pos == len(labels):
        return None
    r = _rank(scores)
    u = r[labels].sum() - pos * (pos + 1) / 2.0
    return float(u / (pos * (len(labels) - pos)))


def load_grid(gdir):
    """reader -> gi -> rung -> {m_bar (orbit mean), forms (per-form score rows)}"""
    out = {}
    for gpath in sorted(glob.glob(os.path.join(gdir, "grid_*.npz"))):
        z = np.load(gpath, allow_pickle=True)
        scores = np.asarray(z["scores"], float)
        meta = [json.loads(s) for s in z["meta"]]
        tag = os.path.basename(gpath)[5:-4]
        per = {}
        for i, m in enumerate(meta):
            per.setdefault(m["gi"], {}).setdefault(m["rung"], []).append(i)
        out[tag] = {gi: {rung: {"m_bar": np.nan_to_num(np.nanmean(scores[idx], 0), nan=0.5),
                                "forms": np.nan_to_num(scores[idx], nan=0.5)}
                         for rung, idx in rungs.items()}
                    for gi, rungs in per.items()}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref-dir", required=True)
    p.add_argument("--grid-a", required=True, help="kin family grid dir (contains 8B=executor)")
    p.add_argument("--grid-b", required=True, help="stranger family grid dir")
    p.add_argument("--exec-a", default="Llama-3.1-8B-Instruct")
    p.add_argument("--exec-b", default="Qwen2.5-7B-Instruct")
    p.add_argument("--tiers", default="Llama-3.2-1B-Instruct:Qwen2.5-1.5B-Instruct,"
                                      "Llama-3.2-3B-Instruct:Qwen2.5-3B-Instruct,"
                                      "Llama-3.1-8B-Instruct:Qwen2.5-7B-Instruct")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    msgs = json.load(open(os.path.join(a.grid_a, "messages.json")))
    refs, masks = {}, {}
    for gi, f in _ckpts(a.ref_dir).items():
        if str(gi) not in msgs:
            continue
        z = np.load(f, allow_pickle=True)
        m_i = np.nan_to_num(np.asarray(z["M_i"], float), nan=0.5)
        refs[gi] = m_i > 0.5
        ex = msgs[str(gi)]["exemplar_idx"]
        mask = np.ones(len(m_i), bool)
        mask[ex["pos"] + ex["neg"]] = False
        masks[gi] = mask

    A, B = load_grid(a.grid_a), load_grid(a.grid_b)
    out = {"T3_ties": {}, "T5a_consensus": {}, "T6c_formvar": {}, "T10a_boot": {},
           "T10b_halves": {}}

    # T3: tie census
    for fam, G in (("A", A), ("B", B)):
        for reader, per in G.items():
            cells = []
            for gi, rungs in per.items():
                for rung in ART:
                    if rung not in rungs or gi not in masks:
                        continue
                    v = rungs[rung]["m_bar"][masks[gi]]
                    _, cnt = np.unique(v, return_counts=True)
                    cells.append({"gi": gi, "rung": rung,
                                  "distinct_frac": round(len(cnt) / len(v), 3),
                                  "mode_share": round(float(cnt.max() / len(v)), 3)})
            ms = [c["mode_share"] for c in cells]
            out["T3_ties"][reader] = {"mean_mode_share": round(float(np.mean(ms)), 3),
                                      "n_cells_mode>0.5": sum(m > 0.5 for m in ms),
                                      "n_cells": len(cells)}

    # T5a: consensus-extension subset per gi
    cons_masks = {}
    for gi in refs:
        fa = A.get(a.exec_a, {}).get(gi, {}).get("full_rubric")
        fb = B.get(a.exec_b, {}).get(gi, {}).get("full_rubric")
        if fa is None or fb is None:
            continue
        base = refs[gi][masks[gi]].mean()
        def binz(m_bar):
            v = m_bar[masks[gi]]
            thr = np.quantile(v, 1 - base)
            return v > thr
        agree = binz(fa["m_bar"]) == binz(fb["m_bar"])
        cons_masks[gi] = agree
    for fam, G in (("A", A), ("B", B)):
        for reader, per in G.items():
            res = {}
            for gi, agree in cons_masks.items():
                for rung in ("name", "definition"):
                    if rung not in per.get(gi, {}):
                        continue
                    v = per[gi][rung]["m_bar"][masks[gi]][agree]
                    lab = refs[gi][masks[gi]][agree]
                    u = auc_mw(v, lab)
                    if u is not None:
                        res.setdefault(str(gi), {})[rung] = round(u, 4)
            out["T5a_consensus"][reader] = res
    out["T5a_consensus"]["_subset_share"] = {str(gi): round(float(m.mean()), 3)
                                             for gi, m in cons_masks.items()}

    # T6c: per-form AUC dispersion (3-form rungs)
    for fam, G in (("A", A), ("B", B)):
        for reader, per in G.items():
            sds = {}
            for gi, rungs in per.items():
                if gi not in refs:
                    continue
                for rung in ("name", "definition"):
                    fz = rungs.get(rung, {}).get("forms")
                    if fz is None or len(fz) < 3:
                        continue
                    aucs = [auc_mw(f[masks[gi]], refs[gi][masks[gi]]) for f in fz]
                    aucs = [x for x in aucs if x is not None]
                    if len(aucs) >= 3:
                        sds.setdefault(rung, []).append(float(np.std(aucs)))
            out["T6c_formvar"][reader] = {r: round(float(np.mean(v)), 4)
                                          for r, v in sds.items()}

    # T10a: probe-clustered bootstrap of tier DiDs; T10b split-half AUC tables
    rng = np.random.default_rng(0)
    tiers = [t.split(":") for t in a.tiers.split(",")]
    for kin, strg in tiers:
        if kin not in A or strg not in B:
            continue
        gis = [gi for gi in refs if gi in A[kin] and gi in B[strg]
               and all(r in A[kin][gi] and r in B[strg][gi] for r in ("name", "definition"))]
        boots = []
        probe_lengths = {len(refs[gi]) for gi in gis}
        if len(probe_lengths) > 1:
            raise ValueError(f"shared-probe bootstrap requires aligned probe pools; lengths={probe_lengths}")
        n_probe = next(iter(probe_lengths), 0)
        for _ in range(a.n_boot):
            dids = []
            # One shared probe resample for every metric in this replicate. Metrics are measured on
            # the same probe pool; drawing separately by metric destroys that dependence and makes
            # the tier contrast look more stable than it is. Per-metric exemplar masks are applied
            # after the common draw.
            shared_draw = rng.integers(0, n_probe, n_probe)
            for gi in gis:
                bs = shared_draw[masks[gi][shared_draw]]
                if not len(bs):
                    continue
                lab = refs[gi][bs]
                if lab.all() or not lab.any():
                    continue
                vals = {}
                for side, G, rd in (("k", A, kin), ("s", B, strg)):
                    for rung in ("name", "definition"):
                        u = auc_mw(G[rd][gi][rung]["m_bar"][bs], lab)
                        vals[f"{side}_{rung}"] = u
                if None in vals.values():
                    continue
                dids.append((vals["s_definition"] - vals["s_name"])
                            - (vals["k_definition"] - vals["k_name"]))
            if dids:
                boots.append(float(np.mean(dids)))
        b = np.array(boots)
        if not len(b):
            out["T10a_boot"][f"{kin}|{strg}"] = {
                "mean_DiD": None, "CI95": None, "p_two_sided": None,
                "n_metrics": len(gis), "n_boot": 0,
                "resampling_unit": "shared probe coordinate; exemplar masks applied after draw"}
            continue
        lower = int(np.sum(b <= 0))
        upper = int(np.sum(b >= 0))
        p_two = min(1.0, 2 * (min(lower, upper) + 1) / (len(b) + 1))
        out["T10a_boot"][f"{kin}|{strg}"] = {
            "mean_DiD": round(float(b.mean()), 4),
            "CI95": [round(float(np.percentile(b, q)), 4) for q in (2.5, 97.5)],
            "p_two_sided": round(float(p_two), 4),
            "n_metrics": len(gis), "n_boot": len(b),
            "resampling_unit": "shared probe coordinate; exemplar masks applied after draw"}
        halves = {}
        # Randomized once, then shared across metrics; even/odd source order can encode corpus
        # batches or genres and is not a stability split.
        half_perm = np.random.default_rng(1).permutation(n_probe)
        half_sets = {"half_a": set(half_perm[::2]), "half_b": set(half_perm[1::2])}
        for gi in gis:
            mk = np.flatnonzero(masks[gi])
            for hname, chosen in half_sets.items():
                hidx = np.asarray([i for i in mk if i in chosen], int)
                lab = refs[gi][hidx]
                for side, G, rd in (("kin", A, kin), ("str", B, strg)):
                    for rung in ("name", "definition"):
                        u = auc_mw(G[rd][gi][rung]["m_bar"][hidx], lab)
                        halves.setdefault(str(gi), {}).setdefault(hname, {})[
                            f"{side}_{rung}"] = None if u is None else round(u, 4)
        out["T10b_halves"][f"{kin}|{strg}"] = {
            "split_seed": 1, "split_method": "one random partition shared across metrics",
            "metrics": halves}

    json.dump(out, open(a.out, "w"), indent=1)
    print(f"-> {a.out}")
    for k, v in out["T10a_boot"].items():
        print(f"T10a {k}: DiD={v['mean_DiD']} CI{v['CI95']} p={v['p_two_sided']}")


if __name__ == "__main__":
    main()
