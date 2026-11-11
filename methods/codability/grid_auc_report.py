#!/usr/bin/env python
"""Threshold-free readout of decompression-grid scores: AUC + Spearman per (reader, gi, rung).

The grid report's bal_acc thresholds orbit-averaged scores at an ABSOLUTE 0.5, which is a
calibration assumption: a reader whose P(yes) is globally shifted below 0.5 (e.g.
Qwen2.5-3B on math, 2026-07-05: all-negative predictions -> bal_acc exactly 0.5 on all 21
metrics while its 1.5B/7B siblings clear the floor) scores as pure chance even if its score
RANKING carries signal. Cross-family comparisons need a calibration-free instrument, so this
emits rank statistics vs the executor reference:

  auc      — Mann-Whitney AUC of m_bar (orbit-averaged rung score) for ref+ vs ref- probes
  spearman — rank correlation of m_bar with the CONTINUOUS executor M_i

Masking and orbit-averaging mirror run_decompression_grid.report() exactly (exemplar probes
excluded; nanmean over a rung's forms; NaN scores -> 0.5). Numpy-only; runs anywhere.

Usage:
  python grid_auc_report.py --ref-dir <ckpt dir with *_sigs.npz> --grid-dir <dir with grid_*.npz>
Writes <grid-dir>/auc_report.json.
"""
import argparse
import glob
import json
import os
import re

import numpy as np

RUNG_ORDER = ["name", "definition", "explanation", "full_rubric", "exemplars",
              "dossier", "dossier_v2"]


def _ckpts(ref_dir):
    out = {}
    for f in sorted(glob.glob(os.path.join(ref_dir, "*_sigs.npz"))):
        m = re.search(r"_(R[123])_metric(\d+)_sigs\.npz$", os.path.basename(f))
        if m:
            out[int(m.group(2))] = f
    return out


def _rank(a):
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), float)
    ranks[order] = np.arange(1, len(a) + 1)
    # midranks for ties
    uniq, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt)
    mid = csum - (cnt - 1) / 2.0
    return mid[inv]


def auc_mw(scores, labels):
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return None
    r = _rank(scores)
    u = r[labels].sum() - len(pos) * (len(pos) + 1) / 2.0
    return float(u / (len(pos) * len(neg)))


def spearman(a, b):
    ra, rb = _rank(a), _rank(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-dir", required=True)
    ap.add_argument("--grid-dir", required=True)
    a = ap.parse_args()

    ckpts = _ckpts(a.ref_dir)
    msgs = json.load(open(os.path.join(a.grid_dir, "messages.json")))
    refs_bin, refs_cont = {}, {}
    for gi, f in ckpts.items():
        z = np.load(f, allow_pickle=True)
        m_i = np.nan_to_num(np.asarray(z["M_i"], float), nan=0.5)
        refs_bin[gi], refs_cont[gi] = m_i > 0.5, m_i

    out = {}
    for gpath in sorted(glob.glob(os.path.join(a.grid_dir, "grid_*.npz"))):
        z = np.load(gpath, allow_pickle=True)
        scores = np.asarray(z["scores"], float)
        meta = [json.loads(s) for s in z["meta"]]
        tag = os.path.basename(gpath)[5:-4]
        per = {}
        for gi in sorted({x["gi"] for x in meta}):
            if gi not in refs_bin or str(gi) not in msgs:
                continue
            ref, cont = refs_bin[gi], refs_cont[gi]
            ex = msgs[str(gi)]["exemplar_idx"]
            mask = np.ones(len(ref), bool)
            mask[ex["pos"] + ex["neg"]] = False
            per_rung = {}
            for rung in RUNG_ORDER:
                idx = [i for i, x in enumerate(meta) if x["gi"] == gi and x["rung"] == rung]
                if not idx:
                    continue
                m_bar = np.nan_to_num(np.nanmean(scores[idx], axis=0), nan=0.5)
                per_rung[rung] = {
                    "auc": (lambda v: round(v, 4) if v is not None else None)(
                        auc_mw(m_bar[mask], ref[mask])),
                    "spearman": (lambda v: round(v, 4) if v is not None else None)(
                        spearman(m_bar[mask], cont[mask])),
                    "n_forms": len(idx),
                    "n_pos": int(ref[mask].sum()), "n_neg": int((~ref[mask]).sum()),
                }
            per[str(gi)] = per_rung
        out[tag] = per
        aucs = [v["auc"] for p in per.values() for v in p.values() if v["auc"] is not None]
        print(f"{tag}: {len(per)} metrics, mean rung AUC "
              f"{np.mean(aucs):.3f}" if aucs else f"{tag}: no scorable rungs")
    path = os.path.join(a.grid_dir, "auc_report.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
