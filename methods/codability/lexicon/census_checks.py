"""Unnamed-rate robustness checks for the author-lexicon census — partition-agnostic.

Three guards on mean_unnamed_rate (2026-07-09 verification; all four domains PASSED — the
unnamed-rate finding is robust, unlike the unguarded dialect contrast, see dialect.py):
  size bins        mean unnamed_rate by construct n_sources bin (size-confound check;
                   humor >= .64 in every bin vs .49-.60 elsewhere, median size 3 all domains)
  junk-excluded    drop junk_doc-bucket sources before the census (moved +-.01)
  mirror-collapsed union-find sources whose quotes share >= .5 token-Jaccard (mirrored
                   canonical texts), one naming vote per component (moved +-.01)

Rerun after any L0->R3 rebuild with the new partition:
  python -m methods.codability.lexicon.census_checks <task> --partition <key->construct json>
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np

from . import census as cz
from .dialect import OUT, bucket_of, load_groups, qtok

BINS = {"2": (2, 2), "3-4": (3, 4), "5-9": (5, 9), "10+": (10, 10 ** 9)}


def unnamed_by_size(task: str, groups: dict) -> dict:
    by_bin = {b: [] for b in BINS}
    sizes = []
    for by_src in groups.values():
        c = cz.concept_census(list(by_src.values()))
        if c is None or c["n_sources"] < 2:
            continue
        sizes.append(c["n_sources"])
        for b, (lo, hi) in BINS.items():
            if lo <= c["n_sources"] <= hi:
                by_bin[b].append(c["unnamed_rate"])
    return {"median_size": float(np.median(sizes)) if sizes else None,
            "bins": {b: (round(float(np.mean(v)), 3), len(v)) for b, v in by_bin.items() if v}}


def unnamed_junk_excluded(task: str, groups: dict) -> tuple:
    rates = []
    for by_src in groups.values():
        rows = [r for r in by_src.values() if bucket_of(
            task, (r.get("strata") or {}).get("subtask_short") or "") != "junk_doc"]
        c = cz.concept_census(rows)
        if c and c["n_sources"] >= 2:
            rates.append(c["unnamed_rate"])
    return (round(float(np.mean(rates)), 3) if rates else None, len(rates))


def unnamed_mirror_collapsed(task: str, groups: dict, thresh: float = .5) -> tuple:
    """Collapse quote-mirror source groups to one naming vote (named if ANY member names)."""
    rates, n_collapsed = [], 0
    for by_src in groups.values():
        rows = list(by_src.values())
        n = len(rows)
        if n < 2:
            continue
        qs = [qtok(r.get("quote")) for r in rows]
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for i in range(n):
            for j in range(i + 1, n):
                if qs[i] and qs[j] and len(qs[i] & qs[j]) / len(qs[i] | qs[j]) >= thresh:
                    parent[find(i)] = find(j)
        comps = defaultdict(list)
        for i in range(n):
            comps[find(i)].append(rows[i])
        n_collapsed += n - len(comps)
        if len(comps) < 2:
            continue
        named = sum(1 for members in comps.values()
                    if any(m.get("named_in_source") and m.get("head_term") for m in members))
        rates.append(1.0 - named / len(comps))
    return (round(float(np.mean(rates)), 3) if rates else None, len(rates), n_collapsed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks", help="comma-sep task names")
    ap.add_argument("--partition", default=None,
                    help="key->construct json; default outputs/lexicon/partition_<task>.json")
    args = ap.parse_args()
    for task in args.tasks.split(","):
        task = task.strip()
        pp = args.partition or os.path.join(OUT, f"partition_{task}.json")
        groups = load_groups(task, pp)
        s = unnamed_by_size(task, groups)
        j = unnamed_junk_excluded(task, groups)
        m = unnamed_mirror_collapsed(task, groups)
        print(f"\n===== {task}  partition={os.path.basename(pp)} =====")
        print(f"  unnamed by size bin: {s['bins']}  (median size {s['median_size']})")
        print(f"  junk-excluded:       {j[0]} (n={j[1]})")
        print(f"  mirror-collapsed:    {m[0]} (n={m[1]}, sources collapsed {m[2]})")


if __name__ == "__main__":
    main()
