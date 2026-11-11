#!/usr/bin/env python
"""T2a scaffold orbit: re-score name/definition rungs under 3 harness scaffolds.

Mechanism under test: paraphrase/format brittleness of the JUDGING HARNESS (not the rung
text) — if cross-family conclusions flip with the scaffold, they were format artifacts.
Scaffolds:
  default    — the grid's rubric-first _YESNO_TEMPLATE (instrument-identical baseline)
  textfirst  — text presented before the criterion (order swap)
  verbose    — elaborated judge framing (persona + care instructions), same slots
Decision rule (notes PART II): scaffold-share < rung-share of variance AND DiD sign stable
across scaffolds.

One reader per process; --gi-list is the fixed 12-metric subset chosen by 3B name-deficit
stratification (4 low + 4 high + 4 random, seed 0).
"""
import argparse
import glob
import json
import os
import re

import numpy as np

from methods.metric_implementer import config as cfgmod
from methods.metric_implementer.vllm_backend import make_judge_backend
from methods.metric_implementer.experiments import alpha_probe as ap
from methods.metric_implementer.experiments.run_real_test import _load_texts
from methods.metric_implementer.recon_channel import _YESNO_TEMPLATE

TEXTFIRST = ("TEXT:\n{text}\n\nCRITERION:\n{rubric}\n\n"
             "Does the TEXT satisfy the CRITERION? Answer with exactly one word, YES or NO.")
VERBOSE = ("You are a careful, experienced judge of writing quality. Your task is to decide "
           "whether a text satisfies a specific evaluative criterion. Read both carefully, "
           "weigh the evidence, and commit to a single verdict.\n\nCRITERION:\n{rubric}\n\n"
           "TEXT:\n{text}\n\nConsidering everything above, does the text satisfy the "
           "criterion? Respond with exactly one word, YES or NO.")
SCAFFOLDS = {"default": None, "textfirst": TEXTFIRST, "verbose": VERBOSE}


def _rank(a):
    u, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    return (np.cumsum(cnt) - (cnt - 1) / 2.0)[inv]


def auc_mw(scores, labels):
    labels = np.asarray(labels, bool)
    pos = int(labels.sum())
    if pos == 0 or pos == len(labels):
        return None
    r = _rank(np.asarray(scores, float))
    return float((r[labels].sum() - pos * (pos + 1) / 2.0) / (pos * (len(labels) - pos)))


def _load_refs(ref_dir):
    out = {}
    for f in sorted(glob.glob(os.path.join(ref_dir, "*_sigs.npz"))):
        m = re.search(r"_metric(\d+)_sigs\.npz$", os.path.basename(f))
        if m:
            z = np.load(f, allow_pickle=True)
            out[m.group(1)] = np.nan_to_num(np.asarray(z["M_i"], float), nan=0.5)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--reader", required=True)
    p.add_argument("--ref-dir", required=True)
    p.add_argument("--msgs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--gi-list", required=True)
    p.add_argument("--n-probes", type=int, default=300)
    p.add_argument("--gepa-reserve", type=int, default=60)
    a = p.parse_args()

    cfg = cfgmod.apply_task_preset(cfgmod.ImplementerConfig(), a.task)
    texts, _ = _load_texts(a.task, a.gepa_reserve + a.n_probes, cfg)
    probes = texts[a.gepa_reserve: a.gepa_reserve + a.n_probes]
    msgs = json.load(open(a.msgs))
    refs = _load_refs(a.ref_dir)
    keep = [g.strip() for g in a.gi_list.split(",") if g.strip()]

    report = json.load(open(a.out)) if os.path.exists(a.out) else {}
    rep = report.setdefault(a.reader, {})
    executor = make_judge_backend(a.reader, cfg, temperature=None)
    for gi in keep:
        m = msgs.get(gi)
        if m is None or gi not in refs or gi in rep:
            continue
        mask = np.ones(len(refs[gi]), bool)
        ex = m.get("exemplar_idx") or {}
        mask[(ex.get("pos") or []) + (ex.get("neg") or [])] = False
        real = refs[gi][mask] > 0.5
        row = {}
        for rung in ("name", "definition"):
            txt = m.get("rungs", {}).get(rung, "")
            if not txt:
                continue
            for sname, tpl in SCAFFOLDS.items():
                sig = np.nan_to_num(np.asarray(
                    ap.signature(executor, txt, probes, cfg.max_text_chars, template=tpl),
                    float), nan=0.5)[mask]
                v = auc_mw(sig, real)
                row.setdefault(rung, {})[sname] = round(v, 4) if v is not None else None
        rep[gi] = row
        json.dump(report, open(a.out, "w"), indent=1)
        print(f"  {gi}: " + " ".join(f"{r}/{s}={row[r][s]}" for r in row for s in row[r]))
    print(f"[{a.reader}] done -> {a.out}")


if __name__ == "__main__":
    main()
