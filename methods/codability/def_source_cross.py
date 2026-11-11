#!/usr/bin/env python
"""T6a definition-source cross: does definition AUTHORSHIP (writer dialect) drive the
family-specific cells?

Mechanism under test: the grid's definitions were written by the A-side (Llama-8B) apparatus;
a stranger might parse A-phrased definitions worse, faking B-only cells (and deflating the
stranger generally — conservative for +DiD, but cell-corrupting).

Phase write: each writer model paraphrases the target definitions (meaning-preserving,
similar length) -> def_sources.json {gi: {orig: ..., <writer>: ...}}.
Phase score: each reader scores every source's definition -> AUC vs refs.
Analysis (CPU, post-hoc): reader-family x def-source interaction = the dialect effect.
Targets: the family-specific cells (CW 11 + math 2), passed via --gi-list.
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

PARA_TMPL = ("Rewrite the following evaluation-criterion definition in your own words. "
             "Preserve the exact meaning and scope; keep a similar length; do not add or "
             "drop requirements; output ONLY the rewritten definition.\n\nDEFINITION:\n{d}")


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
    p.add_argument("--phase", required=True, choices=["write", "score"])
    p.add_argument("--model", required=True, help="writer (phase=write) or reader (phase=score)")
    p.add_argument("--msgs", required=True)
    p.add_argument("--ref-dir", required=True)
    p.add_argument("--sources", required=True, help="def_sources.json path (written/read)")
    p.add_argument("--out", default="", help="scores json (phase=score)")
    p.add_argument("--gi-list", required=True)
    p.add_argument("--n-probes", type=int, default=300)
    p.add_argument("--gepa-reserve", type=int, default=60)
    a = p.parse_args()

    cfg = cfgmod.apply_task_preset(cfgmod.ImplementerConfig(), a.task)
    msgs = json.load(open(a.msgs))
    keep = [g.strip() for g in a.gi_list.split(",") if g.strip()]
    short = a.model.split("/")[-1]

    if a.phase == "write":
        writer = make_judge_backend(a.model, cfg, temperature=None)
        sources = json.load(open(a.sources)) if os.path.exists(a.sources) else {}
        todo = [gi for gi in keep if gi in msgs and short not in sources.get(gi, {})]
        prompts = [PARA_TMPL.format(d=msgs[gi]["rungs"]["definition"]) for gi in todo]
        outs = writer.generate_batch(prompts, max_tokens=260, temperature=0.0)
        for gi, o in zip(todo, outs):
            e = sources.setdefault(gi, {"orig": msgs[gi]["rungs"]["definition"]})
            e[short] = (o or "").strip()
        json.dump(sources, open(a.sources, "w"), indent=1)
        print(f"[write:{short}] {len(todo)} paraphrases -> {a.sources}")
        return

    texts, _ = _load_texts(a.task, a.gepa_reserve + a.n_probes, cfg)
    probes = texts[a.gepa_reserve: a.gepa_reserve + a.n_probes]
    refs = _load_refs(a.ref_dir)
    sources = json.load(open(a.sources))
    report = json.load(open(a.out)) if os.path.exists(a.out) else {}
    rep = report.setdefault(a.model, {})
    executor = make_judge_backend(a.model, cfg, temperature=None)
    for gi in keep:
        if gi not in sources or gi not in refs or gi in rep:
            continue
        mask = np.ones(len(refs[gi]), bool)
        ex = msgs[gi].get("exemplar_idx") or {}
        mask[(ex.get("pos") or []) + (ex.get("neg") or [])] = False
        real = refs[gi][mask] > 0.5
        row = {}
        for src, txt in sources[gi].items():
            if not txt:
                continue
            sig = np.nan_to_num(np.asarray(ap.signature(executor, txt, probes,
                                                         cfg.max_text_chars), float), nan=0.5)[mask]
            v = auc_mw(sig, real)
            row[src] = round(v, 4) if v is not None else None
        rep[gi] = row
        json.dump(report, open(a.out, "w"), indent=1)
        print(f"  {gi}: {row}")
    print(f"[score:{short}] done -> {a.out}")


if __name__ == "__main__":
    main()
