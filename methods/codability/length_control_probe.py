#!/usr/bin/env python
"""T4 length controls: padded-name (T4a) + truncated-dossier (T4b).

Mechanism under test: long rungs penalize small readers independent of CONTENT, faking
name-sufficiency at small N. Controls:
  padded_name    — the bare name + semantically inert filler cycled to the definition's char
                   length. If padded_name ~= name (not ~= definition), the rung effect is
                   content, not length.
  trunc_dossier  — the dossier hard-cut to the definition's length; symmetric check
                   (dossier's extra value is content, not room).
Decision rule (notes PART II): |padded - plain| < .02 AUC across readers.

One reader per process. Probe window MUST match the grid's.
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

FILLER = ("This criterion is one of several used when evaluating texts of this kind. "
          "Read the text carefully and consider it as a whole when forming a judgment. ")


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


def pad_to(name, target_len):
    if len(name) >= target_len:
        return name
    body = name + ". "
    while len(body) < target_len:
        body += FILLER
    return body[:target_len].rstrip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--reader", required=True)
    p.add_argument("--ref-dir", required=True)
    p.add_argument("--msgs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-probes", type=int, default=300)
    p.add_argument("--gepa-reserve", type=int, default=60)
    a = p.parse_args()

    cfg = cfgmod.apply_task_preset(cfgmod.ImplementerConfig(), a.task)
    texts, _ = _load_texts(a.task, a.gepa_reserve + a.n_probes, cfg)
    probes = texts[a.gepa_reserve: a.gepa_reserve + a.n_probes]
    msgs = json.load(open(a.msgs))
    refs = _load_refs(a.ref_dir)

    report = json.load(open(a.out)) if os.path.exists(a.out) else {}
    rep = report.setdefault(a.reader, {})
    executor = make_judge_backend(a.reader, cfg, temperature=None)
    for gi, m in msgs.items():
        if gi not in refs or gi in rep:
            continue
        name = m.get("name", "")
        rdef = m.get("rungs", {}).get("definition", "")
        doss = m.get("rungs", {}).get("dossier", "")
        if not name or not rdef:
            continue
        mask = np.ones(len(refs[gi]), bool)
        ex = m.get("exemplar_idx") or {}
        mask[(ex.get("pos") or []) + (ex.get("neg") or [])] = False
        real = refs[gi][mask] > 0.5
        variants = {"name": name, "definition": rdef,
                    "padded_name": pad_to(name, len(rdef))}
        if doss:
            variants["dossier"] = doss
            variants["trunc_dossier"] = doss[: len(rdef)]
        row = {"len_name": len(name), "len_def": len(rdef), "len_doss": len(doss) or None}
        for rung, txt in variants.items():
            sig = np.nan_to_num(np.asarray(ap.signature(executor, txt, probes,
                                                         cfg.max_text_chars), float), nan=0.5)[mask]
            v = auc_mw(sig, real)
            row[f"{rung}_auc"] = round(v, 4) if v is not None else None
        rep[gi] = row
        json.dump(report, open(a.out, "w"), indent=1)
        print(f"  {gi} {name[:34]:34s} name={row.get('name_auc')} padded={row.get('padded_name_auc')} "
              f"def={row.get('definition_auc')} doss={row.get('dossier_auc')} trunc={row.get('trunc_dossier_auc')}")
    print(f"[{a.reader}] done -> {a.out}")


if __name__ == "__main__":
    main()
