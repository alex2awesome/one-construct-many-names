#!/usr/bin/env python
"""API-served reader probes (sk3-independent): cross-family/generational panels over HTTP.

Two modes sharing one scoring path (greedy YES/NO hard verdicts through the SAME
_YESNO_TEMPLATE as the local grid — instrument-identical prompt, different READOUT:
API gives argmax verdicts, not logprob ranks, so scores here are bal-acc-style and are
compared to local numbers only through the llama-3.1-8b comparability anchor):

  real   — name/definition rungs of real grid metrics vs the LOCAL executor refs
           (m_refs_8b.npz). Requires the probe window to be locally reproducible
           (verified: humor 240/240, math 84/84 exemplar matches; CW is NOT — truncated file).
  pseudo — the pseudo-concept ladder (invented names + programmatic rules): definition-
           execution capacity kappa(reader) with exact ground truth, no refs needed.

Reader spec: "<backend>:<model>", e.g. openrouter:google/gemma-3-12b-it, zai_anthropic:glm-5.2.
Incremental save: (reader, gi, rung) cells already in the output JSON are skipped on re-run.
"""
import argparse
import dataclasses
import json
import os

import numpy as np

from methods.metric_implementer import config as cfgmod
from methods.metric_implementer.backends import LLMBackend
from methods.metric_implementer.recon_channel import _YESNO_TEMPLATE
from methods.metric_implementer.experiments.run_real_test import _load_texts

DATA = "notebooks/data/two_faces_20260702"
DOM = {"humor": ("humor", "r3_humor"), "math": ("math", "r3_math"),
       "creative-writing": ("creative-writing", "r3_cw")}


def load_domain(task, n_probes=300, reserve=60):
    preset, rdir = DOM[task]
    cfg = cfgmod.apply_task_preset(cfgmod.ImplementerConfig(), preset)
    texts, _ = _load_texts(preset, reserve + n_probes, cfg)
    probes = texts[reserve: reserve + n_probes]
    msgs = json.load(open(f"{DATA}/{rdir}/grid_{ 'cw' if task=='creative-writing' else task }_v1/messages.json"))
    refs = None
    rp = f"{DATA}/{rdir}/m_refs_8b.npz"
    if os.path.exists(rp):
        z = np.load(rp, allow_pickle=True)
        refs = {k: np.asarray(z[k], float) for k in z.keys()}
    return cfg, probes, msgs, refs


def pick_metrics(task, msgs, k_extreme=3, k_rand=6, seed=0):
    """3 lowest + 3 highest local 3B name-deficit + k_rand stratified random (fixed seed)."""
    short = "cw" if task == "creative-writing" else task
    rep = json.load(open(f"{DATA}/r3_{short}/grid_{short}_v1/auc_report.json"))
    r3b = rep.get("Llama-3.2-3B-Instruct", {})
    defs = {}
    for gi, rungs in r3b.items():
        a_n, a_d = rungs.get("name", {}).get("auc"), rungs.get("definition", {}).get("auc")
        if a_n is not None and a_d is not None and gi in msgs:
            defs[gi] = a_d - a_n
    ranked = sorted(defs, key=defs.get)
    chosen = ranked[:k_extreme] + ranked[-k_extreme:]
    rest = [g for g in ranked[k_extreme:-k_extreme]]
    rng = np.random.default_rng(seed)
    chosen += list(rng.choice(rest, min(k_rand, len(rest)), replace=False))
    return sorted(set(chosen), key=int), {g: round(defs[g], 4) for g in defs}


def parse_verdicts(outs):
    v = np.full(len(outs), np.nan)
    for i, o in enumerate(outs):
        t = (o or "").strip().lower()
        if t.startswith("yes"):
            v[i] = 1.0
        elif t.startswith("no"):
            v[i] = 0.0
    return v


def bal_acc(pred, lab):
    ok = ~np.isnan(pred)
    if ok.sum() < 10:
        return None
    p, l = pred[ok] > 0.5, lab[ok]
    pos, neg = l, ~l
    if pos.sum() == 0 or neg.sum() == 0:
        return None
    return float(((p & pos).sum() / pos.sum() + (~p & neg).sum() / neg.sum()) / 2)


def make_backend(spec, cfg):
    backend, model = spec.split(":", 1)
    c = dataclasses.replace(cfg, backend=backend)
    return LLMBackend(model, "judge", c, temperature=0.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=list(DOM))
    p.add_argument("--mode", default="real", choices=["real", "pseudo"])
    p.add_argument("--readers", required=True, help="comma list of backend:model specs")
    p.add_argument("--n-probes", type=int, default=300)
    p.add_argument("--subsample", type=int, default=0, help="probe subsample (0 = all), seed 0")
    p.add_argument("--metrics", default="", help="comma gi list; empty = auto 3+3+6 selection")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    cfg, probes, msgs, refs = load_domain(a.task, a.n_probes)
    idx = np.arange(len(probes))
    if a.subsample:
        idx = np.sort(np.random.default_rng(0).choice(idx, a.subsample, replace=False))

    if a.mode == "real":
        gis, all_defs = (a.metrics.split(","), None) if a.metrics else pick_metrics(a.task, msgs)
        items = []
        for gi in gis:
            m = msgs[str(gi)]
            ex = set(m["exemplar_idx"]["pos"] + m["exemplar_idx"]["neg"])
            keep = np.array([i for i in idx if i not in ex])
            lab = refs[str(gi)][keep] > 0.5
            for rung in ("name", "definition"):
                items.append({"gi": str(gi), "rung": rung, "text": m["rungs"][rung],
                              "keep": keep, "lab": lab, "tagname": m["name"]})
        print(f"real probe: {len(gis)} metrics x 2 rungs x ~{len(idx)} probes")
    else:
        from methods.codability.pseudo_concept_ladder import build
        concepts, truths = build([probes[i] for i in idx])
        items = []
        for cname, c in concepts.items():
            for rung, txt in (("name", cname), ("definition", c["definition"])):
                items.append({"gi": cname, "rung": rung, "text": txt,
                              "keep": np.arange(len(idx)), "lab": truths[cname],
                              "tagname": cname})
        print(f"pseudo probe: {len(concepts)} concepts x 2 rungs x {len(idx)} probes")

    report = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for spec in [r.strip() for r in a.readers.split(",") if r.strip()]:
        be = make_backend(spec, cfg)
        rep = report.setdefault(spec, {})
        for it in items:
            if rep.get(it["gi"], {}).get(it["rung"]) is not None:
                continue
            ptexts = ([probes[i] for i in it["keep"]] if a.mode == "real"
                      else [probes[i] for i in idx[it["keep"]]])
            prompts = [_YESNO_TEMPLATE.format(rubric=it["text"], text=t[: cfg.max_text_chars])
                       for t in ptexts]
            outs = be.generate_batch(prompts, max_tokens=3, temperature=0.0)
            pred = parse_verdicts(outs)
            rep.setdefault(it["gi"], {})[it["rung"]] = {
                "bal_acc": (lambda v: round(v, 4) if v is not None else None)(
                    bal_acc(pred, it["lab"])),
                "yes_rate": round(float(np.nanmean(pred)), 3) if (~np.isnan(pred)).any() else None,
                "unparsed": int(np.isnan(pred).sum()), "n": len(pred), "name": it["tagname"]}
            json.dump(report, open(a.out, "w"), indent=1)
            print(f"  {spec} gi={it['gi']} {it['rung']}: {rep[it['gi']][it['rung']]['bal_acc']}"
                  f" (unparsed {rep[it['gi']][it['rung']]['unparsed']})")
        s = be.stats.as_dict()
        rep["_stats"] = s
        json.dump(report, open(a.out, "w"), indent=1)
        print(f"[{spec}] done — {s}")


if __name__ == "__main__":
    main()
