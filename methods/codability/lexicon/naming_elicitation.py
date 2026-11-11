#!/usr/bin/env python3
"""W5a / PREREG-3 — LLM naming mode-collapse, GLM-4.7 family lane.

Blind protocol (E4-resurrect): for each census concept with >=5 human namings, present the
DEFINITION (criterion description from contexts_*.jsonl) of one of the concept's records
with every known human name of the concept redacted, and elicit ONE short name. k = human N
(matched-N pairing, per PREREG-3), temp 1.0, one independent call per sample.

Readout (analyze): per task, PY (d, theta) fit separately on the HUMAN per-concept name
histograms and on the LLM per-concept name histograms (same estimator both sides — W6
winner); per-concept paired posterior-predictive head share (c_max - d)/(theta + N);
one-sided Wilcoxon signed-rank LLM > human per task. Families never pooled.

Subcommands: build | run (resume-safe) | analyze
"""
import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict

ROOT = "ROOT"
LEX = f"{ROOT}/outputs/lexicon"
TASKS = ["humor", "creative-writing", "news-homepages", "math-stackexchange"]
OUT = f"{LEX}/naming_elicitation_glm_20260722.jsonl"
PAY = f"{LEX}/naming_elicitation_payload_20260722.jsonl"

SYSTEM = (
    "You are an expert practitioner asked to NAME an evaluation criterion. You are given "
    "the criterion's definition with its usual name(s) blanked out as [REDACTED]. Reply "
    "with the short name (1-4 words) YOU would naturally use for this criterion — the "
    "term of art you would write in a rubric. Reply STRICT JSON only: "
    '{"name": "..."}')


def _redact(text, names):
    out = text or ""
    for n in sorted(set(names), key=len, reverse=True):
        if len(n) < 3:
            continue
        out = re.sub(re.escape(n), "[REDACTED]", out, flags=re.I)
    return out


def build(_a):
    from methods.codability.lexicon.codability_sampling_model import (
        load_records, mirror_collapse)
    rng = random.Random(20260722)
    rows = []
    for task in TASKS:
        ctx = {}
        for line in open(f"{LEX}/contexts_{task}.jsonl"):
            c = json.loads(line)
            ctx[str(c["key"])] = c
        named = mirror_collapse([r for r in load_records(task)
                                 if r["name"] and r["name"] != "None"])
        bycon = defaultdict(list)
        for r in named:
            bycon[r["con"]].append(r)
        for con, recs in sorted(bycon.items()):
            if len(recs) < 5:
                continue
            allnames = sorted({r["name"] for r in recs})
            pool = [r for r in recs if str(r["key"]) in ctx
                    and (ctx[str(r["key"])].get("description") or "").strip()]
            if len(pool) < 3:
                continue
            for i in range(len(recs)):        # k = human N
                src = pool[i % len(pool)]
                c = ctx[str(src["key"])]
                extra = [c.get("name") or "", src["name"]]
                desc = _redact(c.get("description", ""), allnames + extra)
                rows.append({"task": task, "con": con, "i": i, "n_human": len(recs),
                             "definition": desc[:2000]})
        print(task, "samples:", sum(1 for r in rows if r["task"] == task))
    rng.shuffle(rows)
    with open(PAY, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("total", len(rows), "->", PAY)


def run(_a):
    from methods.metric_implementer import backends as _b, config as _c
    from methods.codability.lexicon.codability_sampling_model import norm_name
    be = _b.LLMBackend("glm-4.7", "naming_elicitation",
                       _c.ImplementerConfig(backend="zai_anthropic"))
    rows = [json.loads(l) for l in open(PAY)]
    done = set()
    if os.path.exists(OUT):
        done = {(r["task"], r["con"], r["i"]) for r in map(json.loads, open(OUT))}
    todo = [r for r in rows if (r["task"], r["con"], r["i"]) not in done]
    print(f"todo {len(todo)}/{len(rows)}")
    with open(OUT, "a") as fo:
        for lo in range(0, len(todo), 100):
            ch = todo[lo:lo + 100]
            ps = [f"DEFINITION:\n{r['definition']}\n\nYour name for this criterion:"
                  for r in ch]
            outs = be.generate_batch(ps, system=SYSTEM, max_tokens=300, temperature=1.0,
                                     seed=None)
            for r, o in zip(ch, outs):
                m = re.search(r"\{.*\}", o or "", re.S)
                name = None
                if m:
                    try:
                        name = norm_name(json.loads(m.group(0)).get("name"))
                    except Exception:
                        pass
                if name:
                    fo.write(json.dumps({"task": r["task"], "con": r["con"], "i": r["i"],
                                         "n_human": r["n_human"], "name": name}) + "\n")
            fo.flush()
            print(f"  {min(lo + 100, len(todo))}/{len(todo)}", flush=True)


def analyze(_a):
    import numpy as np
    from scipy import stats
    from methods.codability.lexicon.codability_sampling_model import (
        fit_py, hists, load_records, mirror_collapse)
    llm = defaultdict(lambda: defaultdict(Counter))
    for line in open(OUT):
        r = json.loads(line)
        llm[r["task"]][r["con"]][r["name"]] += 1
    res = {}
    for task in TASKS:
        named = mirror_collapse([r for r in load_records(task)
                                 if r["name"] and r["name"] != "None"])
        hum = defaultdict(Counter)
        for r in named:
            hum[r["con"]][r["name"]] += 1
        cons = [c for c, cnt in hum.items() if sum(cnt.values()) >= 5
                and c in llm[task] and sum(llm[task][c].values()) >= 5]
        Hh = hists([tuple(sorted(hum[c].values(), reverse=True)) for c in cons])
        Hl = hists([tuple(sorted(llm[task][c].values(), reverse=True)) for c in cons])
        dh, th_h, _ = fit_py(Hh)
        dl, th_l, _ = fit_py(Hl)
        ph, pl = [], []
        for c in cons:
            Nh, Nl = sum(hum[c].values()), sum(llm[task][c].values())
            ph.append((max(hum[c].values()) - dh) / (th_h + Nh))
            pl.append((max(llm[task][c].values()) - dl) / (th_l + Nl))
        w = stats.wilcoxon(pl, ph, alternative="greater")
        res[task] = {"concepts": len(cons),
                     "py_human": [round(dh, 3), round(th_h, 3)],
                     "py_llm": [round(dl, 3), round(th_l, 3)],
                     "mean_head_human": round(float(np.mean(ph)), 4),
                     "mean_head_llm": round(float(np.mean(pl)), 4),
                     "wilcoxon_p_llm_gt_human": round(float(w.pvalue), 6)}
        print(task, res[task])
    path = f"{LEX}/prereg3_results_glm_20260722.json"
    json.dump(res, open(path, "w"), indent=1)
    print("wrote", path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for c in ("build", "run", "analyze"):
        sub.add_parser(c)
    a = p.parse_args()
    {"build": build, "run": run, "analyze": analyze}[a.cmd](a)
