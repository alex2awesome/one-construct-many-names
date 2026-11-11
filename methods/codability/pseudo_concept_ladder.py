#!/usr/bin/env python
"""Pseudo-concept ladder (confounder battery T1c/T9): definition-execution capacity with the
name prior surgically removed.

Each pseudo-concept binds an INVENTED name (zero semantic content) to a fully programmatic
rule over the probe texts, threshold calibrated to the probe median (base rate ~50%). Ground
truth is computed exactly in python. Two rungs only:

  name       — the bare pseudo-word ("the smorbex quality")   -> AUC ~= 0.5 by construction
               (any deviation = leakage / prompt artifact, a validity check in itself)
  definition — the crisp calibrated rule                       -> AUC = the reader's pure
               rule-execution capacity

definition_auc_mean(reader) = mean definition-AUC over concepts = the capability coordinate used to
(a) convert size-matched into capability-matched tiers (T9) and (b) normalize the
articulation-exploitation confound (T1). Two REAL-WORD twins (same rules, real names:
"conciseness", "dialogue-heavy") measure how much a real name primes rule execution.

Scoring reuses the decompression grid's exact machinery (ap.signature via the same backend),
and the probe window MUST match the grid's (--gepa-reserve).
"""
import argparse
import json
import os
import re
import statistics

import numpy as np

from methods.metric_implementer import config as cfgmod
from methods.metric_implementer.vllm_backend import make_judge_backend
from methods.metric_implementer.experiments import alpha_probe as ap
from methods.metric_implementer.experiments.run_real_test import _load_texts

DEF_TMPL = ('A text satisfies the "{name}" criterion if and only if {rule}. '
            'This is a purely mechanical check.')


def _sentences(t):
    return [s for s in re.split(r"[.!?]+\s", t) if s.strip()]


def _words(t):
    return re.findall(r"[A-Za-z']+", t)


STATS = {
    "question_share": lambda t: (lambda s: sum(x.strip().endswith("?") or "?" in x[-3:] for x in s)
                                 / max(len(s), 1))(_sentences(t)),
    "avg_sent_len": lambda t: statistics.mean([len(_words(s)) for s in _sentences(t)] or [0]),
    "numeral_count": lambda t: len(re.findall(r"\d+", t)),
    "second_person": lambda t: (lambda w: sum(x.lower() in ("you", "your", "yours") for x in w)
                                / max(len(w), 1))(_words(t)),
    "paragraphs": lambda t: len([p for p in re.split(r"\n\s*\n", t) if p.strip()]),
    "quote_share": lambda t: (lambda L: sum(('"' in x or "“" in x) for x in L) / max(len(L), 1))(
        [x for x in t.split("\n") if x.strip()]),
    "type_token": lambda t: (lambda w: len({x.lower() for x in w}) / max(len(w), 1))(_words(t)),
    "first_sent_len": lambda t: len(_words(_sentences(t)[0])) if _sentences(t) else 0,
    "median_word_len": lambda t: statistics.median([len(x) for x in _words(t)] or [0]),
    "comma_density": lambda t: t.count(",") / max(len(_words(t)), 1),
}

RULE_TEXT = {
    "question_share": "more than {thr:.3f} of its sentences end with or contain a question mark",
    "avg_sent_len": "its average sentence length exceeds {thr:.1f} words",
    "numeral_count": "it contains more than {thr:.0f} numerals (digit sequences)",
    "second_person": "more than {thr:.4f} of its words are second-person pronouns (you/your/yours)",
    "paragraphs": "it has more than {thr:.0f} paragraphs (blank-line separated blocks)",
    "quote_share": "more than {thr:.3f} of its non-empty lines contain a double-quote character",
    "type_token": "its type-token ratio (distinct words / total words) exceeds {thr:.3f}",
    "first_sent_len": "its first sentence is longer than {thr:.0f} words",
    "median_word_len": "the median length of its words exceeds {thr:.1f} characters",
    "comma_density": "its commas-per-word ratio exceeds {thr:.4f}",
}

PSEUDO_NAMES = ["smorbex", "veltrine", "crandel", "plimsy", "drovic",
                "quenlar", "brintel", "faldore", "morvex", "teslin"]
# real-word twins: same rule machinery, real names — measures real-name priming
REAL_TWINS = {"conciseness (short sentences)": ("avg_sent_len", True),   # True: label INVERTED
              "dialogue-heavy prose": ("quote_share", False)}


def build(probe_texts):
    concepts, truths = {}, {}
    stats_v = {k: np.array([fn(t) for t in probe_texts]) for k, fn in STATS.items()}
    for pseudo, stat in zip(PSEUDO_NAMES, STATS):
        v = stats_v[stat]
        thr = float(np.median(v))
        lab = v > thr
        if not (0.2 <= lab.mean() <= 0.8):
            continue
        name = f"the {pseudo} quality"
        concepts[name] = {"stat": stat, "thr": thr,
                          "definition": DEF_TMPL.format(name=name,
                                                        rule=RULE_TEXT[stat].format(thr=thr))}
        truths[name] = lab
    for rname, (stat, invert) in REAL_TWINS.items():
        v = stats_v[stat]
        thr = float(np.median(v))
        lab = (v < thr) if invert else (v > thr)
        rule = RULE_TEXT[stat].format(thr=thr)
        if invert:
            rule = rule.replace("exceeds", "is below")
        concepts[rname] = {"stat": stat, "thr": thr, "twin": True,
                           "definition": DEF_TMPL.format(name=rname, rule=rule)}
        truths[rname] = lab
    return concepts, truths


def _rank(a):
    uniq, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    mid = np.cumsum(cnt) - (cnt - 1) / 2.0
    return mid[inv]


def auc_mw(scores, labels):
    pos = labels.sum()
    if pos == 0 or pos == len(labels):
        return None
    r = _rank(scores)
    u = r[labels].sum() - pos * (pos + 1) / 2.0
    return float(u / (pos * (len(labels) - pos)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--readers", required=True, help="comma list, one engine at a time")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-probes", type=int, default=300)
    p.add_argument("--gepa-reserve", type=int, default=60, help="MUST match the grid's")
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    cfg = cfgmod.apply_task_preset(cfgmod.ImplementerConfig(), a.task)
    probe_texts, _ = _load_texts(a.task, a.gepa_reserve + a.n_probes, cfg)
    probe_texts = probe_texts[a.gepa_reserve: a.gepa_reserve + a.n_probes]
    # Programmatic truth must be evaluated on exactly the text window the reader receives.
    reader_views = [str(t)[:cfg.max_text_chars] for t in probe_texts]
    concepts, truths = build(reader_views)
    json.dump({k: {kk: (vv if not isinstance(vv, np.ndarray) else None)
                   for kk, vv in v.items()} for k, v in concepts.items()},
              open(os.path.join(a.out_dir, "pseudo_concepts.json"), "w"), indent=1)
    print(f"{len(concepts)} concepts on {len(probe_texts)} probes "
          f"(base rates {[round(float(t.mean()), 2) for t in truths.values()]})")

    rep_path = os.path.join(a.out_dir, "pseudo_ladder_report.json")
    report = json.load(open(rep_path)) if os.path.exists(rep_path) else {}
    for reader in [r.strip() for r in a.readers.split(",") if r.strip()]:
        executor = make_judge_backend(reader, cfg, temperature=None)
        entry = {}
        for cname, c in concepts.items():
            for rung, txt in (("name", cname), ("definition", c["definition"])):
                sig = ap.signature(executor, txt, probe_texts, cfg.max_text_chars)
                sc = np.nan_to_num(np.asarray(sig, float), nan=0.5)
                entry.setdefault(cname, {})[rung] = {
                    "auc": (lambda v: round(v, 4) if v is not None else None)(
                        auc_mw(sc, truths[cname]))}
        defs = [v["definition"]["auc"] for v in entry.values()
                if v.get("definition", {}).get("auc") is not None]
        names = [v["name"]["auc"] for k, v in entry.items()
                 if "twin" not in concepts[k] and v.get("name", {}).get("auc") is not None]
        entry["_definition_auc_mean"] = round(float(np.mean(defs)), 4) if defs else None
        entry["_kappa_def_exec"] = entry["_definition_auc_mean"]  # deprecated artifact compatibility
        entry["_pseudo_name_auc_mean"] = round(float(np.mean(names)), 4) if names else None
        report[reader] = entry
        json.dump(report, open(rep_path, "w"), indent=1)
        print(f"{reader}: definition_auc_mean={entry['_definition_auc_mean']} "
              f"pseudo-name-mean={entry['_pseudo_name_auc_mean']} (leakage check, want ~0.5)")


if __name__ == "__main__":
    main()
