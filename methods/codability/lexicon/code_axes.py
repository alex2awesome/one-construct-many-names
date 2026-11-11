#!/usr/bin/env python3
"""Axes 1-4 of the code-comparison lens menu (notes/2026-07-21__register-instrument-and-lenses.md)
+ Latinate expansion to every field with a complete extraction + selection-audit replication.

Axes computed per variant (normalized head_term):
  dispersion   — number of available fields whose named records use the variant
  termhood_<f> — smoothed log-odds of the variant in field f vs all other fields (jargonness)
  latinate     — detector v2.2 continuous score (0-1)
  nominal_sfx  — deterministic derived-abstract-nominal flag (suffix parser)
  construal    — process (-ing gerund head) / property (nominal) / other
Joined where available: Sonnet-judged formality/stratum/nominalization (1,500 variants).

Outputs: outputs/lexicon/code_axes_20260721.jsonl ; convergence + replication stats printed.
"""
import json
import math
import os
import re
from collections import Counter, defaultdict

import numpy as np
from scipy import stats

from methods.codability.lexicon.latinate_detector import (
    LAT_SUF, V2, latinate_score, words)
from methods.codability.lexicon.codability_sampling_model import LEX, norm_name

FIELDS_ALL = ["humor", "creative-writing", "news-homepages", "math-stackexchange",
              "notice-and-comment", "grant-funding", "peer-review",
              "legal-outcome-prediction", "patents", "press-releases", "code-review"]
NOM_SUF = ["ization", "isation", "ation", "ition", "ility", "ivity", "iency", "ment",
           "ance", "ence", "ancy", "ency", "tion", "sion", "ity", "ness", "ure", "ism"]


def available_fields():
    out = []
    for f in FIELDS_ALL:
        p = f"{LEX}/extract_{f}_glm-4.7.jsonl"
        if os.path.exists(p):
            n = sum(1 for _ in open(p))
            ctx = sum(1 for _ in open(f"{LEX}/contexts_{f}.jsonl"))
            if n >= ctx:          # complete (payload includes anchors, so n >= ctx)
                out.append(f)
    return out


def nominal_sfx(term):
    return int(any(w.endswith(s) for w in words(term) for s in NOM_SUF))


def construal(term):
    ws = words(term)
    if not ws:
        return "other"
    if any(w.endswith("ing") and len(w) > 5 for w in ws):
        return "process"
    return "property" if nominal_sfx(term) or len(ws) <= 3 else "other"


def main():
    fields = available_fields()
    print("complete fields:", fields)
    v2 = V2()
    uses = defaultdict(Counter)          # variant -> field -> n_records
    per_field_tot = Counter()
    sel_audit = {}
    for f in fields:
        deltas = []
        for line in open(f"{LEX}/extract_{f}_glm-4.7.jsonl"):
            r = json.loads(line)
            if r.get("status") != "ok" or str(r.get("key", "")).startswith("ANCHOR"):
                continue
            h = r.get("head_term")
            if not h:
                continue
            v = norm_name(h)
            if not v:
                continue
            uses[v][f] += 1
            per_field_tot[f] += 1
            ks = [k for k in (r.get("key_terms") or [])
                  if k.strip().lower() != h.strip().lower()]
            hs = latinate_score(h, v2.word)
            kk = [x for x in (latinate_score(k, v2.word) for k in ks) if x is not None]
            if hs is not None and kk:
                deltas.append(hs - float(np.mean(kk)))
        if deltas:
            w = stats.wilcoxon(deltas)
            sel_audit[f] = {"n": len(deltas), "mean_delta": round(float(np.mean(deltas)), 4),
                            "p": float(w.pvalue)}
    judged = {}
    for line in open(f"{LEX}/register_height_judgments.jsonl"):
        r = json.loads(line)
        judged[r["variant"]] = r
    G = sum(per_field_tot.values())
    out_path = f"{LEX}/code_axes_20260721.jsonl"
    n_rows = 0
    with open(out_path, "w") as fo:
        for v, uf in uses.items():
            tot = sum(uf.values())
            row = {"variant": v, "total_uses": tot, "dispersion": len(uf),
                   "latinate": latinate_score(v, v2.word),
                   "nominal_sfx": nominal_sfx(v), "construal": construal(v)}
            for f in uf:
                a = uf[f] + .5
                b = per_field_tot[f] - uf[f] + .5
                c = tot - uf[f] + .5
                d = G - per_field_tot[f] - (tot - uf[f]) + .5
                row[f"termhood_{f}"] = round(math.log(a / b) - math.log(c / d), 3)
            j = judged.get(v)
            if j:
                row.update({"judged_stratum": j["stratum"], "judged_formality": j["formality"],
                            "judged_nominalization": j["nominalization"]})
            fo.write(json.dumps(row) + "\n")
            n_rows += 1
    print(f"wrote {n_rows} variant rows -> {out_path}")
    # convergence: judged formality vs latinate score; judged vs suffix nominalization
    xs, ys, na, nn = [], [], 0, 0
    for v, j in judged.items():
        ls = latinate_score(v, v2.word)
        if ls is not None and j.get("formality"):
            xs.append(ls)
            ys.append(j["formality"])
        if j.get("nominalization") is not None:
            nn += 1
            na += int(nominal_sfx(v) == j["nominalization"])
    rho = stats.spearmanr(xs, ys)
    print(f"convergence: judged formality ~ latinate score rho={rho.statistic:.3f} "
          f"(n={len(xs)}, p={rho.pvalue:.1e}); nominalization agreement {na}/{nn}={na/nn:.3f}")
    print("selection-audit replication (head - keys latinate delta):")
    for f, s in sel_audit.items():
        print(f"  {f:26} n={s['n']:5} delta={s['mean_delta']:+.4f} p={s['p']:.1e}")
    disp = Counter(min(len(uf), 5) for uf in uses.values())
    print("dispersion histogram (fields-per-variant, 5=5+):", dict(sorted(disp.items())))


if __name__ == "__main__":
    main()
