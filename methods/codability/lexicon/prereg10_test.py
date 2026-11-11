#!/usr/bin/env python3
"""PREREG-10 / PREREG-10R analysis (registry 2026-07-23): community-rule vs individual-lay
register height. Exact machinery of the original 2026-07-23c run (recovered verbatim from
the session transcript), parameterized over judging sources so the PREREG-10R expanded
wave (p10r chunks) can be appended without touching the frozen recipe.

Sources: each = (out_glob, key_json, anchor_prefix_fmt). Gate: >=8/10 anchor stratum
matches per chunk or the chunk is excluded. Dedup across sources on (cls, normalized
term), FIRST source wins (original 19-chunk judgments retained). Height = mean of pooled
z-scored formality and latinate indicator (pool = all judged terms in the run, as in the
original). PRIMARY: per-domain one-sided MWU community_rule < individual_lay (>=30/side),
Fisher + LOO. SECONDARY: community_rule < official. W5f: LLM-class distances.

Usage: python prereg10_test.py [--expanded]
Writes prereg10_results_20260723.json (original only) or prereg10r_results_20260723.json.
"""
import glob
import json
import re
import sys
from collections import Counter, defaultdict

import numpy as np
from scipy import stats

from methods.codability.lexicon.codability_sampling_model import norm_name

SP = ("/private/tmp/claude-502/-Users-anon-Projects-stanford-research-norm-research/"
      "6eb8228d-7011-4800-87e2-61a172f6003c/scratchpad")
ROOT = "ROOT"
LAT = {"germanic": 0.0, "mixed": 0.5, "latinate": 1.0, "greek": 1.0}
DMAP = {"humor_comedy": "humor", "creative_writing": "creative-writing",
        "news_journalism": "news-homepages", "math": "math-stackexchange",
        "programming_code": "code-review", "law_legal": "legal-outcome-prediction",
        "patents_inventing": "patents", "grants_nonprofit": "grant-funding",
        "press_releases_pr_marketing": "press-releases",
        "peer_review_academia": "peer-review",
        "public_comments_regulation": "notice-and-comment"}


def load_source(out_glob, key_file, prefix):
    key = json.load(open(key_file))
    rows = {}
    gate_ok = gate_fail = 0
    for path in sorted(glob.glob(out_glob)):
        b = int(re.search(r"out_(\d+)", path).group(1))
        ch = {}
        for l in open(path):
            try:
                r = json.loads(l)
                ch[r["id"]] = r
            except Exception:
                pass
        anch = [(k, v) for k, v in key["truth"].items()
                if k.startswith(prefix.format(b=b))]
        hits = sum(1 for k, t in anch if ch.get(k, {}).get("stratum") == t)
        if hits >= 8:
            gate_ok += 1
            for i, r in ch.items():
                if i in key["map"] and r.get("stratum") in LAT and r.get("formality"):
                    m = key["map"][i]
                    rows[i] = {"cls": m["cls"], "term": m["term"],
                               "f": float(r["formality"]), "l": LAT[r["stratum"]],
                               "nom": r.get("nominalization")}
        else:
            gate_fail += 1
            print(f"{out_glob} chunk {b} ANCHOR FAIL {hits}/10 — excluded")
    return rows, gate_ok, gate_fail


def main():
    expanded = "--expanded" in sys.argv
    sources = [(f"{SP}/p10_out_*.jsonl", f"{SP}/p10_key.json", "na{b:02d}")]
    if expanded:
        sources.append((f"{SP}/p10r_out_*.jsonl", f"{SP}/p10r_key.json", "nr{b:02d}"))
    judged, gates = {}, [0, 0]
    seen = set()
    for si, (g, k, pre) in enumerate(sources):
        rows, ok, fail = load_source(g, k, pre)
        gates[0] += ok
        gates[1] += fail
        # dedup ACROSS sources only (skip expansion terms already judged in the
        # original run); within a source keep everything, matching the frozen run.
        src_keys = set()
        for i, v in rows.items():
            dk = (v["cls"], norm_name(v["term"]) or v["term"].strip().lower())
            if si > 0 and dk in seen:
                continue
            src_keys.add(dk)
            judged[i] = v
        seen |= src_keys
    print(f"gates: {gates[0]} pass / {gates[1]} fail | judged terms (deduped): {len(judged)}")
    F = np.array([v["f"] for v in judged.values()])
    L = np.array([v["l"] for v in judged.values()])
    zf, zl = (F - F.mean()) / F.std(), (L - L.mean()) / L.std()
    for (i, v), a, b_ in zip(judged.items(), zf, zl):
        v["h"] = (a + b_) / 2
    bycls = defaultdict(list)
    for v in judged.values():
        bycls[v["cls"]].append(v)
    print(f"\n{'class':16} {'n':>5} {'height_z':>9} {'formality':>9} {'latinate':>8} {'nominal':>7}")
    for c in ["official", "community_rule", "individual_lay", "llm_glm", "llm_gpt56"]:
        vs = bycls[c]
        if not vs:
            continue
        print(f"{c:16} {len(vs):5} {np.mean([v['h'] for v in vs]):+9.3f} "
              f"{np.mean([v['f'] for v in vs]):9.2f} {np.mean([v['l'] for v in vs]):8.3f} "
              f"{np.mean([v['nom'] or 0 for v in vs]):7.0%}")
    # term -> domain
    term_dom = {}
    for l in open(f"{ROOT}/outputs/lexicon/community_rule_criteria_20260723.jsonl"):
        r = json.loads(l)
        doms = [d for d, _, _ in r["uses"]]
        for t in r.get("criterion_terms", []):
            t = (t or "").strip().lower()
            if t:
                term_dom.setdefault(("community_rule", t), Counter()).update(doms)
    for p in glob.glob(f"{SP}/lay_extract_*.jsonl"):
        fld = p.split("lay_extract_")[1][:-6]
        for l in open(p):
            r = json.loads(l)
            if r.get("doc_summary_row") or not r.get("head_term"):
                continue
            nm = norm_name(r["head_term"])
            if nm:
                term_dom.setdefault(("individual_lay", nm), Counter()).update([fld])
    byfield = defaultdict(lambda: defaultdict(list))
    for v in judged.values():
        if v["cls"] not in ("community_rule", "individual_lay"):
            continue
        dc = term_dom.get((v["cls"], v["term"]))
        if not dc:
            continue
        d = DMAP.get(dc.most_common(1)[0][0], dc.most_common(1)[0][0])
        byfield[d][v["cls"]].append(v["h"])
    ps, perdom = [], {}
    print("\nPRIMARY (per-domain one-sided MWU community_rule < individual_lay, >=30/side):")
    for d, cl in sorted(byfield.items()):
        a, b_ = cl.get("community_rule", []), cl.get("individual_lay", [])
        if len(a) < 30 or len(b_) < 30:
            continue
        u = stats.mannwhitneyu(a, b_, alternative="less")
        ps.append(u.pvalue)
        perdom[d] = {"rule_h": round(float(np.mean(a)), 3), "n_rule": len(a),
                     "lay_h": round(float(np.mean(b_)), 3), "n_lay": len(b_),
                     "p": float(u.pvalue)}
        print(f"  {d:26} rule {np.mean(a):+.3f} (n={len(a)}) vs lay {np.mean(b_):+.3f} "
              f"(n={len(b_)}) p={u.pvalue:.4g}")
    X = -2 * sum(np.log(p) for p in ps)
    fp = float(1 - stats.chi2.cdf(X, 2 * len(ps)))
    loo = {}
    for drop in perdom:
        rest = [v["p"] for k, v in perdom.items() if k != drop]
        Xr = -2 * sum(np.log(p) for p in rest)
        loo[drop] = round(float(1 - stats.chi2.cdf(Xr, 2 * len(rest))), 4)
    print(f"  FISHER chi2={X:.2f} df={2*len(ps)} p={fp:.6g} | LOO {loo}")
    u2 = stats.mannwhitneyu([v["h"] for v in bycls["community_rule"]],
                            [v["h"] for v in bycls["official"]], alternative="less")
    print(f"\nSECONDARY community_rule < official: p={u2.pvalue:.4g}")
    print("\nW5f — which human class do LLM names resemble (|mean height gap| / KS):")
    w5f = {}
    for lm in ["llm_glm", "llm_gpt56"]:
        hl = [v["h"] for v in bycls[lm]]
        for hc in ["official", "community_rule", "individual_lay"]:
            hh = [v["h"] for v in bycls[hc]]
            ks = stats.ks_2samp(hl, hh)
            w5f[f"{lm}_vs_{hc}"] = {"d_mean": round(abs(float(np.mean(hl) - np.mean(hh))), 3),
                                    "ks": round(float(ks.statistic), 3)}
            print(f"  {lm:10} vs {hc:16} d_mean={abs(np.mean(hl)-np.mean(hh)):.3f} "
                  f"KS={ks.statistic:.3f}")
    out = {"judged": len(judged), "gates": gates,
           "class_means": {c: {"h": float(np.mean([v["h"] for v in bycls[c]])),
                               "formality": float(np.mean([v["f"] for v in bycls[c]])),
                               "latinate": float(np.mean([v["l"] for v in bycls[c]])),
                               "nominalized": float(np.mean([v["nom"] or 0 for v in bycls[c]])),
                               "n": len(bycls[c])} for c in bycls},
           "per_domain": perdom, "primary_fisher_p": fp, "primary_loo": loo,
           "secondary_p": float(u2.pvalue), "w5f": w5f}
    name = "prereg10r_results_20260723.json" if expanded else "prereg10_check_20260723.json"
    path = f"{ROOT}/outputs/lexicon/{name}"
    json.dump(out, open(path, "w"), indent=1)
    print("\nwrote", path)


if __name__ == "__main__":
    main()
