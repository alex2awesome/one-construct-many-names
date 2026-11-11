#!/usr/bin/env python
"""Name-sufficiency scaling, cross-family enculturation, and a-priori tacitness prediction.

Three analyses over the decompression-grid reports (2026-07-05, user directives):

  scaling  — Direction 1: per-metric name-deficit d(m,r) = bal_acc(definition) - bal_acc(name)
             on the within-family reader ladder (Llama 1B -> 3B -> 8B; 8B reader == the
             executor, i.e. SELF-recovery). S*(m) = smallest scale at which the name alone
             suffices, persisting up-ladder. Survival tables by domain / concept type.
             Ordinal claims only — no continuous extrapolation of bal_acc/MI.
  family   — Direction 2: cross-family (A->B) vs within-family (A->A') contrast on
             byte-identical rung messages. DiD(m) = d_B(tier) - d_A(tier) at size-matched
             tiers; per-metric taxonomy (universal / A-only-lexicalized = enculturation gap /
             B-only / craft-everywhere / A-idiosyncratic-extension / unmeasurable) using
             kin = Llama-3B vs stranger = biggest B reader, conditioned on definition-recovery
             so a missing lexical binding is separated from missing capability.
  prereg   — Freeze the ordinal 70B prediction BEFORE the 70B pass exists: rank metrics by
             8B name-deficit; prediction = 70B name-sufficiency flips are a prefix of that
             ranking (evaluated later as AUC + persistence of already-sufficient metrics).
  apriori  — Direction 3: predict tacitness outcomes from metric TEXT alone (concept-type tag,
             domain class, name/definition length), leave-one-DOMAIN-out. Metric-level
             meta-regression; never label-aware.

Recovery target throughout is the EXECUTOR's verdicts (bal_acc), i.e. transfer of the 8B
writer's metric — not the census self_bits readout (reader-internal consistency), which is
reported alongside for commensurability with the published lexicalization gradient.
"""
import argparse
import glob
import hashlib
import json
import os
from collections import defaultdict

import numpy as np

DATA = "notebooks/data/two_faces_20260702"

# short key -> (grid dir, tag/band domain aliases, domain class)
# Domain classes are assigned from domain knowledge (does the field maintain a formalized,
# textbook-defined professional lexicon? is the value expressive or institutional?); the
# class<->outcome association was first OBSERVED in this data (lexicalization gradient), so
# apriori runs report feature sets with and without the class feature.
DOMAINS = {
    "cw":    ("r3_cw/grid_cw_v1",       ["creative-writing"],                       "expressive"),
    "humor": ("r3_humor/grid_humor_v1", ["humor"],                                  "expressive"),
    "pr":    ("r3_pr/grid_pr_v1",       ["press-releases"],                         "institutional"),
    "news":  ("r3_news/grid_news_v1",   ["news-homepages"],                         "institutional"),
    "math":  ("r3_math/grid_math_v1",   ["math-stackexchange", "math"],             "formal-lexicon"),
    "cr":    ("r3_cr/grid_cr_v1",       ["code-review"],                            "institutional"),
    "peer":  ("r3_peer/grid_peer_v1",   ["peer-review"],                            "institutional"),
    "legal": ("r3_legal/grid_legal_v1", ["legal-outcome-prediction", "law_bridge"], "formal-lexicon"),
    "grant": ("r3_grant/grid_grant_v1", ["grant-funding"],                          "institutional"),
}

LLAMA_LADDER = [("1B", "Llama-3.2-1B-Instruct"), ("3B", "Llama-3.2-3B-Instruct"),
                ("8B", "Llama-3.1-8B-Instruct")]  # 8B = executor = self-recovery
QWEN_TIERS = [("1B", "Qwen2.5-1.5B-Instruct"), ("3B", "Qwen2.5-3B-Instruct"),
              ("8B", "Qwen2.5-7B-Instruct")]      # size-matched tiers (approximate)
ARTIC_RUNGS = ["name", "definition", "explanation", "full_rubric"]  # exclude exemplar/dossier
FLOOR = 0.55          # measurability: max-rung score must clear this (chance = 0.5)
FLOOR_STRICT = 0.60   # sensitivity
EPS_LIST = [0.0, 0.02]
# 'auc' (threshold-free, from auc_report.json) is PRIMARY: report.json's bal_acc thresholds
# orbit-averaged scores at an absolute 0.5, so a reader with globally shifted P(yes)
# (Qwen2.5-3B on math: all-negative -> bal_acc==0.5 on all 21 metrics, while its AUC is a
# healthy 0.649) reads as chance. Calibration is an instrument property, not tacitness.
SCALE = "auc"


def load_tags():
    tags = {}
    for f in ["concept_tags_alldomains.json", "concept_tags_wave2.json", "concept_tags.json"]:
        p = os.path.join(DATA, f)
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        for t in d.get("tags", d if isinstance(d, list) else []):
            tags[(t["domain"], int(t["gi"]))] = t["label"]
    return tags


def load_band():
    """(alias domain, gi) -> band verdict, from band_mode_reread per_metric detail."""
    p = os.path.join(DATA, "band_mode_reread.json")
    out = {}
    if not os.path.exists(p):
        return out
    d = json.load(open(p))
    res = d.get("results", d)
    entries = res if isinstance(res, list) else [dict(domain=k, **v) for k, v in res.items()]
    for e in entries:
        dom = e.get("domain") or e.get("task")
        for pm in e.get("per_metric", []) or []:
            gi = pm.get("gi", pm.get("group"))
            if gi is None:
                continue
            out[(dom, int(gi))] = {"verdict": pm.get("band_verdict", pm.get("verdict")),
                                   "H_M": pm.get("H_M")}
    return out


def build_master():
    """One row per (domain, gi): rung texts/lengths, per-reader per-rung bal_acc/self_bits, tag, band."""
    tags, band = load_tags(), load_band()
    rows = []
    def merged(dirpath):
        """report.json (bal_acc/self_bits) + auc_report.json (auc/spearman), joined per rung."""
        rep = json.load(open(os.path.join(dirpath, "report.json")))
        ap = os.path.join(dirpath, "auc_report.json")
        if os.path.exists(ap):
            for reader, per in json.load(open(ap)).items():
                for gi_s, rungs in per.items():
                    for rung, v in rungs.items():
                        cell = rep.setdefault(reader, {}).setdefault(gi_s, {}).setdefault(rung, {})
                        if isinstance(cell, dict):
                            cell["auc"], cell["spearman"] = v.get("auc"), v.get("spearman")
        return rep

    for short, (gdir, aliases, dclass) in DOMAINS.items():
        rep = merged(os.path.join(DATA, gdir))
        msgs = json.load(open(os.path.join(DATA, gdir, "messages.json")))
        qp = glob.glob(os.path.join(DATA, os.path.dirname(gdir), "grid_*_qwen", "report.json"))
        qrep = None
        if qp and os.path.getsize(qp[0]) > 5000:
            qrep = merged(os.path.dirname(qp[0]))
        for gi_s, m in msgs.items():
            gi = int(gi_s)
            row = {"domain": short, "domain_class": dclass, "gi": gi, "name": m["name"],
                   "word_len": m.get("word_len", {}),
                   "def_text": m["rungs"].get("definition", ""),
                   "tag": None, "band": None, "readers": {}, "qwen": {}}
            for al in aliases:
                row["tag"] = row["tag"] or tags.get((al, gi))
                row["band"] = row["band"] or band.get((al, gi))
            for size, rname in LLAMA_LADDER:
                cell = rep.get(rname, {}).get(gi_s)
                if cell:
                    row["readers"][size] = {r: {"bal_acc": v.get("bal_acc"), "auc": v.get("auc"),
                                                "self_bits": v.get("self_bits")}
                                            for r, v in cell.items() if isinstance(v, dict)}
            if qrep:
                for size, rname in QWEN_TIERS:
                    cell = qrep.get(rname, {}).get(gi_s)
                    if cell:
                        row["qwen"][size] = {r: {"bal_acc": v.get("bal_acc"), "auc": v.get("auc")}
                                             for r, v in cell.items() if isinstance(v, dict)}
            rows.append(row)
    n_tag = sum(1 for r in rows if r["tag"])
    n_band = sum(1 for r in rows if r["band"])
    n_q = sum(1 for r in rows if r["qwen"])
    print(f"master: {len(rows)} metrics / {len(DOMAINS)} domains | tags {n_tag} | "
          f"band {n_band} | qwen-panel rows {n_q}")
    return rows


def acc(row, size, rung, panel="readers"):
    v = row[panel].get(size, {}).get(rung, {})
    return v.get(SCALE)


def measurable(row, size, panel="readers", floor=FLOOR):
    vals = [acc(row, size, r, panel) for r in ARTIC_RUNGS]
    vals = [v for v in vals if v is not None]
    return bool(vals) and max(vals) >= floor


def name_deficit(row, size, panel="readers"):
    a_def, a_name = acc(row, size, "definition", panel), acc(row, size, "name", panel)
    if a_def is None or a_name is None:
        return None
    return a_def - a_name


def name_sufficient(row, size, eps, panel="readers", floor=FLOOR):
    """Name-sufficient = the name ALONE transmits the metric above the floor AND fuller
    articulation adds <= eps. The conjunction matters: near the floor, def~name~chance makes
    d<=0 a coin flip, which faked non-monotone survival curves in the first pass."""
    if not measurable(row, size, panel, floor):
        return None  # can't tell — reader at chance on every articulation rung
    d = name_deficit(row, size, panel)
    a_name = acc(row, size, "name", panel)
    if d is None or a_name is None:
        return None
    return bool(a_name >= floor and d <= eps)


def s_star(row, eps, floor=FLOOR):
    """Smallest ladder scale where name-sufficiency holds and persists up-ladder.
    Returns '1B'/'3B'/'8B' or '>8B' (right-censored) or None (never measurable)."""
    sizes = [s for s, _ in LLAMA_LADDER]
    ns = {s: name_sufficient(row, s, eps, floor=floor) for s in sizes}
    if all(v is None for v in ns.values()):
        return None
    for i, s in enumerate(sizes):
        later = [ns[t] for t in sizes[i:] if ns[t] is not None]
        if ns[s] and later and all(later):
            return s
    return ">8B"


def cmd_scaling(rows, floor):
    out = {"floor": floor, "eps": EPS_LIST, "scale": SCALE,
           "note": "8B reader == executor (self-recovery)",
           "per_metric": [], "survival": {}, "gradient_bal_acc": {}, "gradient_self_bits": {}}
    for row in rows:
        e = {"domain": row["domain"], "gi": row["gi"], "name": row["name"], "tag": row["tag"],
             "deficit": {s: name_deficit(row, s) for s, _ in LLAMA_LADDER},
             "measurable": {s: measurable(row, s, floor=floor) for s, _ in LLAMA_LADDER},
             "s_star": {str(eps): s_star(row, eps, floor) for eps in EPS_LIST}}
        out["per_metric"].append(e)
    # survival: fraction of measurable metrics NOT yet name-sufficient at each scale
    for key_fn, key_name in [(lambda r: r["domain"], "by_domain"),
                             (lambda r: r["domain_class"], "by_class"),
                             (lambda r: r["tag"] or "UNTAGGED", "by_tag")]:
        tab = defaultdict(lambda: {s: [0, 0] for s, _ in LLAMA_LADDER})
        for row in rows:
            for s, _ in LLAMA_LADDER:
                ns = name_sufficient(row, s, EPS_LIST[0], floor=floor)
                if ns is None:
                    continue
                tab[key_fn(row)][s][1] += 1
                if not ns:
                    tab[key_fn(row)][s][0] += 1
        out["survival"][key_name] = {k: {s: {"still_deficient": c[0], "n": c[1],
                                             "frac": round(c[0] / c[1], 3) if c[1] else None}
                                         for s, c in v.items()} for k, v in tab.items()}
    # per-domain mean def-name gap on both scales (bal_acc primary; self_bits = census-commensurable)
    for row in rows:
        for scale_key, val_fn in [("gradient_bal_acc", lambda r, s: name_deficit(r, s)),
                                  ("gradient_self_bits",
                                   lambda r, s: (lambda d, n: None if d is None or n is None else d - n)(
                                       r["readers"].get(s, {}).get("definition", {}).get("self_bits"),
                                       r["readers"].get(s, {}).get("name", {}).get("self_bits")))]:
            for s in ["1B", "3B"]:  # small readers only, matching the census convention
                v = val_fn(row, s)
                if v is not None:
                    out[scale_key].setdefault(row["domain"], {}).setdefault(s, []).append(v)
    for scale_key in ["gradient_bal_acc", "gradient_self_bits"]:
        out[scale_key] = {dom: {s: round(float(np.mean(v)), 4) for s, v in per.items()}
                          for dom, per in out[scale_key].items()}
    return out


def sign_flip_p(diffs, n_perm=10000, seed=0):
    diffs = np.asarray([d for d in diffs if d is not None and np.isfinite(d)])
    if len(diffs) < 3:
        return None, len(diffs)
    rng = np.random.default_rng(seed)
    obs = abs(diffs.mean())
    null = np.abs((diffs[None, :] * rng.choice([-1, 1], (n_perm, len(diffs)))).mean(1))
    return float((null >= obs).mean()), len(diffs)


def cmd_family(rows, floor):
    out = {}
    for dom in sorted({r["domain"] for r in rows if r["qwen"]}):
        drows = [r for r in rows if r["domain"] == dom and r["qwen"]]
        res = {"n": len(drows), "tiers": {}, "taxonomy": [], "taxonomy_counts": defaultdict(int)}
        for size, _ in LLAMA_LADDER:
            dids = []
            for r in drows:
                dl, dq = name_deficit(r, size), name_deficit(r, size, "qwen")
                if dl is None or dq is None:
                    continue
                if not (measurable(r, size, floor=floor) and measurable(r, size, "qwen", floor=floor)):
                    continue
                dids.append(dq - dl)
            p, n = sign_flip_p(dids)
            res["tiers"][size] = {"mean_DiD": round(float(np.mean(dids)), 4) if dids else None,
                                  "n": n, "p_signflip": p,
                                  "self_note": "8B tier: Llama side is self-recovery" if size == "8B" else None}
        # per-metric taxonomy: kin = Llama-3B (small, same family) vs stranger = Qwen-7B ('8B' tier)
        for r in drows:
            kin_ns = name_sufficient(r, "3B", 0.02, floor=floor)
            str_ns = name_sufficient(r, "8B", 0.02, "qwen", floor=floor)
            kin_meas = measurable(r, "3B", floor=floor)
            str_any = any(measurable(r, s, "qwen", floor=floor) for s in ["1B", "3B", "8B"])
            str_def_ok = (acc(r, "8B", "definition", "qwen") or 0) >= floor
            if kin_ns is None and str_ns is None:
                cat = "unmeasurable"
            elif kin_meas and not str_any:
                cat = "A-idiosyncratic-extension"   # kin recovers; stranger fails every rung
            elif kin_ns and str_ns:
                cat = "universal-lexicalized"
            elif kin_ns and str_ns is False and str_def_ok:
                cat = "A-only-lexicalized"          # enculturation gap: B can represent, lacks the binding
            elif kin_ns and str_ns is False:
                cat = "A-only-unclear"              # stranger name-deficient but def below floor too
            elif kin_ns is False and str_ns:
                cat = "B-only-lexicalized"
            elif kin_ns is False and str_ns is False:
                cat = "craft-everywhere"
            else:
                cat = "partial-coverage"
            res["taxonomy"].append({"gi": r["gi"], "name": r["name"], "tag": r["tag"], "cat": cat,
                                    "d_kin_3B": name_deficit(r, "3B"),
                                    "d_stranger_7B": name_deficit(r, "8B", "qwen")})
            res["taxonomy_counts"][cat] += 1
        res["taxonomy_counts"] = dict(res["taxonomy_counts"])
        out[dom] = res
    return out


def cmd_prereg(rows, floor):
    """Ordinal 70B predictions frozen before any 70B reader pass exists."""
    pend, keep = [], []
    for r in rows:
        ns8 = name_sufficient(r, "8B", 0.0, floor=floor)
        d8 = name_deficit(r, "8B")
        if ns8 is True:
            keep.append({"domain": r["domain"], "gi": r["gi"], "name": r["name"]})
        elif ns8 is False and d8 is not None:
            pend.append({"domain": r["domain"], "gi": r["gi"], "name": r["name"],
                         "deficit_8B": round(d8, 4)})
    pend.sort(key=lambda e: e["deficit_8B"])
    body = {
        "frozen": "2026-07-05, before any 70B reader pass (70B rescore lands ~2026-07-10)",
        "scale": SCALE,
        "supersedes_sha256": ("ccf4b80646390c43... (same-day bal_acc-scale freeze, superseded "
                              "pre-publication when the threshold-free AUC readout was adopted; "
                              "reason: absolute-0.5 bal_acc threshold conflates reader calibration "
                              "with tacitness)" if SCALE == "auc" else None),
        "prediction_1_persistence": "every metric name-sufficient at 8B (eps=0) stays name-sufficient at 70B",
        "prediction_2_ordinal": ("metrics flipping to name-sufficient at 70B form a PREFIX of this "
                                 "ranking (ascending 8B name-deficit); evaluate as AUC of rank vs "
                                 "observed flip + violations of prefix property"),
        "guard": "ordinal only — no continuous extrapolation of bal_acc/MI (V-info non-Lipschitz)",
        "floor": floor, "n_name_sufficient_8B": len(keep), "n_pending": len(pend),
        "name_sufficient_8B": keep, "ranking_pending": pend,
    }
    h = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    return {"sha256": h, **body}


def cmd_apriori(rows, floor):
    from sklearn.linear_model import LogisticRegression, Ridge
    from scipy.stats import spearmanr
    try:
        from wordfreq import zipf_frequency
    except ImportError:
        zipf_frequency = None

    TAGS = ["TASTE", "STRUCTURAL_CRAFT", "MECHANICAL"]
    CLASSES = ["expressive", "institutional", "formal-lexicon"]

    def feats(r, use_class, use_lex):
        tag = r["tag"] if r["tag"] in TAGS else None
        f = [1.0 if tag == t else 0.0 for t in TAGS]
        if use_class:
            f += [1.0 if r["domain_class"] == c else 0.0 for c in CLASSES]
        if use_lex:
            wl = r["word_len"]
            f += [wl.get("name", 0) or 0, wl.get("definition", 0) or 0]
            if zipf_frequency:
                ws = [w.strip(",:;()").lower() for w in r["name"].split() if len(w) > 2]
                f += [float(np.mean([zipf_frequency(w, "en") for w in ws])) if ws else 0.0]
        return f

    data = []
    for r in rows:
        # outcome = mean deficit over BOTH small readers (1B+3B): single-reader deficits are
        # reliability-limited (se ~ .04 at n=300 probes vs effects of +-.03), which is what
        # collapsed the first AUC-scale LODO pass
        ds = [name_deficit(r, s) for s in ("1B", "3B") if measurable(r, s, floor=floor)]
        ds = [d for d in ds if d is not None]
        if not ds:
            continue
        d3 = float(np.mean(ds))
        rungs = {ru: acc(r, "3B", ru) for ru in ARTIC_RUNGS}
        rungs = {k: v for k, v in rungs.items() if v is not None}
        best_is_name = max(rungs, key=rungs.get) == "name" if rungs else None
        data.append((r, d3, best_is_name))
    out = {"floor": floor, "n": len(data), "outcome_reader": "Llama-3.2-3B",
           "note": "leave-one-domain-out; zipf name-frequency " +
                   ("INCLUDED" if zipf_frequency else "unavailable (wordfreq not installed)"),
           "feature_sets": {}}
    domains = sorted({r["domain"] for r, _, _ in data})
    for fs_name, (uc, ul) in [("tags_only", (False, False)), ("tags+class", (True, False)),
                              ("tags+class+lex", (True, True))]:
        preds_c, obs_c, preds_b, obs_b = [], [], [], []
        for held in domains:
            tr = [(feats(r, uc, ul), d, b) for r, d, b in data if r["domain"] != held]
            te = [(feats(r, uc, ul), d, b) for r, d, b in data if r["domain"] == held]
            if len(te) < 2:
                continue
            Xtr = np.array([t[0] for t in tr]); Xte = np.array([t[0] for t in te])
            ridge = Ridge(alpha=1.0).fit(Xtr, [t[1] for t in tr])
            preds_c += list(ridge.predict(Xte)); obs_c += [t[1] for t in te]
            ytr = [t[2] for t in tr if t[2] is not None]
            if len(set(ytr)) == 2:
                lr = LogisticRegression(max_iter=1000).fit(
                    np.array([t[0] for t in tr if t[2] is not None]), ytr)
                for t in te:
                    if t[2] is not None:
                        preds_b.append(lr.predict_proba([t[0]])[0][1]); obs_b.append(t[2])
        rho, p = spearmanr(preds_c, obs_c)
        auc = None
        if preds_b and len(set(obs_b)) == 2:
            from sklearn.metrics import roc_auc_score
            auc = round(float(roc_auc_score(obs_b, preds_b)), 3)
        out["feature_sets"][fs_name] = {
            "lodo_spearman_name_deficit_3B": round(float(rho), 3), "p": round(float(p), 5),
            "n_pairs": len(obs_c), "lodo_auc_best_rung_is_name": auc}
    return out


N_PARAMS = {"1B": 1.24, "3B": 3.21, "8B": 8.03}   # actual params, units of 1e9
N_70B = 70.6


def cmd_law(rows, floor):
    """Chinchilla-analogue tacitness scaling law (user 2026-07-05 eve).

    Per (cell, channel): G(N) = G_inf - A * N^(-alpha), G = 2*(AUC-0.5) Gini vs the executor
    ref; alpha and the self-recovery offset delta (8B reader == executor) shared globally.
    G_inf is the EXTRAPOLATED enculturation asymptote ("model_infinity"); per the 2026-06-13
    irreducible-E lit dive, joint-fit intercepts are extrapolated-not-identified, so each
    G_inf ships inside an identified bracket [G(biggest clean reader), C_ceiling] and a
    frozen out-of-sample 70B prediction validates the fit before any asymptote is trusted.
    g_inf = G_inf(def) - G_inf(name) >= 0 is the irreducible articulation gap: what remains
    unnameable at infinite reader scale within this family. Ordinal-prereg remains primary
    until the 70B validation passes (V-info non-Lipschitz guard)."""
    from scipy.optimize import least_squares
    sizes = ["1B", "3B", "8B"]

    def gini(r, s, rung):
        a = acc(r, s, rung)
        return None if a is None else 2 * (a - 0.5)

    def collect(key_fn):
        cells = {}
        for r in rows:
            for ch in ("name", "definition"):
                gs = [gini(r, s, ch) for s in sizes]
                if any(g is None for g in gs):
                    continue
                cells.setdefault((key_fn(r), ch), []).append(gs)
        return {k: np.array(v) for k, v in cells.items() if len(v) >= 5}

    def fit(cells):
        keys = sorted(cells)
        means = {k: cells[k].mean(0) for k in keys}
        ns = np.array([len(cells[k]) for k in keys], float)
        x_n = np.array([N_PARAMS[s] for s in sizes])

        def unpack(p):
            alpha, delta = p[0], p[1]
            gA = p[2:].reshape(len(keys), 2)
            return alpha, delta, gA

        def resid(p):
            alpha, delta, gA = unpack(p)
            out = []
            for i, k in enumerate(keys):
                ginf, aa = gA[i]
                pred = ginf - aa * x_n ** (-alpha)
                pred[2] += delta                     # 8B point: self-recovery offset
                out.append((pred - means[k]) * np.sqrt(ns[i]))
            return np.concatenate(out)

        p0 = [0.5, 0.0]
        for k in keys:
            p0 += [means[k][2], max(means[k][2] - means[k][0], 0.01)]
        lo = [0.01, -0.5] + [-1.0, -5.0] * len(keys)
        hi = [3.00, 0.5] + [1.0, 5.0] * len(keys)
        sol = least_squares(resid, p0, bounds=(lo, hi))
        alpha, delta, gA = unpack(sol.x)
        return keys, alpha, delta, gA, float(np.sqrt(np.mean(sol.fun ** 2)))

    out = {"scale": SCALE, "form": "G(N) = G_inf - A*N^-alpha, N in 1e9 params; shared alpha,"
           " shared delta_self at the 8B (executor==reader) point; Gini G = 2*(AUC-0.5)",
           "guard": "G_inf is EXTRAPOLATED (irreducible-E lit dive 2026-06-13): trust only "
                    "inside the identified bracket and after the frozen 70B prediction "
                    "validates; per-metric laws await 4-point clean ladders (gemma-3 / "
                    "post-70B Llama).", "fits": {}}
    rng = np.random.default_rng(0)
    for label, key_fn in [("class_x_tag", lambda r: f"{r['domain_class']}|{r['tag']}"),
                          ("domain", lambda r: r["domain"])]:
        cells = collect(key_fn)
        if not cells:
            continue
        keys, alpha, delta, gA, rms = fit(cells)
        # metric-level bootstrap for CIs on G_inf and the clean-70B prediction
        boots = {k: [] for k in keys}
        for _ in range(200):
            bcells = {k: cells[k][rng.integers(0, len(cells[k]), len(cells[k]))] for k in keys}
            try:
                bk, ba, bd, bgA, _ = fit(bcells)
                for i, k in enumerate(bk):
                    boots[k].append((bgA[i][0], bgA[i][0] - bgA[i][1] * N_70B ** (-ba)))
            except Exception:
                continue
        entry = {"alpha": round(float(alpha), 4), "delta_self": round(float(delta), 4),
                 "resid_rms": round(rms, 4), "cells": {}}
        for i, k in enumerate(keys):
            ginf, aa = gA[i]
            pred70 = ginf - aa * N_70B ** (-alpha)
            b = np.array(boots[k]) if boots[k] else np.zeros((1, 2))
            key_s = f"{k[0]}::{k[1]}"
            entry["cells"][key_s] = {
                "n_metrics": int(len(cells[k])),
                "G_obs": [round(float(v), 4) for v in cells[k].mean(0)],
                "G_inf": round(float(ginf), 4),
                "G_inf_CI": [round(float(np.percentile(b[:, 0], q)), 4) for q in (2.5, 97.5)],
                "A": round(float(aa), 4),
                "anchor_biggest_clean": round(float(cells[k].mean(0)[1]), 4),  # 3B (8B=self)
                "pred_70B_G": round(float(pred70), 4),
                "pred_70B_AUC": round(float(pred70 / 2 + 0.5), 4),
                "pred_70B_CI_G": [round(float(np.percentile(b[:, 1], q)), 4) for q in (2.5, 97.5)],
            }
        # irreducible articulation gap per cell base (def - name)
        bases = sorted({k[0] for k in keys})
        entry["g_inf_def_minus_name"] = {}
        for base in bases:
            kn, kd = (base, "name"), (base, "definition")
            if kn in keys and kd in keys:
                gn = entry["cells"][f"{base}::name"]["G_inf"]
                gd = entry["cells"][f"{base}::definition"]["G_inf"]
                entry["g_inf_def_minus_name"][base] = round(gd - gn, 4)
        out["fits"][label] = entry
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["scaling", "family", "prereg", "apriori", "law", "all"])
    ap.add_argument("--floor", type=float, default=FLOOR)
    a = ap.parse_args()
    rows = build_master()
    todo = ["scaling", "family", "prereg", "apriori"] if a.cmd == "all" else [a.cmd]
    for cmd in todo:
        res = {"scaling": cmd_scaling, "family": cmd_family, "prereg": cmd_prereg,
               "apriori": cmd_apriori, "law": cmd_law}[cmd](rows, a.floor)
        out_path = os.path.join(DATA, {
            "scaling": "name_sufficiency_scaling.json", "family": "family_enculturation.json",
            "prereg": "prereg_70b_name_sufficiency.json", "apriori": "apriori_tacitness_lodo.json",
            "law": "scaling_law_cells.json"}[cmd])
        json.dump(res, open(out_path, "w"), indent=1, default=str)
        print(f"[{cmd}] -> {out_path}")


if __name__ == "__main__":
    main()
