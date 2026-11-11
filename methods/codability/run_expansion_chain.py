"""Iso-performance expansion chains: the HORIZONTAL decompression measurement.

Vertical reader gaps (grid) are confounded by generic reader deficits. This driver measures the
horizontal quantity instead: *the minimal message expansion at which a weaker reader matches what a
stronger reader extracts from a poorer message* — the exchange rate between articulation and
capacity, per metric.

Design (2026-07-02, user-approved; notes/2026-07-02__iso-performance-expansion-design.md):
  1. NESTED monotone chains: L0 = metric name; each level APPENDS ~25 words of new content of a
     fixed articulation TYPE (definition → mechanism → procedure → boundary → worked_example →
     counterexample → checklist). Nesting makes the expansion dial a scalar and composition
     well-defined. Writer is label-blind (sees name+rubric only; invents any examples itself — no
     probe excerpts, so no leakage and no exemplar-channel confound).
  2. δ-MATCHING WITH CENSORING: match levels computed at a δ grid + paired-bootstrap match
     probabilities; never-matchers are right-censored; Kaplan-Meier-style median costs.
  3. CONFOUND CONTROLS: planted mechanically-checkable items (gold = a python rule on the probe
     text) measure each reader's instruction-following floor separately from knowledge content;
     within-reader ceilings reported.
  4. TYPE-TAGGED increments -> marginal-gain[type x reader] matrix; rank-stability across readers is
     the model-general statement. A small reversed-schedule arm (--reverse-gis) controls for
     type-position confounding.
  5. TRANSITIVITY: for readers C<B<A, h_{C→B}(h_{B→A}(0)) vs h_{C→A}(0) — triangle slack per item.
     Sub-additive slack localizes non-nested background knowledge.
  6. v2 (2026-07-03): executor-consistent PRIMARY readout — each reader also scores the metric's
     full rubric (level -1) so its OWN orbit judgment is the reconstruction target; self_bits via
     i_binary with D1 gate H_self>=0.15. Anchored curves stay as the family-top transmission
     yardstick for iso-matching (labeled as such in the report). Planted gold on the truncated
     reader view; median-split balanced planted rules; compliance-normalized matching arm.

Phases (resumable): chain -> chains.json | score -> chain_<reader>.npz | report ->
expansion_report.json. Smoke: --fake --phase all --gi-list 0 --n-probes 24 --forms 2.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np

from methods.metric_implementer import config as cfgmod
from methods.metric_implementer import vinfo
from methods.metric_implementer.vllm_backend import make_judge_backend
from methods.metric_implementer.experiments import alpha_probe as ap
from methods.metric_implementer.experiments.mine_clusters import r1_groups, r2_groups, r3_groups
from methods.metric_implementer.experiments.run_real_test import _load_texts
from methods.metric_implementer.experiments.value_census import i_binary

STEPS = [
    ("definition", "State the defining property: what must a text have to satisfy this criterion?"),
    ("mechanism", "Describe HOW this quality arises in a text -- the mechanism that produces it."),
    ("procedure", "Give the decision procedure: what a judge should CHECK, in order, to decide."),
    ("boundary", "Describe a boundary case: something that ALMOST satisfies the criterion but "
                 "fails, and why it fails."),
    ("worked_example", "Invent a miniature illustrative example (1-2 invented sentences) that "
                       "clearly satisfies the criterion, and say briefly why."),
    ("counterexample", "Invent a miniature counterexample (1-2 invented sentences) that clearly "
                       "fails the criterion, and say briefly why."),
    ("checklist", "Compress the description into a final checklist of 3-4 short imperative checks."),
]
CHAIN_PROMPT = (
    "You are expanding the description of a criterion for judging {task}, one increment at a time.\n"
    "Criterion name: \"{name}\"\nAuthoritative rubric (reference only, do not quote verbatim):\n"
    "\"{rubric}\"\n\nDescription so far:\n---\n{chain}\n---\n\n"
    "Add the next increment. {instruction}\nWrite ~25 words (never more than 45). Add ONLY new "
    "content; do not repeat or rephrase anything already in the description. Reply with the added "
    "text only.")

# Planted controls: fully-codified rules with programmatic gold. Any competent reader should
# saturate these early; the level at which a reader does = its instruction-following floor.
# v2: gold evaluates on the reader's VIEW (text truncated to max_text_chars) — v1 evaluated the
# full text, capping tail-sensitive rules on long-text domains. Median-threshold rules are
# ~balanced by construction; the computed threshold is baked into the rule text so writer and
# gold share the same number.
PLANTED_STATIC = {
    "planted_question": ("Interrogative presence",
                         "The text contains at least one question mark.",
                         lambda t: "?" in t),
    "planted_dialogue": ("Quoted dialogue presence",
                         "The text contains quoted speech: at least two quotation-mark characters.",
                         lambda t: (t.count('"') + t.count("“") + t.count("”")) >= 2),
    "planted_second_person": ("Second-person address",
                              "The text addresses the reader directly with the word 'you'.",
                              lambda t: re.search(r"\byou\b", str(t), re.I) is not None),
}


def _n_sentences(v: str) -> int:
    return len([s for s in re.split(r"[.!?]+", v) if s.strip()])


def build_planted(probe_texts, max_chars: int) -> dict:
    """Planted dict for THIS probe set: static rules + median-split rules. All gold evaluates on
    the truncated view the reader actually sees; thresholds are corpus medians on that view."""
    views = [str(t)[:max_chars] for t in probe_texts]
    w_med = sorted(len(v.split()) for v in views)[len(views) // 2]
    s_med = sorted(_n_sentences(v) for v in views)[len(views) // 2]
    l_med = round(sorted(float(np.mean([len(w) for w in v.split()])) if v.split() else 0.0
                         for v in views)[len(views) // 2], 1)
    planted = {k: (n, r, (lambda t, rule=rule: bool(rule(str(t)[:max_chars]))))
               for k, (n, r, rule) in PLANTED_STATIC.items()}
    planted["planted_length_median"] = (
        f"Length over {w_med} words", f"The text is longer than {w_med} words.",
        lambda t, n=w_med: len(str(t)[:max_chars].split()) > n)
    planted["planted_sentences_median"] = (
        f"More than {s_med} sentences",
        f"The text contains more than {s_med} sentences (a sentence ends with . ! or ?).",
        lambda t, n=s_med: _n_sentences(str(t)[:max_chars]) > n)
    planted["planted_wordlen_median"] = (
        f"Average word length over {l_med} characters",
        f"The text's average word length exceeds {l_med} characters.",
        lambda t, n=l_med: (lambda ws: bool(ws) and float(np.mean([len(w) for w in ws])) > n)(
            str(t)[:max_chars].split()))
    return planted


# ---------------------------------------------------------------- pure analysis helpers (tested)
def assemble_levels(name: str, adds: list) -> list:
    """Nested chain texts: L0=name; L_i = L_{i-1} + newline + adds[i-1]. Prefix property holds."""
    levels = [str(name).strip()]
    for a in adds:
        levels.append(levels[-1] + "\n" + str(a).strip())
    return levels


def bal_acc(pred: np.ndarray, ref: np.ndarray) -> float:
    pos, neg = ref, ~ref
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    return float(((pred & pos).sum() / pos.sum() + ((~pred) & neg).sum() / neg.sum()) / 2)


def match_level(weak_curve, strong_val, delta=0.0):
    """Minimal level at which the weak reader's curve reaches strong_val - delta; None = censored."""
    if strong_val is None or np.isnan(strong_val):
        return None
    for i, v in enumerate(weak_curve):
        if v is not None and not np.isnan(v) and v >= strong_val - delta:
            return int(i)
    return None


def km_median_level(levels, n_levels: int):
    """Median match level under right-censoring at n_levels (all censoring at the same horizon:
    the KM median equals the ordinary median of censored-as-horizon values, undefined if it lands
    at the horizon). Returns (median_or_None, matched_fraction)."""
    if not levels:
        return None, 0.0
    vals = sorted(n_levels if x is None else x for x in levels)
    med = vals[len(vals) // 2]
    frac = sum(1 for x in levels if x is not None) / len(levels)
    return (None if med >= n_levels else int(med)), round(frac, 3)


def triangle_slack(curve_w, curve_m, curve_s, delta=0.0, level=0):
    """C<B<A readers (weak/mid/strong). Composed: C matches B at B's own match-of-A level;
    direct: C matches A at `level`. Returns dict with h_mid, composed, direct, slack (None where
    censored). Each hop may use ``delta`` tolerance, so the valid direct comparison uses the
    triangle-inequality tolerance ``2 * delta``. Sub-additivity: composed >= direct when both
    are defined (composition can overshoot)."""
    h_mid = match_level(curve_m, curve_s[level], delta)          # B -> A
    composed = None if h_mid is None else match_level(curve_w, curve_m[h_mid], delta)
    direct_delta = 2 * delta
    direct = match_level(curve_w, curve_s[level], direct_delta)
    slack = None if (composed is None or direct is None) else int(composed - direct)
    return {"h_mid": h_mid, "composed": composed, "direct": direct, "slack": slack,
            "hop_delta": delta, "direct_delta": direct_delta}


def size_key(tag: str) -> int:
    for i, s in enumerate(["1B", "3B", "8B", "31B", "70B", "122B"]):
        if s in tag:
            return i
    return 99


# ---------------------------------------------------------------------------------- phases
def _ckpts(ref_dir, gi_list):
    out = {}
    for f in sorted(glob.glob(os.path.join(ref_dir, "*_sigs.npz"))):
        m = re.search(r"_(R[123])_metric(\d+)_sigs\.npz$", os.path.basename(f))
        if m and (not gi_list or int(m.group(2)) in gi_list):
            level, gi = m.group(1), int(m.group(2))
            if gi in out and out[gi][0] != level:
                raise ValueError(f"gi={gi} appears at both {out[gi][0]} and {level} in {ref_dir}; "
                                 f"mixed-level directories do not share metric identities")
            out[gi] = (level, f)
    return out


def build_chains(a, ckpts, planted):
    wcfg = cfgmod.apply_task_preset(cfgmod.ImplementerConfig(), a.task)
    if a.fake:
        wcfg.vllm_fake = True
    writer = make_judge_backend(a.writer_model, wcfg, temperature=None)
    reverse = {int(x) for x in a.reverse_gis.split(",") if x.strip()}
    groups_cache, chains = {}, {}

    def one_item(key, name, rubric, schedule):
        adds, meta = [], []
        chain = str(name).strip()
        for typ, instr in schedule:
            prompt = CHAIN_PROMPT.format(task=a.task.replace("-", " "), name=name, rubric=rubric,
                                         chain=chain, instruction=instr)
            txt = str(writer.generate(prompt, max_tokens=110)).strip()
            if not txt or len(txt.split()) > 60:                     # one stricter retry, then trim
                txt = str(writer.generate(prompt + "\nBe strict: at most 40 words.",
                                          max_tokens=90)).strip()
            trunc = len(txt.split()) > 60
            if trunc:
                txt = " ".join(txt.split()[:45])
            if not txt:
                txt = f"[increment unavailable: {typ}]"
            adds.append(txt)
            meta.append({"type": typ, "words": len(txt.split()), "truncated": trunc})
            chain = chain + "\n" + txt
        return {"name": name, "rubric": rubric, "schedule": [t for t, _ in schedule],
                "levels": assemble_levels(name, adds), "added": meta}

    for gi, (lvl, f) in ckpts.items():
        z = np.load(f, allow_pickle=True)
        if lvl not in groups_cache:
            groups_cache[lvl] = (r1_groups(a.task) if lvl == "R1" else
                                 (r3_groups if lvl == "R3" else r2_groups)(a.task, a.r2_bucket))
        rubric = str(groups_cache[lvl][gi].get("merged_description", "")).strip()
        sched = list(reversed(STEPS)) if gi in reverse else STEPS
        chains[str(gi)] = one_item(gi, str(z["name"]), rubric, sched)
        chains[str(gi)]["level_hierarchy"] = lvl
        print(f"  chained gi={gi} ({chains[str(gi)]['name'][:40]}) reversed={gi in reverse}")
    for key, (name, rule_text, _) in planted.items():
        chains[key] = one_item(key, name, rule_text, STEPS)
        print(f"  chained {key}")
    path = os.path.join(a.out_dir, "chains.json")
    json.dump(chains, open(path, "w"), indent=1)
    print(f"chains: {len(chains)} items x {len(STEPS) + 1} levels -> {path}")
    return chains


def score_reader(a, cfg, reader, chains, probe_texts):
    ecfg = cfgmod.apply_task_preset(cfgmod.ImplementerConfig(), a.task)
    if a.fake:
        ecfg.vllm_fake = True
    executor = make_judge_backend(reader, ecfg, temperature=None)
    rows, meta = [], []
    for key, ch in chains.items():
        for li, text in enumerate(ch["levels"]):
            forms = [("canonical", text)]
            if a.forms > 1:
                forms += list(ap._reformulations(text))[: a.forms - 1]
            for kind, txt in forms:
                rows.append(ap.signature(executor, txt, probe_texts, cfg.max_text_chars))
                meta.append({"item": key, "level": li, "form": kind})
        rub = str(ch.get("rubric", "")).strip()
        if rub and not key.startswith("planted"):
            # v2: the reader's OWN full-rubric orbit judgment = the executor-consistent
            # decompression target (level -1); scored under the same form orbit as the levels.
            rforms = [("canonical", rub)]
            if a.forms > 1:
                rforms += list(ap._reformulations(rub))[: a.forms - 1]
            for kind, txt in rforms:
                rows.append(ap.signature(executor, txt, probe_texts, cfg.max_text_chars))
                meta.append({"item": key, "level": -1, "form": kind})
        print(f"  scored {key} ({ch['name'][:40]})")
    tag = re.sub(r"[^A-Za-z0-9.-]+", "_", os.path.basename(str(reader).rstrip("/")))
    path = os.path.join(a.out_dir, f"chain_{tag}.npz")
    np.savez(path, scores=np.vstack(rows),
             meta=np.array([json.dumps(x) for x in meta], dtype=object),
             reader=reader, ref_dir=a.ref_dir)
    print(f"reader {tag}: {len(rows)} rows -> {path}")
    return path


def report(a, ckpts, probe_texts, planted):
    chains = json.load(open(os.path.join(a.out_dir, "chains.json")))
    n_levels = len(STEPS) + 1
    deltas = [float(x) for x in a.delta_grid.split(",") if x.strip()]

    refs, ref_prov = {}, {}
    for gi, (_, f) in ckpts.items():
        z = np.load(f, allow_pickle=True)
        mi = np.nan_to_num(np.asarray(z["M_i"], float), nan=0.5) > 0.5
        if len(mi) < len(probe_texts):
            raise SystemExit(f"ref M_i for gi={gi} has {len(mi)} probes < requested "
                             f"{len(probe_texts)} -- cannot stretch a reference")
        refs[str(gi)] = mi[: len(probe_texts)]   # same deterministic loader+order => prefix aligns
        if not ref_prov:                          # self-labeling provenance (rescore patch fields)
            ref_prov = {"ref_dir": a.ref_dir,
                        "executor": str(z["executor"]) if "executor" in z.files else None,
                        "retarget_mi_only": str(z["retarget_mi_only"])
                        if "retarget_mi_only" in z.files else None}
    for key, (_, _, rule) in planted.items():
        if key in chains:
            refs[key] = np.array([bool(rule(t)) for t in probe_texts])

    # per-reader per-item per-level predictions + curves (+ v2 self targets from level -1 rows)
    preds, curves, self_tgt = {}, {}, {}
    for gpath in sorted(glob.glob(os.path.join(a.out_dir, "chain_*.npz"))):
        z = np.load(gpath, allow_pickle=True)
        scores = np.asarray(z["scores"], float)
        meta = [json.loads(s) for s in z["meta"]]
        tag = os.path.basename(gpath)[6:-4]
        preds[tag], curves[tag], self_tgt[tag] = {}, {}, {}
        for key in {m["item"] for m in meta}:
            if key not in refs:
                continue
            pv, cv = [], []
            for li in range(n_levels):
                idx = [i for i, m in enumerate(meta) if m["item"] == key and m["level"] == li]
                if not idx:
                    pv.append(None)
                    cv.append(float("nan"))
                    continue
                p = np.nan_to_num(np.nanmean(scores[idx], axis=0), nan=0.5) > 0.5
                pv.append(p)
                cv.append(round(bal_acc(p, refs[key]), 4))
            preds[tag][key], curves[tag][key] = pv, cv
            ridx = [i for i, m in enumerate(meta) if m["item"] == key and m["level"] == -1]
            if ridx:
                self_tgt[tag][key] = np.nan_to_num(np.nanmean(scores[ridx], axis=0), nan=0.5) > 0.5

    tags = sorted(curves, key=size_key)
    print(f"readers (weak->strong): {tags}")

    # v2 PRIMARY (executor-consistent, anchor-free): each reader vs its OWN full-rubric orbit
    # judgment, in bits (exact binary MI), gated on the reader holding a non-degenerate own
    # judgment (D1: H_self >= 0.15). Absent for v1 npz files (no level -1 rows) -> empty dict.
    self_readout = {}
    for tag in tags:
        self_readout[tag] = {}
        for key, tgt in self_tgt.get(tag, {}).items():
            h = float(vinfo._h_bits(float(tgt.mean()))) if len(np.unique(tgt)) > 1 else 0.0
            bits = None
            if h >= 0.15:
                bits = [None if preds[tag][key][li] is None else
                        round(float(i_binary(tgt, preds[tag][key][li])), 4)
                        for li in range(n_levels)]
            self_readout[tag][key] = {"H_self": round(h, 4), "self_bits_by_level": bits}

    # paired bootstrap: P(weak@l >= strong@0) per item for adjacent + extreme pairs
    rng = np.random.default_rng(0)
    B, n = a.n_boot, len(probe_texts)
    boot_idx = [rng.integers(0, n, n) for _ in range(B)]

    def boot_match_prob(wtag, stag, key, delta):
        pw, ps, ref = preds[wtag][key], preds[stag][key], refs[key]
        out = []
        for bi in boot_idx:
            sv = bal_acc(ps[0][bi], ref[bi])
            hit = any(pw[level_i] is not None
                      and bal_acc(pw[level_i][bi], ref[bi]) >= sv - delta
                      for level_i in range(n_levels))
            out.append(hit)
        return round(float(np.mean(out)), 3)

    pairs = [(tags[i], tags[j]) for i in range(len(tags)) for j in range(len(tags)) if i < j]
    matching = {}
    for w, s in pairs:
        pk = f"{w}->{s}"
        matching[pk] = {}
        for d in deltas:
            per = {k: match_level(curves[w][k], curves[s][k][0], d)
                   for k in curves[w] if k in curves[s]}
            km, frac = km_median_level([v for k, v in per.items() if not k.startswith("planted")],
                                       n_levels)
            matching[pk][str(d)] = {"per_item": per, "km_median_level": km,
                                    "matched_fraction_real_metrics": frac}
        if a.n_boot > 0:
            matching[pk]["boot_match_prob_delta0"] = {
                k: boot_match_prob(w, s, k, 0.0) for k in curves[w] if k in curves[s]}

    # v2: compliance normalization. Planted ceiling c_r (median over planted items of the item's
    # best level) maps each reader's mechanical-execution range onto [0,1] via (v-.5)/(c_r-.5);
    # matching on normalized curves separates knowledge content from instruction-following.
    # NOTE deltas are then in normalized units, not raw bal_acc.
    ceilings = {}
    for tag in tags:
        tops = [max(v for v in curves[tag][k] if not np.isnan(v))
                for k in planted if k in curves[tag] and any(not np.isnan(v)
                                                             for v in curves[tag][k])]
        ceilings[tag] = round(float(np.median(tops)), 4) if tops else None

    def norm_curve(tag, key):
        c = ceilings[tag]
        return [float("nan") if np.isnan(v) else (v - 0.5) / (c - 0.5) for v in curves[tag][key]]

    matching_norm = {}
    for w, s in pairs:
        if any(ceilings[t] is None or ceilings[t] <= 0.55 for t in (w, s)):
            continue                                   # ceiling too close to chance to normalize
        pk = f"{w}->{s}"
        matching_norm[pk] = {}
        for d in deltas:
            per = {k: match_level(norm_curve(w, k), norm_curve(s, k)[0], d)
                   for k in curves[w] if k in curves[s]}
            km, frac = km_median_level([v for k, v in per.items() if not k.startswith("planted")],
                                       n_levels)
            matching_norm[pk][str(d)] = {"per_item": per, "km_median_level": km,
                                         "matched_fraction_real_metrics": frac}

    triangles = {}
    if len(tags) >= 3:
        w, m, s = tags[0], tags[1], tags[-1]
        for d in deltas:
            triangles[str(d)] = {k: triangle_slack(curves[w][k], curves[m][k], curves[s][k], d)
                                 for k in curves[w] if k in curves[m] and k in curves[s]}

    # type-tagged marginal gains: gain[type][reader] = median over items of (curve[l]-curve[l-1])
    gains = {}
    for tag in tags:
        gains[tag] = {}
        for key, cv in curves[tag].items():
            if key.startswith("planted"):        # saturated controls would dilute type medians
                continue
            sched = chains[key]["schedule"]
            for li in range(1, n_levels):
                typ = sched[li - 1]
                if not (np.isnan(cv[li]) or np.isnan(cv[li - 1])):
                    gains[tag].setdefault(typ, []).append(cv[li] - cv[li - 1])
        gains[tag] = {t: round(float(np.median(v)), 4) for t, v in gains[tag].items()}

    planted_floor = {}
    for tag in tags:
        floors, flags = {}, {}
        for key in planted:
            if key in curves[tag]:
                ref = refs[key]
                minor = min(ref.mean(), 1 - ref.mean())
                flags[key] = f"imbalanced gold ({minor:.0%} minority)" if minor < 0.10 else "ok"
                lv = next((li for li, v in enumerate(curves[tag][key])
                           if not np.isnan(v) and v >= 0.9), None)
                floors[key] = lv
        planted_floor[tag] = {"first_level_reaching_0.9": floors, "gold_flags": flags}

    out = {"readers": tags, "n_levels": n_levels,
           "reference_provenance": {
               **ref_prov,
               "anchored_semantics": "curves/matching/triangles score every reader against the "
                                     "reference executor's own M_i -- the family-top transmission "
                                     "yardstick iso-performance needs; NOT the decompression "
                                     "estimand"},
           "self_readout_semantics": "PRIMARY executor-consistent decompression: i_binary bits of "
                                     "each reader's OWN full-rubric orbit judgment per level, "
                                     "gated H_self >= 0.15 (D1)",
           "curves": {t: curves[t] for t in tags},
           "self_readout": self_readout,
           "matching": matching,
           "compliance_ceilings": ceilings,
           "matching_compliance_normalized": matching_norm,
           "triangles": triangles,
           "marginal_gain_by_type": gains, "planted_floor": planted_floor,
           "delta_grid": deltas, "n_boot": a.n_boot}
    path = os.path.join(a.out_dir, "expansion_report.json")
    json.dump(out, open(path, "w"), indent=1)
    for pk, mv in matching.items():
        d0 = mv[str(deltas[0])]
        print(f"{pk}: KM median level {d0['km_median_level']}, matched "
              f"{d0['matched_fraction_real_metrics']:.0%} (delta={deltas[0]})")
    print(f"wrote {path}")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--task", default="creative-writing")
    p.add_argument("--ref-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--r2-bucket", default="general")
    p.add_argument("--gi-list", default="", help="comma gi filter (the expansion selection)")
    p.add_argument("--reverse-gis", default="", help="order-control arm: reversed type schedule")
    p.add_argument("--phase", default="all", choices=["chain", "score", "report", "all"])
    p.add_argument("--writer-model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--readers", default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--forms", type=int, default=3)
    p.add_argument("--planted", type=int, default=1)
    p.add_argument("--n-probes", type=int, default=300)
    p.add_argument("--gepa-reserve", type=int, default=60)
    p.add_argument("--delta-grid", default="0,0.02,0.05")
    p.add_argument("--n-boot", type=int, default=500)
    p.add_argument("--fake", action="store_true")
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    cfg = cfgmod.apply_task_preset(cfgmod.ImplementerConfig(), a.task)
    gi_list = [int(x) for x in a.gi_list.split(",") if x.strip()] if a.gi_list else None
    ckpts = _ckpts(a.ref_dir, gi_list)
    if not ckpts:
        raise SystemExit(f"no matching checkpoints in {a.ref_dir}")
    probe_texts, _ = _load_texts(a.task, a.gepa_reserve + a.n_probes, cfg)
    probe_texts = probe_texts[a.gepa_reserve: a.gepa_reserve + a.n_probes]
    planted = build_planted(probe_texts, cfg.max_text_chars) if a.planted else {}
    for k, (_, _, rule) in planted.items():         # gold balance on the truncated view, up front
        g = np.array([bool(rule(t)) for t in probe_texts])
        print(f"  planted gold {k}: {g.mean():.0%} positive")
    print(f"expansion chains over {len(ckpts)} metrics (+{len(planted)} "
          f"planted), {len(probe_texts)} probes, {len(STEPS) + 1} levels")

    if a.phase in ("chain", "all"):
        chains = build_chains(a, ckpts, planted)
    else:
        chains = json.load(open(os.path.join(a.out_dir, "chains.json")))
    if a.phase in ("score", "all"):
        for reader in [r.strip() for r in a.readers.split(",") if r.strip()]:
            score_reader(a, cfg, reader, chains, probe_texts)
    if a.phase in ("report", "all"):
        report(a, ckpts, probe_texts, planted)


if __name__ == "__main__":
    main()
