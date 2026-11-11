#!/usr/bin/env python
"""Stipulation + nonce-definition probe (confounder battery E1/E2, imported from the code-seam
E* battery 2026-07-05).

The decompression grid is ELIMINATIVE — it rules confounds out. These two rungs are the
POSITIVE existence tests that separate "the name is a retrieval key into enculturated
competence" from "the definition is a self-contained spec any instruction-follower executes":

  nonce_def (E1) — the real definition with the construct NAME string swapped for a nonce
      ("the smorbex property"). Isolates the name's retrieval contribution from the
      definition's specification contribution:
        nonce_def ~= def   => the name inside the definition is ceremonial (spec suffices)
        nonce_def <  def   => the name carries content the prose cannot restate (retrieval)

  stip (E2) — a DEVIANT in-prompt redefinition: '"<real name>" means <orthogonal rule>'
      where the rule is a surface property (from pseudo_concept_ladder.STATS) chosen to be
      maximally UNcorrelated with the construct's real extension. Two readouts on the same
      scores:
        compliance = AUC(score, deviant-rule truth)   -- did it obey the stipulation?
        snap_back  = AUC(score, real construct ref)    -- did the name's community meaning
                                                           override the in-prompt text?
      H_spec predicts compliance ~ 1, snap_back ~ chance (a competent follower just executes
      the given rule). Retrieval/enculturation predicts snap_back stays high and compliance is
      dragged down: you cannot re-engineer the field by editing the definition, because the
      name gravitationally pulls back to what the culture means by it (the lexical-Stroop /
      MAGNIFICo prediction, lifted to construct level).

Scoring reuses the exact grid machinery (ap.signature via make_judge_backend). Probe window
MUST match the grid's (--gepa-reserve 60, --n-probes 300). Refs are the executor's own
verdicts (reconstruction-only; never labels). ONE reader per invocation (avoids the
shared-engine OOM); loop readers in the launcher.
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
from methods.codability.pseudo_concept_ladder import STATS, RULE_TEXT

NONCE = "the smorbex property"
STIP_TMPL = ('For the purposes of THIS task only, "{name}" is defined to mean exactly the '
             'following, and nothing else: a text satisfies it if and only if {rule}. '
             'Judge each text strictly by this definition, ignoring any other sense of '
             '"{name}".')
SWAP_TMPL = ('For the purposes of THIS task only, "{name}" is defined to mean exactly the '
             'following, and nothing else: {defn} Judge each text strictly by this '
             'definition, ignoring any other sense of "{name}".')


def _rank(a):
    uniq, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    return (np.cumsum(cnt) - (cnt - 1) / 2.0)[inv]


def auc_mw(scores, labels):
    labels = np.asarray(labels, bool)
    pos = int(labels.sum())
    if pos == 0 or pos == len(labels):
        return None
    r = _rank(np.asarray(scores, float))
    u = r[labels].sum() - pos * (pos + 1) / 2.0
    return float(u / (pos * (len(labels) - pos)))


def _load_refs(ref_dir):
    out = {}
    for f in sorted(glob.glob(os.path.join(ref_dir, "*_sigs.npz"))):
        m = re.search(r"_metric(\d+)_sigs\.npz$", os.path.basename(f))
        if not m:
            continue
        z = np.load(f, allow_pickle=True)
        out[m.group(1)] = np.nan_to_num(np.asarray(z["M_i"], float), nan=0.5)
    return out


def nonce_labeled_definition(name: str, definition: str, nonce: str = NONCE) -> str:
    """Relabel a definition and remove exact, case-insensitive occurrences of the real name.

    Merely changing the label before the colon leaves E1 contaminated whenever the definition
    repeats the construct name.  This helper keeps the surrounding prose fixed while replacing
    every literal name occurrence in the body as well.
    """
    body = re.sub(re.escape(name), nonce, str(definition), flags=re.IGNORECASE)
    return f'"{nonce}": {body}' if body else ""


def build_rungs(msgs, refs, probe_texts, exclude_masks):
    """Per gi: real name/def, nonce_def, and the most-orthogonal deviant stipulation."""
    stat_truth = {k: (v := np.array([fn(t) for t in probe_texts])) > np.median(v)
                  for k, fn in STATS.items()}
    items = {}
    for gi, m in msgs.items():
        if gi not in refs or "name" not in m.get("rungs", {}):
            continue
        name = m["name"]
        rdef = m["rungs"].get("definition", "")
        # E1: present the SAME definition under the real name vs a nonce label. The contrast
        # named_def - nonce_def isolates the name's retrieval value with the full spec held
        # fixed (name-only vs nonce+full-def vs name+full-def, the code-seam E1 KEY test).
        named_def = f'"{name}": {rdef}' if rdef else ""
        nonce_def = nonce_labeled_definition(name, rdef)
        # E2: pick deviant rule least correlated with the real extension (on scored probes)
        mask = exclude_masks[gi]
        real = refs[gi][mask] > 0.5
        scored = []
        for stat, truth in stat_truth.items():
            tv = truth[mask]
            if not (0.25 <= tv.mean() <= 0.75):   # balanced split => clean compliance AUC
                continue
            c = abs(np.corrcoef(real.astype(float), tv.astype(float))[0, 1])
            if np.isfinite(c):   # degenerate refs (constant labels) -> no valid E2 for this gi
                scored.append((c, stat, truth))
        if not scored:
            continue
        scored.sort(key=lambda x: x[0])
        best_abs, stat, truth = scored[0]
        thr = float(np.median(np.array([STATS[stat](t) for t in probe_texts])))
        rule = RULE_TEXT[stat].format(thr=thr)
        rungs = {"name": name, "definition": rdef, "named_def": named_def,
                 "nonce_def": nonce_def, "stip": STIP_TMPL.format(name=name, rule=rule),
                 # E2 control: SAME deviant rule under a nonce name (no real-meaning pull).
                 # compliance(stip_nonce) >> compliance(stip) isolates the real name's
                 # suppression of the stipulation = the snap-back mechanism, airtight.
                 "stip_nonce": STIP_TMPL.format(name=NONCE, rule=rule)}
        item = {"name": name, "rungs": rungs, "deviant_stat": stat, "deviant_truth": truth,
                "deviant_orthogonality": round(best_abs, 3)}
        # GENERALIZATION 1 (rule-robustness): a SECOND orthogonal rule per metric — is
        # suppression invariant to which mechanical rule plays the deviant meaning?
        if len(scored) > 1:
            c2, stat2, truth2 = scored[1]
            thr2 = float(np.median(np.array([STATS[stat2](t) for t in probe_texts])))
            rule2 = RULE_TEXT[stat2].format(thr=thr2)
            rungs["stip2"] = STIP_TMPL.format(name=name, rule=rule2)
            rungs["stip2_nonce"] = STIP_TMPL.format(name=NONCE, rule=rule2)
            item.update(deviant2_stat=stat2, deviant2_truth=truth2,
                        deviant2_orthogonality=round(c2, 3))
        items[gi] = item
    # GENERALIZATION 2 (semantic redefinition): rebind each name to the DEFINITION OF THE
    # MOST-ORTHOGONAL OTHER METRIC — commitment tested against plausible semantic content,
    # not just mechanical rules. Compliance target = the partner metric's own extension.
    gis = list(items)
    for gi in gis:
        best = None
        for gj in gis:
            if gj == gi or not items[gj]["rungs"].get("definition"):
                continue
            lab_j = refs[gj][exclude_masks[gi]] > 0.5
            if not (0.25 <= lab_j.mean() <= 0.75):
                continue
            c = abs(np.corrcoef(refs[gi][exclude_masks[gi]], refs[gj][exclude_masks[gi]])[0, 1])
            if best is None or c < best[0]:
                best = (c, gj)
        if best is None:
            continue
        c, gj = best
        name = items[gi]["name"]
        defn = items[gj]["rungs"]["definition"]
        items[gi]["rungs"]["stip_swap"] = SWAP_TMPL.format(name=name, defn=defn)
        items[gi]["rungs"]["stip_swap_nonce"] = SWAP_TMPL.format(name=NONCE, defn=defn)
        items[gi]["swap_partner"] = gj
        items[gi]["swap_corr"] = round(c, 3)
        items[gi]["swap_truth"] = refs[gj] > 0.5
    return items


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--reader", required=True, help="ONE reader model (per-process, avoids OOM)")
    p.add_argument("--ref-dir", required=True)
    p.add_argument("--msgs", required=True, help="grid messages.json (real name/def rungs)")
    p.add_argument("--out", required=True)
    p.add_argument("--n-probes", type=int, default=300)
    p.add_argument("--gepa-reserve", type=int, default=60)
    p.add_argument("--gi-list", default="", help="comma gi filter; empty=all")
    p.add_argument("--sig-dir", default="", help="save raw per-probe signatures + labels "
                   "(npz per reader) — enables probe-clustered bootstrap of suppress/compliance")
    p.add_argument("--build-only", action="store_true", help="no GPU: emit rung texts + stats")
    a = p.parse_args()

    cfg = cfgmod.apply_task_preset(cfgmod.ImplementerConfig(), a.task)
    probe_texts, _ = _load_texts(a.task, a.gepa_reserve + a.n_probes, cfg)
    probe_texts = probe_texts[a.gepa_reserve: a.gepa_reserve + a.n_probes]

    msgs = json.load(open(a.msgs))
    if a.gi_list:
        keep = set(a.gi_list.split(","))
        msgs = {k: v for k, v in msgs.items() if k in keep}
    refs = _load_refs(a.ref_dir)
    ex = {}
    for gi in refs:
        mask = np.ones(len(probe_texts), bool)
        m = msgs.get(gi) or {}
        held = m.get("exemplar_idx") or {}
        mask[(held.get("pos") or []) + (held.get("neg") or [])] = False
        ex[gi] = mask
    items = build_rungs(msgs, refs, probe_texts, ex)
    print(f"{len(items)} metrics; mean deviant orthogonality "
          f"{np.mean([it['deviant_orthogonality'] for it in items.values()]):.3f} "
          f"(want ~0 = separable compliance/snap-back)")
    if a.build_only:
        json.dump({gi: {"name": it["name"], "deviant_stat": it["deviant_stat"],
                        "orthogonality": it["deviant_orthogonality"],
                        "rungs": it["rungs"]} for gi, it in items.items()},
                  open(a.out, "w"), indent=1)
        print(f"-> {a.out} (build-only)")
        return

    report = json.load(open(a.out)) if os.path.exists(a.out) else {}
    rep = report.setdefault(a.reader, {})
    executor = make_judge_backend(a.reader, cfg, temperature=None)
    sig_acc, sig_path = {}, None
    if a.sig_dir:
        os.makedirs(a.sig_dir, exist_ok=True)
        sig_path = os.path.join(a.sig_dir, a.reader.replace("/", "--") + ".npz")
        if os.path.exists(sig_path):
            z = np.load(sig_path, allow_pickle=True)
            sig_acc = {k: z[k] for k in z.files}
    for gi, it in items.items():
        if gi in rep:
            continue
        include = ex[gi]
        real_full = refs[gi] > 0.5
        real = real_full[include]
        row = {"deviant_stat": it["deviant_stat"], "orthogonality": it["deviant_orthogonality"]}
        for k in ("deviant2_stat", "deviant2_orthogonality", "swap_partner", "swap_corr"):
            if k in it:
                row[k] = it[k]
        STIP_MAP = {"stip": ("stip", "deviant_truth"), "stip_nonce": ("stipNonce", "deviant_truth"),
                    "stip2": ("stip2", "deviant2_truth"), "stip2_nonce": ("stip2Nonce", "deviant2_truth"),
                    "stip_swap": ("stipSwap", "swap_truth"),
                    "stip_swap_nonce": ("stipSwapNonce", "swap_truth")}
        for rung, txt in it["rungs"].items():
            if not txt:
                continue
            sig = np.nan_to_num(np.asarray(ap.signature(executor, txt, probe_texts,
                                                         cfg.max_text_chars), float), nan=0.5)
            sig_eval = sig[include]
            if a.sig_dir:
                sig_acc[f"{gi}|{rung}"] = sig
            if rung in STIP_MAP:
                pfx, tkey = STIP_MAP[rung]
                row[f"{pfx}_compliance"] = _r(auc_mw(sig_eval,
                                                       np.asarray(it[tkey], bool)[include]))
                row[f"{pfx}_snapback"] = _r(auc_mw(sig_eval, real))
            else:
                row[f"{rung}_auc"] = _r(auc_mw(sig_eval, real))
        rep[gi] = row
        json.dump(report, open(a.out, "w"), indent=1)
        if a.sig_dir:
            sig_acc[f"{gi}|LAB_real"] = real_full.astype(float)
            sig_acc[f"{gi}|LAB_include"] = include.astype(float)
            sig_acc[f"{gi}|LAB_deviant"] = np.asarray(it["deviant_truth"], float)
            if "deviant2_truth" in it:
                sig_acc[f"{gi}|LAB_deviant2"] = np.asarray(it["deviant2_truth"], float)
            if "swap_truth" in it:
                sig_acc[f"{gi}|LAB_swap"] = np.asarray(it["swap_truth"], float)
            np.savez(sig_path, **sig_acc)
        print(f"  {gi} {it['name'][:30]:30s} name={row.get('name_auc')} "
              f"def={row.get('definition_auc')} namedDef={row.get('named_def_auc')} "
              f"nonceDef={row.get('nonce_def_auc')} "
              f"| comply={row.get('stip_compliance')} snap={row.get('stip_snapback')}")
    print(f"[{a.reader}] done -> {a.out}")


def _r(v):
    return round(v, 4) if v is not None else None


if __name__ == "__main__":
    main()
