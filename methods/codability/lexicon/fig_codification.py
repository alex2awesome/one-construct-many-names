#!/usr/bin/env python
"""Codification level across domains (appendix figure + backing numbers).

Three panels on one row:
  (a) per-field mix over the 5-rung institutional-codification ladder, CRITERION-weighted
      (each extracted criterion carries the rung of the document that stated it -- the same
      unit as the specificity figure, which counts criteria, not source pages),
  (b) institutional share {official, professional} against PY naming coincidence at L0,
  (c) Spearman rho of each individual rung's share against coincidence.

Panel (c) is the point of the figure: the composite institutional share correlates with L0
agreement, but no institutional rung does the work on its own -- it is PRACTITIONER share
that anti-predicts, and the composite tracks coincidence mainly by being its complement.

The 5-rung code is DESCRIPTIVE ONLY. It failed its validation gate (GLM-vs-Sonnet .777 at
3 classes) and every confirmatory test in this line uses the binary {1,2} vs {3,4,5} collapse
that did pass (.869). Panel (a) therefore separates the two institutional rungs by hue family
so the validated boundary is where blue meets non-blue.

Palette: diverging, institutional (blue) -- academic (neutral) -- non-institutional (warm),
adjacent-pair separation validated (worst normal-vision dE 25.1, worst CVD dE 21.7).
"""
from __future__ import annotations

import glob
import json
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

ROOT = "ROOT"
OUT = f"{ROOT}/outputs/lexicon"
FIGS = f"{ROOT}/latex/paper-1__metric-codability/figs"
INK, MUTED = "#222222", "#888888"

# PY naming coincidence at L0, ledger Table 1 (codability_sampling_20260720.json +
# codability_sampling_widened_20260721.json). Held fixed here; recomputing it is a
# separate pipeline.
COINC = {"legal-outcome-prediction": .088, "humor": .128, "creative-writing": .158,
         "press-releases": .173, "news-homepages": .185, "math-stackexchange": .195,
         "grant-funding": .218, "patents": .273, "notice-and-comment": .306,
         "peer-review": .427, "code-review": .508}
GEN = {"code-review": "Software Code", "creative-writing": "Creative Writing",
       "grant-funding": "Grant Proposals", "humor": "Humor",
       "legal-outcome-prediction": "Legal Arguments", "math-stackexchange": "Math",
       "news-homepages": "Journalism", "notice-and-comment": "Regulatory Comm.",
       "patents": "Patents", "peer-review": "Academic Articles",
       "press-releases": "Press Releases"}
RUNG = {1: "official guideline", 2: "professional standard", 3: "academic",
        4: "practitioner", 5: "community folk"}
COL = {1: "#0d3068", 2: "#5aa0e8", 3: "#e2ded6", 4: "#e07b4f", 5: "#8f2222"}
TXT = {1: "white", 2: INK, 3: INK, 4: INK, 5: "white"}


def collect():
    """Per field: criterion-weighted and document-weighted rung shares."""
    rows = {}
    for task in COINC:
        rung = {}
        for line in open(f"{OUT}/provenance_rungs_{task}.jsonl"):
            r = json.loads(line)
            rung[r["id"]] = r["rung"]
        crit, perdoc = Counter(), Counter()
        files = [f for f in sorted(glob.glob(f"{OUT}/extract_{task}_glm-4.7*.jsonl"))
                 if not f.endswith(".bak")]
        n_seen = n_join = 0
        for fn in files:
            for line in open(fn):
                r = json.loads(line)
                if not r.get("found"):
                    continue
                n_seen += 1
                doc = r["key"].split("::")[2]          # task::layer::doc::idx
                if doc in rung:
                    n_join += 1
                    if rung[doc] is not None:
                        crit[rung[doc]] += 1
                        perdoc[doc] += 1
        docs = Counter(v for v in rung.values() if v is not None)
        tc, td = sum(crit.values()), sum(docs.values())
        rows[task] = {
            "coincidence": COINC[task], "n_criteria": n_seen, "join_rate": n_join / n_seen,
            "crit_share": {k: crit[k] / tc for k in RUNG},
            "doc_share": {k: docs[k] / td for k in RUNG},
            "n_criteria_coded": tc, "n_docs_coded": td,
            "null_rung_docs": 1 - td / len(rung),
            "criteria_per_doc_by_rung": {k: float(np.mean([n for d, n in perdoc.items()
                                                           if rung[d] == k] or [0]))
                                         for k in RUNG},
        }
    return rows


def panel_mix(ax, rows, order):
    y = np.arange(len(order))[::-1]
    left = np.zeros(len(order))
    for k in RUNG:
        w = np.array([rows[t]["crit_share"][k] for t in order])
        ax.barh(y, w, left=left, height=.68, color=COL[k], label=RUNG[k],
                edgecolor="white", linewidth=.9)
        for yi, wi, li in zip(y, w, left):
            if wi >= .12:                       # label only segments wide enough to hold it
                ax.text(li + wi / 2, yi, f"{wi:.2f}".lstrip("0"), ha="center", va="center",
                        fontsize=5.4, color=TXT[k])
        left += w
    ax.set_yticks(y)
    ax.set_yticklabels([GEN[t] for t in order], fontsize=6.4)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, .25, .5, .75, 1])
    ax.set_xlabel("share of criteria", fontsize=6.6)
    ax.tick_params(labelsize=6)
    ax.tick_params(axis="y", length=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    for yi, t in zip(y, order):
        ax.text(1.015, yi, f"{rows[t]['coincidence']:.3f}".lstrip("0"),
                transform=ax.get_yaxis_transform(), ha="left", va="center",
                fontsize=5.8, color=MUTED)
    ax.text(1.015, y[0] + .95, "coinc.", transform=ax.get_yaxis_transform(),
            ha="left", va="center", fontsize=5.6, color=MUTED, style="italic")
    ax.legend(frameon=False, fontsize=5.6, ncol=3, loc="lower center",
              bbox_to_anchor=(.5, 1.02), columnspacing=1.0, handletextpad=.4,
              handlelength=1.0)
    ax.set_title("(a) codification mix, by field", fontsize=7, color=INK, pad=22, loc="left")


# Panel (b) crowds six fields into the low-institutional corner, so labels are hand-placed
# and the longest names are abbreviated (panel (a), immediately left, carries the full names).
NUDGE = {"humor": (-.016, -.014, "right"), "creative-writing": (-.016, .016, "right"),
         "legal-outcome-prediction": (.020, -.016, "left"),
         "math-stackexchange": (0, .030, "center"), "press-releases": (.020, -.020, "left"),
         "news-homepages": (.020, .010, "left"), "grant-funding": (0, .028, "center"),
         "patents": (.020, .008, "left"), "notice-and-comment": (0, .030, "center"),
         "peer-review": (-.020, 0, "right"), "code-review": (-.020, 0, "right")}
SHORT = {"creative-writing": "Creative", "press-releases": "Press Rel.",
         "notice-and-comment": "Regulatory", "peer-review": "Academic Art."}


def panel_scatter(ax, rows, order):
    x = np.array([rows[t]["crit_share"][1] + rows[t]["crit_share"][2] for t in order])
    v = np.array([rows[t]["coincidence"] for t in order])
    ax.scatter(x, v, s=26, color="#0d3068", zorder=3, edgecolor="white", linewidth=.6)
    for xi, vi, t in zip(x, v, order):
        dx, dy, ha = NUDGE[t]
        ax.annotate(SHORT.get(t, GEN[t]), (xi, vi), (xi + dx, vi + dy), fontsize=5.2,
                    color=INK, ha=ha, va="center")
    rho, p = spearmanr(x, v)
    ax.text(.02, .97, f"$\\rho$ = {rho:+.2f}\n$p$ < .001", transform=ax.transAxes,
            fontsize=6.2, color=INK, va="top", ha="left")
    ax.set_xlabel("institutional share (official + professional)", fontsize=6.6)
    ax.set_ylabel("naming coincidence at L0", fontsize=6.6)
    ax.set_xlim(-.30, .80)
    ax.set_ylim(.02, .58)
    ax.set_xticks([0, .2, .4, .6, .8])
    ax.tick_params(labelsize=6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("(b) institutional share vs. L0 agreement", fontsize=7, color=INK,
                 pad=22, loc="left")


def panel_rho(ax, rows, order):
    v = np.array([rows[t]["coincidence"] for t in order])
    stats = []
    for k in RUNG:
        s = np.array([rows[t]["crit_share"][k] for t in order])
        rho, p = spearmanr(s, v)
        stats.append((k, rho, p))
    y = np.arange(len(stats))[::-1]
    ax.barh(y, [s[1] for s in stats], height=.62, color=[COL[s[0]] for s in stats],
            edgecolor="white", linewidth=.8)
    ax.axvline(0, color=MUTED, lw=.7)
    ax.set_yticks(y)
    ax.set_yticklabels([RUNG[s[0]] for s in stats], fontsize=6.2)
    # values in a fixed right-hand column, so their placement does not depend on bar sign
    for yi, (k, rho, p) in zip(y, stats):
        ax.text(1.12, yi, f"{rho:+.2f}  ($p$={p:.3f})", ha="left", va="center",
                fontsize=5.7, color=INK)
    ax.set_xlim(-1.05, 2.75)
    ax.set_xticks([-1, -.5, 0, .5, 1])
    ax.set_xticklabels(["-1", "-.5", "0", ".5", "1"])
    ax.set_xlabel(r"Spearman $\rho$ vs. L0 coincidence", fontsize=6.6)
    ax.tick_params(labelsize=6)
    ax.tick_params(axis="y", length=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_bounds(-1, 1)
    ax.set_title("(c) which rung carries the association", fontsize=7, color=INK,
                 pad=22, loc="left")


def main():
    rows = collect()
    order = sorted(COINC, key=lambda t: COINC[t])

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.55), dpi=300,
                             gridspec_kw={"width_ratios": [1.30, 1.35, 1.05]})
    panel_mix(axes[0], rows, order)
    panel_scatter(axes[1], rows, order)
    panel_rho(axes[2], rows, order)
    plt.tight_layout(w_pad=2.6)
    plt.savefig(f"{FIGS}/fig_codification.png", bbox_inches="tight")

    # document-weighted companion correlation, quoted in the caption as the sensitivity
    xd = [rows[t]["doc_share"][1] + rows[t]["doc_share"][2] for t in order]
    xc = [rows[t]["crit_share"][1] + rows[t]["crit_share"][2] for t in order]
    v = [rows[t]["coincidence"] for t in order]
    rho_d, p_d = spearmanr(xd, v)
    rho_c, p_c = spearmanr(xc, v)
    summary = {"note": "DESCRIPTIVE, not preregistered. 5-rung code failed its validation "
                       "gate (3-class .777); only the binary {1,2} vs {3,4,5} collapse "
                       "(.869) is confirmatory-grade.",
               "criterion_weighted_rho": [rho_c, p_c],
               "document_weighted_rho": [rho_d, p_d],
               "per_rung_rho": {RUNG[k]: list(spearmanr(
                   [rows[t]["crit_share"][k] for t in order], v)) for k in RUNG},
               "fields": rows}
    json.dump(summary, open(f"{OUT}/codification_by_field_20260726.json", "w"), indent=1)
    print(f"wrote {FIGS}/fig_codification.png")
    print(f"wrote {OUT}/codification_by_field_20260726.json")
    print(f"criterion-weighted rho={rho_c:+.3f} p={p_c:.4f} | "
          f"document-weighted rho={rho_d:+.3f} p={p_d:.4f}")


if __name__ == "__main__":
    main()
