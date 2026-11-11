#!/usr/bin/env python
"""Appendix figure (secondary analysis, DESCRIPTIVE / not preregistered): per-domain subfield
diffuseness vs. share of GENERAL-bucket norms, across the 11 census domains.

Motivating question (paper request): "domains that are more diffuse tend to have more general
norms" -- do they? Diffuseness is operationalized three ways in
outputs/lexicon/subfield_age_analyses_20260828.json (per-domain subfield count from the
gpt-parsed census's subtask_short field, treated as "subfield"; Shannon entropy of the
criterion-over-subfield distribution; subfields normalized per 1,000 criteria). Of the three,
only the size-normalized measure (subfields per 1,000 criteria) shows a clear, robust
correlation with %general -- raw subfield COUNT mostly just tracks how many source documents
got parsed per domain (the collection-size confound the paper request itself warned about), and
Shannon entropy is compressed near its ceiling in every domain (subtask_short is close to
per-document-unique) so it has little discriminating spread. This figure therefore plots
subfields-per-1,000-criteria (x) against %general (y), using the canon+newmetrics-merged
%general (the canon-only column has creative-writing pinned at a degenerate 100% because its
specific/hyper_specific labeling pass never ran -- see the JSON's
pct_general_source_files.creative_writing_artifact_warning -- so the merged column is the more
trustworthy one to plot).

All numbers are read from outputs/lexicon/subfield_age_analyses_20260828.json (already computed
by the secondary-analysis task; this script does no new aggregation, only plotting). Style
(colors, spines, fonts, savefig convention) copied from the other paper-1 figure scripts in this
directory, e.g. fig_form_scatter_labels.py / fig_norm_age.py -- single accent color, no seaborn,
no rainbow, direct point labels instead of a legend.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = "ROOT"
FIGS = f"{ROOT}/latex/paper-1__metric-codability/figs"
BLUE, INK, MUTED = "#2a78d6", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.linewidth": .6,
                      "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK,
                      "text.color": INK})

GEN = {"code-review": "Software Code", "creative-writing": "Creative Writing",
       "grant-funding": "Grant Proposals", "humor": "Humor",
       "legal-outcome-prediction": "Legal Arguments", "math-stackexchange": "Math",
       "news-homepages": "Journalism", "notice-and-comment": "Regulatory Comments",
       "patents": "Patents", "peer-review": "Academic Articles",
       "press-releases": "Press Releases"}

D = json.load(open(f"{ROOT}/outputs/lexicon/subfield_age_analyses_20260828.json"))
T1 = D["task1_subfield_diffuseness_vs_norm_generality"]
diff = T1["per_domain_diffuseness"]
pct_general = T1["per_domain_pct_general_canon_plus_newmetrics"]

tasks = sorted(diff.keys())
xs = np.array([diff[t]["subfields_per_1000_criteria"] for t in tasks])
ys = np.array([pct_general[t]["pct_general"] for t in tasks])
rho, p = stats.spearmanr(xs, ys)
# sanity-check against the precomputed correlation table (same measure, same %general column)
stored = T1["spearman_correlations"]["subfields_per_1000_criteria"][
    "spearman_vs_pct_general_canon_plus_newmetrics_n11"]
assert abs(rho - stored["rho"]) < 1e-6 and abs(p - stored["p"]) < 1e-6

fig, ax = plt.subplots(figsize=(5.6, 4.4), dpi=300)
ax.scatter(xs, ys, color=BLUE, s=34, zorder=3, linewidths=0)

# direct labels next to each point; small per-point offsets keep the 11 names from overlapping
# (hand-tuned against this specific point cloud -- 4 domains sit in a tight cluster around
# x=44-49, y=40-47 and need to fan out in different directions)
OFFSETS = {
    "code-review": (0.0, -8.5, "center", "top"),
    "creative-writing": (0.0, 3.2, "center", "bottom"),
    "grant-funding": (1.4, -3.2, "left", "top"),
    "humor": (1.4, 0.0, "left", "center"),
    "legal-outcome-prediction": (-1.4, 3.2, "right", "bottom"),
    "math-stackexchange": (-1.4, -3.2, "right", "top"),
    "news-homepages": (1.4, 3.2, "left", "bottom"),
    "notice-and-comment": (1.4, -4.5, "left", "top"),
    "patents": (0.0, -8.5, "center", "top"),
    "peer-review": (-1.4, -6.0, "right", "top"),
    "press-releases": (-1.4, 4.0, "right", "bottom"),
}
for t, x, y in zip(tasks, xs, ys):
    dx, dy, ha, va = OFFSETS[t]
    ax.annotate(GEN[t], (x, y), xytext=(x + dx, y + dy), fontsize=6.8, color=INK,
                ha=ha, va=va)

# OLS trend line on the plotted (x, y) pairs, for visual reference only (the reported statistic
# is the Spearman rank correlation, not this line's slope)
b1, b0 = np.polyfit(xs, ys, 1)
xg = np.linspace(xs.min() - 3, xs.max() + 3, 50)
ax.plot(xg, b0 + b1 * xg, color=MUTED, lw=.9, ls=":", zorder=1)

ax.set_xlabel("subfields per 1,000 criteria\n(collection-size-normalized diffuseness)")
ax.set_ylabel("share of criteria labeled general (%)")
ax.set_xlim(27, 73)
ax.set_ylim(-8, 112)
ax.spines[["top", "right"]].set_visible(False)
ax.text(0.97, 0.03, f"Spearman $\\rho$={rho:+.2f}, $p$={p:.3f}, $n$=11 domains\n"
                     f"(descriptive; not preregistered)",
        transform=ax.transAxes, fontsize=6.8, color=MUTED, ha="right", va="bottom")

plt.tight_layout()
plt.savefig(f"{FIGS}/fig_subfield_diffuseness.png", bbox_inches="tight")
print(f"wrote {FIGS}/fig_subfield_diffuseness.png  (rho={rho:.3f}, p={p:.4f}, n=11)")
