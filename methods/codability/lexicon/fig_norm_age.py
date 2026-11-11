#!/usr/bin/env python
"""Norm-age figures: main (3 panels, two-column) and appendix (remaining domains).

Extracted from the paper-1 figure notebook so the theme selection is reviewable. The only
change to the plotting logic is EMERGING_PICK: for several domains the automatic
"highest-median" rule surfaced recent-only themes whose NAMES read as dated even though
their sources are recent (math's "Canonical and Normal Forms"). Those rows say little about
what recently entered the domain's norm vocabulary. EMERGING_PICK names the rows to show,
chosen from the same eligible pool (median year >= 2022, n >= 3, earliest source >= 2010) --
it restricts the display, it does not relax the eligibility test.
"""
from __future__ import annotations

import re

import matplotlib.pyplot as plt
import pandas as pd

ROOT = "ROOT"
FIGS = f"{ROOT}/latex/paper-1__metric-codability/figs"
BLUE, RED, INK, MUTED = "#3b82d9", "#c73a3a", "#222222", "#888888"
X0, X1, XB, XBRK = 1418, 2032, 1432, 1458

GEN = {"code-review": "Software Code", "creative-writing": "Creative Writing",
       "grant-funding": "Grant Proposals", "humor": "Humor",
       "legal-outcome-prediction": "Legal Arguments", "math-stackexchange": "Mathematical Writing",
       "news-homepages": "Journalism", "notice-and-comment": "Regulatory Comments",
       "patents": "Patents", "peer-review": "Academic Publishing",
       "press-releases": "Press Releases"}

# recent-only rows to display, by domain (see module docstring)
EMERGING_PICK = {
    "math-stackexchange": ["Benchmarking and Evaluation Protocols",
                           "Sage/Cython Coding Practices",
                           "Test Suite and Benchmark Design"],
    "news-homepages": ["Audience Feedback and Personalization",
                       "Multimedia and Video Assets",
                       "Solutions and Constructive Journalism"],
    "peer-review": ["AI Use and Disclosure",
                    "Citation Integrity and Provenance",
                    "Publication Misconduct and Retractions"],
    "code-review": ["Supply Chain and Dependency Hygiene",
                    "Secrets and Credential Handling",
                    "Container and Deployment Config"],
    "grant-funding": ["Data Management and Sharing Plans",
                      "Broader Impacts and Equity",
                      "Open Access and Dissemination"],
    "creative-writing": ["Sensitivity and Representation",
                         "Content Warnings and Labels",
                         "Platform and Format Conventions"],
}


def _load():
    m = pd.read_parquet(f"{ROOT}/outputs/r2_attr_labeling/agg/"
                        "r2_aspect_year_mentions_distinct_sources.parquet")
    # Gutenberg reprints are dated by fetch, not by the work; they would fake ancient anchors
    m = m[~(m.source_file.str.contains("gutenberg", case=False) & (m.year > 1990))]
    m["stem"] = m.aspect_name.str.split(r" and |, |: ").str[0].str.split().str[:2].str.join(" ")
    return m


# display labels: the generic truncator produces stubs ("Avoidance of", "Sage") for these
NAME_OVERRIDE = {
    "Figures and Tables Quality": "Presentation Quality",
    "Avoidance of Sensationalism and Clickbait": "Anti-Sensationalism",
    "Explanatory Journalism and Contextual Background": "Explanatory Context",
    "Balance, Impartiality and Objectivity": "Balance & Impartiality",
    "Domain-Specific Empirical Formulas and Adjustments": "Empirical Formulas",
    "Logical Soundness and Validity": "Logical Soundness",
    "Definitions and Terminology": "Definitions & Terms",
    "Sage/Cython Coding Practices": "Sage/Cython Practice",
    "Test Suite and Benchmark Design": "Benchmark Design",
    "Benchmarking and Evaluation Protocols": "Benchmarking Protocols",
    "Audience Feedback and Personalization": "Audience Feedback",
    "Solutions and Constructive Journalism": "Constructive Journalism",
    "Multimedia and Video Assets": "Multimedia Assets",
    "Publication Misconduct and Retractions": "Publication Misconduct",
    "Citation Integrity and Provenance": "Citation Integrity",
    "Open Science and Pre-registration": "Open Science",
    "Limitations Acknowledgement": "Limitations",
    "AI Use and Disclosure": "AI Use & Disclosure",
}


def short(nm, maxw=2):
    if nm in NAME_OVERRIDE:
        return NAME_OVERRIDE[nm]
    if "Figures and Tables" in nm:
        return "Presentation Quality"
    seg = re.split(r", |: |—|/", nm)[0].strip()
    first = re.split(r" and ", seg)[0].strip()
    if len(first.split()) == 1 and " and " in seg:
        return " & ".join(x.strip() for x in seg.split(" and ")[:2])
    w = first.split()
    return " ".join(w[:3]) if len(" ".join(w[:2])) < 7 else " ".join(w[:maxw])


def yfmt(y):
    return f"{abs(int(y))} BCE" if y < 0 else str(int(y))


def norm_age_panel(ax, task, mentions, label_fs=7, title=None, group_labels=True):
    df = mentions[mentions.task == task]
    stats = (df.groupby("stem")
               .agg(aspect_name=("aspect_name", "first"), **{"min": ("year", "min"),
                    "max": ("year", "max"), "median": ("year", "median"), "count": ("year", "count")})
               .reset_index())
    per = stats[stats["count"] >= 5].nsmallest(3, "min")
    elig = stats[(stats["median"] >= 2022) & (stats["count"] >= 3) & (stats["min"] >= 2010)]
    pick = EMERGING_PICK.get(task)
    if pick:
        chosen = elig[elig.aspect_name.isin(pick)]
        # any name that is not eligible is dropped, then topped up by the automatic rule
        emer = pd.concat([chosen, elig[~elig.aspect_name.isin(pick)].nlargest(3, "median")]).head(3)
    else:
        emer = elig.nlargest(3, "median")
    if not len(emer):
        emer = stats.nlargest(3, "median").head(3)

    allmin = min(list(per["min"]) + list(emer["min"])) if len(per) else 1500
    ancient = allmin < 1500
    x0 = 1418 if ancient else int((max(allmin, 1500) // 50) * 50) - 30
    yy = 0
    for grp, col in ((per, BLUE), (emer, RED)):
        for r in grp.itertuples():
            ys = df[df.stem == r.stem]["year"]
            lo = max(r.min, 1500) if ancient else r.min
            ax.hlines(-yy, lo, r.max, color=col, lw=5.5, alpha=.25)
            keep = ys >= (1500 if ancient else x0)
            ax.plot(ys[keep], [-yy] * int(keep.sum()), "|", color=col, ms=5, mew=1.0)
            if r.min < 1500:
                ax.plot([XB, XBRK - 8], [-yy, -yy], ls=":", color=col, lw=1.0)
                ax.plot([XB], [-yy], "o", mfc="none", mec=col, ms=4)
                ax.text(XB, -yy + .34, yfmt(r.min), fontsize=label_fs - 1.5, color=col,
                        style="italic", ha="left")
            ax.text(-0.01, -yy, short(r.aspect_name), transform=ax.get_yaxis_transform(),
                    ha="right", va="center", fontsize=label_fs)
            ax.text(1.01, -yy, f"n={r.count}", transform=ax.get_yaxis_transform(),
                    ha="left", va="center", fontsize=label_fs - 1.5, color=MUTED)
            yy += 1
        if col is BLUE:
            ax.axhline(-yy + .5, color="#cccccc", lw=.6, ls="--")
            if not group_labels:
                continue
            # stacked either side of the divider, in the left margin of the plot area:
            # every bar starts well right of x=.02, so nothing is overprinted
            ax.text(.015, -yy + .74, r"$\uparrow$ perennial", transform=ax.get_yaxis_transform(),
                    ha="left", va="center", fontsize=label_fs - 1.8, color=BLUE,
                    style="italic", zorder=6)
            ax.text(.015, -yy + .26, r"$\downarrow$ recent-only",
                    transform=ax.get_yaxis_transform(), ha="left", va="center",
                    fontsize=label_fs - 1.8, color=RED, style="italic", zorder=6)
    ax.set_xlim(x0, X1)
    ax.set_ylim(-yy + .4, .85)
    ax.set_yticks([])
    ax.set_xticks([1500, 1750] if ancient
                  else [t for t in (1800, 1900, 2000) if t > x0 + 10])
    ax.tick_params(labelsize=label_fs - 1)
    ax.spines[["top", "right", "left"]].set_visible(False)
    for dx in ((-4, 4) if ancient else ()):
        ax.plot([XBRK + dx - 3, XBRK + dx + 3], [ax.get_ylim()[0]] * 2, color="white", lw=3,
                clip_on=False, zorder=4)
        ax.plot([XBRK + dx - 4, XBRK + dx + 4],
                [ax.get_ylim()[0] - .12, ax.get_ylim()[0] + .12], color=MUTED, lw=.8,
                clip_on=False, zorder=5)
    ax.set_title(title or GEN[task], fontsize=label_fs + 1.5, color=INK, pad=3)


def main_figure(tasks=("peer-review", "news-homepages", "math-stackexchange")):
    m = _load()
    fig, axes = plt.subplots(1, len(tasks), figsize=(7.1, 1.85), dpi=300)
    for ax, t in zip(axes, tasks):
        norm_age_panel(ax, t, m, label_fs=6.2)
        ax.set_xlabel("source year", fontsize=6.5)
    plt.tight_layout(w_pad=3.2)
    plt.savefig(f"{FIGS}/fig_norm_age_main.png", bbox_inches="tight")
    print(f"wrote {FIGS}/fig_norm_age_main.png ({len(tasks)} panels)")


def appendix_figure():
    m = _load()
    rest = [t for t in GEN if t in set(m.task)
            and t not in ("peer-review", "news-homepages", "math-stackexchange")]
    ncol = 2                      # 3 columns squeezed the labels unreadably
    nrow = -(-len(rest) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.1, 1.72 * nrow), dpi=300)
    for ax, t in zip(axes.ravel(), rest):
        norm_age_panel(ax, t, m, label_fs=6.2)
        ax.set_xlabel("source year", fontsize=6.5)
    for ax in axes.ravel()[len(rest):]:
        ax.axis("off")
    plt.tight_layout(w_pad=4.0, h_pad=1.9)
    plt.savefig(f"{FIGS}/fig_norm_age_appendix.png", bbox_inches="tight")
    print(f"wrote {FIGS}/fig_norm_age_appendix.png ({len(rest)} panels)")


if __name__ == "__main__":
    main_figure()
    appendix_figure()
