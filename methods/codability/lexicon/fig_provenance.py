#!/usr/bin/env python
"""Figure 13: source pages by page type, stacked by domain.

Change from the notebook version: bars are sorted by total height rather than grouped into
contiguous institutional / academic / community bands. Sorting and banding cannot both hold,
so the band a page type belongs to is now carried by the colour of its tick label instead of
by its position, and the dividers are gone.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = "ROOT"
FIGS = f"{ROOT}/latex/paper-1__metric-codability/figs"
MUTED = "#888888"

LBL = {"formal_guideline": "Formal guideline", "professional_standard": "Professional standard",
       "stylebook": "Stylebook", "textbook_excerpt": "Textbook excerpt",
       "course_syllabus": "Course syllabus", "tutorial": "Tutorial",
       "research_article": "Research article", "academic_page": "Academic page",
       "dataset": "Dataset", "wiki": "Wiki", "how_to": "How-to",
       "news_article": "News article", "blog_post": "Blog post", "forum_post": "Forum post",
       "contest_criteria": "Contest criteria", "other": "Other"}
BAND = {**{k: "inst" for k in ("formal_guideline", "professional_standard", "stylebook",
                               "textbook_excerpt", "course_syllabus", "tutorial")},
        **{k: "acad" for k in ("research_article", "academic_page", "dataset", "wiki", "how_to")},
        **{k: "comm" for k in ("news_article", "blog_post", "forum_post", "contest_criteria",
                               "other")}}
BANDC = {"inst": "#1e5fae", "acad": "#2a9d5a", "comm": "#c73a3a"}
BANDN = {"inst": "institutional", "acad": "academic / reference", "comm": "community / informal"}

TASKORD = ["patents", "notice-and-comment", "legal-outcome-prediction", "grant-funding",
           "code-review", "peer-review", "press-releases", "math-stackexchange",
           "news-homepages", "creative-writing", "humor"]
PAL = ["#1e5fae", "#3b82d9", "#5f9ce0", "#8fbceb", "#2a9d5a", "#7fc8a0", "#eda100", "#f3c96b",
       "#eb6834", "#f3a683", "#c73a3a"]
GEN = {"code-review": "Software Code", "creative-writing": "Creative Writing",
       "grant-funding": "Grant Proposals", "humor": "Humor",
       "legal-outcome-prediction": "Legal Arguments", "math-stackexchange": "Math",
       "news-homepages": "Journalism", "notice-and-comment": "Regulatory Comments",
       "patents": "Patents", "peer-review": "Academic Articles",
       "press-releases": "Press Releases"}


def main(rotation=45):
    df = pd.read_parquet(f"{ROOT}/notebooks/_explore_cache/pages.parquet")
    ct = pd.crosstab(df.task, df.orientation).drop(columns=["error"], errors="ignore")
    cols = [c for c in LBL if c in ct.columns]
    order = sorted(cols, key=lambda c: -ct[c].sum())          # tallest first

    fig, ax = plt.subplots(figsize=(7.0, 2.9), dpi=300)
    x = np.arange(len(order))
    bot = np.zeros(len(order))
    for t, col in zip(TASKORD, PAL):
        if t not in ct.index:
            continue
        v = ct.loc[t, order].values
        ax.bar(x, v, bottom=bot, color=col, width=.72, label=GEN[t],
               edgecolor="white", linewidth=.4)
        bot += v

    ax.set_xticks(x)
    ax.set_xticklabels([LBL[c] for c in order], fontsize=6, rotation=rotation, ha="right")
    for tick, c in zip(ax.get_xticklabels(), order):
        tick.set_color(BANDC[BAND[c]])
    ax.set_ylabel("source pages", fontsize=7.5)
    ax.tick_params(labelsize=6.5)
    ax.spines[["top", "right"]].set_visible(False)

    leg1 = ax.legend(frameon=False, fontsize=5.6, ncol=4, loc="upper right",
                     columnspacing=.9, handletextpad=.4, handlelength=1.1)
    ax.add_artist(leg1)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="none", edgecolor="none", label=BANDN[b])
                       for b in ("inst", "acad", "comm")],
              labelcolor=[BANDC[b] for b in ("inst", "acad", "comm")],
              frameon=False, fontsize=5.8, ncol=3, loc="upper right",
              bbox_to_anchor=(1.0, .78), handlelength=0, handletextpad=0, columnspacing=1.2)

    plt.tight_layout()
    plt.savefig(f"{FIGS}/fig_source_provenance.png", bbox_inches="tight")
    print(f"wrote {FIGS}/fig_source_provenance.png (sorted, rotation={rotation})")


if __name__ == "__main__":
    main()
