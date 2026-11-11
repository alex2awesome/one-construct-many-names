#!/usr/bin/env python
"""Main-body scatter (paper-1 Fig 9): formal-form AUC vs informal-form AUC per metric.

Extracted from the "Main-body scatter" cell of
notebooks/2026-07-23__paper-1-figures.ipynb (the cell that writes fig_form_scatter.png), with
one change: highlighted-point labels now show BOTH attested phrasings instead of a single short
paraphrase, so a reader can see how differently-worded the formal and informal phrasings of the
same metric actually are. Each label is two stacked lines:

    formal: <formal phrasing>
    informal: <informal phrasing>

Phrasings longer than ~45 characters are truncated with an ellipsis so labels stay readable at
single-column width. Everything else -- colors, marker sizes, fonts, the diagonal, the two
shaded win-regions and their labels/arrows, axis limits/aspect, and which 6 metric ids are
highlighted -- is copied verbatim from the notebook. Only the highlighted-label positions
changed, because the old short-name offsets were tuned for one-line labels and a 45-char,
two-line label is much wider (measured ~0.6-0.9 of the whole 1.02-unit axis span at the
notebook's fontsize=5.8) than any one-line label ever was.

Layout: the 3 "formal phrasing wins" labels (a64, a118, a2) sit in a band above the axis
(y > 1.02); the 3 "informal phrasing wins" labels (a196, a0, a32) sit in a band below the axis
(y < 0), each with a thin leader line back to its point. This keeps the plot's own WIDTH
unchanged from the original (the paper includes this figure at width=.98\\columnwidth, so width
is what "single-column" actually constrains) -- the added label content instead grows the
canvas's HEIGHT, which LaTeX's width-only \\includegraphics scaling doesn't penalize. The
alternative (a wide two-column layout flanking the plot) was tried first and rejected: it grew
the canvas ~2.7x wider than tall, which would have shrunk the plot itself to a sliver once
scaled to columnwidth in print.

The 88-point cloud (xs/ys) comes from outputs/lexicon/prereg15_results_20260723.json
(`per_metric`), same as the notebook. That file has auc_low/auc_high/mid per point but NOT the
phrasing text. The full formal/informal phrasing strings for the 6 highlighted mids are not in
the notebook cell at all (it hardcodes short human paraphrases in LBL) -- they were recovered
from outputs/lexicon/paper1_figdata.json's `table3` (top-5 low_wins + top-5 high_wins by |dAUC|),
which is the same underlying table the notebook's Table 3 markdown table renders elsewhere in
the notebook. All 6 highlighted mids (a64, a118, a2, a196, a0, a32) are present in table3, and
their auc_low/auc_high values there match prereg15_results exactly (cross-checked below).

Known compromise: plt.tight_layout() is NOT called before savefig (the notebook cell calls it).
With labels placed this far outside the axes' default extent, tight_layout()'s auto-margin
solver tries to shrink the drawn axes to fit everything inside the original (3.3, 3.1) inch
figsize and produces a degenerate near-empty plot. savefig(bbox_inches='tight') alone still
crops the canvas correctly around every artist (axes + outside labels), so the axes' own
internal spacing is unaffected -- this only changes how the *outer* canvas margin is computed,
not any of the requested-to-preserve style (colors/markers/fonts/aspect).
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt

ROOT = "ROOT"
FIGS = f"{ROOT}/latex/paper-1__metric-codability/figs"
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.linewidth": .6,
                      "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK,
                      "text.color": INK})

D = json.load(open(f"{ROOT}/outputs/lexicon/paper1_figdata.json"))
p15 = json.load(open(f"{ROOT}/outputs/lexicon/prereg15_results_20260723.json"))["per_metric"]

# Same 6 highlighted metric ids as the notebook's LBL dict (mid -> short paraphrase, offset, ha).
# Full phrasing text for these mids lives in paper1_figdata.json's table3, not in prereg15.
t3_rows = {r["mid"]: r for r in D["table3"]["low_wins"] + D["table3"]["high_wins"]}
HIGHLIGHT_MIDS = ["a64", "a118", "a2", "a196", "a0", "a32"]
assert set(HIGHLIGHT_MIDS) <= set(t3_rows), "missing table3 phrasing row for a highlighted mid"
p15_by_mid = {r["mid"]: r for r in p15}
for _m in HIGHLIGHT_MIDS:
    assert abs(p15_by_mid[_m]["auc_low"] - t3_rows[_m]["auc_low"]) < 1e-6
    assert abs(p15_by_mid[_m]["auc_high"] - t3_rows[_m]["auc_high"]) < 1e-6


def trunc(s: str, maxlen: int = 45) -> str:
    """Truncate a phrasing to ~45 chars with a trailing ellipsis, matching the paper author's spec."""
    s = s.strip()
    if len(s) <= maxlen:
        return s
    return s[: maxlen - 1].rstrip() + "…"


def label_text(mid: str) -> str:
    row = t3_rows[mid]
    return f"formal: {trunc(row['high'])}\ninformal: {trunc(row['low'])}"


# anchor (x,y in data coords), text alignment, per highlighted point. Anchor x tracks each
# point's own x (left/center/right-aligned as needed) so leader lines stay short and close to
# vertical rather than raking all the way across the plot. Row y-spacing (top band: .14 apart;
# bottom band: .14 apart, starting past y=-0.24 to clear the x-axis tick labels/xlabel, which
# extend down to y=-0.168) was chosen so no two label boxes' rendered bounding boxes touch --
# verified below by the renderer, not just by eye.
LBL_POS = {
    "a64":  {"anchor": (0.02, 1.06),  "ha": "left",   "va": "center"},
    "a118": {"anchor": (0.06, 1.20),  "ha": "left",   "va": "center"},
    "a2":   {"anchor": (0.51, 1.34),  "ha": "center", "va": "center"},
    "a196": {"anchor": (1.10, -0.26), "ha": "right",  "va": "center"},
    "a0":   {"anchor": (1.07, -0.40), "ha": "right",  "va": "center"},
    "a32":  {"anchor": (0.99, -0.54), "ha": "right",  "va": "center"},
}

fig, ax = plt.subplots(figsize=(3.3, 3.1), dpi=300)
ax.fill_between([0, 1.02], [0, 1.02], 1.02, color=BLUE, alpha=.075, linewidth=0, zorder=0)
ax.fill_between([0, 1.02], 0, [0, 1.02], color=ORANGE, alpha=.07, linewidth=0, zorder=0)
ax.plot([0, 1], [0, 1], color=MUTED, lw=.8, ls=':')
xs = [r["auc_low"] for r in p15]
ys = [r["auc_high"] for r in p15]
ax.scatter(xs, ys, c="#b9c2cc", s=13, alpha=.8, linewidths=0)

text_objs = {}
for r in p15:
    if r["mid"] in LBL_POS:
        pos = LBL_POS[r["mid"]]
        pt = (r["auc_low"], r["auc_high"])
        ax.scatter([pt[0]], [pt[1]], c=ORANGE, s=26, zorder=3, linewidths=0)
        t = ax.text(pos["anchor"][0], pos["anchor"][1], label_text(r["mid"]),
                    fontsize=5.8, color=INK, ha=pos["ha"], va=pos["va"], linespacing=1.35,
                    zorder=5, clip_on=False)
        text_objs[r["mid"]] = t
        # leader line: from just past the near edge of the label block to the point. Text is
        # zorder=5 (drawn on top), the line is zorder=2, so wherever the two happen to cross,
        # the line renders behind the letters and never obscures the label text.
        edge_y = pos["anchor"][1] + (-0.06 if pos["anchor"][1] > 1.0 else 0.06)
        ax.annotate('', xy=pt, xytext=(pos["anchor"][0], edge_y),
                    arrowprops=dict(arrowstyle='->', color=MUTED, lw=.7, shrinkA=1.0, shrinkB=2.5),
                    zorder=2, annotation_clip=False)

ax.set_xlabel('AUC under the informal phrasing')
ax.set_ylabel('AUC under the formal phrasing')
ax.set_xlim(0, 1.02)
ax.set_ylim(0, 1.02)
ax.set_aspect('equal')
ax.spines[['top', 'right']].set_visible(False)
# region labels riding the diagonal, near its top end
rt1 = ax.text(.055, .125, 'formal phrasing wins', fontsize=6.2, color=MUTED, rotation=45,
              rotation_mode='anchor', ha='left', va='bottom')
rt2 = ax.text(.125, .055, 'informal phrasing wins', fontsize=6.2, color=MUTED, rotation=45,
              rotation_mode='anchor', ha='left', va='top')
d = .06  # perpendicular arrows beside the region labels, clear of the text
ax.annotate('', xy=(.075 - 0, .315 + d), xytext=(.105, .285),
            arrowprops=dict(arrowstyle='->', color=BLUE, lw=.9, alpha=.65))
ax.annotate('', xy=(.315 + d, .075), xytext=(.285, .105),
            arrowprops=dict(arrowstyle='->', color=ORANGE, lw=.9, alpha=.75))
# NOTE: no plt.tight_layout() call here -- see module docstring "Known compromise".
# savefig(bbox_inches='tight') below still crops the canvas correctly around every artist.

fig.canvas.draw()
renderer = fig.canvas.get_renderer()

# --- verification: confirm no highlighted label overlaps another label, an axis decoration
# (tick labels / axis labels / the diagonal region labels), or any of the 88 plotted points. ---
check_objs = dict(text_objs)
check_objs['_region_lbl1'] = rt1
check_objs['_region_lbl2'] = rt2
check_objs['_yaxis_label'] = ax.yaxis.label
check_objs['_xaxis_label'] = ax.xaxis.label
for i, tl in enumerate(ax.get_yticklabels()):
    check_objs[f'_ytick{i}'] = tl
for i, tl in enumerate(ax.get_xticklabels()):
    check_objs[f'_xtick{i}'] = tl
bboxes = {k: v.get_window_extent(renderer) for k, v in check_objs.items()}
names = list(bboxes.keys())
overlaps = []
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a, b = names[i], names[j]
        if a not in text_objs and b not in text_objs:
            continue  # only care about pairs involving a highlight label
        if bboxes[a].overlaps(bboxes[b]):
            overlaps.append((a, b))
all_pts_disp = ax.transData.transform(list(zip(xs, ys)))
marker_half_px = 4.2
point_overlaps = []
for mid, t in text_objs.items():
    tb = bboxes[mid]
    for (px, py) in all_pts_disp:
        if (tb.x0 - marker_half_px <= px <= tb.x1 + marker_half_px
                and tb.y0 - marker_half_px <= py <= tb.y1 + marker_half_px):
            point_overlaps.append((mid, px, py))

plt.savefig(f'{FIGS}/fig_form_scatter.png', bbox_inches='tight')
print('saved', f'{FIGS}/fig_form_scatter.png')
print('label/axis overlap check:', 'NONE' if not overlaps else overlaps)
print('label/point overlap check:', 'NONE' if not point_overlaps else point_overlaps)
for m in HIGHLIGHT_MIDS:
    print(m, '|', label_text(m).replace(chr(10), ' // '))
