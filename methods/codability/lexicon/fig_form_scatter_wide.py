#!/usr/bin/env python
"""Landscape version of the paper-1 form scatter (formal-form AUC vs informal-form AUC per
humor construct) for a two-column float: the square plot sits on the left and the six
highlighted constructs are numbered on the plot and listed, with both phrasings, in a key on
the right. Data, colors, highlighted ids and the region labels are those of
fig_form_scatter_labels.py; only the layout differs. Writes figs/fig_form_scatter_wide.png
(the single-column fig_form_scatter.png is left untouched)."""
import json
import matplotlib.pyplot as plt

ROOT = "ROOT"
FIGS = f"{ROOT}/latex/paper-1__metric-codability/figs"
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.linewidth": .6,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK,
                     "text.color": INK, "xtick.labelsize": 7, "ytick.labelsize": 7})

D = json.load(open(f"{ROOT}/outputs/lexicon/paper1_figdata.json"))
p15 = json.load(open(f"{ROOT}/outputs/lexicon/prereg15_results_20260723.json"))["per_metric"]
t3 = {r["mid"]: r for r in D["table3"]["low_wins"] + D["table3"]["high_wins"]}
FORMAL_WINS = ["a2", "a118", "a64"]      # above the diagonal
INFORMAL_WINS = ["a32", "a0", "a196"]    # below the diagonal
ORDER = FORMAL_WINS + INFORMAL_WINS
by_mid = {r["mid"]: r for r in p15}
for m in ORDER:
    assert abs(by_mid[m]["auc_low"] - t3[m]["auc_low"]) < 1e-6
    assert abs(by_mid[m]["auc_high"] - t3[m]["auc_high"]) < 1e-6


def trunc(s, n=52):
    s = s.strip()
    return s if len(s) <= n else s[:n - 1].rstrip() + "…"


fig = plt.figure(figsize=(4.4, 2.7), dpi=300)
ax = fig.add_axes([0.105, 0.145, 0.47, 0.83])
ax.fill_between([0, 1.02], [0, 1.02], 1.02, color=BLUE, alpha=.075, linewidth=0, zorder=0)
ax.fill_between([0, 1.02], 0, [0, 1.02], color=ORANGE, alpha=.07, linewidth=0, zorder=0)
ax.plot([0, 1], [0, 1], color=MUTED, lw=.8, ls=':')
xs = [r["auc_low"] for r in p15]
ys = [r["auc_high"] for r in p15]
ax.scatter(xs, ys, c="#b9c2cc", s=11, alpha=.8, linewidths=0)
# number offsets (points) chosen to sit clear of the grey cloud
OFF = {"a2": (-9, -8), "a118": (4, -8), "a64": (5, -3), "a32": (4, 2), "a0": (4, -8), "a196": (-9, -4)}
for k, m in enumerate(ORDER, 1):
    pt = (by_mid[m]["auc_low"], by_mid[m]["auc_high"])
    ax.scatter([pt[0]], [pt[1]], c=ORANGE, s=24, zorder=3, linewidths=0)
    ax.annotate(str(k), xy=pt, xytext=OFF[m], textcoords="offset points", fontsize=6.5,
                fontweight="bold", color=INK, ha="center", va="center", zorder=5)
ax.set_xlabel('AUC under the informal phrasing', fontsize=7.5)
ax.set_ylabel('AUC under the formal phrasing', fontsize=7.5)
ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02); ax.set_aspect('equal')
ax.spines[['top', 'right']].set_visible(False)
ax.text(.055, .125, 'formal phrasing wins', fontsize=5.6, color=MUTED, rotation=45,
        rotation_mode='anchor', ha='left', va='bottom')
ax.text(.125, .055, 'informal phrasing wins', fontsize=5.6, color=MUTED, rotation=45,
        rotation_mode='anchor', ha='left', va='top')
d = .06
ax.annotate('', xy=(.075, .315 + d), xytext=(.105, .285), arrowprops=dict(arrowstyle='->', color=BLUE, lw=.9, alpha=.65))
ax.annotate('', xy=(.315 + d, .075), xytext=(.285, .105), arrowprops=dict(arrowstyle='->', color=ORANGE, lw=.9, alpha=.75))

# key on the right
x0, y = 0.60, 0.965
fs = 6.3
fig.text(x0, y, 'formal phrasing wins', fontsize=fs, color=BLUE, va='top', fontweight='bold')
y -= 0.075
for k, m in enumerate(ORDER, 1):
    if k == 4:
        y -= 0.03
        fig.text(x0, y, 'informal phrasing wins', fontsize=fs, color=ORANGE, va='top', fontweight='bold')
        y -= 0.075
    fig.text(x0, y, f'{k}', fontsize=fs, color=INK, va='top', fontweight='bold')
    fig.text(x0 + 0.03, y, f'formal: {trunc(t3[m]["high"])}\ninformal: {trunc(t3[m]["low"])}',
             fontsize=fs, color=INK, va='top', linespacing=1.3)
    y -= 0.135
plt.savefig(f'{FIGS}/fig_form_scatter_wide.png', bbox_inches='tight', pad_inches=0.02)
print('saved')
for k, m in enumerate(ORDER, 1):
    print(k, m, '|', t3[m]['high'], '||', t3[m]['low'])
