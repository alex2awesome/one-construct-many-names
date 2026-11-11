"""Emit the appendix "Hierarchy subtrees by domain" figures (fig:treewide-<task>)
from outputs/lexicon/prereg23_task_subtrees_v3.json, using the compact 4-row
geometry of the known-good OLD wide template (main.tex, \\iffalse block,
\\label{fig:treewide}): R3 y=4.35, R2 y=3.45, R1 y=2.55, L0 y=1.15, and a
FIXED-size l0 style (text width=1.72cm, minimum height=.72cm) so every leaf
box is identical. Unlike the old template (one single path per figure), each
theme here may show 1 or 2 R1 constructs, per the branch_supported flag
computed by build_task_subtrees_v3.py.
"""
import json

REPO = 'ROOT'
L = f'{REPO}/outputs/lexicon'

data = json.load(open(f'{L}/prereg23_task_subtrees_v3.json'))

GEN = {"code-review": "Software Code", "creative-writing": "Creative Writing",
       "grant-funding": "Grant Proposals", "humor": "Humor",
       "legal-outcome-prediction": "Legal Arguments", "math-stackexchange": "Mathematical Writing",
       "news-homepages": "Journalism", "notice-and-comment": "Regulatory Comments",
       "patents": "Patents", "peer-review": "Academic Publishing",
       "press-releases": "Press Releases"}

TASK_ORDER = ["code-review", "creative-writing", "grant-funding", "humor",
              "legal-outcome-prediction", "math-stackexchange", "news-homepages",
              "notice-and-comment", "patents", "peer-review", "press-releases"]


def esc(s):
    if s is None:
        return ''
    s = str(s)
    s = s.replace('\\', r'\textbackslash{}')
    s = s.replace('&', r'\&').replace('%', r'\%').replace('$', r'\$')
    s = s.replace('#', r'\#').replace('_', r'\_')
    s = s.replace('{', r'\{').replace('}', r'\}')
    s = s.replace('~', r'\textasciitilde{}').replace('^', r'\textasciicircum{}')
    return s


def truncate(s, cap):
    s = s.strip()
    if len(s) <= cap:
        return s
    cut = s[:cap].rsplit(' ', 1)[0]
    if not cut:
        cut = s[:cap]
    return cut + '\\,\\ldots'


# ---- geometry: compact 4-row layout copied from the OLD wide template
# (main.tex \iffalse block, \label{fig:treewide}) ----
Y_CAT = 4.35
Y_THEME = 3.45
Y_CON = 2.55
Y_LEAF = 1.15
Y_ANN = 0.62
Y_KEY = Y_CAT + 0.20

DX_LEAF = 1.98      # leaf-to-leaf step within one construct (== old template's step)
CON_GAP = 0.50       # gap ADDED ON TOP OF one DX_LEAF step, between two constructs of the SAME theme
THEME_GAP = 1.20     # gap ADDED ON TOP OF one DX_LEAF step, between different themes' groups
ELLIPSIS_GAP = 1.80  # gap from the rightmost real leaf to the decorative "..." R3 child
X0 = 1.30            # first leaf x (== old template)

CAT_CAP, THEME_CAP, CON_CAP, LEAF_CAP = 60, 62, 66, 46


def r3(v):
    return round(v, 3)


def layout(themes):
    """Returns leaf_x[ti][ci] (list of x per leaf), con_x[ti][ci] (construct
    center), theme_x[ti] (theme center). `cursor` always tracks the x of the
    LAST leaf placed; a new construct/theme group's first leaf starts one
    full DX_LEAF step past the cursor (so it never overlaps the previous
    group), plus an additional CON_GAP/THEME_GAP visual break."""
    leaf_x, con_x, theme_x = [], [], []
    cursor = None
    for ti, th in enumerate(themes):
        theme_leaf_groups = []
        theme_con_centers = []
        for ci, c in enumerate(th['constructs']):
            n = len(c['forms'])
            if cursor is None:
                start = X0
            else:
                extra = THEME_GAP if ci == 0 else CON_GAP
                start = cursor + DX_LEAF + extra
            xs = [r3(start + j * DX_LEAF) for j in range(n)]
            theme_leaf_groups.append(xs)
            theme_con_centers.append(r3(sum(xs) / len(xs)))
            cursor = xs[-1]
        leaf_x.append(theme_leaf_groups)
        con_x.append(theme_con_centers)
        theme_x.append(r3(sum(theme_con_centers) / len(theme_con_centers)))
    return leaf_x, con_x, theme_x


def build_figure(task):
    v = data[task]
    dom = GEN[task]
    themes = v['themes']
    leaf_x, con_x, theme_x = layout(themes)
    cat_x = r3(sum(theme_x) / len(theme_x))
    rightmost_leaf = max(x for grp in leaf_x for xs in grp for x in xs)
    ellipsis_x = r3(rightmost_leaf + ELLIPSIS_GAP)

    all_forms = [f for th in themes for c in th['constructs'] for f in c['forms']]
    n_forms_total = len(all_forms)
    n_reg = sum(1 for f in all_forms if f['formality'] is not None)
    any_register = n_reg > 0
    branch_flags = [th['branch_supported'] for th in themes]
    n_branch = sum(branch_flags)

    lines = []
    lines.append(r'\begin{figure*}[t]')
    lines.append(r'\centering')
    lines.append(r'\resizebox{\textwidth}{!}{%')
    lines.append(r'	\begin{tikzpicture}[x=1cm,y=1cm,')
    lines.append(r'		lvl/.style={font=\tiny\bfseries, text=gray, anchor=east},')
    lines.append(rf'		nd/.style={{draw, rounded corners=2pt, align=center, inner sep=2.5pt, font=\tiny, text width=3.0cm}},')
    lines.append(r'		l0/.style={draw=none, rounded corners=2pt, align=center, inner sep=2.5pt, font=\tiny, text width=1.72cm, minimum height=.72cm},')
    lines.append(r'		ann/.style={font=\tiny, text=gray},')
    lines.append(r'		e/.style={draw=gray!60}]')
    lines.append(rf'		\node[lvl] at (-0.2,{Y_CAT}) {{R3 category}};')
    lines.append(rf'		\node[lvl] at (-0.2,{Y_THEME}) {{R2 theme}};')
    lines.append(rf'		\node[lvl] at (-0.2,{Y_CON}) {{R1 construct}};')
    lines.append(rf'		\node[lvl] at (-0.2,{Y_LEAF}) {{L0 surface forms}};')

    if any_register:
        lines.append(rf'		\node[font=\tiny, text=gray, anchor=east] at ({cat_x + 2.0},{Y_KEY}) {{less formal}};')
        lines.append(r'		\foreach \i/\c in {0/form2,1/form3,2/form4,3/form5,4/form6,5/form7}')
        lines.append(rf'		\node[draw=none, fill=\c, minimum width=.34cm, minimum height=.22cm, inner sep=0] at ({cat_x + 2.3}+\i*0.36,{Y_KEY}) {{}};')
        lines.append(rf'		\node[font=\tiny, text=gray, anchor=west] at ({cat_x + 4.5},{Y_KEY}) {{more formal}};')

    cat_name = truncate(esc(v['category_name']), CAT_CAP)
    lines.append(rf'		\node[nd, fill=gray!12] (c1) at ({cat_x},{Y_CAT}) {{{cat_name}}};')

    for ti, th in enumerate(themes):
        tname = truncate(esc(th['theme_name']), THEME_CAP)
        lines.append(rf'		\node[nd, fill=gray!5] (t{ti}) at ({theme_x[ti]},{Y_THEME}) {{{tname}}};')

    # decorative third R3 child: a plain, unboxed "..." signaling the category
    # continues beyond the 2 R2 themes actually shown. No box, no fill, no
    # children -- visually unmistakably not a data node.
    lines.append(rf'		\node[font=\small\bfseries, text=gray!55] (tdots) at ({ellipsis_x},{Y_THEME}) {{$\cdots$}};')
    lines.append(r'		\draw[e, dashed, gray!45] (c1.south) -- (tdots.north);')

    for ti, th in enumerate(themes):
        for ci, c in enumerate(th['constructs']):
            coname = truncate(esc(c['construct_name']), CON_CAP)
            lines.append(rf'		\node[nd] (g{ti}_{ci}) at ({con_x[ti][ci]},{Y_CON}) {{{coname}}};')

    ann_nodes = []
    for ti, th in enumerate(themes):
        for ci, c in enumerate(th['constructs']):
            for li, (x, f) in enumerate(zip(leaf_x[ti][ci], c['forms'])):
                color = f['color']
                if color is None:
                    fill = 'white'
                    textcolor = ''
                else:
                    fill = color
                    textcolor = ', text=white' if color in ('form6', 'form7') else ''
                txt = truncate(esc(f['text']), LEAF_CAP)
                lines.append(rf'		\node[l0, fill={fill}{textcolor}] (a{ti}_{ci}_{li}) at ({x},{Y_LEAF}) {{{txt}}};')
                if f['annotation']:
                    ann_nodes.append((x, f['annotation']))

    for x, ann in ann_nodes:
        lines.append(rf'		\node[ann] at ({x},{Y_ANN}) {{\textit{{{esc(ann)}}}}};')

    # edges
    for ti in range(len(themes)):
        lines.append(rf'		\draw[e] (c1.south) -- (t{ti}.north);')
    for ti, th in enumerate(themes):
        for ci in range(len(th['constructs'])):
            lines.append(rf'		\draw[e] (t{ti}.south) -- (g{ti}_{ci}.north);')
    for ti, th in enumerate(themes):
        for ci, c in enumerate(th['constructs']):
            for li in range(len(c['forms'])):
                lines.append(rf'		\draw[e] (g{ti}_{ci}.south) -- (a{ti}_{ci}_{li}.north);')

    lines.append(r'	\end{tikzpicture}}')

    # ---- caption: short, per-figure facts only. The shared selection rule and
    # register-coverage explanation live once in the section intro. ----
    cap = []
    cap.append(rf"{esc(dom)}, rooted at the R3 category \emph{{{esc(v['category_name'])}}} "
               rf"({v['n_themes_shown']} of its {v['n_themes_in_category_total']} R2 themes shown).")
    cap.append(rf"{n_reg} of {n_forms_total} forms carry measured formality; the rest are unshaded.")

    caption = ' '.join(cap)

    lines.append(rf'\caption{{{caption}}}')
    lines.append(rf'\label{{fig:treewide-{task}}}')
    lines.append(r'\end{figure*}')
    return '\n'.join(lines)


intro = r"""\section{Hierarchy subtrees by domain}
\label{app:treewide}
For each of the 11 domains, Fig.~\ref{fig:treewide-code-review}--\ref{fig:treewide-press-releases}
below show a subtree of the certified hierarchy (Appendix~\ref{app:hierarchy}), in the same
compact, two-column layout as the single humor example in Fig.~\ref{fig:tree} above. For each
domain we pick the R3 evaluative category with the most member R2 themes and show its two
richest themes (by number of member R1 constructs); a dashed edge to an unboxed ``$\cdots$''
marks that the category has further member themes not drawn here. Each shown theme in turn
displays up to two of its richest R1 constructs (ranked by number of named L0 surface forms)
whenever the theme has at least two named constructs that each carry at least two named L0
forms; where that support does not exist, the theme instead shows only its single richest
construct, with as many named L0 forms as the data provides (2--3 where available). Nine of the
eleven domains support this two-construct branching in both shown themes (code-review,
creative-writing, grant-funding, humor, legal-outcome-prediction, math-stackexchange,
news-homepages, notice-and-comment, patents); peer-review and press-releases branch in only one
of their two shown themes, because the other theme's R1 partition is here fully degenerate
(every construct a singleton surface form) --- a genuine property of those cells, not a display
choice, and visible directly in those two panels as a theme with only one child construct. All
category, theme, and construct names are taken verbatim from the frozen taxonomies produced by
the classification pipeline of Appendix~\ref{app:hierarchy}
(\texttt{outputs/lexicon/derive\_then\_classify\_v1/<task>/\{R2,R3\}/taxonomy\_*.json}); none
are generated for this figure. Register shading and etymology/metaphoricity tags come from the
register bank (\S\ref{app:instruments}) where it covers a given surface form (shading gives
judged formality, darker $=$ more formal, and the short italic tag under a form is its dominant
etymological stratum or metaphoricity); elsewhere they are omitted rather than inferred, and the
form is drawn unshaded in plain white rather than in a color that would imply a measurement we
do not have. The machine-readable data behind every panel, including the full (untruncated)
strings, the per-theme branch-support audit, and the member-theme audit trail for each chosen R3
category, is released at \texttt{outputs/lexicon/prereg23\_task\_subtrees\_v3.json} and
\texttt{outputs/lexicon/prereg23\_r3\_names\_v3.json}.

"""

figs = [build_figure(t) for t in TASK_ORDER]
full = intro + '\n\n'.join(figs) + '\n'

out_path = '/private/tmp/claude-502/-Users-anon-Projects-stanford-research-norm-research/6eb8228d-7011-4800-87e2-61a172f6003c/scratchpad/treewide_section_v3.tex'
open(out_path, 'w').write(full)
print('wrote', out_path, len(full), 'chars')
