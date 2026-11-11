"""Extractor for the appendix "Hierarchy subtrees by domain" figures
(fig:treewide-<task>). Rebuilds outputs/lexicon/prereg23_task_subtrees_v3.json.

Per task: pick the R3 category with the most member R2 themes; show its 3
richest themes (by number of member R1 constructs). For EACH theme, take up
to 2 R1 constructs where the data supports branching (>=2 named constructs
each having >=2 named L0 forms); otherwise fall back to the single richest
named construct available. Then show 2-3 named L0 forms per chosen
construct (as many as exist, capped at 3).

All names are verbatim joins against the frozen partition/taxonomy/name
files -- nothing is invented. Register shading/annotation is attached only
when the register bank actually covers a given L0 id; otherwise the form
gets a neutral (unshaded) fill and no annotation.
"""
import json, os, glob, collections

REPO = 'ROOT'
L = f'{REPO}/outputs/lexicon'
DERIVE = f'{L}/derive_then_classify_v1'

TASK_FN = {'math-stackexchange': 'math'}  # taxonomy filename uses "math" not "math-stackexchange"

TASKS = ['code-review', 'creative-writing', 'grant-funding', 'humor',
         'legal-outcome-prediction', 'math-stackexchange', 'news-homepages',
         'notice-and-comment', 'patents', 'peer-review', 'press-releases']


def load_raw(p):
    d = json.load(open(p))
    if isinstance(d, dict) and 'partition' in d and isinstance(d['partition'], dict):
        d = d['partition']
    return {str(k): str(v) for k, v in d.items()}


def load_names(p):
    if not os.path.exists(p):
        return {}
    d = json.load(open(p))
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[str(k)] = v.get('name')
        else:
            out[str(k)] = str(v)
    return out


def load_taxonomy(p, arr_key):
    d = json.load(open(p))
    arr = d.get(arr_key) or d.get('themes') or d.get('categories')
    return {row['id']: row['name'] for row in arr}


def load_taxonomy_full(p, arr_key):
    d = json.load(open(p))
    arr = d.get(arr_key) or d.get('themes') or d.get('categories')
    return {row['id']: row for row in arr}


# ---- register bank: task -> con(L0 id) -> best (head) row ----
register = collections.defaultdict(dict)
_rows_by_con = collections.defaultdict(list)
with open(f'{L}/register_bank_20260722.jsonl') as f:
    for line in f:
        r = json.loads(line)
        _rows_by_con[(r['task'], str(r['con']))].append(r)
for (task, con), rows in _rows_by_con.items():
    head = [r for r in rows if r.get('is_head')]
    pick = head[0] if head else max(rows, key=lambda r: r.get('count', 0))
    register[task][con] = pick


def formality_color(f):
    f = max(2, min(7, int(round(f))))
    return f'form{f}'


_END_CHARS = ('.', '!', '?', '"', "'", ')', '”', '’', '...')


def looks_truncated(s):
    s = (s or '').strip()
    return len(s) >= 90 and not s.endswith(_END_CHARS)


def register_annotation(row):
    if row is None:
        return None
    meta = row.get('metaphoricity')
    strat = row.get('stratum')
    if meta == 1:
        return 'metaphor'
    if strat == 'latinate':
        return 'Latin-derived'
    if strat == 'greek':
        return 'Greek-derived'
    if strat == 'germanic':
        return 'Germanic'
    return None


def pick_forms(l0_ids, l0n, reg_task, cap=3):
    """Rank named L0 children (prefer register-covered, then non-truncated,
    then numeric id) and return up to `cap` form dicts."""
    def sort_key(i):
        has_reg = 1 if i in reg_task else 0
        trunc = 1 if looks_truncated(l0n.get(i)) else 0
        try:
            num = int(i)
        except ValueError:
            num = 0
        return (-has_reg, trunc, num)
    named_l0 = [i for i in l0_ids if l0n.get(i)]
    named_l0.sort(key=sort_key)
    forms = []
    for i in named_l0[:cap]:
        row = reg_task.get(i)
        forms.append({
            'l0_id': i,
            'text': l0n[i],
            'formality': row.get('formality') if row else None,
            'color': formality_color(row['formality']) if row else None,
            'annotation': register_annotation(row) if row else None,
            'truncated_in_source': looks_truncated(l0n[i]),
        })
    return forms, len(named_l0)


def build_task(t):
    fn = TASK_FN.get(t, t)
    r1 = load_raw(f'{L}/partition_{t}_R1.json')   # L0 -> R1construct
    r2 = load_raw(f'{L}/partition_{t}_R2.json')   # R1construct -> T##
    r3 = load_raw(f'{L}/partition_{t}_R3.json')   # T## -> C##

    r2_tax = load_taxonomy(f'{DERIVE}/{t}/R2/taxonomy_{fn}_R2.json', 'themes')
    r3_tax = load_taxonomy(f'{DERIVE}/{t}/R3/taxonomy_{fn}_R3.json', 'categories')

    l0f = sorted(glob.glob(f'{L}/cluster_names_{t}_L0v*.json'))[-1]
    l0n = load_names(l0f)
    r1n = load_names(f'{L}/node_names_{t}_R1.json')

    cat_members = collections.defaultdict(list)      # C## -> [T##]
    for th, cat in r3.items():
        cat_members[cat].append(th)
    th_members = collections.defaultdict(list)        # T## -> [R1con]
    for con, th in r2.items():
        th_members[th].append(con)
    con_members = collections.defaultdict(list)        # R1con -> [L0id]
    for l0id, con in r1.items():
        con_members[con].append(l0id)

    # pick category with most member themes (ties -> lowest id)
    cat = sorted(cat_members, key=lambda c: (-len(cat_members[c]), c))[0]
    all_theme_ids_in_cat = cat_members[cat]

    # rank themes within category by number of member R1 constructs (richness)
    themes_sorted = sorted(all_theme_ids_in_cat, key=lambda th: (-len(th_members.get(th, [])), th))
    themes_shown = themes_sorted[:2]

    reg_task = register.get(t, {})

    out_themes = []
    for th in themes_shown:
        cons = th_members.get(th, [])
        # per-construct: named? and how many named L0 children?
        con_info = []
        for c in cons:
            nm = r1n.get(c)
            if not nm:
                continue
            l0_ids = con_members.get(c, [])
            named_l0 = [i for i in l0_ids if l0n.get(i)]
            con_info.append({
                'id': c, 'name': nm, 'trunc': looks_truncated(nm),
                'n_named_l0': len(named_l0), 'l0_ids': l0_ids,
            })
        qualifying = [c for c in con_info if c['n_named_l0'] >= 2]
        # rank: prefer non-truncated name, then more named L0 children, then id
        def crank(c):
            return (1 if c['trunc'] else 0, -c['n_named_l0'], c['id'])
        qualifying.sort(key=crank)
        branch_supported = len(qualifying) >= 2

        if branch_supported:
            chosen = qualifying[:2]
        else:
            all_named = sorted(con_info, key=crank)
            chosen = all_named[:1]

        theme_constructs = []
        for c in chosen:
            forms, n_named_available = pick_forms(c['l0_ids'], l0n, reg_task, cap=3)
            theme_constructs.append({
                'construct_id': c['id'],
                'construct_name': c['name'],
                'n_l0_in_construct': len(c['l0_ids']),
                'n_l0_named_available': n_named_available,
                'forms': forms,
            })

        out_themes.append({
            'theme_id': th,
            'theme_name': r2_tax.get(th),
            'n_constructs_in_theme': len(cons),
            'n_named_constructs_in_theme': len(con_info),
            'n_constructs_qualifying_for_branch': len(qualifying),
            'branch_supported': branch_supported,
            'constructs': theme_constructs,
        })

    return {
        'task': t,
        'category_id': cat,
        'category_name': r3_tax.get(cat),
        'n_themes_in_category_total': len(all_theme_ids_in_cat),
        'n_themes_shown': len(out_themes),
        'all_theme_names_in_category': [r2_tax.get(th) for th in all_theme_ids_in_cat],
        'themes': out_themes,
    }


if __name__ == '__main__':
    result = {}
    r3_audit = {}
    for t in TASKS:
        result[t] = build_task(t)

        fn = TASK_FN.get(t, t)
        r3_path = f'{DERIVE}/{t}/R3/taxonomy_{fn}_R3.json'
        r2_path = f'{DERIVE}/{t}/R2/taxonomy_{fn}_R2.json'
        r3_full = load_taxonomy_full(r3_path, 'categories')
        v = result[t]
        cat = v['category_id']
        row = r3_full.get(cat, {})
        r3_audit[t] = {
            'category_id': cat,
            'category_name': row.get('name'),
            'category_definition': row.get('definition'),
            'name_source': os.path.relpath(r3_path, REPO),
            'n_member_themes_total': v['n_themes_in_category_total'],
            'member_theme_ids_and_names_source': os.path.relpath(r2_path, REPO),
            'member_theme_names': v['all_theme_names_in_category'],
            'note': ('R3 category name is taken verbatim from the frozen taxonomy file above '
                     '(joined on the category id via partition_<task>_R3.json); it is NOT '
                     'model- or hand-generated by this build script. Member theme names are '
                     'listed for audit only (all themes in the category, not just the 3 shown '
                     'in the figure).'),
        }

    out_path = f'{L}/prereg23_task_subtrees_v3.json'
    json.dump(result, open(out_path, 'w'), indent=1)
    print('wrote', out_path)

    r3_out_path = f'{L}/prereg23_r3_names_v3.json'
    json.dump(r3_audit, open(r3_out_path, 'w'), indent=1)
    print('wrote', r3_out_path)

    print()
    print(f"{'task':26s} {'themes branch/shown':22s} {'construct counts per theme'}")
    for t, v in result.items():
        n_branch = sum(1 for th in v['themes'] if th['branch_supported'])
        counts = [len(th['constructs']) for th in v['themes']]
        forms_per_con = [[len(c['forms']) for c in th['constructs']] for th in v['themes']]
        n_reg = sum(1 for th in v['themes'] for c in th['constructs'] for f in c['forms']
                    if f['annotation'] or f['formality'] is not None)
        n_forms_total = sum(1 for th in v['themes'] for c in th['constructs'] for f in c['forms'])
        print(f"{t:26s} branch_themes={n_branch}/{len(v['themes'])} constructs_per_theme={counts} "
              f"forms_per_construct={forms_per_con} reg_forms={n_reg}/{n_forms_total}")
