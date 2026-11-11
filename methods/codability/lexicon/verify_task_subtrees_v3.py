"""Correctness check for outputs/lexicon/prereg23_task_subtrees_v3.json.

For every task, every theme, every construct, every L0 form shown in the
figure JSON, verify against the raw partition files that:
  - the theme's R3 parent (via partition_<task>_R3.json) equals the figure's
    stated category_id;
  - the construct's R2 parent (via partition_<task>_R2.json) equals the
    figure's stated theme_id;
  - the L0 form's R1 parent (via partition_<task>_R1.json) equals the
    figure's stated construct_id.
Also verifies no name/text was invented: every category/theme/construct name
and every L0 form's text matches verbatim what the taxonomy/node-name/
cluster-name source files give for that id (a direct re-lookup, independent
of the build script).
Prints a violation count; must be 0 before any LaTeX is emitted.
"""
import json, os, glob

REPO = 'ROOT'
L = f'{REPO}/outputs/lexicon'
DERIVE = f'{L}/derive_then_classify_v1'
TASK_FN = {'math-stackexchange': 'math'}


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
        out[str(k)] = v.get('name') if isinstance(v, dict) else str(v)
    return out


def load_taxonomy(p, arr_key):
    d = json.load(open(p))
    arr = d.get(arr_key) or d.get('themes') or d.get('categories')
    return {row['id']: row['name'] for row in arr}


data = json.load(open(f'{L}/prereg23_task_subtrees_v3.json'))

violations = []
checked = {'theme_under_cat': 0, 'construct_under_theme': 0, 'form_under_construct': 0,
           'category_name': 0, 'theme_name': 0, 'construct_name': 0, 'form_text': 0}

for task, v in data.items():
    r1 = load_raw(f'{L}/partition_{task}_R1.json')  # L0 -> R1construct
    r2 = load_raw(f'{L}/partition_{task}_R2.json')  # R1construct -> theme
    r3 = load_raw(f'{L}/partition_{task}_R3.json')  # theme -> category
    cat = v['category_id']

    fn = TASK_FN.get(task, task)
    r2_tax = load_taxonomy(f'{DERIVE}/{task}/R2/taxonomy_{fn}_R2.json', 'themes')
    r3_tax = load_taxonomy(f'{DERIVE}/{task}/R3/taxonomy_{fn}_R3.json', 'categories')
    r1n = load_names(f'{L}/node_names_{task}_R1.json')
    l0f = sorted(glob.glob(f'{L}/cluster_names_{task}_L0v*.json'))[-1]
    l0n = load_names(l0f)

    checked['category_name'] += 1
    if r3_tax.get(cat) != v['category_name']:
        violations.append(f"{task}: category_name mismatch for {cat}: json={v['category_name']!r} src={r3_tax.get(cat)!r}")

    for th in v['themes']:
        th_id = th['theme_id']
        checked['theme_under_cat'] += 1
        actual_cat = r3.get(th_id)
        if actual_cat != cat:
            violations.append(f"{task}: theme {th_id} parent={actual_cat!r} != stated category {cat!r}")

        checked['theme_name'] += 1
        if r2_tax.get(th_id) != th['theme_name']:
            violations.append(f"{task}: theme_name mismatch for {th_id}: json={th['theme_name']!r} src={r2_tax.get(th_id)!r}")

        for c in th['constructs']:
            con_id = c['construct_id']
            checked['construct_under_theme'] += 1
            actual_th = r2.get(con_id)
            if actual_th != th_id:
                violations.append(f"{task}: construct {con_id} parent={actual_th!r} != stated theme {th_id!r}")

            checked['construct_name'] += 1
            if r1n.get(con_id) != c['construct_name']:
                violations.append(f"{task}: construct_name mismatch for {con_id}: json={c['construct_name']!r} src={r1n.get(con_id)!r}")

            for f in c['forms']:
                l0_id = f['l0_id']
                checked['form_under_construct'] += 1
                actual_con = r1.get(l0_id)
                if actual_con != con_id:
                    violations.append(f"{task}: L0 {l0_id} parent={actual_con!r} != stated construct {con_id!r}")

                checked['form_text'] += 1
                if l0n.get(l0_id) != f['text']:
                    violations.append(f"{task}: L0 text mismatch for {l0_id}: json={f['text']!r} src={l0n.get(l0_id)!r}")

print("checked:", checked)
print("VIOLATIONS:", len(violations))
for x in violations[:50]:
    print("  ", x)

if violations:
    raise SystemExit(1)
print("ZERO VIOLATIONS -- all theme/construct/form parentage confirmed against partition_<task>_R{1,2,3}.json")
