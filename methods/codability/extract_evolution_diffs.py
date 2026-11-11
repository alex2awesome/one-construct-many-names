"""Extract per-step transitions from (A) tacit decompression ladders and (B) unsupervised
prompt-evolution trajectories, in ONE shared format, for blinded change-type labeling
(user directive 2026-07-28: localize the tacit-vs-reconstruction difference to WHAT CHANGES
per evolution step, within unsupervised metrics only).

(A) two_faces vertical ladders  name->definition->explanation (the clean rungs; exemplars/
    dossier are ostension-confounded per isomorphism_census.py) across 10 domains.
(B) gepa_h2h criterion rounds (multi-domain, overlaps A), gepa_nc rubric rounds r0..r5,
    metric_implementer version registries (operator-labeled lineage kept in the KEY only).

Output: transitions JSONL {id, before, after, added} with corpus identity held OUT of the
labeled file (separate key file), plus planted anchors.
"""
import json, glob, random, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TF = ROOT / "notebooks/data/two_faces_20260702"
SCRATCH = Path("/private/tmp/claude-502/-Users-anon-Projects-stanford-research-norm-research/"
               "5dab3f74-48cc-4698-a04d-3d07440a91bf/scratchpad")

def sents(t):
    return [s.strip() for s in re.split(r"(?<=[.!?;])\s+|\n+", t or "") if len(s.strip()) > 3]

def toks(s):
    return set(re.findall(r"[a-z0-9']+", s.lower()))

def added_sents(before, after):
    bs = [toks(s) for s in sents(before)]
    out = []
    for s in sents(after):
        ts = toks(s)
        if not ts:
            continue
        if all(len(ts & b) / max(1, len(ts | b)) < 0.6 for b in bs):
            out.append(s)
    return out

rows = []  # (corpus_tag, lineage, step_name, before, after)

# ---------- (A) vertical ladders ----------
for mj in sorted(glob.glob(str(TF / "r3_*/grid_*_v1/messages.json"))):
    dom = Path(mj).parent.name.replace("grid_", "").replace("_v1", "")
    for gi, m in json.load(open(mj)).items():
        r = m.get("rungs", {})
        for a, b in (("name", "definition"), ("definition", "explanation")):
            if r.get(a) and r.get(b):
                rows.append(("A_ladder", f"{dom}:{gi}", f"{a}->{b}", r[a], r[b]))

# ---------- (B1) gepa_h2h states ----------
for sj in ("methods/metric_seam/battery/gepa_h2h/state.json",
           "methods/metric_seam/battery/gepa_h2h_legal/state.json",
           "methods/metric_seam/battery/gepa_h2h_multi/state.json"):
    st = json.load(open(ROOT / sj))
    for key, c in st["criteria"].items():
        hist = sorted(c.get("history", []), key=lambda h: h["round"])
        for h0, h1 in zip(hist, hist[1:]):
            p0, p1 = h0.get("prompt_used"), h1.get("prompt_used")
            if p0 and p1 and p0 != p1:
                rows.append(("B_gepa_h2h", key, f"r{h0['round']}->r{h1['round']}", p0, p1))

# ---------- (B2) gepa_nc rounds ----------
banks = {}
for rf in sorted(glob.glob(str(ROOT / "datasets/notice-and-comment/v4/gepa_nc/bank_r*.jsonl"))):
    rnd = int(re.search(r"bank_r(\d+)", rf).group(1))
    banks[rnd] = {json.loads(l)["rubric_id"]: json.loads(l)["description"] for l in open(rf)}
for rid in banks.get(0, {}):
    for rnd in sorted(banks)[:-1]:
        p0, p1 = banks[rnd].get(rid), banks[rnd + 1].get(rid)
        if p0 and p1 and p0 != p1:
            rows.append(("B_gepa_nc", f"nc:{rid}", f"r{rnd}->r{rnd+1}", p0, p1))

# ---------- (B3) metric_implementer registries ----------
for md in sorted(glob.glob(str(ROOT / "outputs/metric_implementer/*/registry/metrics/*"))):
    vs = sorted(glob.glob(md + "/versions/v*__prompt.json"))
    seq = []
    for vf in vs:
        v = json.load(open(vf))
        op = (v.get("lineage") or {}).get("operator", "?")
        seq.append((Path(vf).name, op, v.get("body", "")))
    for (n0, o0, b0), (n1, o1, b1) in zip(seq, seq[1:]):
        if b0 and b1 and b0 != b1:
            rows.append(("B_registry", Path(md).name[:40], f"{n0[:4]}->{n1[:4]}[{o1}]", b0, b1))

random.Random(11).shuffle(rows)
# stratified cap: keep all h2h + registry; sample ladders and nc
caps = {"A_ladder": 200, "B_gepa_nc": 150, "B_gepa_h2h": 10**9, "B_registry": 10**9}
kept, seen = [], {}
for r in rows:
    c = seen.get(r[0], 0)
    if c < caps[r[0]]:
        kept.append(r); seen[r[0]] = c + 1
print("kept per corpus:", seen)

ANCHORS = [
    ("CONCEPT_SEMANTICS", "Wit and brevity.",
     "Wit and brevity: the text achieves its effect through compact, surprising turns of phrase, where humor emerges from precision of word choice rather than elaboration."),
    ("MECHANISM", "Tight editing creates this quality.",
     "Tight editing creates this quality. It arises because every removed redundancy raises the information density the reader experiences, so each remaining word carries comedic load."),
    ("PROCEDURE", "Judge whether the argument is rigorous.",
     "Judge whether the argument is rigorous. Step 1: list each claim. Step 2: for each claim, check whether evidence appears before it. Step 3: count unsupported claims and score 1 minus their fraction."),
    ("SCORING_MECHANICS", "Rate the persuasiveness of the essay.",
     "Rate the persuasiveness of the essay. Output a single integer 1-5; use 3 only when genuinely torn; respond in JSON as {\"score\": N} with no other text."),
    ("INPUT_HYGIENE", "Score the depth of legal analysis in the comment.",
     "Score the depth of legal analysis in the comment. Score only the visible body text; ignore letterhead, signature blocks, docket numbers, and any attachment placeholders."),
    ("EXAMPLE", "Reward vivid imagery.",
     "Reward vivid imagery. For instance, 'the fog pressed its gray thumb against the window' should score high; 'it was very foggy outside' should score low."),
    ("BOUNDARY", "Penalize purple prose.",
     "Penalize purple prose. Do not penalize ornate style when the genre is gothic pastiche; the criterion targets unmotivated ornament, not register."),
]
out, key = [], {}
for i, (corpus, lineage, step, before, after) in enumerate(kept):
    tid = f"T{i:04d}"
    out.append({"id": tid, "before": before[:900], "after": after[:1100],
                "added": added_sents(before, after)[:12]})
    key[tid] = {"corpus": corpus, "lineage": lineage, "step": step}
for j, (lab, before, after) in enumerate(ANCHORS):
    tid = f"T9{j:03d}"
    out.append({"id": tid, "before": before, "after": after,
                "added": added_sents(before, after)[:12]})
    key[tid] = {"corpus": "ANCHOR", "lineage": lab, "step": "anchor"}
random.Random(13).shuffle(out)
with open(SCRATCH / "evolution_transitions.jsonl", "w") as f:
    for r in out:
        f.write(json.dumps(r) + "\n")
json.dump(key, open(SCRATCH / "evolution_transitions_key.json", "w"), indent=1)
print("wrote", len(out), "transitions (+key)")
