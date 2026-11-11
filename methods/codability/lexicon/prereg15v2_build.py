#!/usr/bin/env python3
"""PREREG-15-v2 builder — pure data construction, no LLM judging, no API calls.

Implements the frozen design in
notes/2026-08-28__prereg15v2-community-matching.md:

Stage A (formality-wave chunk materials):
  1. metric set = the 88 mids from outputs/lexicon/prereg15_results_20260723.json
     per_metric.
  2. form pool = outputs/lexicon/silver_join/humor_forms_manifest.json restricted to
     those mids, deduped by identical rubric string, fresh minted ids f000..f857.
  3. chunks of ~45 forms + 10 camouflaged anchors (from the p10 anchor bank, 190
     items) each; every form appears in exactly TWO different chunks (k=2
     independent random assignments); every line (form or anchor) gets a fresh
     minted id so anchors are indistinguishable from forms; each chunk is shuffled.
  4. judge instructions = verbatim copy of p10_judge_instructions.txt.

Stage B (item draws):
  5. per surviving metric: 20 pos + 20 neg candidate documents drawn by stable hash
     SHA-256(doc_id + "p15v2-20260828"); metrics with <25 pos or <25 neg candidates
     are excluded.
  6. doc ids resolved to text via the local silver-text corpora; unresolvable docs
     are replaced by the next doc in hash order; metrics that cannot reach 20
     resolvable pos + 20 resolvable neg are excluded.
  7. v2_items_<mid>.json per surviving metric.

Everything is deterministic given SEED=20260828 (single numpy Generator, drawn from
in a fixed, documented order). Never modifies outputs/lexicon/prereg15* (v1).

Run:  python3 methods/codability/lexicon/prereg15v2_build.py
Output: outputs/lexicon/prereg15v2_20260828/
"""
import glob
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
LEX = ROOT / "outputs/lexicon"
SILVER_JOIN = LEX / "silver_join"
REGISTER = LEX / "register_corpus_20260723"
SPLITS = ROOT / "outputs/silver_label_splits"
STANDUP = ROOT / "datasets/humor/standup_reddit"
OUT = LEX / "prereg15v2_20260828"

SEED = 20260828
HASH_SALT = "p15v2-20260828"
CHUNK_TARGET = 45
N_ANCHORS_PER_CHUNK = 10
N_POS = 20
N_NEG = 20
TRUNC_CHARS = 4000

POS_FILES = ["humor_ypos_v3.json", "humor_ypos_v4.json", "humor_ypos_v5.json",
             "humor_ypos_v6.json", "humor_r2_ypos.json", "humor_r3_ypos.json"]
NEG_FILES = ["humor_r2_yneg.json", "humor_r3_yneg.json"]


def jload(p):
    with open(p) as f:
        return json.load(f)


def jlload(p):
    with open(p) as f:
        return [json.loads(line) for line in f]


def stable_hash(doc_id):
    return hashlib.sha256((doc_id + HASH_SALT).encode()).hexdigest()


def mint_id(rng, used):
    """Fresh random id, same format for forms and anchors alike (camouflage)."""
    while True:
        digits = rng.integers(0, 16, size=10)
        cid = "n" + "".join(f"{d:x}" for d in digits)
        if cid not in used:
            used.add(cid)
            return cid


# ---------------------------------------------------------------------------
# Stage A materials
# ---------------------------------------------------------------------------

def load_mids():
    res = jload(LEX / "prereg15_results_20260723.json")
    mids = [r["mid"] for r in res["per_metric"]]
    assert len(mids) == 88, f"expected 88 mids, got {len(mids)}"
    return mids


def build_form_pool(mids):
    """Restrict humor_forms_manifest.json to `mids`, dedupe identical rubric
    strings, mint fresh sequential ids f000..f857 in file-encounter order
    (deterministic regardless of PYTHONHASHSEED — dict insertion order only)."""
    manifest = jload(SILVER_JOIN / "humor_forms_manifest.json")
    mids_set = set(mids)
    rubric_to_formid = {}
    rubric_to_metrics = {}
    order = []
    for r in manifest:
        if r["metric_id"] not in mids_set:
            continue
        rub = r["rubric"]
        if rub not in rubric_to_formid:
            fid = f"f{len(rubric_to_formid):03d}"
            rubric_to_formid[rub] = fid
            rubric_to_metrics[rub] = set()
            order.append(rub)
        rubric_to_metrics[rub].add(r["metric_id"])
    pool = [{"form_id": rubric_to_formid[rub], "rubric": rub,
             "metric_ids": sorted(rubric_to_metrics[rub])}
            for rub in order]
    return pool


def load_anchor_bank():
    """p10 anchor bank: naXXXX -> {term, truth (etymology stratum)}. Anchor terms
    are recovered by scanning the p10 chunk files for rows whose id is a key of
    p10_key.json's 'truth' dict (those rows kept their literal naXXXX id in p10;
    v2 re-mints camouflaged ids for them instead, see build_chunks)."""
    key = jload(REGISTER / "p10_key.json")
    truth = key["truth"]
    terms = {}
    for fn in sorted(glob.glob(str(REGISTER / "p10_chunk_*.jsonl"))):
        for row in jlload(fn):
            if row["id"] in truth:
                terms[row["id"]] = row["term"]
    missing = set(truth) - set(terms)
    assert not missing, f"anchor terms missing for {missing}"
    return {aid: {"term": terms[aid], "truth": truth[aid]} for aid in truth}


def build_chunks(form_pool, anchor_bank, rng):
    """Two independent full passes over the form pool (each pass = one random
    permutation split into chunks of ~45), so every form appears in exactly two
    different chunks with two different minted ids. Each chunk then draws 10
    anchors without replacement from the 190-item bank (reuse across chunks is
    expected: 38 chunks * 10 > 190) and every line (form or anchor) gets a fresh
    minted id, then the chunk is shuffled."""
    form_ids = [f["form_id"] for f in form_pool]
    formid_to_rubric = {f["form_id"]: f["rubric"] for f in form_pool}
    n_forms = len(form_ids)
    n_chunks_per_pass = round(n_forms / CHUNK_TARGET)
    form_id_arr = np.array(form_ids)

    anchor_ids = sorted(anchor_bank.keys())
    n_anchors = len(anchor_ids)

    used_ids = set()
    key_private = {}
    chunks = []          # list[list[{"id","term"}]]
    chunk_provenance = []  # list[{"pass","n_forms","n_anchors"}]

    for pass_i in range(2):
        perm = rng.permutation(n_forms)
        groups = np.array_split(perm, n_chunks_per_pass)
        for g in groups:
            chunk_idx = len(chunks)
            group_form_ids = form_id_arr[g].tolist()
            lines = []
            for fid in group_form_ids:
                cid = mint_id(rng, used_ids)
                lines.append({"id": cid, "term": formid_to_rubric[fid]})
                key_private[cid] = {"kind": "form", "form_id": fid, "chunk": chunk_idx}
            anchor_pick = rng.choice(n_anchors, size=N_ANCHORS_PER_CHUNK, replace=False)
            for ai in anchor_pick:
                aid = anchor_ids[ai]
                cid = mint_id(rng, used_ids)
                lines.append({"id": cid, "term": anchor_bank[aid]["term"]})
                key_private[cid] = {"kind": "anchor", "anchor_truth": anchor_bank[aid]["truth"],
                                     "anchor_id": aid, "chunk": chunk_idx}
            order = rng.permutation(len(lines))
            lines = [lines[i] for i in order]
            chunks.append(lines)
            chunk_provenance.append({"pass": pass_i, "n_forms": len(group_form_ids),
                                      "n_anchors": N_ANCHORS_PER_CHUNK})

    # verify every form appears in exactly 2 distinct chunks
    form_chunk_count = {}
    for cid, row in key_private.items():
        if row["kind"] == "form":
            form_chunk_count.setdefault(row["form_id"], set()).add(row["chunk"])
    assert set(form_chunk_count) == set(form_ids)
    assert all(len(v) == 2 for v in form_chunk_count.values()), \
        "every form must land in exactly 2 distinct chunks"

    return chunks, key_private, chunk_provenance


# ---------------------------------------------------------------------------
# Stage B materials
# ---------------------------------------------------------------------------

def load_label_pools():
    pos_dicts = [jload(SPLITS / f) for f in POS_FILES]
    neg_dicts = [jload(SPLITS / f) for f in NEG_FILES]
    return pos_dicts, neg_dicts


def candidate_sets(mid, pos_dicts, neg_dicts):
    pos = set()
    for d in pos_dicts:
        pos.update(d.get(mid, []))
    neg = set()
    for d in neg_dicts:
        neg.update(d.get(mid, []))
    neg -= pos
    return pos, neg


def build_text_index(all_referenced):
    """Primary source: outputs/lexicon/silver_join/humor_all_silver_texts.jsonl
    (thread_id -> text, covers reddit_standupcomedy/standupshots/standupworkshop
    and ast prefixes). Supplemented with datasets/humor/standup_reddit's local
    filtered_threads.jsonl (150 raw threads, unprefixed ids) reconstructed against
    whichever of the three reddit_* prefixes the referenced-doc-id universe
    disambiguates to (skipped if a raw id matches more than one prefix)."""
    idx = {}
    with open(SILVER_JOIN / "humor_all_silver_texts.jsonl") as f:
        for line in f:
            d = json.loads(line)
            idx[d["thread_id"]] = d["text"]
    n_primary = len(idx)

    n_supplement = 0
    n_ambiguous = 0
    prefixes = ["reddit_standupcomedy", "reddit_standupshots", "reddit_standupworkshop"]
    fthreads = STANDUP / "filtered_threads.jsonl"
    if fthreads.exists():
        for row in jlload(fthreads):
            tid, text = row.get("thread_id"), row.get("text")
            if not tid or not text:
                continue
            hits = [p for p in prefixes if f"{p}__{tid}" in all_referenced]
            if len(hits) > 1:
                n_ambiguous += 1
                continue
            for p in hits:
                full_id = f"{p}__{tid}"
                if full_id not in idx:
                    idx[full_id] = text
                    n_supplement += 1
    return idx, {"n_primary": n_primary, "n_supplement_added": n_supplement,
                 "n_supplement_ambiguous": n_ambiguous, "n_total": len(idx)}


def draw_resolved(cand_set, n_target, text_idx):
    """Walk candidates in stable-hash order; skip unresolvable docs (recording a
    replacement) and keep going until n_target resolvable docs are collected (or
    the candidate pool is exhausted). The returned list is therefore naturally in
    hash order, satisfying both the draw rule and the 'each group in hash order'
    output requirement."""
    ranked = sorted(cand_set, key=stable_hash)
    chosen, n_replacements = [], 0
    for doc_id in ranked:
        if doc_id in text_idx:
            chosen.append(doc_id)
        else:
            n_replacements += 1
        if len(chosen) == n_target:
            break
    return chosen, n_replacements


def build_stage_b(mids, out_dir):
    pos_dicts, neg_dicts = load_label_pools()

    all_referenced = set()
    for d in pos_dicts + neg_dicts:
        for v in d.values():
            all_referenced.update(v)
    text_idx, text_idx_stats = build_text_index(all_referenced)

    survivors, exclusions = [], []
    replacement_report = {}

    for mid in mids:
        pos_cand, neg_cand = candidate_sets(mid, pos_dicts, neg_dicts)
        if len(pos_cand) < 25 or len(neg_cand) < 25:
            exclusions.append({"mid": mid, "reason": "candidate_pool_too_small",
                                "n_pos_candidates": len(pos_cand),
                                "n_neg_candidates": len(neg_cand)})
            continue

        pos_chosen, pos_repl = draw_resolved(pos_cand, N_POS, text_idx)
        neg_chosen, neg_repl = draw_resolved(neg_cand, N_NEG, text_idx)
        if len(pos_chosen) < N_POS or len(neg_chosen) < N_NEG:
            exclusions.append({"mid": mid, "reason": "insufficient_resolvable_text",
                                "n_pos_candidates": len(pos_cand),
                                "n_neg_candidates": len(neg_cand),
                                "n_pos_resolved": len(pos_chosen),
                                "n_neg_resolved": len(neg_chosen)})
            continue

        items = []
        for i, doc_id in enumerate(pos_chosen):
            items.append({"item_idx": i, "doc_id": doc_id, "label": 1,
                           "text": text_idx[doc_id][:TRUNC_CHARS]})
        for j, doc_id in enumerate(neg_chosen):
            items.append({"item_idx": N_POS + j, "doc_id": doc_id, "label": 0,
                           "text": text_idx[doc_id][:TRUNC_CHARS]})

        with open(out_dir / f"v2_items_{mid}.json", "w") as f:
            json.dump({"metric_id": mid, "items": items}, f, indent=1)

        survivors.append(mid)
        replacement_report[mid] = {"pos_replacements": pos_repl, "neg_replacements": neg_repl,
                                    "n_pos_candidates": len(pos_cand),
                                    "n_neg_candidates": len(neg_cand)}

    return survivors, exclusions, replacement_report, text_idx_stats


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    mids = load_mids()

    # --- Stage A ---
    form_pool = build_form_pool(mids)
    with open(OUT / "v2_form_pool.json", "w") as f:
        json.dump(form_pool, f, indent=1)

    anchor_bank = load_anchor_bank()
    chunks, key_private, chunk_provenance = build_chunks(form_pool, anchor_bank, rng)

    for i, lines in enumerate(chunks):
        with open(OUT / f"v2_chunk_{i:02d}.jsonl", "w") as f:
            for row in lines:
                f.write(json.dumps(row) + "\n")

    with open(OUT / "v2_key_private.json", "w") as f:
        json.dump(key_private, f, indent=1)

    judge_instructions = (REGISTER / "p10_judge_instructions.txt").read_text()
    (OUT / "v2_judge_instructions.txt").write_text(judge_instructions)

    key_sha256 = hashlib.sha256((OUT / "v2_key_private.json").read_bytes()).hexdigest()

    # --- Stage B ---
    survivors, exclusions, replacement_report, text_idx_stats = build_stage_b(mids, OUT)

    # --- report ---
    n_forms = len(form_pool)
    n_chunks = len(chunks)
    n_lines_total = sum(len(c) for c in chunks)
    total_repl = sum(v["pos_replacements"] + v["neg_replacements"]
                      for v in replacement_report.values())
    max_repl_mid = max(replacement_report,
                        key=lambda m: replacement_report[m]["pos_replacements"]
                        + replacement_report[m]["neg_replacements"]) if replacement_report else None

    lines = []
    lines.append("# PREREG-15-v2 build report")
    lines.append("")
    lines.append("Pure data construction (no LLM judging, no API calls) per "
                  "`notes/2026-08-28__prereg15v2-community-matching.md`, built by "
                  "`methods/codability/lexicon/prereg15v2_build.py`.")
    lines.append("")
    lines.append("## Stage A — formality-wave chunk materials")
    lines.append("")
    lines.append(f"- Metric set: {len(mids)} mids from `prereg15_results_20260723.json` per_metric.")
    lines.append(f"- Form pool: {n_forms} distinct rubric strings restricted to those mids "
                  "(deduped from 901 raw metric-form rows), minted `f000`..`f{:03d}`."
                  .format(n_forms - 1))
    lines.append(f"- Chunks: {n_chunks} total (2 independent passes x "
                  f"{n_chunks // 2} chunks/pass), {n_lines_total} lines "
                  f"({2*n_forms} form-lines + {n_chunks*N_ANCHORS_PER_CHUNK} anchor-lines). "
                  "Every form appears in exactly 2 distinct chunks (verified in-script).")
    lines.append(f"- Anchor bank: 190 items from `p10_key.json['truth']`; {n_chunks} chunks x "
                  f"{N_ANCHORS_PER_CHUNK} anchors = {n_chunks*N_ANCHORS_PER_CHUNK} anchor-slots "
                  "(reuse across chunks expected and required, since 190 < total slots; no reuse "
                  "within a chunk).")
    lines.append(f"- v2_key_private.json sha256: `{key_sha256}`")
    lines.append("")
    lines.append("## Stage B — item draws")
    lines.append("")
    lines.append(f"- Doc-text index: {text_idx_stats['n_primary']} docs from "
                  "`silver_join/humor_all_silver_texts.jsonl` + "
                  f"{text_idx_stats['n_supplement_added']} net-new from "
                  "`datasets/humor/standup_reddit/filtered_threads.jsonl` "
                  f"({text_idx_stats['n_supplement_ambiguous']} skipped as prefix-ambiguous) "
                  f"= {text_idx_stats['n_total']} resolvable docs total.")
    lines.append(f"- Surviving metrics: {len(survivors)} / {len(mids)}")
    lines.append(f"- Excluded: {len(exclusions)}")
    for e in exclusions:
        if e["reason"] == "candidate_pool_too_small":
            lines.append(f"  - `{e['mid']}`: candidate_pool_too_small "
                          f"(pos_candidates={e['n_pos_candidates']}, "
                          f"neg_candidates={e['n_neg_candidates']}, need >=25 each)")
        else:
            lines.append(f"  - `{e['mid']}`: insufficient_resolvable_text "
                          f"(pos_resolved={e['n_pos_resolved']}/20, "
                          f"neg_resolved={e['n_neg_resolved']}/20, "
                          f"pos_candidates={e['n_pos_candidates']}, "
                          f"neg_candidates={e['n_neg_candidates']})")
    lines.append(f"- Replacement draws (unresolvable doc -> next by hash order): "
                  f"{total_repl} total across {len(survivors)} surviving metrics "
                  f"(mean {total_repl/max(1,len(survivors)):.2f} per metric out of 40 items; "
                  f"max at `{max_repl_mid}`: "
                  f"{replacement_report[max_repl_mid]['pos_replacements'] + replacement_report[max_repl_mid]['neg_replacements'] if max_repl_mid else 0})")
    lines.append("")
    lines.append("### Per-metric replacement counts (surviving metrics)")
    lines.append("")
    lines.append("| mid | pos_candidates | neg_candidates | pos_replacements | neg_replacements |")
    lines.append("|---|---|---|---|---|")
    for mid in survivors:
        r = replacement_report[mid]
        lines.append(f"| {mid} | {r['n_pos_candidates']} | {r['n_neg_candidates']} | "
                      f"{r['pos_replacements']} | {r['neg_replacements']} |")
    lines.append("")

    with open(OUT / "v2_build_report.md", "w") as f:
        f.write("\n".join(lines))

    print(f"mids={len(mids)} forms={n_forms} chunks={n_chunks} lines={n_lines_total}")
    print(f"stage B survivors={len(survivors)} excluded={len(exclusions)}")
    print(f"key_private sha256={key_sha256}")
    print(f"output dir: {OUT}")


if __name__ == "__main__":
    main()
