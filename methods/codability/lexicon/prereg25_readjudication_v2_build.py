#!/usr/bin/env python
"""PREREG-25 re-adjudication v2 (author-calibrated standard): batches, few-shot split, key.

Prompt v2 = the lane-A prompt (prereg25_adjudication_build.PROMPT_HEAD) with a calibration
paragraph (verbatim from the coordinator brief, 2026-10-12) and a block of author-labeled
examples inserted after the PART 1 question. PART 2 (orientation) is unchanged.

Few-shot split (stratified third of the author's snapshot-2 labels, 97 pairs, 32 examples), per
cell quotas, each cell ordered by (number of original strict families disagreeing with the author,
desc; sha256(seed + doc id)):
  different: all 3;
  overlap, original verdict disagreed with the author (12): 5;
  overlap, original agreed or unjudged, comedy names (v2/t3/PREREG-11 humor; 5): 2;
  same, original disagreed (28): 16;
  same, original agreed or unjudged, comedy names, not PREREG-25 (7): 3;
  same, PREREG-25 pairs (35; all labeled same by the author and by both strict families): 3.
Creative-writing PREREG-11 pairs (7..13) are never examples (the prompt is about comedy). The
rest (65) is held out for evaluation; it keeps 12 same pairs and 7 overlap pairs on which the
strict judges disagreed with the author, and 32 of the 35 PREREG-25 pairs.
Batches: the 27 lane-A batches unchanged in content and order (same anchors, same ids), with
the new head; the 7 humor PREREG-11 pairs are appended one each to b00..b06 (ids P61.. /
F31..), and the 7 creative-writing PREREG-11 pairs form b27 with the same anchors design and
the domain words of the head switched from comedy to creative writing.

Writes outputs/lexicon/prereg25_20261012/readjudication_v2/{batches.json, key_private.json,
prompt_v2_head.txt, prompt_v2_head_creative_writing.txt, fewshot_split.json}.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys

ROOT = "ROOT"
sys.path.insert(0, f"{ROOT}/methods/codability/lexicon")
import prereg25_adjudication_build as B0  # noqa: E402

V2 = f"{ROOT}/outputs/lexicon/prereg15v2_20260828"
P25 = f"{ROOT}/outputs/lexicon/prereg25_20261012"
OUT = f"{P25}/readjudication_v2"
SCR = "/private/tmp/claude-502/-Users-anon-Projects-stanford-research-norm-research/6eb8228d-7011-4800-87e2-61a172f6003c/scratchpad"
SNAP = f"{SCR}/author_annotations_snapshot2.json"
ANNOT = f"{SCR}/annot_data.json"
SEED = "p25-readj-v2-20261012"
QUOTA = [("different", "any", 99), ("overlap", "dis", 5), ("overlap", "agree_comedy", 2),
         ("same", "dis", 16), ("same", "agree_comedy", 3), ("same", "p25", 3)]

CALIB = ("Treat each name as a pointer to the quality a comedy community cares about, not as a literal "
         "checklist. Two names are the SAME requirement when a rater applying either would attend to the same "
         "aspect of the piece, even if one is phrased as a rule and the other as a quality, one is a slogan and "
         "the other a definition, one is narrower or more concrete in wording, or the two sit in different "
         "registers. They are OVERLAPPING when they share a concern but one checks a materially different "
         "property or adds a second concern. They are DIFFERENT when they target different properties of the "
         "piece.")

Q1 = ("PART 1 - EQUIVALENCE. For each pair, answer this question: would two raters, one applying name A and one "
      "applying name B to the same piece of comedy, be checking the same requirement?\n")


def h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    old_b = json.load(open(f"{P25}/adjudication/batches.json"))
    old_k = json.load(open(f"{P25}/adjudication/key_private.json"))
    pool = json.load(open(f"{V2}/v2_form_pool.json"))
    rub = {r["form_id"]: r["rubric"] for r in pool}
    by_rub = {r["rubric"]: r["form_id"] for r in pool}
    arms = json.load(open(f"{V2}/v2_arms.json"))
    p25 = {p["mid"]: p for p in json.load(open(f"{P25}/p25_pairs.json"))["pairs"]}
    snap = json.load(open(SNAP))
    annot = {p["id"]: p for s in json.load(open(ANNOT)) for p in s["pairs"]}

    def pair_key(doc: str) -> str | None:
        st, mid = doc.split(":", 1)
        if st == "v2":
            return "|".join(sorted((arms[mid]["low_form_id"], arms[mid]["high_form_id"])))
        if st == "p25":
            return "|".join(sorted((p25[mid]["form_id_a"], p25[mid]["form_id_b"])))
        if st == "t3":
            a = annot[doc]
            return "|".join(sorted((by_rub[a["a"]], by_rub[a["b"]])))
        return None

    # ---- few-shot split
    cand = {}
    for doc, v in snap.items():
        a = annot[doc]
        assert a["a"] and a["b"]
        pk = pair_key(doc)
        if pk is not None:
            assert pk in old_k["pairs"], (doc, pk)
            assert {rub[f] for f in pk.split("|")} == {a["a"], a["b"]}, doc
        j = a.get("judges") or {}
        dis = sum(j.get(f) not in (None, v["label"]) for f in ("gpt", "claude"))
        cand[doc] = {"doc": doc, "set": v["set"], "label": v["label"], "a": a["a"], "b": a["b"],
                     "pair_key": pk, "orig_gpt": j.get("gpt"), "orig_claude": j.get("claude"),
                     "n_orig_families_disagreeing": dis}
    def cell(c: dict) -> str:
        if c["set"] == "p25":
            return "p25"
        if c["n_orig_families_disagreeing"] > 0:
            return "dis"
        if c["set"] == "p11" and int(c["doc"].split(":")[1]) >= 7:
            return "creative_writing"
        return "agree_comedy"

    few = []
    for lab, cl, n in QUOTA:
        pool_l = [c for c in cand.values() if c["label"] == lab and (cl == "any" or cell(c) == cl)
                  and cell(c) != "creative_writing"]
        pool_l.sort(key=lambda c: (-c["n_orig_families_disagreeing"], h(SEED + c["doc"])))
        few += pool_l[:n]
    for c in cand.values():
        c["cell"] = cell(c)
    few_ids = {c["doc"] for c in few}
    rng = random.Random(SEED)
    few_order = sorted(few, key=lambda c: h(SEED + "order" + c["doc"]))
    held = sorted((c for c in cand.values() if c["doc"] not in few_ids), key=lambda c: c["doc"])
    from collections import Counter
    split = {"source_snapshot": SNAP, "rule": __doc__.split("Few-shot split")[1].split("Batches:")[0].strip(),
             "fewshot": few_order, "heldout": held,
             "counts": {"fewshot": dict(Counter(c["label"] for c in few)),
                        "heldout": dict(Counter(c["label"] for c in held)),
                        "fewshot_by_set": dict(Counter(c["set"] for c in few)),
                        "heldout_by_set": dict(Counter(c["set"] for c in held)),
                        "heldout_by_label_cell": {f"{k[0]}|{k[1]}": v for k, v in
                                                  Counter((c["label"], c["cell"]) for c in held).items()}}}
    json.dump(split, open(f"{OUT}/fewshot_split.json", "w"), indent=1, ensure_ascii=False)

    ex_lines = "\n".join(f"A = {json.dumps(c['a'], ensure_ascii=False)} | B = {json.dumps(c['b'], ensure_ascii=False)}"
                         f" -> {c['label']}" for c in few_order)
    insert = (CALIB + "\n\nLabeled examples of this standard:\n" + ex_lines + "\n\n")
    head0 = B0.PROMPT_HEAD
    assert head0.count(Q1) == 1
    head = head0.replace(Q1, Q1 + insert)
    open(f"{OUT}/prompt_v2_head.txt", "w").write(head)
    cw = {"comedy communities (stand-up, sketch, humor writing) use to judge comedy":
          "creative-writing communities (fiction, screenwriting, writing workshops) use to judge creative writing",
          "the same piece of comedy": "the same piece of writing",
          "is it better or worse comedy": "is it better or worse writing",
          "the quality a comedy community cares about": "the quality a writing community cares about"}
    head_cw = head
    for k, v in cw.items():
        assert head_cw.count(k) == 1, k
        head_cw = head_cw.replace(k, v)
    open(f"{OUT}/prompt_v2_head_creative_writing.txt", "w").write(head_cw)

    # ---- PREREG-11 pairs
    p11 = [annot[f"p11:{i}"] for i in range(14)]
    batches, key = [], {}
    for ob in old_b:
        bid = ob["batch"]
        body = ob["prompt"][len(head0):]
        assert ob["prompt"].startswith(head0)
        kb = json.loads(json.dumps(old_k["key"][bid]))
        i = int(bid[1:])
        if i < 7:
            p = p11[i]
            assert "humor" in p["mid"]
            pre, post = body.split("\n\nCRITERION NAMES:\n")
            npid = f"P{len(kb['pairs']) + 1:02d}"
            pre += f"\n{npid}: A = {json.dumps(p['a'], ensure_ascii=False)} | B = {json.dumps(p['b'], ensure_ascii=False)}"
            kb["pairs"][npid] = {"kind": "p11", "doc": p["id"], "names": [p["a"], p["b"]]}
            post = post.rstrip("\n")
            for side in ("a", "b"):
                nfid = f"F{len(kb['forms']) + 1:02d}"
                post += f"\n{nfid}: {json.dumps(p[side], ensure_ascii=False)}"
                kb["forms"][nfid] = {"kind": "p11", "doc": p["id"], "side": side, "name": p[side]}
            body = pre + "\n\nCRITERION NAMES:\n" + post + "\n"
        batches.append({"batch": bid, "prompt": head + body, "n_pairs": len(kb["pairs"]), "n_forms": len(kb["forms"])})
        key[bid] = kb
    # b27: creative-writing PREREG-11 pairs, same anchor design
    p_items = [("p11", p) for p in p11[7:]] + [("anchor_same", a) for a in rng.sample(B0.PARAPHRASE_SAME, 3)] + \
              [("anchor_diff", a) for a in rng.sample(B0.CROSS_METRIC_DIFF, 3)]
    f_items = [("p11", (p, s)) for p in p11[7:] for s in ("a", "b")] + \
              [("anchor_fault", a) for a in rng.sample(B0.FAULT_NAMES, 4)]
    rng.shuffle(p_items)
    rng.shuffle(f_items)
    kb, lp, lf = {"pairs": {}, "forms": {}}, [], []
    for i, (kind, it) in enumerate(p_items, 1):
        pid = f"P{i:02d}"
        if kind == "p11":
            assert "creative" in it["mid"]
            na, nb = it["a"], it["b"]
            kb["pairs"][pid] = {"kind": "p11", "doc": it["id"], "names": [na, nb]}
        else:
            na, nb = it if rng.random() < 0.5 else (it[1], it[0])
            kb["pairs"][pid] = {"kind": kind, "answer": "same" if kind == "anchor_same" else "different",
                                "names": [na, nb]}
        lp.append(f"{pid}: A = {json.dumps(na, ensure_ascii=False)} | B = {json.dumps(nb, ensure_ascii=False)}")
    for i, (kind, it) in enumerate(f_items, 1):
        fid = f"F{i:02d}"
        if kind == "p11":
            p, s = it
            kb["forms"][fid] = {"kind": "p11", "doc": p["id"], "side": s, "name": p[s]}
            nm = p[s]
        else:
            kb["forms"][fid] = {"kind": kind, "answer": "worse", "name": it}
            nm = it
        lf.append(f"{fid}: {json.dumps(nm, ensure_ascii=False)}")
    batches.append({"batch": "b27", "prompt": head_cw + "\nPAIRS:\n" + "\n".join(lp) + "\n\nCRITERION NAMES:\n" +
                    "\n".join(lf) + "\n", "n_pairs": len(lp), "n_forms": len(lf)})
    key["b27"] = kb
    json.dump(batches, open(f"{OUT}/batches.json", "w"), indent=1, ensure_ascii=False)
    json.dump({"key": key, "pairs": old_k["pairs"], "forms": old_k["forms"], "seed": SEED,
               "p11": {p["id"]: {"a": p["a"], "b": p["b"], "mid": p["mid"]} for p in p11}},
              open(f"{OUT}/key_private.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(split["counts"]), "batches", len(batches),
          "max_chars", max(len(b["prompt"]) for b in batches))


if __name__ == "__main__":
    main()
