#!/usr/bin/env python3
"""Assemble the REGISTER BANK — one canonical, reusable row per (task, concept, variant)
joining every register instrument produced by the metric-lexicon campaign, so register can
be incorporated as a covariate in other streams (recovery, scoring-sensitivity, tacitness,
preference ladders).

Columns per row:
  task, con, variant, count, n_namings, is_head, py_head_share      (usage, W1/W6)
  stratum, formality, nominalization                                 (W4 judge, full 2,195)
  latinate_v22                                                       (lexical detector v2.2)
  inst_share, inst_uses, total_uses                                  (W3 institutionality)
  metaphoricity, transparency, thickthin                             (GLM axes, where judged)
  height_z  = mean(z(formality), z(latinate indicator))              (W4 composite)

Output: outputs/lexicon/register_bank_20260722.jsonl  (+ .meta.json provenance sidecar)
"""
import json

import numpy as np

ROOT = "ROOT"
LEX = f"{ROOT}/outputs/lexicon"
LATMAP = {"germanic": 0.0, "mixed": 0.5, "latinate": 1.0, "greek": 1.0}


def jload(path):
    return [json.loads(l) for l in open(path)]


def main():
    from methods.codability.lexicon.latinate_detector import V2, latinate_score
    v2 = V2()

    reg = {}
    for r in jload(f"{LEX}/register_height_judgments.jsonl"):
        if r.get("stratum") in LATMAP:
            reg.setdefault(r["variant"], r)      # first (Sonnet-batch) row wins
    inst = {r["variant"]: r for r in jload(f"{LEX}/variant_institutionality_20260720.jsonl")}
    axes = {}
    for ax in ("metaphoricity", "transparency", "thickthin"):
        for r in jload(f"{LEX}/axis_{ax}_20260721.jsonl"):
            axes.setdefault(r["variant"], {})[ax] = r["score"]

    # composite height z on the full judged inventory
    vs = sorted(reg)
    f = np.array([float(reg[v].get("formality") or 4) for v in vs])
    l = np.array([LATMAP[reg[v]["stratum"]] for v in vs])
    zf, zl = (f - f.mean()) / f.std(), (l - l.mean()) / l.std()
    height = {v: float((a + b) / 2) for v, a, b in zip(vs, zf, zl)}

    rows = []
    miss_reg = 0
    for r in jload(f"{LEX}/name_variants_20260720.jsonl"):
        v = r["variant"]
        j = reg.get(v)
        if j is None:
            miss_reg += 1
        row = {"task": r["task"], "con": r["concept"], "variant": v,
               "count": r["count"], "n_namings": r["n_namings"],
               "is_head": r["is_head"], "py_head_share": r["py_head_share"],
               "stratum": j and j["stratum"], "formality": j and j.get("formality"),
               "nominalization": j and j.get("nominalization"),
               "height_z": height.get(v),
               "latinate_v22": latinate_score(v, v2.word)}
        i = inst.get(v)
        if i:
            row.update({"inst_share": i["inst_share"], "inst_uses": i["inst_uses"],
                        "total_uses": i["total_uses"]})
        row.update(axes.get(v, {}))
        rows.append(row)

    out = f"{LEX}/register_bank_20260722.jsonl"
    with open(out, "w") as fo:
        for r in rows:
            fo.write(json.dumps(r) + "\n")
    meta = {"built": "2026-07-22", "rows": len(rows),
            "unique_variants": len({r['variant'] for r in rows}),
            "judged_register": len(reg), "missing_register_rows": miss_reg,
            "with_institutionality": sum(1 for r in rows if "inst_share" in r),
            "with_axes": sum(1 for r in rows if "metaphoricity" in r),
            "sources": ["name_variants_20260720", "register_height_judgments (2,195 = full)",
                        "variant_institutionality_20260720", "axis_*_20260721",
                        "latinate_detector v2.2"]}
    json.dump(meta, open(out.replace(".jsonl", ".meta.json"), "w"), indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
