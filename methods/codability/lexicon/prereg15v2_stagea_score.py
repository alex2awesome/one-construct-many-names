#!/usr/bin/env python
"""PREREG-15-v2 Stage A scorer: anchor checks, formality aggregation, arm selection.

Runs AFTER the judging wave. Per chunk: >=8/10 anchors must have the correct stratum
(the p10 discipline); failing chunks are excluded and reported for fresh re-judging.
Per form: formality = mean over its (up to 2) passing-chunk judgments. Arms per metric:
lowest and highest mean formality; metrics whose high-low difference is under 1.0 SD of
the pooled form-formality distribution are excluded from the primary test (frozen rule,
notes/2026-08-28__prereg15v2-community-matching.md).

Also writes label-free, deterministically shuffled scoring copies of the item files so
the scoring judge never sees a label or a label-ordered list (disclosed run detail; the
shuffle is identical across arms and replicates of a metric).

Out (all in outputs/lexicon/prereg15v2_20260828/):
  v2_stagea_report.md, v2_form_formality.json, v2_arms.json, v2_scoring_items_<mid>.json
"""
from __future__ import annotations

import glob
import json
import os
import random
import statistics as st

D = "ROOT/outputs/lexicon/prereg15v2_20260828"
GATE = 8


def main() -> None:
    key = json.load(open(f"{D}/v2_key_private.json"))
    pool = {r["form_id"]: r for r in json.load(open(f"{D}/v2_form_pool.json"))}

    chunk_pass, chunk_fail, form_scores = {}, {}, {}
    for out in sorted(glob.glob(f"{D}/v2_out_*.jsonl")):
        cx = out.split("_")[-1].split(".")[0]
        rows = [json.loads(l) for l in open(out)]
        judged = {r["id"]: r for r in rows}
        anchors = [(mid, meta) for mid, meta in key.items()
                   if meta["kind"] == "anchor" and int(meta["chunk"]) == int(cx)]
        ok = sum(1 for mid, meta in anchors
                 if judged.get(mid, {}).get("stratum") == meta["anchor_truth"])
        (chunk_pass if ok >= GATE else chunk_fail)[cx] = f"{ok}/{len(anchors)}"
        if ok < GATE:
            continue
        for mid, meta in key.items():
            if meta["kind"] == "form" and int(meta["chunk"]) == int(cx) and mid in judged:
                f = judged[mid].get("formality")
                if isinstance(f, (int, float)) and 1 <= f <= 7:
                    form_scores.setdefault(meta["form_id"], []).append(float(f))

    means = {fid: st.mean(v) for fid, v in form_scores.items()}
    both = [fid for fid, v in form_scores.items() if len(v) >= 2]
    agree = (st.mean([abs(form_scores[f][0] - form_scores[f][1]) for f in both])
             if both else float("nan"))
    sd = st.pstdev(list(means.values()))

    metric_forms = {}
    for fid, rec in pool.items():
        if fid not in means:
            continue
        for mid in rec["metric_ids"]:
            metric_forms.setdefault(mid, []).append((means[fid], fid))

    arms, excluded = {}, {}
    for mid, forms in sorted(metric_forms.items()):
        forms.sort()
        (f_lo, lo), (f_hi, hi) = forms[0], forms[-1]
        if f_hi - f_lo < 1.0 * sd:
            excluded[mid] = round(f_hi - f_lo, 3)
            continue
        arms[mid] = {"low_form_id": lo, "low_rubric": pool[lo]["rubric"], "f_low": round(f_lo, 3),
                     "high_form_id": hi, "high_rubric": pool[hi]["rubric"], "f_high": round(f_hi, 3)}

    for path in sorted(glob.glob(f"{D}/v2_items_*.json")):
        mid = os.path.basename(path)[len("v2_items_"):-len(".json")]
        items = json.load(open(path))["items"]
        rng = random.Random(f"p15v2-shuffle-{mid}")
        order = list(range(len(items)))
        rng.shuffle(order)
        blind = [{"item_idx": items[i]["item_idx"], "text": items[i]["text"]} for i in order]
        json.dump({"metric_id": mid, "items": blind},
                  open(f"{D}/v2_scoring_items_{mid}.json", "w"), indent=0)

    json.dump({"form_means": {k: round(v, 3) for k, v in means.items()},
               "n_judgments_per_form": {k: len(v) for k, v in form_scores.items()},
               "pooled_sd": round(sd, 4), "pass1_vs_pass2_mean_absdiff": round(agree, 3)},
              open(f"{D}/v2_form_formality.json", "w"), indent=0)
    json.dump(arms, open(f"{D}/v2_arms.json", "w"), indent=1)

    with open(f"{D}/v2_stagea_report.md", "w") as fh:
        fh.write("# PREREG-15-v2 Stage A report\n\n")
        fh.write(f"- chunks passing anchor check: {len(chunk_pass)}; failing: {len(chunk_fail)}\n")
        if chunk_fail:
            fh.write(f"- FAILED chunks (need fresh re-judging): {chunk_fail}\n")
        fh.write(f"- forms with >=1 judgment: {len(means)} of {len(pool)}; with both passes: {len(both)}\n")
        fh.write(f"- pass1-vs-pass2 mean |diff|: {agree:.3f}; pooled SD of form formality: {sd:.3f}\n")
        fh.write(f"- metrics with arms: {len(arms)}; excluded for gap under 1.0 SD ({sd:.2f}): {len(excluded)} {excluded}\n")
    print(open(f"{D}/v2_stagea_report.md").read())


if __name__ == "__main__":
    main()
