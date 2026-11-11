"""Face-2 decompression grid: articulation-TYPE rungs x readers (theory Q3; roadmap Phase 3).

Rungs are TYPES of telling, not lengths (user decision 2026-07-02; prior result: length alone is a
poor predictor). Per metric the writer unpacks the SAME concept into:

    name         the R3 cluster name alone -- a pure index into the reader's prior
    definition   intension ("which means...")            } length-matched (~<=50 words)
    explanation  mechanism/recognition ("happens when...") } so type is not secretly length
    full_rubric  the R3 merged_description (the Face-1 anchor channel)
    exemplars    k confident positives + k negatives shown, NO verbal content
    dossier      definition + explanation + exemplars (telling+showing ceiling)

Readings: name->definition gain = lexical/indexical gap; definition->explanation = knowing-that ->
knowing-how (Ryle); explanation->exemplars = ostensive content (Polanyi); dossier plateau below the
ceiling with a saturated census = tacit-within-frame (levels.py L4). The strong-weak reader gap at
`name` measures the reader's enculturated stock. RUNGS is a registry: new axes (contrast pairs,
worked exemplars, program form, emic writer) = new entries, no structural change.

Phases (resumable, artifacts on disk):
    messages  writer model unpacks each metric once   -> <out>/messages.json  (auditable/editable)
    score     each reader scores probes under every rung x form -> <out>/grid_<reader>.npz
    report    balanced-accuracy vs the reference target M_i, normalized by dossier -> report.json

Verbal rungs are orbit-averaged over deterministic reformulations (form control; the 12%-flip
finding makes single-phrasing rungs uninterpretable). Exemplar-bearing rungs stay canonical (the
reformulator would mangle example blocks). Exemplar probes are excluded from evaluation.

Smoke (no GPU):  python -m methods.codability.run_decompression_grid --fake --phase all \
    --ref-dir <ckpts> --out-dir <scratch> --gi-list 0,47 --forms 2
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re

import numpy as np

from methods.metric_implementer import config as cfgmod
from methods.metric_implementer import vinfo
from methods.metric_implementer.vllm_backend import make_judge_backend
from methods.metric_implementer.experiments import alpha_probe as ap
from methods.metric_implementer.experiments.mine_clusters import r1_groups, r2_groups, r3_groups
from methods.metric_implementer.experiments.run_real_test import _load_texts
from methods.metric_implementer.experiments.value_census import i_binary

RUNG_ORDER = ["name", "definition", "explanation", "full_rubric", "exemplars", "dossier",
              "dossier_v2"]
VERBAL = {"name", "definition", "explanation", "full_rubric"}      # reformulation-safe rungs
# dossier_v2 (2026-07-04 audit fix, task #17): v1 dossier embeds the exemplars block whose first
# line is "Judge by these examples ONLY." -- a self-contradicting prompt (tell everything, then
# order the reader to ignore it). v2 = identical content with the examples reframed as
# "Illustrative examples:" so telling+showing compose instead of conflicting. v1 rungs are kept
# byte-identical for comparability with the CW/humor v1 grids.

DEF_PROMPT = ("A criterion for judging {task} is named: \"{name}\".\nIts full rubric reads:\n"
              "\"{rubric}\"\n\nState a precise DEFINITION of this criterion in at most 50 words: "
              "what property must a text have to satisfy it? Do not give examples and do not "
              "describe how to detect it. Reply with the definition only.")
EXP_PROMPT = ("A criterion for judging {task} is named: \"{name}\".\nIts full rubric reads:\n"
              "\"{rubric}\"\n\nExplain in at most 50 words HOW this quality arises in a text and "
              "how a reader RECOGNIZES it: the mechanism and the observable signs. Do not restate "
              "the definition. Reply with the explanation only.")


def _ckpts(ref_dir: str, gi_list):
    out = {}
    for f in sorted(glob.glob(os.path.join(ref_dir, "*_sigs.npz"))):
        m = re.search(r"_(R[123])_metric(\d+)_sigs\.npz$", os.path.basename(f))
        if not m:
            continue
        lvl, gi = m.group(1), int(m.group(2))
        if gi_list and gi not in gi_list:
            continue
        if gi in out and out[gi][0] != lvl:                    # same gi at two levels = identity
            raise ValueError(f"gi={gi} appears at both {out[gi][0]} and {lvl} in {ref_dir} -- "
                             f"mixed-level dir; refusing (levels index different hierarchies)")
        out[gi] = (lvl, f)
    return out


def _groups(task: str, bucket: str, level: str):
    if level == "R1":
        return r1_groups(task)
    return (r3_groups if level == "R3" else r2_groups)(task, bucket)


def _excerpt(text: str, n: int) -> str:
    t = " ".join(str(text).split())
    return t[:n] + ("..." if len(t) > n else "")


def build_messages(a, cfg, ckpts, probe_texts) -> dict:
    wcfg = cfgmod.apply_task_preset(cfgmod.ImplementerConfig(), a.task)
    if a.fake:
        wcfg.vllm_fake = True
    writer = make_judge_backend(a.writer_model, wcfg, temperature=None)
    groups_cache, msgs = {}, {}
    for gi, (lvl, f) in ckpts.items():
        z = np.load(f, allow_pickle=True)
        if lvl not in groups_cache:
            groups_cache[lvl] = _groups(a.task, a.r2_bucket, lvl)
        name = str(z["name"])
        rubric = str(groups_cache[lvl][gi].get("merged_description", "")).strip()
        def gen(tpl):
            return str(writer.generate(
                tpl.format(task=a.task.replace("-", " "), name=name, rubric=rubric),
                max_tokens=90)).strip()
        definition = gen(DEF_PROMPT) or f"[definition unavailable for {name}]"
        explanation = gen(EXP_PROMPT) or f"[explanation unavailable for {name}]"
        # exemplars: most-confident positives/negatives under the REFERENCE target, held out of eval
        M = np.nan_to_num(np.asarray(z["M_i"], float), nan=0.5)
        order = np.argsort(M)
        neg_idx = [int(i) for i in order[: a.k_exemplars]]
        pos_idx = [int(i) for i in order[::-1][: a.k_exemplars]]
        ex_body = ("Texts that SATISFY the criterion:\n"
                   + "\n".join(f"[+{j+1}] {_excerpt(probe_texts[i], a.max_ex_chars)}"
                               for j, i in enumerate(pos_idx))
                   + "\nTexts that VIOLATE the criterion:\n"
                   + "\n".join(f"[-{j+1}] {_excerpt(probe_texts[i], a.max_ex_chars)}"
                               for j, i in enumerate(neg_idx)))
        ex_block = "Judge by these examples ONLY.\n" + ex_body      # v1 ostension pole (unchanged)
        msgs[str(gi)] = {
            "name": name, "level": lvl, "rubric": rubric,
            "rungs": {"name": name,
                      "definition": definition,
                      "explanation": explanation,
                      "full_rubric": rubric,
                      "exemplars": ex_block,
                      "dossier": (f"{definition}\nHow to recognize it: {explanation}\n{ex_block}"),
                      "dossier_v2": (f"{definition}\nHow to recognize it: {explanation}\n"
                                     f"Illustrative examples:\n{ex_body}")},
            "exemplar_idx": {"pos": pos_idx, "neg": neg_idx},
            "word_len": {},
        }
        msgs[str(gi)]["word_len"] = {r: len(str(t).split())
                                     for r, t in msgs[str(gi)]["rungs"].items()}
    path = os.path.join(a.out_dir, "messages.json")
    json.dump(msgs, open(path, "w"), indent=1)
    print(f"messages: {len(msgs)} metrics x {len(RUNG_ORDER)} rungs -> {path}")
    return msgs


def score_reader(a, cfg, reader: str, msgs: dict, probe_texts) -> str:
    ecfg = cfgmod.apply_task_preset(cfgmod.ImplementerConfig(), a.task)
    if a.fake:
        ecfg.vllm_fake = True
    executor = make_judge_backend(reader, ecfg, temperature=None)
    rows, meta = [], []
    for gi, m in msgs.items():
        for rung in RUNG_ORDER:
            if rung not in m["rungs"]:      # older messages.json (e.g. CW v1 predates
                continue                    # dossier_v2) must stay byte-identical — skip
            base = m["rungs"][rung]
            forms = [("canonical", base)]
            if rung in VERBAL and a.forms > 1:
                forms += list(ap._reformulations(base))[: a.forms - 1]
            for kind, txt in forms:
                rows.append(ap.signature(executor, txt, probe_texts, cfg.max_text_chars))
                meta.append({"gi": int(gi), "rung": rung, "form": kind})
        print(f"  scored gi={gi} ({m['name'][:40]})")
    tag = re.sub(r"[^A-Za-z0-9.-]+", "_", os.path.basename(reader.rstrip("/")))
    path = os.path.join(a.out_dir, f"grid_{tag}.npz")
    probe_sha256 = np.asarray([hashlib.sha256(str(t).encode()).hexdigest()
                               for t in probe_texts])
    probe_set_sha256 = hashlib.sha256("\n".join(probe_sha256).encode()).hexdigest()
    np.savez(path, scores=np.vstack(rows), meta=np.array([json.dumps(x) for x in meta], dtype=object),
             reader=reader, ref_dir=a.ref_dir, probe_sha256=probe_sha256,
             probe_set_sha256=probe_set_sha256)
    print(f"reader {tag}: {len(rows)} rows -> {path}")
    return path


def _span_r2(sigs, m_bar, mask):
    """In-span vs out-of-span classification of a rung judge: 5-fold CV R^2 of ridge-regressing the
    rung signature on the metric's criteria/species basis (same probes). HIGH = the rung is an
    assembly of census-known units (better addressing); LOW while the rung carries value = content
    no articulated criterion in the census induces (a genuinely new unit; check vs OPT+eps)."""
    try:
        from sklearn.linear_model import RidgeCV
        from sklearn.model_selection import cross_val_score
    except Exception:
        return None
    X = np.nan_to_num(np.asarray(sigs, float), nan=0.5).T[mask]
    y = np.nan_to_num(np.asarray(m_bar, float), nan=0.5)[mask]
    if X.shape[0] < 50 or float(np.std(y)) < 1e-6:
        return None
    r2 = cross_val_score(RidgeCV(alphas=np.logspace(0, 4, 9)), X, y, cv=5, scoring="r2")
    return round(float(np.mean(r2)), 3)


def report(a, ckpts) -> dict:
    msgs = json.load(open(os.path.join(a.out_dir, "messages.json")))
    refs, sig_cache = {}, {}
    for gi, (_, f) in ckpts.items():
        z = np.load(f, allow_pickle=True)
        refs[gi] = np.nan_to_num(np.asarray(z["M_i"], float), nan=0.5) > 0.5
        sig_cache[gi] = np.asarray(z["sigs"], float)
    out = {}
    for gpath in sorted(glob.glob(os.path.join(a.out_dir, "grid_*.npz"))):
        z = np.load(gpath, allow_pickle=True)
        scores = np.asarray(z["scores"], float)
        meta = [json.loads(s) for s in z["meta"]]
        tag = os.path.basename(gpath)[5:-4]
        per = {}
        for gi in {x["gi"] for x in meta}:
            ref = refs.get(gi)
            if ref is None:
                continue
            ex = msgs[str(gi)]["exemplar_idx"]
            mask = np.ones(len(ref), bool)
            mask[ex["pos"] + ex["neg"]] = False                  # exemplars never evaluated
            per_rung, preds = {}, {}
            for rung in RUNG_ORDER:
                idx = [i for i, x in enumerate(meta) if x["gi"] == gi and x["rung"] == rung]
                if not idx:
                    continue
                m_bar = np.nanmean(scores[idx], axis=0)          # orbit-average over forms
                pred = np.nan_to_num(m_bar, nan=0.5) > 0.5
                preds[rung] = pred
                pos, neg = ref & mask, (~ref) & mask
                acc_p = float((pred & pos).sum() / max(pos.sum(), 1))
                acc_n = float((~pred & neg).sum() / max(neg.sum(), 1))
                per_rung[rung] = {"bal_acc": round((acc_p + acc_n) / 2, 4),
                                  "agree": round(float((pred == ref)[mask].mean()), 4),
                                  "n_forms": len(idx),
                                  "span_r2": _span_r2(sig_cache[gi], m_bar, mask)}
            d_lift = per_rung.get("dossier", {}).get("bal_acc", float("nan")) - 0.5
            for rung, v in per_rung.items():
                v["rel_to_dossier"] = (round((v["bal_acc"] - 0.5) / d_lift, 3)
                                       if d_lift > 0.02 else None)   # dossier at chance: rel undefined
            # within-reader curve R_E(w) vs R_E(rich): agreement with the SAME reader's dossier
            # judgment -- no cross-executor conflation (bal_acc vs the external ref carries the
            # executor-indexed target; self_agree carries the pure decompression shape)
            if "dossier" in preds:
                for rung, v in per_rung.items():
                    v["self_agree"] = round(float((preds[rung] == preds["dossier"])[mask].mean()), 4)
            # executor-consistent (anchor-free) PRIMARY readout, 2026-07-03: the reader's OWN
            # full_rubric orbit judgment is the target; bits via the census's exact i_binary.
            # The bal_acc-vs-external-reference fields above remain as the SECONDARY, separately
            # named quantity (cross-executor transmission) — never report them as "decompression".
            if "full_rubric" in preds:
                tgt = preds["full_rubric"][mask].astype(int)
                h_self = float(vinfo._h_bits(float(tgt.mean()))) if len(np.unique(tgt)) > 1 else 0.0
                for rung, v in per_rung.items():
                    v["H_self"] = round(h_self, 4)
                    v["self_bits"] = (round(float(i_binary(tgt, preds[rung][mask].astype(int))), 4)
                                      if rung != "full_rubric" and h_self >= 0.15 else None)
            per[str(gi)] = per_rung
        out[tag] = per
    path = os.path.join(a.out_dir, "report.json")
    json.dump(out, open(path, "w"), indent=1)
    for tag, per in out.items():
        print(f"\nreader {tag} (balanced acc; rel = vs dossier):")
        print(f"{'gi':>4} " + " ".join(f"{r[:9]:>10}" for r in RUNG_ORDER))
        for gi, pr in sorted(per.items(), key=lambda kv: int(kv[0])):
            def _cell(r):
                if r not in pr:
                    return " " * 10
                rel = pr[r]["rel_to_dossier"]
                return f"{pr[r]['bal_acc']:>6.3f}/{rel:>3.1f}" if rel is not None else \
                       f"{pr[r]['bal_acc']:>6.3f}/ --"
            print(f"{gi:>4} " + " ".join(_cell(r) for r in RUNG_ORDER))
    print(f"\nwrote {path}")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--task", default="creative-writing")
    p.add_argument("--ref-dir", required=True,
                   help="ckpt dir supplying metric names + reference M_i (use the ORBIT-corrected "
                        "dir once available; single-form src is the provisional reference)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--r2-bucket", default="general")
    p.add_argument("--gi-list", default="", help="comma gi filter; empty = all in ref-dir")
    p.add_argument("--phase", default="all", choices=["messages", "score", "report", "all"])
    p.add_argument("--writer-model", default="meta-llama/Llama-3.1-8B-Instruct",
                   help="unpacks concepts ONCE; use local 70B for the real run (not GLM -- quota)")
    p.add_argument("--readers", default="meta-llama/Llama-3.1-8B-Instruct",
                   help="comma list, scored sequentially (one engine resident at a time)")
    p.add_argument("--forms", type=int, default=3, help="orbit size for VERBAL rungs (form control)")
    p.add_argument("--k-exemplars", type=int, default=2)
    p.add_argument("--max-ex-chars", type=int, default=400)
    p.add_argument("--n-probes", type=int, default=300)
    p.add_argument("--gepa-reserve", type=int, default=60)
    p.add_argument("--fake", action="store_true", help="FakeVLLM wiring smoke, no GPU")
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    cfg = cfgmod.apply_task_preset(cfgmod.ImplementerConfig(), a.task)
    gi_list = [int(x) for x in a.gi_list.split(",") if x.strip()] if a.gi_list else None
    ckpts = _ckpts(a.ref_dir, gi_list)
    if not ckpts:
        raise SystemExit(f"no matching checkpoints in {a.ref_dir}")
    probe_texts, _ = _load_texts(a.task, a.gepa_reserve + a.n_probes, cfg)
    probe_texts = probe_texts[a.gepa_reserve: a.gepa_reserve + a.n_probes]
    print(f"grid over {len(ckpts)} metrics, {len(probe_texts)} probes, rungs={RUNG_ORDER}")

    if a.phase in ("messages", "all"):
        msgs = build_messages(a, cfg, ckpts, probe_texts)
    else:
        msgs = json.load(open(os.path.join(a.out_dir, "messages.json")))
    if a.phase in ("score", "all"):
        for reader in [r.strip() for r in a.readers.split(",") if r.strip()]:
            score_reader(a, cfg, reader, msgs, probe_texts)
    if a.phase in ("report", "all"):
        report(a, ckpts)


if __name__ == "__main__":
    main()
