"""Address-segment grid: how much tacit construct knowledge must be made explicit at each scale?

Addendum to the tacit-scaling line (2026-07-11, user request): the rung grid measures articulation
TYPE (name/definition/...); this grid measures the quantity of construct knowledge externalized in
deterministic text
segments. Per metric the
verbal dossier content (full_rubric + definition + explanation from the existing grid's
messages.json -- byte-identical to what every prior reader saw) is decomposed into leaf units with
the CUF address lattice (unit_certificate._segment_sentences + _CLAUSE_SPLIT, min_words=3), deduped
by normalized text, ordered rubric -> definition -> explanation (Face-1 anchor first). Rungs:

    u0        the metric name alone (byte-identical to the rung grid's `name`)
    uk        name + first k segments, k = 1..n_segments
    fk        name + length-matched inert filler (CUF FILLER_BANK) matching uk's added words --
              mechanical length control: separates articulated construct knowledge from prompt
              length; articulation and filler receive the same deterministic prompt-form orbit

Readout: same instrument as grid_auc_report.py -- Mann-Whitney AUC of the orbit-averaged score
vs the frozen 8B-executor reference M_i (threshold-free; exemplar probes masked). Downstream
analysis: k*(reader, tau) = min k with best-so-far AUC >= tau; the unit deficit
Delta-k(small, big) = k*(small) - k*(big) answers "how many MORE address segments does the smaller
model need". These segments are NOT certified CUF Omega units: no per-segment necessity,
sufficiency, or stability certificate has been run. 8B reader == executor (self-recovery) -- flag
as SELF, as everywhere in the line.

Phases (resumable): build -> unit_messages.json; score -> unitgrid_<reader>.npz;
report -> unit_auc_report.json.

Smoke (no GPU): python -m methods.codability.unit_count_grid --fake --phase all \
    --task humor --grid-dir <dir with messages.json> --ref-dir <ckpts> --out-dir <scratch>
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import itertools
import json
import os
import re

import numpy as np

from methods.metric_implementer import config as cfgmod
from methods.metric_implementer.vllm_backend import make_judge_backend
from methods.metric_implementer.experiments.unit_certificate import (_segment_sentences,
                                                                     _CLAUSE_SPLIT, FILLER_BANK)
from methods.codability.grid_auc_report import auc_mw, spearman, _ckpts

UNIT_SOURCES = ["full_rubric", "definition", "explanation"]     # exemplars excluded: verbal units only


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", str(s).lower()).strip()


def leaf_units(text: str, min_words: int = 3):
    """Uncertified leaf address segments (legacy function name retained for artifact compatibility).

    These share CUF's segmentation regexes but have not passed the tests required to call them
    certified Omega units.
    """
    out = []
    for seg in _segment_sentences(str(text)):
        if len(seg.split()) < min_words:
            continue
        parts = [p.strip(" ,") for p in _CLAUSE_SPLIT.split(seg) if p.strip(" ,")]
        parts = [p for p in parts if len(p.split()) >= min_words]
        out.extend(parts if len(parts) >= 2 else [seg])
    return out


def _filler_words(n: int) -> str:
    """Inert discourse glue of ~n words from the graded CUF filler bank (cycled, deterministic)."""
    words = []
    for f in itertools.cycle(FILLER_BANK[::-1]):        # longest first: fewer seams
        words.extend(f.split())
        if len(words) >= n:
            break
    return " ".join(words[:n]) + "."


def _join(units) -> str:
    return " ".join(u if u.endswith((";", ".", "!", "?")) else u + ";" for u in units)


def build(a) -> dict:
    msgs = json.load(open(os.path.join(a.grid_dir, "messages.json")))
    out, truncated = {}, []
    for gi, m in msgs.items():
        seen, units = set(), []
        for src in UNIT_SOURCES:
            for u in leaf_units(m["rungs"].get(src, "")):
                k = _norm(u)
                if k and k not in seen:
                    seen.add(k)
                    units.append({"src": src, "text": u})
        if len(units) > a.max_units:
            truncated.append((gi, len(units)))
            units = units[: a.max_units]
        name = m["rungs"]["name"]
        rungs = {"u0": name}
        for k in range(1, len(units) + 1):
            body = _join([u["text"] for u in units[:k]])
            rungs[f"u{k}"] = f"{name}. {body}"
            rungs[f"f{k}"] = f"{name}. {_filler_words(len(body.split()))}"
        out[gi] = {"name": name, "level": m.get("level"),
                   "segment_method": "cuf_address_lattice_leaf_uncertified",
                   "n_segments": len(units), "segments": units,
                   # Legacy aliases keep old downstream readers working; do not interpret as
                   # certified Omega units.
                   "n_units": len(units), "units": units,
                   "rungs": rungs, "exemplar_idx": m["exemplar_idx"],
                   "word_len": {r: len(t.split()) for r, t in rungs.items()}}
    if truncated:
        print(f"NOTE: {len(truncated)} metrics truncated at --max-units={a.max_units}: {truncated}")
    path = os.path.join(a.out_dir, "unit_messages.json")
    json.dump(out, open(path, "w"), indent=1)
    n_r = sum(len(v["rungs"]) for v in out.values())
    print(f"unit messages: {len(out)} metrics, {n_r} rungs -> {path}")
    return out

def score_reader(a, cfg, reader: str, msgs: dict, probe_texts) -> str:
    from methods.metric_implementer.experiments import alpha_probe as ap

    ecfg = cfgmod.apply_task_preset(cfgmod.ImplementerConfig(), a.task)
    if a.fake:
        ecfg.vllm_fake = True
    executor = make_judge_backend(reader, ecfg, temperature=None)
    rows, meta = [], []
    for gi, m in msgs.items():
        for rung, base in m["rungs"].items():
            # Every content and filler arm receives the same form operators. Otherwise a uk-fk
            # contrast mixes semantic content with orbit averaging / instruction boilerplate.
            forms = [("canonical", base)]
            if a.forms > 1:
                forms += list(ap._reformulations(base))[: a.forms - 1]
            for kind, txt in forms:
                rows.append(ap.signature(executor, txt, probe_texts, cfg.max_text_chars))
                meta.append({"gi": int(gi), "rung": rung, "form": kind})
        print(f"  scored gi={gi} ({m['name'][:40]})", flush=True)
    tag = re.sub(r"[^A-Za-z0-9.-]+", "_", os.path.basename(reader.rstrip("/")))
    path = os.path.join(a.out_dir, f"unitgrid_{tag}.npz")
    probe_sha256 = np.asarray([hashlib.sha256(str(t).encode()).hexdigest()
                               for t in probe_texts])
    probe_set_sha256 = hashlib.sha256("\n".join(probe_sha256).encode()).hexdigest()
    np.savez(path, scores=np.vstack(rows),
             meta=np.array([json.dumps(x) for x in meta], dtype=object),
             reader=reader, ref_dir=a.ref_dir,
             protocol_schema="address_segment_grid/v2_form_matched",
             segment_method="cuf_address_lattice_leaf_uncertified",
             probe_sha256=probe_sha256, probe_set_sha256=probe_set_sha256)
    print(f"reader {tag}: {len(rows)} rows -> {path}")
    return path


def report(a) -> dict:
    msgs = json.load(open(os.path.join(a.out_dir, "unit_messages.json")))
    refs_bin, refs_cont = {}, {}
    for gi, f in _ckpts(a.ref_dir).items():
        z = np.load(f, allow_pickle=True)
        m_i = np.nan_to_num(np.asarray(z["M_i"], float), nan=0.5)
        refs_bin[gi], refs_cont[gi] = m_i > 0.5, m_i
    out = {"_protocol": {
        "schema": "address_segment_auc_report/v2",
        "quantity": "uncertified address segments (not certified CUF Omega units)",
        "treatment": "explicit articulation of the target construct's tacit knowledge",
        "required_form_control": "identical reformulation operators for uk and fk",
    }, "_reader_protocols": {}}
    for gpath in sorted(glob.glob(os.path.join(a.out_dir, "unitgrid_*.npz"))):
        z = np.load(gpath, allow_pickle=True)
        scores = np.asarray(z["scores"], float)
        meta = [json.loads(s) for s in z["meta"]]
        tag = os.path.basename(gpath)[9:-4]
        source_protocol = (str(z["protocol_schema"]) if "protocol_schema" in z.files else
                           "legacy_v1_no_embedded_protocol")
        out["_reader_protocols"][tag] = {
            "source_protocol": source_protocol,
            "form_matched_claim": source_protocol == "address_segment_grid/v2_form_matched",
        }
        per = {}
        for gi in sorted({x["gi"] for x in meta}):
            if gi not in refs_bin or str(gi) not in msgs:
                continue
            ref, cont = refs_bin[gi], refs_cont[gi]
            ex = msgs[str(gi)]["exemplar_idx"]
            mask = np.ones(len(ref), bool)
            mask[ex["pos"] + ex["neg"]] = False
            per_rung = {}
            for rung in sorted({x["rung"] for x in meta if x["gi"] == gi}):
                idx = [i for i, x in enumerate(meta) if x["gi"] == gi and x["rung"] == rung]
                m_bar = np.nan_to_num(np.nanmean(scores[idx], axis=0), nan=0.5)
                per_rung[rung] = {
                    "auc": (lambda v: round(v, 4) if v is not None else None)(
                        auc_mw(m_bar[mask], ref[mask])),
                    "spearman": (lambda v: round(v, 4) if v is not None else None)(
                        spearman(m_bar[mask], cont[mask])),
                    "n_forms": len(idx),
                    "words": msgs[str(gi)]["word_len"].get(rung),
                }
            segments = msgs[str(gi)].get("segments", msgs[str(gi)]["units"])
            per[str(gi)] = {"n_segments": msgs[str(gi)].get("n_segments",
                                                                   msgs[str(gi)]["n_units"]),
                            "segment_method": msgs[str(gi)].get(
                                "segment_method", "legacy_uncertified_address_segments"),
                            # Legacy aliases.
                            "n_units": msgs[str(gi)]["n_units"],
                            "segment_srcs": [u["src"] for u in segments],
                            "unit_srcs": [u["src"] for u in segments],
                            "rungs": per_rung}
        out[tag] = per
        aucs = [v["auc"] for p in per.values() for r, v in p["rungs"].items()
                if v["auc"] is not None and r.startswith("u")]
        print(f"{tag}: {len(per)} metrics, mean unit-rung AUC "
              f"{np.mean(aucs):.3f}" if aucs else f"{tag}: no scorable rungs")
    path = os.path.join(a.out_dir, "unit_auc_report.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"-> {path}")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--task", required=True)
    p.add_argument("--grid-dir", required=True, help="existing rung-grid dir (messages.json source)")
    p.add_argument("--ref-dir", required=True, help="ckpt dir with *_sigs.npz (M_i reference)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--phase", default="all", choices=["build", "score", "report", "all"])
    p.add_argument("--readers", default="meta-llama/Llama-3.2-1B-Instruct",
                   help="exactly one reader for score/all; launch separate processes per reader")
    p.add_argument("--forms", type=int, default=3,
                   help="shared orbit size for both content and filler rungs")
    p.add_argument("--max-units", type=int, default=14)
    p.add_argument("--n-probes", type=int, default=300)
    p.add_argument("--gepa-reserve", type=int, default=60)
    p.add_argument("--fake", action="store_true")
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    cfg = cfgmod.apply_task_preset(cfgmod.ImplementerConfig(), a.task)
    if a.phase in ("build", "all"):
        msgs = build(a)
    else:
        msgs = json.load(open(os.path.join(a.out_dir, "unit_messages.json")))
    if a.phase in ("score", "all"):
        from methods.metric_implementer.experiments.run_real_test import _load_texts

        readers = [r.strip() for r in a.readers.split(",") if r.strip()]
        if len(readers) != 1:
            p.error("score/all requires exactly one --readers value; launch one process per reader")
        probe_texts, _ = _load_texts(a.task, a.gepa_reserve + a.n_probes, cfg)
        probe_texts = probe_texts[a.gepa_reserve: a.gepa_reserve + a.n_probes]
        print(f"unit grid over {len(msgs)} metrics, {len(probe_texts)} probes")
        for reader in readers:
            tag = re.sub(r"[^A-Za-z0-9.-]+", "_", os.path.basename(reader.rstrip("/")))
            if os.path.exists(os.path.join(a.out_dir, f"unitgrid_{tag}.npz")):
                print(f"reader {tag}: npz exists — skip")
                continue
            # NOTE: one engine per PROCESS — a second in-process engine inits against the
            # first one's unreleased memory (vLLM teardown lag); drive one reader per call
            score_reader(a, cfg, reader, msgs, probe_texts)
    if a.phase in ("report", "all"):
        report(a)


if __name__ == "__main__":
    main()
