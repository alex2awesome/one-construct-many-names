"""Blind cross-family validation for R-level arbiter labels.

This module deliberately separates payload construction from scoring: ``emit`` writes node-pair
texts without the reference-family label, and ``report`` joins an independently produced vote file
back to the frozen arbiter votes.  The default sample is balanced across arbiter scores 0/1/2 so the
three kinds of disagreement are visible; consequently, raw agreement from this diagnostic is not a
population estimate for the original eval-set mixture.

Vote rows are strict JSONL ``{"pair_id": str, "score": 0|1|2}``.  Malformed, duplicate, missing,
and unexpected votes are fatal by default; ``--allow-incomplete`` is diagnostic-only.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
from collections import Counter
from typing import Dict, Iterable, List

from .build_level import OUT


def _strict_votes(paths: Iterable[str]) -> tuple[Dict[str, int], dict]:
    rows_by_pid: Dict[str, List[int]] = {}
    duplicates: Counter[str] = Counter()
    malformed = 0
    for path in sorted(paths):
        with open(path) as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    malformed += 1
                    continue
                if not isinstance(row, dict) or set(row) != {"pair_id", "score"}:
                    malformed += 1
                    continue
                pid, score = row["pair_id"], row["score"]
                if not isinstance(pid, str) or type(score) is not int or score not in (0, 1, 2):
                    malformed += 1
                    continue
                duplicates[pid] += 1
                rows_by_pid.setdefault(pid, []).append(score)
    duplicate_ids = sorted(p for p, n in duplicates.items() if n > 1)
    conflicts = sorted(p for p, scores in rows_by_pid.items() if len(set(scores)) > 1)
    # Never choose a last/first winner for a duplicate: ambiguous provenance is not a valid vote.
    votes = {p: scores[0] for p, scores in rows_by_pid.items() if len(scores) == 1}
    return votes, {"malformed": malformed,
                   "duplicate_rows": sum(n - 1 for n in duplicates.values()),
                   "duplicate_pair_ids": duplicate_ids, "conflicting_pair_ids": conflicts}


def _rank(task: str, level: str, score: int, pair_id: str, seed: int) -> str:
    key = f"{seed}|{task}|{level}|{score}|{pair_id}"
    return hashlib.sha256(key.encode()).hexdigest()


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _manifest_path(payload_path: str) -> str:
    return payload_path.replace(".jsonl", "_manifest_private.json")


def _load_manifest(task: str, level: str, manifest_path: str | None) -> tuple[dict, str]:
    payload = os.path.join(OUT, "codex_val", f"{task}_{level}_blind_reaudit.jsonl")
    path = manifest_path or _manifest_path(payload)
    manifest = json.load(open(path))
    if manifest.get("task") != task or manifest.get("level") != level:
        raise ValueError(f"manifest {path} is for {manifest.get('task')}/{manifest.get('level')}, "
                         f"not {task}/{level}")
    if manifest.get("payload_sha256") != _sha256(manifest["payload_path"]):
        raise ValueError(f"blind payload hash no longer matches frozen manifest: {path}")
    return manifest, path


def _semantic_text_ok(text: str) -> bool:
    s = str(text or "").strip().strip(".")
    return bool(s and not s.isdigit()
                and not re.fullmatch(r"[\w-]*_R[0-9]+_(?:g|solo_)?[\w-]+", s))


def emit(task: str, level: str = "R2", *, n_per_score: int = 40, seed: int = 0,
         out_path: str | None = None) -> dict:
    """Emit a deterministic score-balanced blind payload without reference labels."""
    eval_path = os.path.join(OUT, f"level_eval_{task}_{level}.jsonl")
    eval_rows = {r["pair_id"]: r for line in open(eval_path) if line.strip()
                 for r in (json.loads(line),)}
    reference, diag = _strict_votes(
        glob.glob(os.path.join(OUT, "level_votes", f"arb_{task}_{level}_[0-9]*.jsonl")))
    if diag["malformed"] or diag["duplicate_rows"]:
        raise ValueError(f"[{task}/{level}] corrupt reference votes: malformed={diag['malformed']}, "
                         f"duplicates={diag['duplicate_rows']}")
    selected: List[dict] = []
    available = Counter()
    for score in (0, 1, 2):
        pids = [p for p, s in reference.items() if s == score and p in eval_rows]
        available[score] = len(pids)
        pids.sort(key=lambda p: _rank(task, level, score, p, seed))
        if len(pids) < n_per_score:
            raise ValueError(f"[{task}/{level}] only {len(pids)} score-{score} rows; "
                             f"cannot draw requested {n_per_score}")
        for pid in pids[:n_per_score]:
            row = eval_rows[pid]
            if not (_semantic_text_ok(row.get("canonical_a"))
                    and _semantic_text_ok(row.get("canonical_b"))):
                raise ValueError(f"[{task}/{level}] pair {pid} has an opaque/bare-ID concept; "
                                 f"name/backfill it before LLM validation")
            selected.append({"pair_id": pid, "task": task, "level": level,
                             "relation": "same theme" if level == "R2" else level,
                             "node_a": row["node_a"], "node_b": row["node_b"],
                             "canonical_a": row["canonical_a"],
                             "canonical_b": row["canonical_b"]})
    selected.sort(key=lambda r: _rank(task, level, -1, r["pair_id"], seed))
    out_path = out_path or os.path.join(OUT, "codex_val", f"{task}_{level}_blind_reaudit.jsonl")
    if os.path.dirname(out_path):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        for row in selected:
            fh.write(json.dumps(row) + "\n")
    manifest_path = _manifest_path(out_path)
    manifest = {"task": task, "level": level, "seed": seed, "n_per_score": n_per_score,
                "pair_ids_ordered": [r["pair_id"] for r in selected],
                "reference_scores": {r["pair_id"]: reference[r["pair_id"]] for r in selected},
                "payload_path": os.path.abspath(out_path), "payload_sha256": _sha256(out_path),
                "eval_path": os.path.abspath(eval_path), "eval_sha256": _sha256(eval_path),
                "reference_files": [{"path": os.path.abspath(p), "sha256": _sha256(p)} for p in
                                    sorted(glob.glob(os.path.join(
                                        OUT, "level_votes", f"arb_{task}_{level}_[0-9]*.jsonl")))]}
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=1)
    return {"task": task, "level": level, "path": out_path, "manifest_private": manifest_path,
            "n": len(selected), "n_per_score": n_per_score, "available": dict(available),
            "reference_load": diag}


def _cohen_kappa(a: List[int], b: List[int], labels: Iterable[int]) -> float | None:
    if not a:
        return None
    labs = list(labels)
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[x] / n) * (cb[x] / n) for x in labs)
    return (po - pe) / (1 - pe) if pe < 1 else None


def report(task: str, vote_path: str, level: str = "R2", *, n_per_score: int = 40,
           seed: int = 0, require_complete: bool = True,
           manifest_path: str | None = None) -> dict:
    """Score independent votes against the exact deterministic blind sample emitted above."""
    manifest, manifest_path = _load_manifest(task, level, manifest_path)
    if manifest["n_per_score"] != n_per_score or manifest["seed"] != seed:
        raise ValueError("requested sampling parameters do not match the frozen manifest")
    expected_order = manifest["pair_ids_ordered"]
    expected = set(expected_order)
    reference = {p: int(s) for p, s in manifest["reference_scores"].items()}
    ref_diag = {"source": "frozen-private-manifest", "manifest": manifest_path}
    candidate, cand_diag = _strict_votes([vote_path])
    missing = sorted(expected - set(candidate))
    unexpected = sorted(set(candidate) - expected)
    bad_candidate = (cand_diag["malformed"] or cand_diag["duplicate_rows"]
                     or cand_diag["conflicting_pair_ids"])
    if require_complete and (len(expected) != 3 * n_per_score or missing or unexpected or bad_candidate):
        raise ValueError(f"[{task}/{level}] incomplete cross-family votes: missing={len(missing)}, "
                         f"unexpected={len(unexpected)}, malformed={cand_diag['malformed']}, "
                         f"duplicates={cand_diag['duplicate_rows']}")
    pids = [p for p in expected_order if p in candidate]
    a, b = [reference[p] for p in pids], [candidate[p] for p in pids]
    confusion = {str(i): {str(j): 0 for j in (0, 1, 2)} for i in (0, 1, 2)}
    for x, y in zip(a, b):
        confusion[str(x)][str(y)] += 1
    same_a, same_b = [int(x == 2) for x in a], [int(x == 2) for x in b]
    arb_pos = sum(same_a)
    cand_pos = sum(same_b)
    both_pos = sum(x and y for x, y in zip(same_a, same_b))
    n = len(pids)
    return {
        "task": task, "level": level, "n_expected": len(expected), "n_scored": n,
        "missing": missing, "unexpected": unexpected,
        "reference_load": ref_diag, "candidate_load": cand_diag,
        "confusion_reference_rows_candidate_cols": confusion,
        "exact_3way_agreement": (sum(x == y for x, y in zip(a, b)) / n) if n else None,
        "same_binary_agreement": (sum(x == y for x, y in zip(same_a, same_b)) / n) if n else None,
        "cohen_kappa_3way": _cohen_kappa(a, b, (0, 1, 2)),
        "cohen_kappa_same_binary": _cohen_kappa(same_a, same_b, (0, 1)),
        "candidate_confirms_reference_same": (both_pos / arb_pos) if arb_pos else None,
        "reference_confirms_candidate_same": (both_pos / cand_pos) if cand_pos else None,
        "candidate_same_rate_balanced_sample": (cand_pos / n) if n else None,
        "note": "Score-balanced diagnostic sample: aggregate agreement/kappa and candidate-SAME "
                "precision reflect this artificial mixture. P(candidate SAME | reference SAME) "
                "is the directly class-conditional confirmation quantity.",
    }


def emit_disagreements(task: str, vote_path: str, level: str = "R2", *,
                       n_per_score: int = 40, seed: int = 0,
                       out_path: str | None = None, manifest_path: str | None = None) -> dict:
    """Emit blind A/B disagreement rows for an independent LLM adjudicator.

    Reference-family and candidate-family identities are deterministically swapped per pair so the
    adjudicator cannot favor either source. The private ``orientation`` sidecar is needed only for
    later bookkeeping and must not be given to the adjudicator.
    """
    manifest, manifest_path = _load_manifest(task, level, manifest_path)
    eval_rows = {r["pair_id"]: r for line in open(manifest["payload_path"]) if line.strip()
                 for r in (json.loads(line),)}
    reference = {p: int(s) for p, s in manifest["reference_scores"].items()}
    candidate, diag = _strict_votes([vote_path])
    expected = manifest["pair_ids_ordered"]
    missing = sorted(set(expected) - set(candidate))
    unexpected = sorted(set(candidate) - set(expected))
    if (missing or unexpected or diag["malformed"] or diag["duplicate_rows"]):
        raise ValueError(f"[{task}/{level}] invalid candidate votes for disagreement emission: "
                         f"missing={len(missing)}, unexpected={len(unexpected)}, "
                         f"malformed={diag['malformed']}, duplicates={diag['duplicate_rows']}")
    rows, orientation = [], {}
    for pid in expected:
        if pid not in candidate or candidate[pid] == reference[pid]:
            continue
        swap = int(hashlib.sha256(f"ab|{seed}|{pid}".encode()).hexdigest(), 16) % 2 == 1
        a, b = ((candidate[pid], reference[pid]) if swap
                else (reference[pid], candidate[pid]))
        r = eval_rows[pid]
        rows.append({"pair_id": pid, "task": task, "level": level, "relation": "same theme",
                     "canonical_a": r["canonical_a"], "canonical_b": r["canonical_b"],
                     "score_A": a, "score_B": b})
        orientation[pid] = {"A": "candidate" if swap else "reference",
                            "B": "reference" if swap else "candidate"}
    rows.sort(key=lambda r: _rank(task, level, -2, r["pair_id"], seed))
    out_path = out_path or os.path.join(OUT, "codex_val", f"{task}_{level}_disagreements_blind.jsonl")
    if os.path.dirname(out_path):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    sidecar = out_path.replace(".jsonl", "_orientation_private.json")
    with open(sidecar, "w") as fh:
        json.dump(orientation, fh, indent=1)
    return {"task": task, "level": level, "n_disagreements": len(rows), "path": out_path,
            "orientation_private": sidecar, "candidate_load": diag}


def adjudication_report(task: str, adjudication_path: str, level: str = "R2",
                        *, n_total: int = 120) -> dict:
    """Bookkeeping over independent LLM adjudications of the blinded A/B disagreements."""
    base = os.path.join(OUT, "codex_val", f"{task}_{level}_disagreements_blind")
    disagreements = {r["pair_id"]: r for line in open(base + ".jsonl") if line.strip()
                     for r in (json.loads(line),)}
    orientation = json.load(open(base + "_orientation_private.json"))
    adjudicated = {}
    malformed = 0
    for line in open(adjudication_path):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1; continue
        if not isinstance(row, dict) or not isinstance(row.get("pair_id"), str):
            malformed += 1; continue
        adjudicated[row["pair_id"]] = row
    missing = sorted(set(disagreements) - set(adjudicated))
    unexpected = sorted(set(adjudicated) - set(disagreements))
    if malformed or missing or unexpected:
        raise ValueError(f"[{task}/{level}] invalid adjudications: malformed={malformed}, "
                         f"missing={len(missing)}, unexpected={len(unexpected)}")
    counts = Counter(reference=0, candidate=0, neither=0, unjudgeable=0)
    for pid, row in adjudicated.items():
        score = row.get("adjudicated_score")
        if row.get("status") == "unjudgeable" or score is None:
            counts["unjudgeable"] += 1
            continue
        if type(score) is not int or score not in (0, 1, 2):
            raise ValueError(f"bad adjudicated_score for {pid}: {score!r}")
        pair = disagreements[pid]
        matched = set()
        if score == pair["score_A"]:
            matched.add(orientation[pid]["A"])
        if score == pair["score_B"]:
            matched.add(orientation[pid]["B"])
        if not matched:
            counts["neither"] += 1
        else:
            counts.update(matched)
    n_disagree = len(disagreements)
    n_agree = n_total - n_disagree
    denom = n_total - counts["unjudgeable"]
    return {"task": task, "level": level, "n_total": n_total,
            "n_initial_agreements": n_agree, "n_disagreements": n_disagree,
            "disagreement_adjudication": dict(counts),
            "sonnet_match_to_adjudicated_or_agreed": ((n_agree + counts["reference"]) / denom
                                                       if denom else None),
            "codex_match_to_adjudicated_or_agreed": ((n_agree + counts["candidate"]) / denom
                                                      if denom else None),
            "note": "Agreement rows were not re-judged; the two models already assigned the same "
                    "score there. Match rates exclude adjudicator-marked unjudgeable rows."}


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    ep = sub.add_parser("emit")
    ep.add_argument("--task", required=True); ep.add_argument("--level", default="R2")
    ep.add_argument("--n-per-score", type=int, default=40); ep.add_argument("--seed", type=int, default=0)
    ep.add_argument("--out", default=None)
    rp = sub.add_parser("report")
    rp.add_argument("--task", required=True); rp.add_argument("--votes", required=True)
    rp.add_argument("--level", default="R2"); rp.add_argument("--n-per-score", type=int, default=40)
    rp.add_argument("--seed", type=int, default=0); rp.add_argument("--allow-incomplete", action="store_true")
    rp.add_argument("--out", default=None); rp.add_argument("--manifest", default=None)
    dp = sub.add_parser("disagreements")
    dp.add_argument("--task", required=True); dp.add_argument("--votes", required=True)
    dp.add_argument("--level", default="R2"); dp.add_argument("--n-per-score", type=int, default=40)
    dp.add_argument("--seed", type=int, default=0); dp.add_argument("--out", default=None)
    dp.add_argument("--manifest", default=None)
    ap = sub.add_parser("adjudication-report")
    ap.add_argument("--task", required=True); ap.add_argument("--adjudications", required=True)
    ap.add_argument("--level", default="R2"); ap.add_argument("--n-total", type=int, default=120)
    ap.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    if args.cmd == "emit":
        result = emit(args.task, args.level, n_per_score=args.n_per_score, seed=args.seed,
                      out_path=args.out)
    elif args.cmd == "report":
        result = report(args.task, args.votes, args.level, n_per_score=args.n_per_score,
                        seed=args.seed, require_complete=not args.allow_incomplete,
                        manifest_path=args.manifest)
    elif args.cmd == "disagreements":
        result = emit_disagreements(args.task, args.votes, args.level,
                                    n_per_score=args.n_per_score, seed=args.seed,
                                    out_path=args.out, manifest_path=args.manifest)
    else:
        result = adjudication_report(args.task, args.adjudications, args.level,
                                     n_total=args.n_total)
    if getattr(args, "out", None) and args.cmd in ("report", "adjudication-report"):
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
