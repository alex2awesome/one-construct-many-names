"""Replicated strong-LLM re-audit of suspect R1 evaluation truth.

Existing eval rows stay frozen.  Two blind judges score every row; disagreements are routed to a
third blind judge.  Final truth is the ordinal median (equivalently majority for score-2 SAME), and
is written to an isolated candidate directory until explicitly promoted.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

from .build_level import OUT, _binary_kappa, _file_sha256


ROOT = Path(OUT) / "r1_truth_reaudit"
VERSION = "r1-truth-reaudit-v1"


def _load(path: str | Path, expected: set[str] | None = None) -> dict[str, int]:
    votes = {}
    malformed = 0
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            continue
        pid, score = row.get("pair_id"), row.get("score")
        if (set(row) != {"pair_id", "score"} or not isinstance(pid, str) or pid in votes
                or type(score) is not int or score not in (0, 1, 2)
                or (expected is not None and pid not in expected)):
            malformed += 1
            continue
        votes[pid] = score
    missing = (expected - set(votes)) if expected is not None else set()
    if malformed or missing:
        raise ValueError(f"invalid reaudit votes: malformed={malformed} missing={len(missing)}")
    return votes


def prepare(task: str, per_agent: int = 150) -> dict:
    level = "R1"
    eval_path = (Path(OUT) / f"level_eval_{task}_{level}.jsonl").resolve()
    protocol = (Path(OUT) / f"ARBITER_PROTOCOL_{level}.txt").resolve()
    rows = [json.loads(x) for x in eval_path.read_text().splitlines() if x.strip()]
    if len({row.get("pair_id") for row in rows}) != len(rows):
        raise ValueError(f"[{task}] duplicate eval pair IDs")
    payload_dir = ROOT / "payloads"
    payload_dir.mkdir(parents=True, exist_ok=True)
    for old in payload_dir.glob(f"{task}_R1_*.jsonl"):
        old.unlink()
    paths = []
    for start in range(0, len(rows), per_agent):
        path = payload_dir / f"{task}_R1_{start // per_agent:03d}.jsonl"
        path.write_text("".join(json.dumps({
            "pair_id": row["pair_id"], "concept_a": row["canonical_a"],
            "concept_b": row["canonical_b"]}, ensure_ascii=False) + "\n"
            for row in rows[start:start + per_agent]))
        paths.append(str(path.resolve()))
    old_votes = sorted(glob.glob(str(Path(OUT) / "level_votes" / f"arb_{task}_R1_[0-9]*.jsonl")))
    manifest = {
        "version": VERSION, "task": task, "level": level, "n_pairs": len(rows),
        "eval_path": str(eval_path), "eval_sha256": _file_sha256(str(eval_path)),
        "protocol_path": str(protocol), "protocol_sha256": _file_sha256(str(protocol)),
        "payload_paths": paths,
        "prior_vote_fingerprints": [{"path": str(Path(p).resolve()),
                                     "sha256": _file_sha256(p)} for p in old_votes],
        "decision": "two complete blind LLM passes; third LLM adjudicates every disagreement",
        "final_same_rule": "at least two of three strict integer scores equal 2",
    }
    ROOT.mkdir(exist_ok=True)
    (ROOT / f"{task}_R1_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def stage_disagreements(task: str, votes_a_path: str, votes_b_path: str,
                        per_agent: int = 150) -> dict:
    manifest = json.loads((ROOT / f"{task}_R1_manifest.json").read_text())
    if (_file_sha256(manifest["eval_path"]) != manifest["eval_sha256"]
            or _file_sha256(manifest["protocol_path"]) != manifest["protocol_sha256"]):
        raise ValueError(f"[{task}] frozen reaudit input changed")
    rows = {row["pair_id"]: row for row in
            (json.loads(x) for x in Path(manifest["eval_path"]).read_text().splitlines()) if row}
    expected = set(rows)
    a, b = _load(votes_a_path, expected), _load(votes_b_path, expected)
    disagreements = [pid for pid in rows if a[pid] != b[pid]]
    payload_dir = ROOT / "adjudicate_payloads"
    payload_dir.mkdir(exist_ok=True)
    for old in payload_dir.glob(f"{task}_R1_*.jsonl"):
        old.unlink()
    paths = []
    staged = [{"pair_id": pid, "concept_a": rows[pid]["canonical_a"],
               "concept_b": rows[pid]["canonical_b"]} for pid in disagreements]
    for start in range(0, len(staged), per_agent):
        path = payload_dir / f"{task}_R1_{start // per_agent:03d}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                for row in staged[start:start + per_agent]))
        paths.append(str(path.resolve()))
    aa, bb = [a[p] == 2 for p in rows], [b[p] == 2 for p in rows]
    kappa = _binary_kappa(aa, bb)
    result = {
        "task": task, "n": len(rows), "n_disagreements": len(disagreements),
        "exact_3way_agreement": round(sum(a[p] == b[p] for p in rows) / len(rows), 3),
        "binary_same_agreement": round(sum(x == y for x, y in zip(aa, bb)) / len(rows), 3),
        "binary_same_kappa": round(kappa, 3) if kappa is not None else None,
        "judge_a_same_rate": round(sum(aa) / len(aa), 3),
        "judge_b_same_rate": round(sum(bb) / len(bb), 3),
        "adjudicate_payload_paths": paths,
        "votes_a_path": str(Path(votes_a_path).resolve()),
        "votes_a_sha256": _file_sha256(votes_a_path),
        "votes_b_path": str(Path(votes_b_path).resolve()),
        "votes_b_sha256": _file_sha256(votes_b_path),
    }
    (ROOT / f"{task}_R1_agreement.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def finalize(task: str, adjudication_path: str) -> dict:
    manifest = json.loads((ROOT / f"{task}_R1_manifest.json").read_text())
    agreement = json.loads((ROOT / f"{task}_R1_agreement.json").read_text())
    for label in ("a", "b"):
        if _file_sha256(agreement[f"votes_{label}_path"]) != agreement[f"votes_{label}_sha256"]:
            raise ValueError(f"[{task}] frozen judge {label} votes changed")
    eval_rows = [json.loads(x) for x in Path(manifest["eval_path"]).read_text().splitlines() if x]
    expected = {row["pair_id"] for row in eval_rows}
    a = _load(agreement["votes_a_path"], expected)
    b = _load(agreement["votes_b_path"], expected)
    disputed = {pid for pid in expected if a[pid] != b[pid]}
    c = _load(adjudication_path, disputed)
    final = {}
    for pid in expected:
        if pid in disputed:
            values = sorted((a[pid], b[pid], c[pid]))
            final[pid] = values[1]
        else:
            final[pid] = a[pid]
    out_dir = ROOT / "final_votes"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"arb_{task}_R1.jsonl"
    out.write_text("".join(json.dumps({"pair_id": row["pair_id"],
                                        "score": final[row["pair_id"]]}) + "\n"
                           for row in eval_rows))
    result = {"task": task, "n": len(final), "n_adjudicated": len(disputed),
              "score_counts": {str(score): sum(x == score for x in final.values())
                               for score in (0, 1, 2)},
              "same_rate": round(sum(x == 2 for x in final.values()) / len(final), 3),
              "final_votes_path": str(out.resolve()), "final_votes_sha256": _file_sha256(str(out)),
              "status": "isolated candidate truth; not promoted to level_votes"}
    (ROOT / f"{task}_R1_final_report.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
