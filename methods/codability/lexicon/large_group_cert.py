"""Independent LLM certification gate for oversized upper-hierarchy communities."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .build_level import OUT, _file_sha256, _load_partition, nodes_from_level

ROOT = Path(OUT) / "large_group_cert"


def _stem(task: str, level: str, tag: str = "") -> str:
    """Return the artifact stem, preserving the legacy name when ``tag`` is empty."""
    return f"{task}_{level}" + (f"_{tag}" if tag else "")


def prepare(task: str, level: str, partition_path: str, threshold: int = 30,
            required_judges: int = 1, tag: str = "") -> dict:
    if required_judges not in (1, 2):
        raise ValueError("required_judges must be 1 or 2")
    part = {str(k): str(v) for k, v in _load_partition(partition_path).items()}
    nodes, _ = nodes_from_level(task, level)
    by_id = {str(n["node_id"]): n for n in nodes}
    missing, extra = set(by_id) - set(part), set(part) - set(by_id)
    if missing or extra:
        raise ValueError(f"[{task}/{level}] certification partition coverage failure: "
                         f"missing={len(missing)} extra={len(extra)}")
    members: dict[str, list[str]] = defaultdict(list)
    for node, group in part.items(): members[group].append(node)
    protocol = Path(OUT) / f"ARBITER_PROTOCOL_{level}.txt"
    payload_dir = ROOT / "payloads"; payload_dir.mkdir(parents=True, exist_ok=True)
    stem = _stem(task, level, tag)
    # Match only this stem's numbered payloads.  In particular, preparing the legacy empty-tag
    # stem must not delete payloads belonging to tagged candidates that share its prefix.
    for old in payload_dir.glob(f"{stem}_[0-9][0-9][0-9].json"):
        old.unlink()
    paths = []
    for group, ids in sorted(members.items()):
        if len(ids) <= threshold: continue
        row = {"group_id": group, "level": level, "n_nodes": len(ids),
               "nodes": [{"node_id": n, "name": str(by_id[n].get("name") or n),
                          "gloss": str(by_id[n].get("gloss") or "")} for n in sorted(ids)]}
        path = payload_dir / f"{stem}_{len(paths):03d}.json"
        path.write_text(json.dumps(row, ensure_ascii=False, indent=1) + "\n"); paths.append(str(path))
    manifest = {"task": task, "level": level, **({"tag": tag} if tag else {}),
                "threshold": threshold,
                "required_judges": required_judges,
                "partition_path": partition_path, "partition_sha256": _file_sha256(partition_path),
                "protocol_path": str(protocol), "protocol_sha256": _file_sha256(str(protocol)),
                "n_groups": len(members), "n_oversized": len(paths), "payload_paths": paths,
                "payload_fingerprints": [
                    {"path": path, "sha256": _file_sha256(path)} for path in paths],
                "decision": ("independent LLM must certify full-group coherence or repartition"
                             if required_judges == 1 else
                             "Sonnet and GPT-5 independently certify or repartition; a third "
                             "frontier judge is required only when their decisions differ"),
                "output_schema": {"group_id": "string", "certified": "boolean",
                                  "shared_concept": ("nonempty string when certified=true; "
                                                     "null allowed when certified=false"),
                                  "rationale": "nonempty string",
                                  "groups": "required exact partition when certified=false"}}
    ROOT.mkdir(exist_ok=True)
    (ROOT / f"{stem}_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _load_decision(payload: dict, vote_path: Path) -> tuple[None | list[list[str]], bool]:
    """Return (partition-or-None-for-certified, valid) for one complete semantic vote."""
    if not vote_path.exists():
        return None, False
    try:
        vote = json.loads(vote_path.read_text())
    except Exception:
        return None, False
    gid = str(payload["group_id"])
    if str(vote.get("group_id")) != gid or type(vote.get("certified")) is not bool:
        return None, False
    if not str(vote.get("rationale") or "").strip():
        return None, False
    if vote["certified"]:
        if not str(vote.get("shared_concept") or "").strip():
            return None, False
        return None, True
    groups = vote.get("groups")
    expected = {str(x["node_id"]) for x in payload["nodes"]}
    flat = [str(x) for group in groups or [] if isinstance(group, list) for x in group]
    if (not isinstance(groups, list) or any(not isinstance(group, list) or not group
                                            for group in groups)
            or len(flat) != len(set(flat)) or set(flat) != expected):
        return None, False
    return [list(map(str, group)) for group in groups], True


def _common_refinement(expected: set[str], left: None | list[list[str]],
                       right: None | list[list[str]]) -> None | list[list[str]]:
    """Conservatively combine two LLM partitions without inventing a semantic merge.

    ``None`` denotes a full-group certificate.  The intersection of two equivalence relations is
    itself a partition: nodes remain together only when every judge who requested a split kept
    them together.  Thus a certificate can never erase the other judge's semantic distinction.
    """
    if left is None and right is None:
        return None
    if left is None:
        return right
    if right is None:
        return left
    left_id = {node: index for index, group in enumerate(left) for node in group}
    right_id = {node: index for index, group in enumerate(right) for node in group}
    if set(left_id) != expected or set(right_id) != expected:
        raise ValueError("cannot refine incomplete LLM partitions")
    buckets: dict[tuple[int, int], list[str]] = defaultdict(list)
    for node in sorted(expected):
        buckets[(left_id[node], right_id[node])].append(node)
    return list(buckets.values())


def _decision_key(decision: None | list[list[str]]) -> tuple:
    """Canonical equality key independent of group and member ordering."""
    if decision is None:
        return ("certified",)
    return ("split", tuple(sorted(tuple(sorted(group)) for group in decision)))


def _adjudicate_decisions(expected: set[str], first, second, third=None, *, third_supplied=False):
    """Use two judges directly when they agree; otherwise require a third frontier decision."""
    if _decision_key(first) == _decision_key(second):
        if third is not None:
            raise ValueError("third large-cluster decision supplied despite judge agreement")
        return first
    if not third_supplied:
        raise ValueError("third large-cluster decision required for judge disagreement")
    decisions = (first, second, third)
    if sum(decision is None for decision in decisions) >= 2:
        return None
    splits = [decision for decision in decisions if decision is not None]
    combined = splits[0]
    for decision in splits[1:]:
        combined = _common_refinement(expected, combined, decision)
    return combined


def _apply_decisions(part: dict[str, str],
                     decisions: dict[str, None | list[list[str]]]) -> dict[str, str]:
    out = dict(part)
    for gid, groups in decisions.items():
        if groups is None:
            continue
        for index, group in enumerate(groups):
            for node in group:
                out[node] = f"{gid}_c{index}"
    return out


def _partition_diagnostics(partition: dict[str, str], threshold: int) -> dict:
    sizes: dict[str, int] = defaultdict(int)
    for group in partition.values():
        sizes[group] += 1
    return {"n_groups": len(sizes),
            "remaining_over_threshold": sum(n > threshold for n in sizes.values()),
            "max_group_size": max(sizes.values(), default=0)}


def apply(task: str, level: str, tag: str = "") -> dict:
    stem = _stem(task, level, tag)
    manifest = json.loads((ROOT / f"{stem}_manifest.json").read_text())
    if (_file_sha256(manifest["partition_path"]) != manifest["partition_sha256"]
            or _file_sha256(manifest["protocol_path"]) != manifest["protocol_sha256"]):
        raise ValueError(f"[{task}/{level}] certification inputs changed")
    fingerprints = manifest.get("payload_fingerprints")
    if fingerprints is not None:
        if ([row["path"] for row in fingerprints] != manifest["payload_paths"]
                or any(_file_sha256(row["path"]) != row["sha256"] for row in fingerprints)):
            raise ValueError(f"[{task}/{level}] certification payloads changed")
    part = {str(k): str(v) for k, v in _load_partition(manifest["partition_path"]).items()}
    required_judges = int(manifest.get("required_judges", 1))
    decisions = {}; judge_a_decisions = {}; judge_b_decisions = {}; malformed = 0
    tiebreak_decisions = {}
    exact_certification_agreement = 0
    common_refinements = 0
    for p in manifest["payload_paths"]:
        payload = json.loads(Path(p).read_text()); gid = str(payload["group_id"])
        expected = {str(x["node_id"]) for x in payload["nodes"]}
        first, valid = _load_decision(payload, ROOT / "votes" / Path(p).name)
        if not valid:
            malformed += 1
            continue
        judge_a_decisions[gid] = first
        if required_judges == 1:
            decisions[gid] = first
            continue
        second, valid = _load_decision(payload, ROOT / "replicate_votes" / Path(p).name)
        if not valid:
            malformed += 1
            continue
        judge_b_decisions[gid] = second
        exact_agreement = _decision_key(first) == _decision_key(second)
        exact_certification_agreement += int(exact_agreement)
        third = None
        if not exact_agreement:
            third, valid = _load_decision(payload, ROOT / "tiebreak_votes" / Path(p).name)
            if not valid:
                malformed += 1
                continue
            tiebreak_decisions[gid] = third
        try:
            combined = _adjudicate_decisions(
                expected, first, second, third, third_supplied=not exact_agreement)
        except ValueError:
            malformed += 1
            continue
        common_refinements += int(combined is not None and sum(
            decision is not None for decision in (first, second, third)) >= 2)
        decisions[gid] = combined
    if len(decisions) != manifest["n_oversized"] or malformed:
        raise ValueError(f"[{task}/{level}] incomplete certificates "
                         f"{len(decisions)}/{manifest['n_oversized']} malformed={malformed}")
    out = _apply_decisions(part, decisions)
    out_path = Path(OUT) / f"partition_{stem}_certified.json"
    out_path.write_text(json.dumps(out) + "\n")
    diagnostics = _partition_diagnostics(out, manifest["threshold"])
    candidates = {"consensus": {"partition_path": str(out_path),
                                  "partition_sha256": _file_sha256(str(out_path)),
                                  **diagnostics}}
    if required_judges == 2:
        for label, candidate_decisions in (("judge_a", judge_a_decisions),
                                           ("judge_b", judge_b_decisions)):
            candidate = _apply_decisions(part, candidate_decisions)
            candidate_path = Path(OUT) / f"partition_{stem}_certified_{label}.json"
            candidate_path.write_text(json.dumps(candidate) + "\n")
            candidates[label] = {"partition_path": str(candidate_path),
                                 "partition_sha256": _file_sha256(str(candidate_path)),
                                 **_partition_diagnostics(candidate, manifest["threshold"])}
    report = {"task": task, "level": level, **({"tag": tag} if tag else {}),
              "threshold": manifest["threshold"],
              "required_judges": required_judges,
              "exact_certification_agreement": (round(exact_certification_agreement /
                                                        manifest["n_oversized"], 3)
                                                  if required_judges == 2 and
                                                  manifest["n_oversized"] else None),
              "common_refinements": common_refinements,
              "tiebreaks_used": len(tiebreak_decisions),
              "certified_oversized": sum(v is None for v in decisions.values()),
              "repartitioned_oversized": sum(v is not None for v in decisions.values()),
              "groups_before": manifest["n_groups"], "groups_after": diagnostics["n_groups"],
              "remaining_over_threshold": diagnostics["remaining_over_threshold"],
              "max_group_size": diagnostics["max_group_size"],
              "partition_path": str(out_path), "partition_sha256": _file_sha256(str(out_path)),
              "candidate_partitions": candidates,
              "selection_rule": ("two independent frontier judges; invoke a third only when their "
                                  "full-group certificate/repartition decisions differ; majority "
                                  "decides intact-vs-split and split proposals use common refinement"
                                  if required_judges == 2 else
                                  "single certified candidate")}
    (ROOT / f"{stem}_apply_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prep = subparsers.add_parser("prepare")
    prep.add_argument("task")
    prep.add_argument("level")
    prep.add_argument("partition_path")
    prep.add_argument("--threshold", type=int, default=30)
    prep.add_argument("--required-judges", type=int, choices=(1, 2), default=2)
    prep.add_argument("--tag", default="")
    app = subparsers.add_parser("apply")
    app.add_argument("task")
    app.add_argument("level")
    app.add_argument("--tag", default="")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.task, args.level, args.partition_path,
                         threshold=args.threshold, required_judges=args.required_judges,
                         tag=args.tag)
    else:
        result = apply(args.task, args.level, tag=args.tag)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
