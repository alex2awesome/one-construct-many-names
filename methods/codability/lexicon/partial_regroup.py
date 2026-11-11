"""Apply independently certified member-level regroupings without whole-community merges."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .build_level import OUT, _file_sha256, _load_partition


ROOT = Path(OUT) / "semantic_group_merge"
VERSION = "partial-regroup-v1"


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"non-object row at {path}:{line_no}")
        rows.append(row)
    return rows


def freeze(task: str, level: str, partition_path: str, proposals_path: str,
           audit_path: str) -> dict:
    """Freeze a construction proposal and its independent subgroup audit before application."""
    partition = Path(partition_path).expanduser().resolve()
    proposals = Path(proposals_path).expanduser().resolve()
    audit = Path(audit_path).expanduser().resolve()
    protocol = (Path(OUT) / f"ARBITER_PROTOCOL_{level}.txt").resolve()
    semantic_manifest = ROOT / f"{task}_{level}_manifest.json"
    manifest = {
        "version": VERSION, "task": task, "level": level,
        "partition_path": str(partition), "partition_sha256": _file_sha256(str(partition)),
        "proposals_path": str(proposals), "proposals_sha256": _file_sha256(str(proposals)),
        "audit_path": str(audit), "audit_sha256": _file_sha256(str(audit)),
        "protocol_path": str(protocol), "protocol_sha256": _file_sha256(str(protocol)),
        "semantic_retrieval_manifest_path": str(semantic_manifest.resolve()),
        "semantic_retrieval_manifest_sha256": _file_sha256(str(semantic_manifest)),
        "acceptance_rule": "constructor proposes exact subgroup AND independent audit score is 2",
    }
    path = ROOT / f"{task}_{level}_partial_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def apply(task: str, level: str) -> dict:
    manifest_path = ROOT / f"{task}_{level}_partial_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("version") != VERSION or manifest.get("task") != task or manifest.get("level") != level:
        raise ValueError("partial regroup manifest identity mismatch")
    for field in ("partition", "proposals", "audit", "protocol", "semantic_retrieval_manifest"):
        if _file_sha256(manifest[f"{field}_path"]) != manifest[f"{field}_sha256"]:
            raise ValueError(f"[{task}/{level}] frozen {field} changed")

    source = {str(k): str(v) for k, v in _load_partition(manifest["partition_path"]).items()}
    semantic = json.loads(Path(manifest["semantic_retrieval_manifest_path"]).read_text())
    payload_by_pair = {}
    for item in semantic.get("payload_fingerprints", []):
        path = Path(item["path"])
        if _file_sha256(str(path)) != item["sha256"]:
            raise ValueError(f"semantic payload changed: {path}")
        for row in _load_jsonl(path):
            pid = str(row.get("pair_id"))
            if pid in payload_by_pair:
                raise ValueError(f"duplicate semantic pair_id {pid}")
            payload_by_pair[pid] = row

    proposals = _load_jsonl(Path(manifest["proposals_path"]))
    proposed = {}
    used_nodes = set()
    for row in proposals:
        if set(row) != {"pair_id", "subgroups"} or not isinstance(row["subgroups"], list):
            raise ValueError("invalid partial proposal row")
        pid = str(row["pair_id"])
        if pid not in payload_by_pair or pid in proposed:
            raise ValueError(f"unknown or duplicate proposal pair_id {pid}")
        payload = payload_by_pair[pid]
        side_a = {str(x["node_id"]) for x in payload["group_a"]["all_members"]}
        side_b = {str(x["node_id"]) for x in payload["group_b"]["all_members"]}
        groups = []
        for index, subgroup in enumerate(row["subgroups"]):
            if set(subgroup) != {"node_ids_a", "node_ids_b"}:
                raise ValueError(f"invalid subgroup schema for {pid}/{index}")
            a = [str(x) for x in subgroup["node_ids_a"]]
            b = [str(x) for x in subgroup["node_ids_b"]]
            members = a + b
            if (not a or not b or len(members) != len(set(members))
                    or not set(a) <= side_a or not set(b) <= side_b
                    or not set(members) <= set(source)):
                raise ValueError(f"invalid subgroup membership for {pid}/{index}")
            if used_nodes.intersection(members):
                raise ValueError(f"node proposed more than once at {pid}/{index}")
            used_nodes.update(members)
            groups.append(members)
        proposed[pid] = groups

    audit_rows = _load_jsonl(Path(manifest["audit_path"]))
    audits = {}
    for row in audit_rows:
        if set(row) != {"pair_id", "subgroup_index", "score"}:
            raise ValueError("invalid partial audit row")
        pid, index, score = str(row["pair_id"]), row["subgroup_index"], row["score"]
        if (pid not in proposed or type(index) is not int or not 0 <= index < len(proposed[pid])
                or type(score) is not int or score not in (0, 1, 2)
                or (pid, index) in audits):
            raise ValueError(f"invalid partial audit decision {pid}/{index}")
        audits[(pid, index)] = score
    expected = {(pid, i) for pid, groups in proposed.items() for i in range(len(groups))}
    if set(audits) != expected:
        raise ValueError(f"partial audit coverage mismatch: {len(audits)}/{len(expected)}")

    out = dict(source)
    accepted = 0
    accepted_nodes = 0
    for pid, groups in proposed.items():
        for index, members in enumerate(groups):
            if audits[(pid, index)] != 2:
                continue
            digest = hashlib.sha1(
                f"{VERSION}||{task}||{level}||{pid}||{index}".encode()).hexdigest()[:10]
            group_id = f"{task}_{level}_partial_{digest}"
            for node in members:
                out[node] = group_id
            accepted += 1
            accepted_nodes += len(members)
    if set(out) != set(source):
        raise ValueError("partial regroup changed node coverage")
    out_path = Path(OUT) / f"partition_{task}_{level}_partial_candidate.json"
    out_path.write_text(json.dumps(out) + "\n")
    report = {
        "task": task, "level": level, "n_source_groups": len(set(source.values())),
        "n_candidate_groups": len(set(out.values())), "n_proposed_subgroups": len(expected),
        "n_accepted_subgroups": accepted, "n_accepted_nodes": accepted_nodes,
        "partition_path": str(out_path.resolve()), "partition_sha256": _file_sha256(str(out_path)),
        "semantic_decider": "constructor plus independent score-2 LLM audit",
    }
    (ROOT / f"{task}_{level}_partial_apply_report.json").write_text(
        json.dumps(report, indent=2) + "\n")
    return report
