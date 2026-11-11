"""Semantic cross-community recall recovery with LLM-only merge decisions.

``prepare`` uses BGE embeddings only to retrieve plausible *cross-community* group pairs.  It
never turns a cosine similarity into a semantic label.  The resulting payloads expose the complete
membership of both source communities, representative members, and the strongest retrieved child
node pairs so two independent LLM passes can judge the frozen level relation.

``apply`` is deliberately fail-closed.  The screen directory must cover every emitted pair exactly
once with strict integer labels.  A staged confirmation pass may cover only the screen-positive
pairs (the only pairs that could possibly be admitted), while the legacy mode still accepts a full
second pass.  A source-community edge is admitted only when both independent LLM passes score it
``2``.  Connected components of those doubly-confirmed edges are then composed with the frozen
source partition; cosine values are not consulted during application.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np

from .build_level import (
    OUT,
    _file_sha256,
    _load_partition,
    _validate_semantic_nodes,
    nodes_from_level,
    rep_text,
)
from .sources import ROOT as REPO_ROOT


MODEL_ID = "BAAI/bge-small-en-v1.5"
POOLING = "cls"  # BGE encoder model cards use the first-token/CLS representation.
SHARD_SIZE = 100
EVIDENCE_PER_GROUP_PAIR = 5
REPRESENTATIVE_MEMBERS = 8
VERSION = "semantic-group-merge-v1"


class FrozenInputError(RuntimeError):
    """A manifest-frozen preparation input changed before apply."""


class VoteCoverageError(RuntimeError):
    """A vote directory is malformed or does not exactly cover the candidate set."""


def _artifact_root() -> Path:
    return Path(OUT) / "semantic_group_merge"


def _stem(task: str, level: str, tag: str = "") -> str:
    return f"{task}_{level}" + (f"_{tag}" if tag else "")


def _manifest_path(task: str, level: str, tag: str = "") -> Path:
    return _artifact_root() / f"{_stem(task, level, tag)}_manifest.json"


def _confirm_manifest_path(task: str, level: str, tag: str = "") -> Path:
    return _artifact_root() / f"{_stem(task, level, tag)}_confirm_manifest.json"


def _stable_pair_id(task: str, level: str, partition_sha256: str,
                    group_a: str, group_b: str) -> str:
    a, b = sorted((str(group_a), str(group_b)))
    raw = f"{VERSION}||{task}||{level}||{partition_sha256}||{a}||{b}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def _canonical_pair(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((str(a), str(b))))


def _jsonl_rows(path: Path) -> Iterator[dict]:
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read JSONL file {path}: {exc}") from exc
    for line_no, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"malformed JSON in {path}:{line_no}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"non-object JSON row in {path}:{line_no}")
        yield row


def _pair_set(paths: Iterable[Path]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for path in paths:
        for row in _jsonl_rows(path):
            if "node_a" not in row or "node_b" not in row:
                raise ValueError(f"pair row in {path} lacks node_a/node_b")
            a, b = str(row["node_a"]), str(row["node_b"])
            if a == b:
                raise ValueError(f"self-pair in exclusion source {path}: {a}")
            pairs.add(_canonical_pair(a, b))
    return pairs


def _resolve_manifest_relative(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else Path(REPO_ROOT) / candidate


def _eval_path(task: str, level: str) -> Path:
    level_manifest = Path(OUT) / f"level_manifest_{task}_{level}.json"
    if level_manifest.exists():
        manifest = json.loads(level_manifest.read_text())
        if manifest.get("eval_path"):
            path = _resolve_manifest_relative(str(manifest["eval_path"]))
            frozen = manifest.get("eval_sha256")
            if frozen and _file_sha256(str(path)) != frozen:
                raise FrozenInputError(f"[{task}/{level}] frozen eval changed: {path}")
            return path
    return Path(OUT) / f"level_eval_{task}_{level}.jsonl"


def _source_fingerprints(paths: Sequence[Path]) -> list[dict]:
    return [{"path": str(path.resolve()), "sha256": _file_sha256(str(path))}
            for path in paths]


def _pool_embeddings(last_hidden_state, attention_mask, pooling: str = POOLING):
    """Pool transformer token states; BGE defaults to CLS, with masked mean available for tests.

    This helper intentionally does not normalize so pooling and normalization can be tested
    independently; ``_embed_bge`` always L2-normalizes the pooled vectors before retrieval.
    """
    if pooling == "cls":
        return last_hidden_state[:, 0]
    if pooling == "mean":
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    raise ValueError(f"unsupported pooling mode: {pooling}")


def _tokenizer_sha256(tokenizer) -> str:
    h = hashlib.sha256()
    vocab = tokenizer.get_vocab()
    h.update(json.dumps(sorted((str(token), int(index)) for token, index in vocab.items()),
                        ensure_ascii=False, separators=(",", ":")).encode())
    h.update(json.dumps(tokenizer.special_tokens_map, sort_keys=True,
                        ensure_ascii=False, default=str).encode())
    return h.hexdigest()


def _model_state_sha256(model) -> str:
    """Hash actual loaded weights, not a mutable model alias or cache path."""
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(value.dtype).encode())
        h.update(json.dumps(list(value.shape)).encode())
        # Viewing bytes avoids NumPy's lack of support for some torch dtypes (for example bfloat16).
        h.update(value.view(-1).view(__import__("torch").uint8).numpy().tobytes())
    return h.hexdigest()


def _embed_bge(texts: Sequence[str], batch_size: int = 64) -> tuple[np.ndarray, dict]:
    """Embed text with transformers directly; sentence-transformers is intentionally unused."""
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID)
    state_hash = _model_state_sha256(model)
    tokenizer_hash = _tokenizer_sha256(tokenizer)
    combined = hashlib.sha256(
        f"{MODEL_ID}||{POOLING}||{state_hash}||{tokenizer_hash}".encode()).hexdigest()
    revision = (getattr(model.config, "_commit_hash", None)
                or getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
                or "unresolved")

    requested = os.environ.get("SEMANTIC_GROUP_MERGE_DEVICE", "").strip()
    if requested:
        device = torch.device(requested)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model.to(device).eval()

    batches = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(list(texts[start:start + batch_size]), padding=True,
                                truncation=True, max_length=512, return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            output = model(**encoded)
            pooled = _pool_embeddings(output.last_hidden_state,
                                      encoded["attention_mask"], POOLING)
            batches.append(F.normalize(pooled, p=2, dim=1).cpu().numpy().astype(np.float32))
    vectors = (np.concatenate(batches, axis=0) if batches
               else np.empty((0, int(getattr(model.config, "hidden_size", 0))), dtype=np.float32))
    return vectors, {
        "model_id": MODEL_ID,
        "model_revision": str(revision),
        "model_sha256": combined,
        "model_state_sha256": state_hash,
        "tokenizer_sha256": tokenizer_hash,
        "pooling": POOLING,
        "normalized": True,
    }


def _validate_vectors(vectors: np.ndarray, n_rows: int) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != n_rows or values.shape[1] < 1:
        raise ValueError(f"embedding shape {values.shape} does not match {n_rows} nodes")
    if not np.isfinite(values).all():
        raise ValueError("embeddings contain non-finite values")
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms <= 0):
        raise ValueError("embeddings contain zero vectors")
    # Enforce normalization even for injected/test embedders and record only normalized vectors.
    return values / norms[:, None]


def _member_row(node: dict) -> dict:
    return {"node_id": str(node["node_id"]),
            "name": str(node.get("name") or node["node_id"]),
            "gloss": str(node.get("gloss") or "")}


def _representatives(member_ids: Sequence[str], evidence: Sequence[dict], side: str,
                     by_id: dict[str, dict]) -> list[dict]:
    key = f"node_{side}"
    ordered = []
    for row in evidence:
        node_id = str(row[key])
        if node_id not in ordered:
            ordered.append(node_id)
    for node_id in sorted(member_ids):
        if node_id not in ordered:
            ordered.append(node_id)
    return [_member_row(by_id[node_id]) for node_id in ordered[:REPRESENTATIVE_MEMBERS]]


def _write_payloads(task: str, level: str, rows: Sequence[dict],
                    tag: str = "") -> tuple[list[str], list[dict]]:
    payload_dir = _artifact_root() / "payloads"
    payload_dir.mkdir(parents=True, exist_ok=True)
    stem = _stem(task, level, tag)
    for old in payload_dir.iterdir():
        if old.is_file() and re.fullmatch(rf"{re.escape(stem)}_\d{{3}}\.jsonl", old.name):
            old.unlink()
    paths: list[str] = []
    fingerprints: list[dict] = []
    for shard, start in enumerate(range(0, len(rows), SHARD_SIZE)):
        path = payload_dir / f"{stem}_{shard:03d}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                for row in rows[start:start + SHARD_SIZE]))
        paths.append(str(path.resolve()))
        fingerprints.append({"path": str(path.resolve()), "sha256": _file_sha256(str(path))})
    return paths, fingerprints


def prepare(task: str, level: str, partition_path: str, k: int = 50,
            cap: int = 2000, tag: str = "",
            exclude_manifest_paths: Sequence[str] | None = None) -> dict:
    """Prepare frozen cross-community semantic-retrieval payloads.

    ``cap`` limits emitted *group-pair* candidates after child-node kNN evidence is aggregated.
    ``exclude_manifest_paths`` removes exact source-group pairs already routed by earlier frozen
    semantic-merge manifests.  This is judgment deduplication only: it never treats an earlier
    score, cosine, or lexical comparison as a semantic decision.  Similarity determines routing
    and evidence order only; every merge decision is deferred to the two independent LLM vote
    passes consumed by :func:`apply`.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    if cap < 0:
        raise ValueError("cap cannot be negative")
    partition_file = Path(partition_path).expanduser().resolve()
    if not partition_file.exists():
        raise FileNotFoundError(partition_file)
    partition_sha = _file_sha256(str(partition_file))
    partition = {str(node): str(group)
                 for node, group in _load_partition(str(partition_file)).items()}
    nodes, _ = nodes_from_level(task, level)
    _validate_semantic_nodes(task, level, nodes)
    by_id = {str(node["node_id"]): node for node in nodes}
    if set(partition) != set(by_id):
        missing = sorted(set(by_id) - set(partition))[:8]
        extra = sorted(set(partition) - set(by_id))[:8]
        raise ValueError(f"[{task}/{level}] partition/node coverage mismatch; "
                         f"missing={missing} extra={extra}")

    protocol = (Path(OUT) / f"ARBITER_PROTOCOL_{level}.txt").resolve()
    if not protocol.exists():
        raise FileNotFoundError(f"missing arbiter protocol: {protocol}")
    eval_file = _eval_path(task, level).resolve()
    if not eval_file.exists():
        raise FileNotFoundError(f"missing frozen level eval: {eval_file}")
    lexical_files = sorted((Path(OUT) / "level_arbiter").glob(
        f"{task}_{level}_verify_*.jsonl"))
    eval_pairs = _pair_set([eval_file])
    lexical_pairs = _pair_set(lexical_files)
    excluded = eval_pairs | lexical_pairs

    excluded_group_pairs: set[tuple[str, str]] = set()
    exclusion_sources: list[dict] = []
    for raw_path in exclude_manifest_paths or ():
        prior_path = Path(raw_path).expanduser().resolve()
        prior = _load_manifest(task, level, str(prior_path))
        prior_pairs: set[tuple[str, str]] = set()
        for payload_path in prior.get("payload_paths") or []:
            for row in _jsonl_rows(Path(payload_path)):
                group_a = str((row.get("group_a") or {}).get("group_id") or "")
                group_b = str((row.get("group_b") or {}).get("group_id") or "")
                if not group_a or not group_b or group_a == group_b:
                    raise FrozenInputError(
                        f"malformed excluded group pair in prior manifest: {prior_path}")
                prior_pairs.add(_canonical_pair(group_a, group_b))
        excluded_group_pairs.update(prior_pairs)
        exclusion_sources.append({
            "path": str(prior_path),
            "sha256": _file_sha256(str(prior_path)),
            "n_group_pairs": len(prior_pairs),
        })

    ordered_ids = sorted(by_id)
    texts = [rep_text(by_id[node_id]) for node_id in ordered_ids]
    vectors, model = _embed_bge(texts)
    vectors = _validate_vectors(vectors, len(ordered_ids))
    if not isinstance(model, dict) or len(str(model.get("model_sha256") or "")) != 64:
        raise ValueError("embedder did not provide a frozen 64-character model_sha256")

    members: dict[str, list[str]] = defaultdict(list)
    for node_id, group_id in partition.items():
        members[group_id].append(node_id)
    for group_ids in members.values():
        group_ids.sort()

    node_candidates: dict[tuple[str, str], float] = {}
    if len(ordered_ids) >= 2 and cap:
        from sklearn.neighbors import NearestNeighbors
        neighbors = min(k + 1, len(ordered_ids))
        distances, indices = NearestNeighbors(
            n_neighbors=neighbors, metric="cosine").fit(vectors).kneighbors(vectors)
        for row_index, node_a in enumerate(ordered_ids):
            for position in range(indices.shape[1]):
                other_index = int(indices[row_index, position])
                if other_index == row_index:
                    continue
                node_b = ordered_ids[other_index]
                if partition[node_a] == partition[node_b]:
                    continue
                node_pair = _canonical_pair(node_a, node_b)
                if node_pair in excluded:
                    continue
                similarity = 1.0 - float(distances[row_index, position])
                node_candidates[node_pair] = max(node_candidates.get(node_pair, -1.0), similarity)

    evidence_by_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for (node_x, node_y), similarity in node_candidates.items():
        group_x, group_y = partition[node_x], partition[node_y]
        group_a, group_b = sorted((group_x, group_y))
        node_a, node_b = ((node_x, node_y) if group_x == group_a else (node_y, node_x))
        evidence_by_groups[(group_a, group_b)].append({
            "node_a": node_a,
            "node_b": node_b,
            "canonical_a": rep_text(by_id[node_a]),
            "canonical_b": rep_text(by_id[node_b]),
            "cosine": round(float(similarity), 6),
        })
    for evidence in evidence_by_groups.values():
        evidence.sort(key=lambda row: (-row["cosine"], row["node_a"], row["node_b"]))
        del evidence[EVIDENCE_PER_GROUP_PAIR:]

    eligible_group_pairs = [
        pair for pair in evidence_by_groups if pair not in excluded_group_pairs
    ]
    ranked_group_pairs = sorted(
        eligible_group_pairs,
        key=lambda pair: (-evidence_by_groups[pair][0]["cosine"], pair[0], pair[1]))[:cap]
    payload_rows = []
    for group_a, group_b in ranked_group_pairs:
        evidence = evidence_by_groups[(group_a, group_b)]
        payload_rows.append({
            "pair_id": _stable_pair_id(task, level, partition_sha, group_a, group_b),
            "task": task,
            "level": level,
            "group_a": {
                "group_id": group_a,
                "representative_members": _representatives(
                    members[group_a], evidence, "a", by_id),
                "all_members": [_member_row(by_id[node]) for node in members[group_a]],
            },
            "group_b": {
                "group_id": group_b,
                "representative_members": _representatives(
                    members[group_b], evidence, "b", by_id),
                "all_members": [_member_row(by_id[node]) for node in members[group_b]],
            },
            "evidence": evidence,
            "retrieval_notice": "Cosine is routing evidence only; assign semantics from the frozen LLM protocol.",
        })

    pair_ids = [row["pair_id"] for row in payload_rows]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("candidate pair_id collision")
    payload_paths, payload_fingerprints = _write_payloads(task, level, payload_rows, tag)
    manifest = {
        "version": VERSION,
        "task": task,
        "level": level,
        "tag": tag,
        "k": k,
        "cap": cap,
        "n_nodes": len(nodes),
        "n_source_groups": len(members),
        "n_node_pair_candidates": len(node_candidates),
        "n_group_pair_candidates": len(payload_rows),
        "partition_path": str(partition_file),
        "partition_sha256": partition_sha,
        "protocol_path": str(protocol),
        "protocol_sha256": _file_sha256(str(protocol)),
        "model": model,
        "eval_sources": _source_fingerprints([eval_file]),
        "lexical_verify_sources": _source_fingerprints(lexical_files),
        "n_eval_pairs_excluded": len(eval_pairs),
        "n_lexical_verify_pairs_excluded": len(lexical_pairs),
        "exclude_manifest_sources": exclusion_sources,
        "n_prior_group_pairs_listed": len(excluded_group_pairs),
        "n_prior_group_pairs_excluded": len(evidence_by_groups) - len(eligible_group_pairs),
        "payload_paths": payload_paths,
        "payload_fingerprints": payload_fingerprints,
        "vote_schema": {"pair_id": "string", "score": "strict integer 0|1|2"},
        "semantic_decider": "LLM only; cosine retrieves/ranks candidates and never authorizes a merge",
        "apply_rule": "merge a group pair only when independent screen and confirm votes both equal 2",
    }
    root = _artifact_root()
    root.mkdir(parents=True, exist_ok=True)
    _manifest_path(task, level, tag).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _validate_frozen_file(row: dict, label: str) -> None:
    path = Path(str(row.get("path") or ""))
    expected = str(row.get("sha256") or "")
    if not path.exists() or not expected or _file_sha256(str(path)) != expected:
        raise FrozenInputError(f"frozen {label} changed or disappeared: {path}")


def _load_manifest(task: str, level: str, manifest_path: str | None) -> dict:
    path = Path(manifest_path).resolve() if manifest_path else _manifest_path(task, level)
    try:
        manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenInputError(f"cannot read semantic merge manifest {path}") from exc
    if not isinstance(manifest, dict) or manifest.get("version") != VERSION:
        raise FrozenInputError(f"unsupported or malformed manifest: {path}")
    if manifest.get("task") != task or manifest.get("level") != level:
        raise FrozenInputError(f"manifest identity mismatch: expected {task}/{level}")
    for key, label in (("partition", "partition"), ("protocol", "protocol")):
        _validate_frozen_file({"path": manifest.get(f"{key}_path"),
                               "sha256": manifest.get(f"{key}_sha256")}, label)
    for row in manifest.get("eval_sources") or []:
        _validate_frozen_file(row, "eval source")
    for row in manifest.get("lexical_verify_sources") or []:
        _validate_frozen_file(row, "lexical verify source")
    for row in manifest.get("exclude_manifest_sources") or []:
        _validate_frozen_file(row, "excluded prior manifest")
    for row in manifest.get("payload_fingerprints") or []:
        _validate_frozen_file(row, "payload")
    return manifest


def _payload_candidates(manifest: dict) -> dict[str, tuple[str, str]]:
    expected_fingerprints = {str(Path(row["path"]).resolve()): row["sha256"]
                             for row in manifest.get("payload_fingerprints") or []}
    paths = [Path(path).resolve() for path in manifest.get("payload_paths") or []]
    if {str(path) for path in paths} != set(expected_fingerprints):
        raise FrozenInputError("payload path/fingerprint sets differ")
    candidates: dict[str, tuple[str, str]] = {}
    for path in paths:
        for row in _jsonl_rows(path):
            pair_id = row.get("pair_id")
            group_a = (row.get("group_a") or {}).get("group_id")
            group_b = (row.get("group_b") or {}).get("group_id")
            if not isinstance(pair_id, str) or group_a is None or group_b is None:
                raise FrozenInputError(f"malformed candidate payload row in {path}")
            if pair_id in candidates:
                raise FrozenInputError(f"duplicate candidate pair_id: {pair_id}")
            groups = _canonical_pair(str(group_a), str(group_b))
            if groups[0] == groups[1]:
                raise FrozenInputError(f"same-group candidate in {path}: {groups[0]}")
            candidates[pair_id] = groups
    if len(candidates) != int(manifest.get("n_group_pair_candidates", -1)):
        raise FrozenInputError("manifest candidate count does not match frozen payloads")
    return candidates


def _vote_file_rows(path: Path) -> Iterator[dict]:
    if path.suffix == ".jsonl":
        yield from _jsonl_rows(path)
        return
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise VoteCoverageError(f"malformed vote file: {path}") from exc
    rows = value if isinstance(value, list) else [value]
    for row in rows:
        if not isinstance(row, dict):
            raise VoteCoverageError(f"non-object vote in {path}")
        yield row


def _load_vote_directory(path: str, expected: set[str], stage: str) -> dict[str, int]:
    directory = Path(path).expanduser().resolve()
    if not directory.is_dir():
        raise VoteCoverageError(f"{stage} vote directory does not exist: {directory}")
    files = sorted([item for item in directory.iterdir()
                    if item.is_file() and item.suffix in (".json", ".jsonl")])
    votes: dict[str, int] = {}
    for file_path in files:
        for row in _vote_file_rows(file_path):
            if set(row) != {"pair_id", "score"}:
                raise VoteCoverageError(
                    f"{stage} vote must have exact pair_id/score schema: {file_path}")
            pair_id, score = row["pair_id"], row["score"]
            if not isinstance(pair_id, str) or type(score) is not int or score not in (0, 1, 2):
                raise VoteCoverageError(f"invalid {stage} vote in {file_path}: {row!r}")
            if pair_id in votes:
                raise VoteCoverageError(f"duplicate {stage} vote for {pair_id}")
            votes[pair_id] = score
    present = set(votes)
    if present != expected:
        missing = sorted(expected - present)[:8]
        extra = sorted(present - expected)[:8]
        raise VoteCoverageError(
            f"{stage} vote coverage is not exact: {len(present)}/{len(expected)}; "
            f"missing={missing} extra={extra}")
    return votes


class _DisjointSet:
    def __init__(self, values: Iterable[str]):
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        low, high = sorted((ra, rb))
        self.parent[high] = low


def _directory_sha256(path: str) -> str:
    directory = Path(path).expanduser().resolve()
    h = hashlib.sha256()
    for file_path in sorted(item for item in directory.iterdir()
                            if item.is_file() and item.suffix in (".json", ".jsonl")):
        h.update(file_path.name.encode())
        h.update(_file_sha256(str(file_path)).encode())
    return h.hexdigest()


def stage_confirm(task: str, level: str, screen_vote_dir: str, *,
                  manifest_path: str | None = None, tag: str = "",
                  per_agent: int = SHARD_SIZE) -> dict:
    """Freeze and emit an independent confirmation pass for screen-positive pairs only.

    A screen-negative pair cannot be merged under the dual-score-2 rule, so asking the second LLM
    to rescore it consumes judgment without changing any possible partition.  This function keeps
    the first pass complete and fail-closed, then routes *every and only* score-2 screen decision to
    an independent confirmer with the original full semantic payload.
    """
    if per_agent < 1:
        raise ValueError("per_agent must be at least 1")
    manifest_file = (Path(manifest_path).expanduser().resolve() if manifest_path
                     else _manifest_path(task, level, tag).resolve())
    manifest = _load_manifest(task, level, str(manifest_file))
    candidates = _payload_candidates(manifest)
    screen_path = Path(screen_vote_dir).expanduser().resolve()
    screen = _load_vote_directory(str(screen_path), set(candidates), "screen")
    positive_ids = [pair_id for pair_id in candidates if screen[pair_id] == 2]

    payload_by_id = {}
    for path in manifest.get("payload_paths") or []:
        for row in _jsonl_rows(Path(path)):
            payload_by_id[row["pair_id"]] = row
    if set(payload_by_id) != set(candidates):
        raise FrozenInputError("confirmation staging could not recover every frozen payload")

    payload_dir = _artifact_root() / "confirm_payloads"
    payload_dir.mkdir(parents=True, exist_ok=True)
    stem = _stem(task, level, tag)
    for old in payload_dir.glob(f"{stem}_*.jsonl"):
        old.unlink()
    paths, fingerprints = [], []
    for shard, start in enumerate(range(0, len(positive_ids), per_agent)):
        path = payload_dir / f"{stem}_{shard:03d}.jsonl"
        path.write_text("".join(
            json.dumps(payload_by_id[pair_id], ensure_ascii=False) + "\n"
            for pair_id in positive_ids[start:start + per_agent]))
        paths.append(str(path.resolve()))
        fingerprints.append({"path": str(path.resolve()), "sha256": _file_sha256(str(path))})

    result = {
        "version": VERSION,
        "task": task,
        "level": level,
        "tag": tag,
        "source_manifest_path": str(manifest_file),
        "source_manifest_sha256": _file_sha256(str(manifest_file)),
        "screen_vote_dir": str(screen_path),
        "screen_vote_sha256": _directory_sha256(str(screen_path)),
        "n_candidates": len(candidates),
        "n_screen_positive": len(positive_ids),
        "confirm_pair_ids": positive_ids,
        "payload_paths": paths,
        "payload_fingerprints": fingerprints,
        "rule": "confirm every and only screen score-2 pair; merge only on independent 2+2",
    }
    destination = _confirm_manifest_path(task, level, tag)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    return result


def _load_staged_confirm(path: str, task: str, level: str, manifest_path: str,
                         screen_path: str, screen: dict[str, int]) -> set[str]:
    confirmation_path = Path(path).expanduser().resolve()
    try:
        staged = json.loads(confirmation_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenInputError(f"cannot read staged confirmation manifest: {confirmation_path}") from exc
    if (staged.get("version") != VERSION or staged.get("task") != task
            or staged.get("level") != level):
        raise FrozenInputError("staged confirmation identity/version mismatch")
    if (_file_sha256(manifest_path) != staged.get("source_manifest_sha256")
            or str(Path(manifest_path).resolve()) != str(Path(staged.get("source_manifest_path", "")).resolve())):
        raise FrozenInputError("staged confirmation source manifest changed")
    if (_directory_sha256(screen_path) != staged.get("screen_vote_sha256")
            or str(Path(screen_path).resolve()) != str(Path(staged.get("screen_vote_dir", "")).resolve())):
        raise FrozenInputError("staged confirmation screen votes changed")
    for row in staged.get("payload_fingerprints") or []:
        _validate_frozen_file(row, "confirmation payload")
    expected = {pair_id for pair_id, score in screen.items() if score == 2}
    recorded = staged.get("confirm_pair_ids")
    if (not isinstance(recorded, list) or len(recorded) != len(set(recorded))
            or set(recorded) != expected or staged.get("n_screen_positive") != len(expected)):
        raise FrozenInputError("staged confirmation does not exactly cover screen-positive pairs")
    return expected


def apply(task: str, level: str, screen_vote_dir: str, confirm_vote_dir: str, *,
          manifest_path: str | None = None, output_path: str | None = None,
          confirm_manifest_path: str | None = None) -> dict:
    """Apply only doubly-confirmed score-2 group edges to the frozen source partition."""
    screen_path = Path(screen_vote_dir).expanduser().resolve()
    confirm_path = Path(confirm_vote_dir).expanduser().resolve()
    if screen_path == confirm_path:
        raise VoteCoverageError("screen and confirm must be distinct vote directories")
    resolved_manifest_path = str(Path(manifest_path).expanduser().resolve()) if manifest_path else str(
        _manifest_path(task, level).resolve())
    manifest = _load_manifest(task, level, resolved_manifest_path)
    candidates = _payload_candidates(manifest)
    expected = set(candidates)
    # Both complete directories are loaded before any output is written.
    screen = _load_vote_directory(str(screen_path), expected, "screen")
    confirm_expected = (_load_staged_confirm(confirm_manifest_path, task, level,
                                              resolved_manifest_path, str(screen_path), screen)
                        if confirm_manifest_path else expected)
    confirm = _load_vote_directory(str(confirm_path), confirm_expected, "confirm")

    source = {str(node): str(group) for node, group in _load_partition(
        str(manifest["partition_path"])).items()}
    source_groups = set(source.values())
    dsu = _DisjointSet(source_groups)
    confirmed_pair_ids = []
    for pair_id, (group_a, group_b) in candidates.items():
        if group_a not in source_groups or group_b not in source_groups:
            raise FrozenInputError(f"candidate {pair_id} references a stale source group")
        # Semantic authorization comes only from independent LLM labels.  Cosine is not read here.
        if screen[pair_id] == 2 and confirm.get(pair_id) == 2:
            dsu.union(group_a, group_b)
            confirmed_pair_ids.append(pair_id)

    components: dict[str, list[str]] = defaultdict(list)
    for group_id in sorted(source_groups):
        components[dsu.find(group_id)].append(group_id)
    composed_group: dict[str, str] = {}
    for groups in components.values():
        groups.sort()
        if len(groups) == 1:
            new_group = groups[0]
        else:
            digest = hashlib.sha1("||".join(groups).encode()).hexdigest()[:12]
            new_group = f"{task}_{level}_sgm_{digest}"
        for group_id in groups:
            composed_group[group_id] = new_group
    composed = {node: composed_group[group] for node, group in source.items()}
    if set(composed) != set(source):
        raise FrozenInputError("composed partition lost source nodes")

    destination = (Path(output_path).expanduser().resolve() if output_path else
                   (Path(OUT) / f"partition_{task}_{level}_semantic_merged.json").resolve())
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(composed, sort_keys=True) + "\n")
    report = {
        "version": VERSION,
        "task": task,
        "level": level,
        "source_partition_path": manifest["partition_path"],
        "source_partition_sha256": manifest["partition_sha256"],
        "screen_vote_dir": str(screen_path),
        "screen_vote_sha256": _directory_sha256(str(screen_path)),
        "confirm_vote_dir": str(confirm_path),
        "confirm_vote_sha256": _directory_sha256(str(confirm_path)),
        "confirm_manifest_path": (str(Path(confirm_manifest_path).expanduser().resolve())
                                  if confirm_manifest_path else None),
        "n_confirm_judgments": len(confirm),
        "n_candidates": len(candidates),
        "n_dual_score2_edges": len(confirmed_pair_ids),
        "dual_score2_pair_ids": sorted(confirmed_pair_ids),
        "groups_before": len(source_groups),
        "groups_after": len(set(composed.values())),
        "partition_path": str(destination),
        "partition_sha256": _file_sha256(str(destination)),
        "semantic_rule": "only pair_ids scored 2 in both screen and confirm authorize edges",
    }
    report_path = _artifact_root() / f"{task}_{level}_apply_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("task")
    prep.add_argument("level")
    prep.add_argument("partition_path")
    prep.add_argument("--k", type=int, default=50)
    prep.add_argument("--cap", type=int, default=2000)
    prep.add_argument("--tag", default="")
    stage = sub.add_parser("stage-confirm")
    stage.add_argument("task")
    stage.add_argument("level")
    stage.add_argument("screen_vote_dir")
    stage.add_argument("--manifest-path")
    stage.add_argument("--tag", default="")
    stage.add_argument("--per-agent", type=int, default=SHARD_SIZE)
    app = sub.add_parser("apply")
    app.add_argument("task")
    app.add_argument("level")
    app.add_argument("screen_vote_dir")
    app.add_argument("confirm_vote_dir")
    app.add_argument("--manifest-path")
    app.add_argument("--output-path")
    app.add_argument("--confirm-manifest-path")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.task, args.level, args.partition_path, k=args.k, cap=args.cap,
                         tag=args.tag)
    elif args.command == "stage-confirm":
        result = stage_confirm(args.task, args.level, args.screen_vote_dir,
                               manifest_path=args.manifest_path, tag=args.tag,
                               per_agent=args.per_agent)
    else:
        result = apply(args.task, args.level, args.screen_vote_dir, args.confirm_vote_dir,
                       manifest_path=args.manifest_path, output_path=args.output_path,
                       confirm_manifest_path=args.confirm_manifest_path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
