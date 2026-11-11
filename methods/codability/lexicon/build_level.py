#!/usr/bin/env python
"""General hierarchy level builder L -> L+1 (R1=construct, R2=theme, R3=category), all tasks.

Nodes at level L+1 are the GROUPS of level L, each identified by its GEPA-named label+gloss. The
DIFFUSE net (user 2026-07-07: lexical decays up-tree) is an LLM GROUP-PROPOSER over node reps:
  - if #nodes is small (<= SINGLE_CALL_MAX) -> one prompt sees ALL nodes -> a global grouping
    (natural for R2/R3, where nodes are few hundred);
  - else -> TF-IDF-bucketed neighborhoods so related nodes co-occur (R1, thousands of L0 clusters);
    each node is assigned to its group in its PRIMARY (self-centered) bucket -> no cross-bucket
    union-find chaining.
The grouping is the hypothesis P_{L+1}; a frozen LEVEL EVAL (node pairs judged at the level relation
by an arbiter panel) MEASURES it: recall P(co | same-at-level) / precision P(same-at-level | co).

Files (per task/level): level_payloads/<task>_<L>_group_NNN.jsonl (proposer in),
level_votes/<task>_<L>_group_NNN.jsonl (proposer out: {bucket_id, groups:[[node_id,...]]}),
level_eval_<task>_<L>.jsonl (frozen node-pair eval), level_arbiter/<task>_<L>_NNN.jsonl (verify in),
partition_<task>_<L>.json (node_id -> group_id), node_names_<task>_<L>.json.
"""
from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import random
import re
import tempfile
from collections import defaultdict
from typing import Dict, List, Tuple

from .sources import ROOT

OUT = os.path.join(ROOT, "outputs", "lexicon")
SINGLE_CALL_MAX = 220  # <= this many nodes -> one global group-proposer prompt

# relation name + the sameness guidance shown to proposer & arbiter for each level
RELATIONS = {
    "R1": ("same construct",
           "Two clusters are the SAME CONSTRUCT when they are interchangeable criteria or direct "
           "facets/indicators of ONE narrow latent evaluative quality. They may inspect different "
           "evidence; operational interchangeability is L0 and is sufficient but not necessary here. "
           "A shared umbrella topic without one coherent latent dimension is only R2."),
    "R2": ("same theme",
           "Two constructs share the SAME THEME if they belong to one broad evaluative area/family "
           "of the domain. Different themes address different areas."),
    "R3": ("same category",
           "Two themes share the SAME TOP-LEVEL CATEGORY — the coarsest grouping the domain uses."),
}
PREV = {"R1": "L0v2", "R2": "R1", "R3": "R2"}
_PARENT_MANIFEST_FIELDS = (
    "parent_partition_path", "parent_partition_sha256",
    "parent_names_path", "parent_names_sha256",
)
_CANONICAL_PARTITION_RE = re.compile(r"^partition_.+_(?:L0v\d+|R[123])\.json$")


class LevelManifestError(RuntimeError):
    pass


def _h(*p: str) -> str:
    return hashlib.sha1("||".join(p).encode()).hexdigest()


def _load_partition(path: str) -> Dict[str, str]:
    d = json.load(open(path))
    return {k: str(v) for k, v in (d.get("partition", d)).items()}


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_json_write(path: str, payload) -> None:
    """Replace a JSON artifact atomically; never expose a half-written lineage file."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{os.path.basename(path)}.", dir=directory)
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh, indent=1)
            fh.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.remove(temporary)
        except FileNotFoundError:
            pass
        raise


def _manifest_path(task: str, level: str) -> str:
    if level not in PREV:
        raise ValueError(f"unknown hierarchy level: {level!r}")
    return os.path.join(OUT, f"level_manifest_{task}_{level}.json")


def _canonical_partition_path(task: str, level: str) -> str:
    if level not in PREV:
        raise ValueError(f"unknown hierarchy level: {level!r}")
    return os.path.join(OUT, f"partition_{task}_{level}.json")


def _candidate_output_path(task: str, level: str, output_path: str) -> str:
    """Accept only an explicit, non-canonical candidate destination."""
    if not output_path or not str(output_path).strip():
        raise ValueError("an explicit non-canonical output_path is required")
    destination = os.fspath(output_path)
    canonical = _canonical_partition_path(task, level)
    if (_CANONICAL_PARTITION_RE.fullmatch(os.path.basename(destination))
            or os.path.realpath(destination) == os.path.realpath(canonical)):
        raise ValueError(
            f"refusing to write a build directly to canonical-looking destination {destination!r}; "
            "write an immutable candidate name, then use promote_partition/partition-promote")
    return destination


def _validate_manifest_identity(manifest: dict, task: str, level: str) -> None:
    if manifest.get("task", task) != task or manifest.get("level", level) != level:
        raise LevelManifestError(
            f"[{task}/{level}] manifest identity disagrees with its filename: "
            f"task={manifest.get('task')!r}, level={manifest.get('level')!r}")


def _parent_partition(task: str, level: str) -> tuple[str, str]:
    """Exact parent partition path/hash, frozen by a level manifest when available."""
    manifest_path = _manifest_path(task, level)
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {}
    _validate_manifest_identity(manifest, task, level)
    path = manifest.get("parent_partition_path")
    frozen_digest = manifest.get("parent_partition_sha256")
    if bool(path) != bool(frozen_digest):
        raise LevelManifestError(
            f"[{task}/{level}] parent partition pin is partial; path and sha256 are both required")
    if path and not os.path.isabs(path):
        path = os.path.join(ROOT, path)
    if not path:
        prev = PREV[level]
        if prev == "L0v2":
            v3 = os.path.join(OUT, f"partition_{task}_L0v3.json")
            path = v3 if os.path.exists(v3) else os.path.join(OUT, f"partition_{task}_L0v2.json")
        else:
            path = os.path.join(OUT, f"partition_{task}_{prev}.json")
    digest = _file_sha256(path)
    if frozen_digest and frozen_digest != digest:
        raise RuntimeError(f"[{task}/{level}] frozen parent partition changed on disk: {path}")
    return path, digest


def _default_parent_names(task: str, level: str, parent_path: str) -> str:
    """Resolve the semantic name/gloss artifact corresponding to an exact parent partition."""
    prev = PREV[level]
    if prev == "L0v2":
        version = re.search(r"_(L0v\d+)\.json$", parent_path)
        candidates = ([os.path.join(OUT, f"cluster_names_{task}_{version.group(1)}.json")]
                      if version else [])
        candidates.append(os.path.join(OUT, f"cluster_names_{task}_L0v2.json"))
    else:
        candidates = [os.path.join(OUT, f"node_names_{task}_{prev}.json")]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if not path:
        raise FileNotFoundError(f"[{task}/{level}] no semantic names for parent {parent_path}; "
                                f"tried {candidates}")
    return path


def _parent_names(task: str, level: str, parent_path: str) -> tuple[str, str]:
    manifest_path = _manifest_path(task, level)
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {}
    _validate_manifest_identity(manifest, task, level)
    path = manifest.get("parent_names_path")
    frozen = manifest.get("parent_names_sha256")
    if bool(path) != bool(frozen):
        raise LevelManifestError(
            f"[{task}/{level}] parent names pin is partial; path and sha256 are both required")
    if path and not os.path.isabs(path):
        path = os.path.join(ROOT, path)
    path = path or _default_parent_names(task, level, parent_path)
    digest = _file_sha256(path)
    if frozen and frozen != digest:
        raise RuntimeError(f"[{task}/{level}] frozen parent semantic names changed on disk: {path}")
    return path, digest


def _validate_upper_parent_material(task: str, level: str, parent_path: str,
                                    names_path: str) -> None:
    """Fail closed before a legacy R1/R2 canonical becomes an upper-level parent."""
    previous = PREV[level]
    if previous not in PREV:
        return
    directory, basename = os.path.split(os.path.abspath(parent_path))
    if basename not in os.listdir(directory):
        raise LevelManifestError(
            f"[{task}/{level}] upper parent path is not present with exact case: {parent_path}")

    lower_parent_path, _ = _parent_partition(task, previous)
    expected_ids = set(_load_partition(lower_parent_path).values())
    parent = _load_partition(parent_path)
    missing = sorted(expected_ids - set(parent))
    extra = sorted(set(parent) - expected_ids)
    if missing or extra:
        raise LevelManifestError(
            f"[{task}/{level}] upper parent node inventory mismatch: missing={len(missing)}, "
            f"extra={len(extra)}, missing_sample={missing[:5]}, extra_sample={extra[:5]}")

    names = json.load(open(names_path))
    group_ids = set(parent.values())
    missing_names = sorted(group_ids - set(names))
    if missing_names:
        raise LevelManifestError(
            f"[{task}/{level}] upper parent has {len(missing_names)} unnamed groups; "
            f"sample={missing_names[:5]}")
    semantic_nodes = []
    for group_id in sorted(group_ids):
        row = names[group_id]
        if not isinstance(row, dict):
            raise LevelManifestError(
                f"[{task}/{level}] semantic name for {group_id!r} must be a JSON object")
        semantic_nodes.append({"node_id": group_id, "name": row.get("name", group_id),
                               "gloss": row.get("gloss", "")})
    _validate_semantic_nodes(task, level, semantic_nodes)


def _freeze_parent_for_new_build(task: str, level: str) -> None:
    """Freeze the complete L0->parent lineage before creating a build at ``level``."""
    if level not in PREV:
        raise ValueError(f"unknown hierarchy level: {level!r}")
    previous = PREV[level]
    if previous in PREV:
        _freeze_parent_for_new_build(task, previous)

    manifest_path = _manifest_path(task, level)
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {
        "task": task, "level": level}
    _validate_manifest_identity(manifest, task, level)
    manifest.setdefault("task", task)
    manifest.setdefault("level", level)

    parent_path_present = bool(manifest.get("parent_partition_path"))
    parent_hash_present = bool(manifest.get("parent_partition_sha256"))
    if parent_path_present != parent_hash_present:
        raise LevelManifestError(
            f"[{task}/{level}] refusing to complete a partial parent partition pin; "
            "path and sha256 must be created together")
    if parent_path_present:
        parent, parent_digest = _parent_partition(task, level)
    elif previous == "L0v2":
        versions = []
        for path in glob.glob(os.path.join(OUT, f"partition_{task}_L0v*.json")):
            m = re.search(r"_L0v(\d+)\.json$", path)
            if m:
                versions.append((int(m.group(1)), path))
        if not versions:
            raise FileNotFoundError(f"no L0vN parent exists for {task}/{level}")
        parent = max(versions)[1]
        parent_digest = _file_sha256(parent)
    else:
        parent = _canonical_partition_path(task, previous)
        parent_digest = _file_sha256(parent)

    names_path_present = bool(manifest.get("parent_names_path"))
    names_hash_present = bool(manifest.get("parent_names_sha256"))
    if names_path_present != names_hash_present:
        raise LevelManifestError(
            f"[{task}/{level}] refusing to complete a partial parent names pin; "
            "path and sha256 must be created together")
    if names_path_present:
        names, names_digest = _parent_names(task, level, parent)
    else:
        names = _default_parent_names(task, level, parent)
        names_digest = _file_sha256(names)

    _validate_upper_parent_material(task, level, parent, names)

    updated = dict(manifest)
    updated.update(parent_partition_path=os.path.relpath(parent, ROOT),
                   parent_partition_sha256=parent_digest,
                   parent_names_path=os.path.relpath(names, ROOT),
                   parent_names_sha256=names_digest)
    if updated != manifest:
        _atomic_json_write(manifest_path, updated)
    _validate_level_manifest(task, level, require_frozen_parent=True)


def _validate_level_manifest(task: str, level: str, *,
                             require_frozen_parent: bool = True) -> None:
    manifest_path = _manifest_path(task, level)
    if not os.path.exists(manifest_path):
        if require_frozen_parent:
            raise LevelManifestError(
                f"[{task}/{level}] no level manifest; freeze lineage before reading or building")
        return
    manifest = json.load(open(manifest_path))
    _validate_manifest_identity(manifest, task, level)
    missing = [field for field in _PARENT_MANIFEST_FIELDS if not manifest.get(field)]
    if require_frozen_parent and missing:
        raise LevelManifestError(
            f"[{task}/{level}] manifest does not fully freeze its parent; missing={missing}")
    parent, _ = _parent_partition(task, level)
    if require_frozen_parent or manifest.get("parent_names_path") or manifest.get("parent_names_sha256"):
        _parent_names(task, level, parent)
    eval_rel, eval_hash = manifest.get("eval_path"), manifest.get("eval_sha256")
    if eval_rel and eval_hash:
        eval_path = eval_rel if os.path.isabs(eval_rel) else os.path.join(ROOT, eval_rel)
        if _file_sha256(eval_path) != eval_hash:
            raise RuntimeError(f"[{task}/{level}] frozen eval changed on disk: {eval_path}")
    protocol_rel = manifest.get("verify_protocol_path")
    protocol_hash = manifest.get("verify_protocol_sha256")
    if protocol_rel and protocol_hash:
        protocol_path = (protocol_rel if os.path.isabs(protocol_rel)
                         else os.path.join(ROOT, protocol_rel))
        if _file_sha256(protocol_path) != protocol_hash:
            raise RuntimeError(f"[{task}/{level}] frozen verify protocol changed on disk: "
                               f"{protocol_path}")
    previous = PREV[level]
    if require_frozen_parent and previous in PREV:
        _validate_level_manifest(task, previous, require_frozen_parent=True)


def _update_level_manifest(task: str, level: str, **fields) -> None:
    path = _manifest_path(task, level)
    _validate_level_manifest(task, level, require_frozen_parent=True)
    manifest = json.load(open(path))
    parent, digest = _parent_partition(task, level)
    names, names_digest = _parent_names(task, level, parent)
    manifest.update(parent_partition_path=os.path.relpath(parent, ROOT),
                    parent_partition_sha256=digest,
                    parent_names_path=os.path.relpath(names, ROOT),
                    parent_names_sha256=names_digest, **fields)
    _atomic_json_write(path, manifest)


def _validate_semantic_nodes(task: str, level: str, nodes: List[dict]) -> None:
    def opaque(n):
        if str(n.get("gloss") or "").strip():
            return False
        name, node_id = str(n.get("name") or "").strip(), str(n["node_id"])
        return (not name or name == node_id or name.isdigit()
                or bool(re.fullmatch(r"[\w-]*_R[0-9]+_(?:g|solo_)?[\w-]+", name)))
    empty = [n["node_id"] for n in nodes if opaque(n)]
    if empty:
        raise ValueError(f"[{task}/{level}] {len(empty)} nodes have only a bare ID and cannot be "
                         f"LLM-judged semantically; name/backfill them first. sample={empty[:8]}")


def nodes_from_level(task: str, level: str) -> Tuple[List[dict], Dict[str, set]]:
    """Return (nodes, node->original-keys). nodes = [{node_id,name,gloss}] = the GROUPS of the
    previous level, named. Composes keys down to L0 rubric items for later census."""
    _validate_level_manifest(task, level, require_frozen_parent=True)
    prev = PREV[level]
    if prev == "L0v2":
        # Use the manifest-frozen parent when present; legacy builds retain their historical v3->v2
        # fallback. Starting a new build records the exact parent path and hash.
        # (star-1-round; never mints a new id), so cluster_names_<task>_L0v2.json already has a name
        # for every id that can appear in L0v3 -- no separate L0v3 names file is needed.
        l0_path, _ = _parent_partition(task, level)
        part = _load_partition(l0_path)   # key -> cluster
        # New tasks may receive their first complete naming pass only after the append-only L0v3
        # repair. Prefer names matching the frozen parent vintage, while retaining L0v2 fallback
        # for historical builds whose v3 reused already-named v2 cluster IDs.
        name_path, _ = _parent_names(task, level, l0_path)
        names = json.load(open(name_path))
        keys_of: Dict[str, set] = defaultdict(set)
        for k, c in part.items():
            keys_of[c].add(k)
    else:
        parent_path, _ = _parent_partition(task, level)
        part = _load_partition(parent_path)  # prevnode -> group
        name_path, _ = _parent_names(task, level, parent_path)
        names = json.load(open(name_path))
        # compose original keys through the earlier level's key map
        _, prev_keys = nodes_from_level(task, prev)
        keys_of = defaultdict(set)
        for pn, g in part.items():
            keys_of[g] |= prev_keys.get(pn, set())
    nodes = [{"node_id": g, "name": (names.get(g) or {}).get("name", g)[:90],
              "gloss": (names.get(g) or {}).get("gloss", "")}
             for g in sorted(keys_of)]
    return nodes, keys_of


def rep_text(n: dict) -> str:
    g = n.get("gloss") or ""
    return f"{n['name']}. {g}".strip()


def make_buckets(nodes: List[dict], bucket_size: int = 40) -> List[List[str]]:
    """NON-overlapping buckets (each node in exactly one) so the group-proposer runs once per bucket.
    Small n -> a single global bucket. Large n -> ~ceil(n/bucket_size) KMeans buckets over TF-IDF of
    the node reps, so similar (candidate-same-construct) nodes co-occur in a bucket."""
    ids = [n["node_id"] for n in nodes]
    if len(nodes) <= SINGLE_CALL_MAX:
        return [ids]
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import MiniBatchKMeans
    X = TfidfVectorizer(min_df=1, max_features=40000, sublinear_tf=True).fit_transform([rep_text(n) for n in nodes])
    k = max(2, -(-len(nodes) // bucket_size))
    labels = MiniBatchKMeans(n_clusters=k, random_state=0, n_init=3).fit_predict(X)
    buckets: Dict[int, List[str]] = defaultdict(list)
    for i, lab in enumerate(labels):
        buckets[int(lab)].append(ids[i])
    return [b for b in buckets.values() if b]


def emit_group_payloads(task: str, level: str, per_bucket: int = 1) -> int:
    _freeze_parent_for_new_build(task, level)
    nodes, _ = nodes_from_level(task, level)
    _validate_semantic_nodes(task, level, nodes)
    by_id = {n["node_id"]: n for n in nodes}
    buckets = make_buckets(nodes)
    pd = os.path.join(OUT, "level_payloads"); os.makedirs(pd, exist_ok=True)
    for f in glob.glob(os.path.join(pd, f"{task}_{level}_group_*.jsonl")):
        os.remove(f)
    n = 0
    for bi, b in enumerate(buckets):
        rows = [{"node_id": nid, "name": by_id[nid]["name"], "gloss": by_id[nid]["gloss"]} for nid in b]
        with open(os.path.join(pd, f"{task}_{level}_group_{bi:04d}.jsonl"), "w") as fh:
            fh.write(json.dumps({"bucket_id": bi, "nodes": rows}) + "\n")
        n += 1
    _update_level_manifest(task, level, n_nodes=len(nodes), n_buckets=n,
                           single_call=len(nodes) <= SINGLE_CALL_MAX)
    print(f"[{task}/{level}] {len(nodes)} nodes -> {n} group-proposer bucket(s) "
          f"({'single-call' if len(nodes) <= SINGLE_CALL_MAX else 'TF-IDF buckets'})")
    return n


def ingest_groups(task: str, level: str, *, output_path: str) -> Dict[str, str]:
    """Proposer outputs -> P_{level} (node_id -> group_id). Buckets are non-overlapping, so each
    node appears in exactly one bucket's groups -> one group. Unassigned nodes become singletons.
    This always writes an immutable candidate; canonical replacement is promotion-only."""
    destination = _candidate_output_path(task, level, output_path)
    nodes, _ = nodes_from_level(task, level)
    ids = [n["node_id"] for n in nodes]
    assign: Dict[str, str] = {}
    for f in sorted(glob.glob(os.path.join(OUT, "level_votes", f"{task}_{level}_group_*.jsonl"))):
        for line in open(f):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            bi = r.get("bucket_id")
            for gi, grp in enumerate(r.get("groups") or []):
                for nid in grp:
                    if nid in ids and nid not in assign:
                        assign[nid] = f"{task}_{level}_g{bi}_{gi}"
    for nid in ids:
        assign.setdefault(nid, f"{task}_{level}_solo_{nid}")
    _atomic_json_write(destination, assign)
    ng = len(set(assign.values()))
    print(f"[{task}/{level}] grouped {len(ids)} nodes -> {ng} {RELATIONS[level][0]} groups "
          f"(collapse {len(ids)}->{ng}, {100*(1-ng/len(ids)):.0f}%)")
    return assign


def emit_level_eval(task: str, level: str, n_pairs: int = 900, n_anchor: int = 0) -> str:
    """Frozen node-pair eval, stratified by TF-IDF sim (high-sim within-bucket + random cross)."""
    path = os.path.join(OUT, f"level_eval_{task}_{level}.jsonl")
    if os.path.exists(path):
        # Existing legacy evals may predate manifests; never silently rewrite them here.
        _validate_level_manifest(task, level)
        print(f"[{task}/{level}] eval frozen — not rebuilding"); return path
    _freeze_parent_for_new_build(task, level)
    nodes, _ = nodes_from_level(task, level)
    _validate_semantic_nodes(task, level, nodes)
    by_id = {n["node_id"]: n for n in nodes}
    ids = [n["node_id"] for n in nodes]
    max_pairs = len(ids) * (len(ids) - 1) // 2
    if n_pairs > max_pairs:
        raise ValueError(f"[{task}/{level}] requested {n_pairs} unique eval pairs but only "
                         f"{max_pairs} exist for {len(ids)} nodes")
    if len(ids) < 2:
        raise ValueError(f"[{task}/{level}] need at least two nodes for pair evaluation")
    reps = [rep_text(by_id[i]) for i in ids]
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors
    X = TfidfVectorizer(min_df=1, max_features=40000, sublinear_tf=True).fit_transform(reps)
    k = min(11, len(ids))
    nn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(X)
    dist, idx = nn.kneighbors(X)
    pool = {}

    def add(a, b, stratum, sim=None):
        if a == b:
            return
        pid = _h(*sorted((a, b)))[:16]
        if pid not in pool:
            pool[pid] = {"pair_id": pid, "task": task, "level": level, "stratum": stratum,
                         "node_a": a, "node_b": b, "tfidf_cos": sim,
                         "canonical_a": rep_text(by_id[a]), "canonical_b": rep_text(by_id[b])}
    for i in range(len(ids)):
        for jp in range(1, k):
            j = int(idx[i][jp]); sim = 1.0 - float(dist[i][jp])
            if sim >= 0.15:
                add(ids[i], ids[j], "highsim", round(sim, 3))
    rng = random.Random(0)
    hs = [p for p in pool.values() if p["stratum"] == "highsim"]
    rng.shuffle(hs); hs = hs[: n_pairs // 2]
    pool = {p["pair_id"]: p for p in hs}
    while len(pool) < n_pairs:
        a, b = ids[rng.randrange(len(ids))], ids[rng.randrange(len(ids))]
        add(a, b, "random")
    rows = sorted(pool.values(), key=lambda r: _h(r["pair_id"], "s"))
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[{task}/{level}] {len(rows)} node-pair eval -> {path}")
    _update_level_manifest(task, level, n_nodes=len(nodes), eval_path=os.path.relpath(path, ROOT),
                           eval_sha256=_file_sha256(path))
    return path


def emit_arbiter_payloads(task: str, level: str, per_agent: int = 130, n_anchor: int = 6) -> int:
    """2026-07-10 Codex fix (round 2): the heuristic-anchor duplication (round-1 fix) injected the
    SAME anchor pair_id, drawn from `rows` itself, into EVERY shard -- since harvesters key votes by
    pair_id, that made the "official" vote for those specific eval pairs an order-dependent
    last-write-wins artifact (a real correctness bug, not just redundant judging). No gold truth
    exists at this level yet (the arbiter IS what defines it), so a heuristic anchor here is weak
    value anyway. DROPPED per Codex's option 2 rather than fixed with a dedup pass -- simplicity over
    a heuristic-only QC signal. Emits the frozen eval pairs unmodified, chunked into shards."""
    _validate_level_manifest(task, level)
    rows = [json.loads(l) for l in open(os.path.join(OUT, f"level_eval_{task}_{level}.jsonl"))]
    pd = os.path.join(OUT, "level_arbiter"); os.makedirs(pd, exist_ok=True)
    # Delete only numeric arbiter shards. A rerun after verify emission must never destroy the
    # independently generated <task>_<level>_verify_*.jsonl payloads.
    for f in glob.glob(os.path.join(pd, f"{task}_{level}_[0-9][0-9][0-9].jsonl")):
        os.remove(f)
    n = 0
    for a in range(0, len(rows), per_agent):
        with open(os.path.join(pd, f"{task}_{level}_{n:03d}.jsonl"), "w") as fh:
            for r in rows[a:a + per_agent]:
                fh.write(json.dumps({"pair_id": r["pair_id"], "node_a": r["node_a"],
                                     "node_b": r["node_b"], "canonical_a": r["canonical_a"],
                                     "canonical_b": r["canonical_b"]}) + "\n")
        n += 1
    print(f"[{task}/{level}] {len(rows)} eval pairs -> {n} arbiter shards (no anchors -- see docstring)")
    return n


class IncompleteArbiterVotesError(RuntimeError):
    pass


def _binary_kappa(a: List[bool], b: List[bool]) -> float | None:
    if not a:
        return None
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else None


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    if n <= 0:
        return None
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return max(0.0, center - half), min(1.0, center + half)


def _strict_level_votes(paths: List[str]) -> tuple[Dict[str, bool], dict]:
    rows: Dict[str, List[int]] = defaultdict(list)
    malformed = 0
    for path in sorted(paths):
        for line in open(path):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1; continue
            if not isinstance(row, dict):
                malformed += 1; continue
            pid, score = row.get("pair_id"), row.get("score")
            if not isinstance(pid, str) or type(score) is not int or score not in (0, 1, 2):
                malformed += 1; continue
            rows[pid].append(score)
    duplicate = sorted(p for p, scores in rows.items() if len(scores) > 1)
    conflicting = sorted(p for p, scores in rows.items() if len(set(scores)) > 1)
    votes = {p: scores[0] == 2 for p, scores in rows.items() if len(scores) == 1}
    return votes, {"malformed": malformed, "duplicate_pair_ids": duplicate,
                   "conflicting_pair_ids": conflicting}


def score(task: str, level: str, *, require_complete: bool = True,
          partition_path: str | None = None,
          arbiter_vote_paths: List[str] | None = None) -> dict:
    """Matched fixed-mixture diagnostics versus arbiter-judged level truth.

    IMPORTANT: the frozen eval deliberately contains 50% representation neighbors and 50% random
    pairs.  The unweighted conditional rates below describe that evaluation mixture; they are not
    population recall/precision.  In particular, `precision` must not be presented as global
    partition precision.  Use ``upper_precision_audit`` for a uniform predicted-positive sample and
    independent replicated LLM judgments.  The historical keys remain for matched-run continuity
    and are explicitly scope-labeled in the returned artifact.

    ★ Why chance-correction is mandatory for CROSS-LEVEL comparison (2026-07-11): raw recall is NOT
    comparable across levels because partitions have wildly different granularity — R1 has hundreds of
    groups (null co-label rate p0~.02), R3 has 4-7 (p0~.25). A 4-group R3 partition trivially co-labels
    ~25% of ANY pairs, so its raw recall .75 is only ~3x its null, while R1's .48 is 16-37x its null.
    Comparing raw recall up-tree manufactured a spurious "U-shape". Report `recall_kappa` (chance-
    corrected, = (recall-p0)/(1-p0)) and `recall_lift` (=recall/p0) as PRIMARY; raw recall is granularity-
    contaminated. `p0` = sum_g C(s_g,2)/C(N,2) over the FULL partition = expected recall/precision of a
    RANDOM partition of the same group sizes. `precision_lift` = precision/base_rate (base_rate = share of
    scored eval pairs that are truth-SAME = precision's null). See ledger 2026-07-11 self-correction.
    `coverage` = voted eval pairs / total eval pairs. By default any missing vote or stale eval node
    raises ``IncompleteArbiterVotesError``; pass ``require_complete=False`` only for an explicitly
    provisional diagnostic.

    Historical compatibility: ``recall_kappa`` remains as a deprecated alias, but the quantity is
    now named ``chance_corrected_recall`` because it is NOT Cohen's kappa. A real binary Cohen kappa
    between arbiter-SAME and partition-co-label decisions is reported separately on the eval sample."""
    _validate_level_manifest(task, level)
    from math import comb
    from collections import Counter
    selected_partition_path = (partition_path or
                               os.path.join(OUT, f"partition_{task}_{level}.json"))
    part_all = _load_partition(selected_partition_path)
    current_nodes, _ = nodes_from_level(task, level)
    current_ids = {n["node_id"] for n in current_nodes}
    missing_partition_nodes = sorted(current_ids - set(part_all))
    extra_partition_nodes = sorted(set(part_all) - current_ids)
    if missing_partition_nodes:
        raise IncompleteArbiterVotesError(
            f"[{task}/{level}] partition misses {len(missing_partition_nodes)} current nodes; "
            f"sample={missing_partition_nodes[:5]}")
    part = {node: part_all[node] for node in current_ids}
    selected_vote_paths = (list(arbiter_vote_paths) if arbiter_vote_paths is not None else
                           glob.glob(os.path.join(
                               OUT, "level_votes", f"arb_{task}_{level}_[0-9]*.jsonl")))
    votes, vote_diag = _strict_level_votes(selected_vote_paths)
    ev = {r["pair_id"]: r for r in (json.loads(l) for l in open(os.path.join(OUT, f"level_eval_{task}_{level}.jsonl")))}
    missing_votes = sorted(set(ev) - set(votes))
    unexpected_votes = sorted(set(votes) - set(ev))
    stale_eval_pairs = sorted(pid for pid, r in ev.items()
                              if r["node_a"] not in current_ids or r["node_b"] not in current_ids)
    invalid_vote_files = (vote_diag["malformed"] or vote_diag["duplicate_pair_ids"]
                          or unexpected_votes)
    if require_complete and (missing_votes or stale_eval_pairs or invalid_vote_files):
        raise IncompleteArbiterVotesError(
            f"[{task}/{level}] incomplete eval: missing_votes={len(missing_votes)}, "
            f"stale_node_pairs={len(stale_eval_pairs)}, unexpected_votes={len(unexpected_votes)}, "
            f"malformed={vote_diag['malformed']}, duplicates={len(vote_diag['duplicate_pair_ids'])}; "
            f"pass require_complete=False only for a "
            f"provisional diagnostic")
    n_same = same_co = n_co = co_same = n_scored = 0
    truth_binary: List[bool] = []
    pred_binary: List[bool] = []
    by_stratum: Dict[str, List[tuple[bool, bool]]] = defaultdict(list)
    for pid, r in ev.items():
        if pid not in votes or pid in stale_eval_pairs:
            continue
        same = votes[pid]
        ca, cb = part[r["node_a"]], part[r["node_b"]]
        n_scored += 1
        co = ca == cb
        truth_binary.append(bool(same)); pred_binary.append(bool(co))
        by_stratum[str(r.get("stratum") or "unknown")].append((bool(same), bool(co)))
        if same:
            n_same += 1; same_co += int(co)
        if co:
            n_co += 1; co_same += int(same)
    recall = (same_co / n_same) if n_same else None
    precision = (co_same / n_co) if n_co else None
    # null pair-positive rate p0 over the FULL partition (granularity baseline)
    sizes = Counter(part.values()); N = sum(sizes.values())
    p0 = (sum(comb(s, 2) for s in sizes.values()) / comb(N, 2)) if N >= 2 else None
    base_rate = (n_same / n_scored) if n_scored else None  # precision's chance level
    chance_corrected = ((recall - p0) / (1 - p0)) if (recall is not None and p0 not in (None, 1)) else None
    recall_lift = (recall / p0) if (recall is not None and p0) else None
    precision_lift = (precision / base_rate) if (precision is not None and base_rate) else None
    # completeness: every eval pair should have a well-formed arbiter vote
    voted_in_eval = sum(1 for pid in ev if pid in votes)
    coverage = round(voted_in_eval / len(ev), 3) if ev else None
    scorable_coverage = round(n_scored / len(ev), 3) if ev else None
    rec_ci = _wilson(same_co, n_same)
    cc_ci = (((rec_ci[0] - p0) / (1 - p0), (rec_ci[1] - p0) / (1 - p0))
             if rec_ci is not None and p0 not in (None, 1) else None)
    strata_report = {}
    for stratum, pairs in sorted(by_stratum.items()):
        ns = sum(s for s, _ in pairs); nc = sum(c for _, c in pairs)
        tp = sum(s and c for s, c in pairs)
        strata_report[stratum] = {"n": len(pairs), "n_same": ns, "n_colabeled": nc,
                                   "recall": round(tp / ns, 3) if ns else None,
                                   "precision": round(tp / nc, 3) if nc else None}
    complete = not missing_votes and not stale_eval_pairs and not invalid_vote_files
    binary_kappa = _binary_kappa(truth_binary, pred_binary)
    return {"task": task, "level": level, "relation": RELATIONS[level][0],
            "partition_path": selected_partition_path,
            "partition_sha256": _file_sha256(selected_partition_path),
            "arbiter_vote_paths": [str(path) for path in selected_vote_paths],
            "arbiter_vote_sha256": [_file_sha256(path) for path in selected_vote_paths],
            "n_same": n_same, "recall": round(recall, 3) if recall is not None else None,
            "n_colabeled": n_co, "precision": round(precision, 3) if precision is not None else None,
            "recall_scope": "unweighted fixed 50% neighbor + 50% random evaluation mixture",
            "precision_scope": "unweighted fixed evaluation mixture; NOT global partition precision",
            "global_precision": None,
            "global_precision_required_artifact": "upper_precision_audit replicated LLM report",
            "n_groups": len(sizes), "p0": round(p0, 4) if p0 is not None else None,
            "chance_corrected_recall": round(chance_corrected, 3) if chance_corrected is not None else None,
            "chance_corrected_recall_ci95": ([round(x, 3) for x in cc_ci] if cc_ci else None),
            "recall_kappa": round(chance_corrected, 3) if chance_corrected is not None else None,
            "recall_kappa_deprecated_alias": True,
            "cohen_kappa_same_binary_eval": (round(binary_kappa, 3)
                                               if binary_kappa is not None else None),
            "recall_lift": round(recall_lift, 1) if recall_lift is not None else None,
            "precision_lift": round(precision_lift, 2) if precision_lift is not None else None,
            "n_eval": len(ev), "n_voted": voted_in_eval, "n_scored": n_scored,
            "vote_coverage": coverage, "scorable_coverage": scorable_coverage,
            "coverage": scorable_coverage, "complete": complete,
            "missing_vote_pair_ids": missing_votes, "unexpected_vote_pair_ids": unexpected_votes,
            "vote_file_diagnostics": vote_diag,
            "stale_eval_pair_ids": stale_eval_pairs,
            "partition_extra_node_ids": extra_partition_nodes,
            "by_eval_stratum": strata_report}


def emit_verify_net(task: str, level: str, k: int = 20, min_cos: float = 0.12,
                    cap: int = 9000, per_agent: int = 300, n_anchor: int = 6) -> dict:
    """PAIRWISE candidate net (replaces bucketed group-proposer; humor R1 measured: KMeans ceiling
    .30 vs global TF-IDF kNN ceiling .976). Global kNN over node reps (name+gloss), EXCLUDE the
    held-out eval pairs (2026-07-10 Codex fix: verify and arbiter are now the SAME Sonnet-5 family
    — the 2026-07-07 evaluator-independence decision retired Opus-as-arbiter for cost — so
    including eval pairs in the build net is correlated-error optimism, not independent
    measurement; the old inline comment assumed a stale Opus/Sonnet split), rank by cosine, take
    top `cap`, emit Sonnet-verify payloads at the level relation + blinded QC anchors drawn from
    arbiter-judged eval pairs. Anchor pair_ids are persisted (level_anchor_ids_<task>_<level>.json)
    so apply_pairwise can exclude them from ever contributing a partition edge, mirroring the L0
    screen/confirm anchor discipline. Also reports the top-band ceiling vs the eval-SAME pairs."""
    _validate_level_manifest(task, level)
    verify_protocol = os.path.join(OUT, f"STRICT_BUILD_PROTOCOL_{level}.txt")
    if os.path.exists(verify_protocol):
        _update_level_manifest(
            task, level,
            verify_protocol_path=os.path.relpath(verify_protocol, ROOT),
            verify_protocol_sha256=_file_sha256(verify_protocol))
    nodes, _ = nodes_from_level(task, level)
    _validate_semantic_nodes(task, level, nodes)
    ids = [n["node_id"] for n in nodes]
    by_id = {n["node_id"]: n for n in nodes}
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors
    X = TfidfVectorizer(min_df=1, max_features=40000, sublinear_tf=True).fit_transform([rep_text(n) for n in nodes])
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(ids)), metric="cosine").fit(X)
    dist, idx = nn.kneighbors(X)
    evalp = set()
    same_eval = set()
    eval_by_pair = {}
    ep = os.path.join(OUT, f"level_eval_{task}_{level}.jsonl")
    votes = {}
    for f in glob.glob(os.path.join(OUT, "level_votes", f"arb_{task}_{level}_[0-9]*.jsonl")):
        for line in open(f):
            if line.strip():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sc = r.get("score")
                # strict int (2026-07-10 Codex round-5, final hole): unvalidated here meant a float
                # 2.0 or bool could be SELECTED into same2/same0 (anchor candidates) below via bare
                # `==`, then get silently dropped when apply_pairwise re-loads arb labels with its
                # own strict-int check -- an anchor picked but never validly labeled. Strict-int end
                # to end so selection and validation always agree.
                if type(sc) is int and sc in (0, 1, 2):
                    votes[r["pair_id"]] = sc
    if os.path.exists(ep):
        for line in open(ep):
            r = json.loads(line)
            fs = frozenset((r["node_a"], r["node_b"]))
            evalp.add(fs)
            eval_by_pair[r["pair_id"]] = r
            if votes.get(r["pair_id"]) == 2:
                same_eval.add(fs)
    cand = {}
    for i in range(len(ids)):
        for jp in range(1, idx.shape[1]):
            j = int(idx[i][jp])
            # Exact duplicate vectors can tie at distance zero and displace the query row from
            # neighbor position 0.  Never assume jp>=1 implies a distinct node.
            if j == i:
                continue
            cos = 1.0 - float(dist[i][jp])
            if cos >= min_cos:
                fs = frozenset((ids[i], ids[j]))
                if fs in evalp:
                    continue  # EXCLUDE eval pairs from the build net -- see docstring
                cand[fs] = max(cand.get(fs, 0.0), cos)
    ranked = sorted(cand.items(), key=lambda kv: -kv[1])[:cap]
    pd = os.path.join(OUT, "level_arbiter"); os.makedirs(pd, exist_ok=True)
    for f in glob.glob(os.path.join(pd, f"{task}_{level}_verify_*.jsonl")):
        os.remove(f)
    rows = []
    for fs, cos in ranked:
        a, b = sorted(fs)
        rows.append({"pair_id": _h(a, b)[:16], "node_a": a, "node_b": b,
                     "canonical_a": rep_text(by_id[a]), "canonical_b": rep_text(by_id[b])})
    # blinded QC anchors: a few clear score==2 + score==0 arbiter-judged pairs, ride-along in every
    # shard but excluded from apply_pairwise's edge set so they can never affect the partition even
    # if a verify judge scores one SAME.
    import random
    same2 = [p for p, s in votes.items() if s == 2 and p in eval_by_pair]
    same0 = [p for p, s in votes.items() if s == 0 and p in eval_by_pair]
    # When a replicated truth re-audit exists, gate on the high-confidence intersection rather
    # than randomly selecting borderline median decisions.  The construction judge must reproduce
    # pairs that BOTH independent truth judges called SAME (or DIFFERENT).  This remains independent
    # of construction votes and avoids letting 2-vs-1 adjudication disputes dominate a six-anchor
    # calibration gate.
    replicate_root = os.path.join(OUT, "r1_truth_reaudit")
    replicate_paths = [os.path.join(replicate_root, side, f"{task}_{level}.jsonl")
                       for side in ("votes_a", "votes_b")]
    anchor_source = "final arbiter score"
    if all(os.path.exists(path) for path in replicate_paths):
        replicate_votes = []
        for path in replicate_paths:
            current = {}
            for line in open(path):
                if not line.strip():
                    continue
                row = json.loads(line); score = row.get("score")
                if (set(row) == {"pair_id", "score"} and type(score) is int
                        and score in (0, 1, 2) and row["pair_id"] not in current):
                    current[row["pair_id"]] = score
            replicate_votes.append(current)
        expected_eval = set(eval_by_pair)
        if all(set(current) == expected_eval for current in replicate_votes):
            a_votes, b_votes = replicate_votes
            same2 = [p for p in eval_by_pair if a_votes[p] == 2 and b_votes[p] == 2]
            same0 = [p for p in eval_by_pair if a_votes[p] == 0 and b_votes[p] == 0]
            anchor_source = "unanimous intersection of two independent truth re-audit judges"
    rng = random.Random(0)
    rng.shuffle(same2); rng.shuffle(same0)
    anchor_pids = (same2[:max(1, n_anchor // 2)] + same0[:max(1, n_anchor // 2)])[:n_anchor]
    anchor_rows = [{"pair_id": p, "node_a": eval_by_pair[p]["node_a"], "node_b": eval_by_pair[p]["node_b"],
                    "canonical_a": eval_by_pair[p]["canonical_a"], "canonical_b": eval_by_pair[p]["canonical_b"]}
                   for p in anchor_pids]
    anchor_path = os.path.join(OUT, f"level_anchor_ids_{task}_{level}.json")
    n = 0
    if rows:
        # 2026-07-10 Codex round-4 fix: only persist anchor_ids when anchors are actually DISPATCHED.
        # The old unconditional dump wrote a non-empty anchor_ids file even when `rows` is empty (a
        # genuinely empty net -- zero candidates, a valid "no build evidence" result) -- the shard
        # loop below never runs for an empty `rows`, so no shard ever carries those anchors, yet
        # apply_pairwise still required votes for every persisted id and would falsely raise
        # AnchorGateFailure on a clean no-merge case. If a stale anchor_ids file exists from a
        # previous (non-empty) run, remove it so an empty rebuild doesn't inherit orphaned ids.
        json.dump(anchor_pids, open(anchor_path, "w"))
        for x in range(0, len(rows), per_agent):
            chunk = rows[x:x + per_agent] + anchor_rows
            rng.shuffle(chunk)
            with open(os.path.join(pd, f"{task}_{level}_verify_{n:03d}.jsonl"), "w") as fh:
                for r in chunk:
                    fh.write(json.dumps(r) + "\n")
            n += 1
    elif os.path.exists(anchor_path):
        os.remove(anchor_path)
    # top-band ceiling vs held-out eval-SAME: 2026-07-10 Codex fix (round 2) -- since `cand`/`netset`
    # now correctly EXCLUDE eval pairs (fix round 1), same_eval subset of evalp can never intersect
    # netset by construction -> the old ceiling formula always reads 0.0 (dead gate, not a real
    # measurement). Fix the DIAGNOSTIC, not the build: recompute a SEPARATE, diagnostic-only
    # candidate ranking (`cand_diag`) from the SAME raw kNN structure but WITHOUT the evalp
    # exclusion, and ask whether each held-out eval-SAME pair WOULD have ranked in the top `cap` band
    # had it been a normal candidate. This never touches `cand`/`ranked`/`rows` (the actual build).
    cand_diag = {}
    for i in range(len(ids)):
        for jp in range(1, idx.shape[1]):
            j = int(idx[i][jp])
            if j == i:
                continue
            cos = 1.0 - float(dist[i][jp])
            if cos >= min_cos:
                fs = frozenset((ids[i], ids[j]))
                cand_diag[fs] = max(cand_diag.get(fs, 0.0), cos)
    netset_diag = {fs for fs, _ in sorted(cand_diag.items(), key=lambda kv: -kv[1])[:cap]}
    ceil = (sum(1 for fs in same_eval if fs in netset_diag) / len(same_eval)) if same_eval else None
    res = {"task": task, "level": level, "n_nodes": len(ids), "net": len(cand),
           "band": len(ranked), "shards": n, "n_anchor": len(anchor_pids),
           "anchor_source": anchor_source,
           "eval_same_excluded": len(same_eval),
           "topband_ceiling_vs_eval_same": round(ceil, 3) if ceil is not None else None}
    print(json.dumps(res))
    return res


class IncompleteVotesError(RuntimeError):
    pass


class AnchorGateFailure(RuntimeError):
    pass


def apply_pairwise(task: str, level: str, pos_gate: float = 0.8, neg_gate: float = 0.9,
                   exclude_from_gate=None, *, resolution: float = 1.0,
                   related_weight: float = 0.0,
                   output_path: str) -> dict:
    """Verified node pairs -> weighted LOUVAIN community detection over the node graph.

    Score-2 SAME pairs always have weight 1.  By default score-1 RELATED pairs are absent, exactly
    preserving the historical hard-edge build.  A positive ``related_weight`` admits score-1
    judgments only as weaker community-structure evidence; it does not relabel them SAME.  This is
    useful when a sparse hard-edge graph cannot recover held-out members of otherwise coherent
    constructs.  Any such candidate still has to pass the independent global LLM precision audit
    and whole-group certification gates. ``output_path`` is required and must be an immutable,
    non-canonical candidate name. Canonical replacement is available only through
    :func:`promote_partition`.

    exclude_from_gate: optional set of anchor pair_ids to drop from the GATE's accuracy/coverage
    math only (e.g. a disputed gold label, confirmed by independent multi-judge agreement to be a
    bad anchor pick, not a degraded verifier). These pids stay in `anchor_ids` and are therefore
    STILL never allowed to contribute a build edge -- excluding a pair from the gate must never
    let it leak into the partition, or it reintroduces exactly the eval/build circularity the
    round-1 fix exists to prevent. Document every use of this in the ledger with the reasoning.

    2026-07-10 Codex fixes (round 2, on top of round 1's fail-closed Louvain + anchor exclusion):
    (1) COMPLETENESS CHECK (fail-closed hole): round 1 excluded anchor votes but never verified
        every REAL candidate pair actually got voted on. No files / all-empty / all-malformed /
        all-non-SAME every produce the identical same=[] -> Louvain would silently write an
        ALL-SINGLETON partition and report it as a normal, successful result. Now: before touching
        Louvain, every non-anchor pair_id emitted into the verify net (`pay` minus `anchor_ids`)
        must have at least one well-formed vote on disk (by pair_id coverage, not shard-name count,
        malformed/unscored lines don't count) -- else raise IncompleteVotesError and refuse to
        freeze a partition off incomplete data.
    (2) ANCHOR GATE IS LIVE (round 1 only counted+discarded anchor votes -- a verifier returning
        all-2 would have passed silently). Every vote cast for a KNOWN anchor pair_id (drawn from
        arbiter score==2 / score==0 eval pairs, see emit_verify_net) is now compared against its
        known label: positive (expected SAME) accuracy must be >= pos_gate (default .8, mirrors the
        L0 anchor gate); negative (expected UNRELATED) accuracy -- fraction NOT scored 2 -- must be
        >= neg_gate (default .9, "clean": false-merges on a known-unrelated pair are the worse
        error). Below either threshold -> raise AnchorGateFailure. Anchor votes never contribute an
        edge regardless of gate outcome.
    (3) Div-by-zero guard if a level ever has 0 nodes.
    """
    destination = _candidate_output_path(task, level, output_path)
    _validate_level_manifest(task, level, require_frozen_parent=True)
    nodes, _ = nodes_from_level(task, level)
    ids = [n["node_id"] for n in nodes]
    pay = {}
    for f in glob.glob(os.path.join(OUT, "level_arbiter", f"{task}_{level}_verify_*.jsonl")):
        for line in open(f):
            if line.strip():
                r = json.loads(line); pay[r["pair_id"]] = (r["node_a"], r["node_b"])
    anchor_path = os.path.join(OUT, f"level_anchor_ids_{task}_{level}.json")
    anchor_ids = set(json.load(open(anchor_path))) if os.path.exists(anchor_path) else set()
    arb = {}
    for f in glob.glob(os.path.join(OUT, "level_votes", f"arb_{task}_{level}_[0-9]*.jsonl")):
        for line in open(f):
            if line.strip():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sc = r.get("score")
                # strict int (2026-07-10 Codex round-4): an unvalidated label load meant a
                # missing/malformed arb score for every anchor silently produced has_pos_label=
                # has_neg_label=False below -> the WHOLE anchor gate no-ops (pos_acc/neg_acc stay
                # None and neither raise branch fires) -- the gate ran OPEN. type(sc) is int also
                # blocks the bool/float equality quirk from manufacturing a fake label.
                if type(sc) is int and sc in (0, 1, 2):
                    arb[r["pair_id"]] = sc
    anchor_label = {p: arb.get(p) for p in anchor_ids}  # 2 = expect SAME, 0 = expect UNRELATED
    exclude_from_gate = set(exclude_from_gate or ())
    gate_anchor_ids = anchor_ids - exclude_from_gate  # anchor_ids (full) still used for build exclusion below
    has_pos_label = any(anchor_label.get(p) == 2 for p in gate_anchor_ids)
    has_neg_label = any(anchor_label.get(p) == 0 for p in gate_anchor_ids)
    if gate_anchor_ids and not has_pos_label and not has_neg_label:
        raise AnchorGateFailure(
            f"[{task}/{level}] apply_pairwise: {len(gate_anchor_ids)} gate-relevant anchor pair_ids "
            f"exist but NONE have a valid (strict-int, score 0 or 2) arbiter label -- cannot "
            f"establish gate ground truth. This is a setup failure (corrupt/missing arb_* labels "
            f"for the anchors), not an absence of anchors -- refusing to run the gate open.")

    expected = set(pay.keys()) - anchor_ids
    voted_pids = set()
    same = []
    anchor_votes: Dict[str, List[int]] = defaultdict(list)
    n_malformed = 0
    for f in glob.glob(os.path.join(OUT, "level_votes", f"vrf_{task}_{level}_*.jsonl")):
        for line in open(f):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                n_malformed += 1
                continue
            pid = r.get("pair_id")
            sc = r.get("score")
            # STRICT int check (2026-07-10 Codex round-3): Python's `==` makes True==1, False==0,
            # 2.0==2, so `sc in (0,1,2)` alone silently accepts bool/float scores as valid votes --
            # `{"score": true}` or `{"score": 2.0}` would count as complete AND 2.0 could add a real
            # merge edge. `type(sc) is int` rejects bool (bool is an int subclass but type() != int)
            # and float outright.
            if pid is None or type(sc) is not int or sc not in (0, 1, 2):
                n_malformed += 1
                continue
            if pid in anchor_ids:
                anchor_votes[pid].append(sc)
                continue  # blinded QC anchor -- never a partition edge
            voted_pids.add(pid)
            if sc == 2 and pid in pay:
                same.append(pay[pid])

    missing = expected - voted_pids
    if missing:
        raise IncompleteVotesError(
            f"[{task}/{level}] apply_pairwise: {len(missing)}/{len(expected)} verify-net pairs have "
            f"NO well-formed vote on disk ({n_malformed} malformed lines seen) -- refusing to freeze "
            f"a partition off incomplete votes. Sample missing pair_ids: {sorted(missing)[:5]}")

    # ANCHOR COVERAGE (2026-07-10 Codex round-3 fix): round-2's gate went FAIL-OPEN if the verify
    # fleet simply omitted every anchor vote -- expected was built as `pay.keys() - anchor_ids`, so
    # missing anchor votes never tripped the completeness check either; pos_acc/neg_acc both come
    # back None and NEITHER gate branch below fires. Assert coverage FIRST: every anchor pair_id
    # must have >=1 well-formed vote, and every labeled class that actually exists among the anchors
    # (some score==2 label and/or some score==0 label) must produce a non-empty accuracy -- None is
    # now a hard failure, not a silent pass.
    if gate_anchor_ids:
        uncovered = sorted(p for p in gate_anchor_ids if not anchor_votes.get(p))
        if uncovered:
            raise AnchorGateFailure(
                f"[{task}/{level}] apply_pairwise: {len(uncovered)}/{len(gate_anchor_ids)} "
                f"gate-relevant anchor pair_ids got ZERO well-formed votes (verify fleet silently "
                f"skipped/malformed them) -- fail-closed. Sample: {uncovered[:5]}")

    pos_votes = [s for p, votes in anchor_votes.items() if p in gate_anchor_ids and anchor_label.get(p) == 2 for s in votes]
    neg_votes = [s for p, votes in anchor_votes.items() if p in gate_anchor_ids and anchor_label.get(p) == 0 for s in votes]
    pos_acc = (sum(1 for s in pos_votes if s == 2) / len(pos_votes)) if pos_votes else None
    neg_acc = (sum(1 for s in neg_votes if s != 2) / len(neg_votes)) if neg_votes else None
    # has_pos_label/has_neg_label computed earlier (right after anchor_label) -- reused here.
    if has_pos_label and pos_acc is None:
        raise AnchorGateFailure(f"[{task}/{level}] apply_pairwise: positive-labeled anchors exist "
                                 f"but pos_acc is None (no scorable votes) -- fail-closed, not open.")
    if has_neg_label and neg_acc is None:
        raise AnchorGateFailure(f"[{task}/{level}] apply_pairwise: negative-labeled anchors exist "
                                 f"but neg_acc is None (no scorable votes) -- fail-closed, not open.")
    if pos_acc is not None and pos_acc < pos_gate:
        raise AnchorGateFailure(
            f"[{task}/{level}] apply_pairwise: POSITIVE anchor accuracy {pos_acc:.2f} < gate "
            f"{pos_gate} ({len(pos_votes)} votes) -- verify fleet looks degraded, refusing to apply.")
    if neg_acc is not None and neg_acc < neg_gate:
        raise AnchorGateFailure(
            f"[{task}/{level}] apply_pairwise: NEGATIVE anchor accuracy {neg_acc:.2f} < gate "
            f"{neg_gate} ({len(neg_votes)} votes) -- verify fleet looks degraded, refusing to apply.")

    # LOUVAIN community detection on the verified-SAME graph. Connected-components chains true
    # constructs together through a few wrong bridge edges (humor R1 13-shard: CC recall .53 but
    # precision .43); star-1-round under-groups (.16/.87); triangle-filter over-filters (.13/.85).
    # Louvain finds DENSE communities and won't merge two dense clusters joined by one bad edge
    # (.45/.80) — the recall/precision balance. NO fallback: raise if this ever fails.
    if resolution <= 0:
        raise ValueError("Louvain resolution must be positive")
    if (isinstance(related_weight, bool) or not isinstance(related_weight, (int, float))
            or not math.isfinite(related_weight) or not 0 <= related_weight < 1):
        raise ValueError("related_weight must be finite and in [0, 1)")
    import networkx as nx
    from networkx.algorithms.community import louvain_communities
    G = nx.Graph(); G.add_nodes_from(ids)
    related = []
    if related_weight:
        # Re-read only already validated, complete, non-anchor candidate votes.  Weak edges go in
        # first so an accidental duplicate score-2 occurrence deterministically wins at weight 1.
        for f in glob.glob(os.path.join(OUT, "level_votes", f"vrf_{task}_{level}_*.jsonl")):
            for line in open(f):
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                pid, sc = r.get("pair_id"), r.get("score")
                if pid not in anchor_ids and type(sc) is int and sc == 1 and pid in pay:
                    related.append(pay[pid])
        for a, b in related:
            G.add_edge(a, b, weight=float(related_weight))
    for a, b in same:
        G.add_edge(a, b, weight=1.0)
    group = {}
    for i, comm in enumerate(louvain_communities(
            G, seed=0, resolution=resolution, weight="weight")):
        for nd in comm:
            group[nd] = f"{task}_{level}_g{i}"
    _atomic_json_write(destination, group)
    ng = len(set(group.values()))
    collapse = round(1 - ng / len(ids), 3) if ids else None
    pct = f"{100 * (1 - ng / len(ids)):.0f}%" if ids else "n/a"
    print(f"[{task}/{level}] pairwise Louvain: {len(same)} SAME edges + {len(related)} RELATED "
          f"edges @ {related_weight} ({sum(len(v) for v in anchor_votes.values())} "
          f"anchor votes excluded, pos_acc={pos_acc}, neg_acc={neg_acc}) -> {len(ids)} nodes into {ng} "
          f"{RELATIONS[level][0]} groups (collapse {pct})")
    return {"n_edges": len(same), "n_related_edges": len(related),
            "related_weight": related_weight, "n_groups": ng, "collapse": collapse,
            "resolution": resolution, "partition_path": destination,
            "anchor_pos_acc": round(pos_acc, 3) if pos_acc is not None else None,
            "anchor_neg_acc": round(neg_acc, 3) if neg_acc is not None else None}


def promote_partition(task: str, level: str, candidate_path: str, *,
                      replace: bool = False) -> dict:
    """Validate and atomically promote one immutable candidate to the canonical partition.

    Replacing a different canonical partition requires ``replace=True`` and automatically banks the
    old file under a content-addressed ``precanon`` name. Direct builders cannot use this path.
    """
    _validate_level_manifest(task, level, require_frozen_parent=True)
    candidate_path = _candidate_output_path(task, level, candidate_path)
    if not os.path.isfile(candidate_path):
        raise FileNotFoundError(candidate_path)
    candidate = _load_partition(candidate_path)
    nodes, _ = nodes_from_level(task, level)
    expected = {n["node_id"] for n in nodes}
    missing = sorted(expected - set(candidate))
    extra = sorted(set(candidate) - expected)
    if missing or extra:
        raise ValueError(
            f"[{task}/{level}] candidate node inventory mismatch: missing={len(missing)}, "
            f"extra={len(extra)}, missing_sample={missing[:5]}, extra_sample={extra[:5]}")
    if not all(isinstance(group_id, str) and group_id.strip() for group_id in candidate.values()):
        raise ValueError(f"[{task}/{level}] every promoted node needs a non-empty string group id")

    canonical = _canonical_partition_path(task, level)
    candidate_digest = _file_sha256(candidate_path)
    prior_digest = _file_sha256(canonical) if os.path.exists(canonical) else None
    prior = _load_partition(canonical) if prior_digest else None
    replacing_different_content = prior is not None and prior != candidate
    backup_path = None
    if replacing_different_content:
        if not replace:
            raise FileExistsError(
                f"[{task}/{level}] canonical partition already exists with different content; "
                "pass replace=True/--replace-canonical for an intentional, backed-up promotion")
        backup_path = os.path.join(
            OUT, f"partition_{task}_{level}_precanon_{prior_digest[:12]}.json")
        if not os.path.exists(backup_path):
            _atomic_json_write(backup_path, prior)

    _atomic_json_write(canonical, candidate)
    canonical_digest = _file_sha256(canonical)
    _update_level_manifest(
        task, level,
        canonical_partition_path=os.path.relpath(canonical, ROOT),
        canonical_partition_sha256=canonical_digest,
        promoted_from_path=os.path.relpath(os.path.abspath(candidate_path), ROOT),
        promoted_from_sha256=candidate_digest,
        previous_canonical_partition_sha256=prior_digest,
        previous_canonical_backup_path=(os.path.relpath(backup_path, ROOT)
                                            if backup_path else None),
    )
    return {"task": task, "level": level, "candidate_path": candidate_path,
            "candidate_sha256": candidate_digest, "canonical_path": canonical,
            "canonical_sha256": canonical_digest, "replaced": replacing_different_content,
            "backup_path": backup_path}


def bucket_ceiling(task: str, level: str) -> dict:
    """Does the (KMeans/TF-IDF) BATCHING cap recall? Of the arbiter-SAME node pairs, what fraction
    share a bucket (only those CAN be grouped)? ceiling = P(same-bucket | arbiter-SAME). Independent
    of the group-proposer: low ceiling => upgrade the batcher (bigger/semantic/overlapping)."""
    node_bucket = {}
    for f in sorted(glob.glob(os.path.join(OUT, "level_payloads", f"{task}_{level}_group_*.jsonl"))):
        try:
            r = json.loads(open(f).read().strip().splitlines()[0])
        except Exception:
            continue
        for nd in r.get("nodes", []):
            node_bucket[nd["node_id"]] = r.get("bucket_id")
    votes = {}
    for f in glob.glob(os.path.join(OUT, "level_votes", f"arb_{task}_{level}_*.jsonl")):
        for line in open(f):
            if line.strip():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sc = r.get("score")
                if type(sc) is int and sc in (0, 1, 2):  # strict int, not isinstance (bool quirk)
                    votes[r["pair_id"]] = (sc == 2)
    ev = {r["pair_id"]: r for r in (json.loads(l) for l in open(os.path.join(OUT, f"level_eval_{task}_{level}.jsonl")))}
    same = [ev[p] for p, s in votes.items() if s and p in ev]
    in_bucket = sum(1 for r in same if node_bucket.get(r["node_a"]) is not None
                    and node_bucket.get(r["node_a"]) == node_bucket.get(r["node_b"]))
    n_nodes = len(node_bucket)
    n_buckets = len(set(node_bucket.values()))
    return {"task": task, "level": level, "n_nodes": n_nodes, "n_buckets": n_buckets,
            "n_arbiter_same": len(same), "same_in_bucket": in_bucket,
            "bucket_ceiling": round(in_bucket / len(same), 3) if same else None}


def write_protocols():
    """Per-level proposer + arbiter protocol files (analogous to JUDGE_PROTOCOL.txt)."""
    for lvl, (rel, guid) in RELATIONS.items():
        gp = (f"You organize a domain's evaluation concepts into groups that share the SAME "
              f"higher-level meaning. {guid}\n\nYou are given a list of nodes, each a {{node_id, "
              f"name, gloss}}. Partition them into groups where every node in a group has the "
              f"'{rel}' relation to the others. A node may be alone. Merge only genuine "
              f"'{rel}' matches — do NOT force unrelated nodes together.\n\nReturn ONE JSON object: "
              f'{{"groups": [["<node_id>", "<node_id>", ...], ...]}} covering EVERY node_id exactly once.')
        open(os.path.join(OUT, f"GROUP_PROTOCOL_{lvl}.txt"), "w").write(gp)
        ap = (f"You compare two domain concepts (each a NAME plus a one-sentence gloss). Score how "
              f"much they have the '{rel}' relation: 0, 1, or 2.\n{guid}\n\n2 = clearly '{rel}'. "
              f"1 = related but NOT '{rel}' (a different {rel.split()[-1]}). 0 = unrelated.\n"
              f"Surface wording is irrelevant. Respond ONE JSON object: "
              f'{{"reasoning": "<1 sentence>", "score": 0|1|2}}.')
        open(os.path.join(OUT, f"ARBITER_PROTOCOL_{lvl}.txt"), "w").write(ap)
    print("wrote GROUP_PROTOCOL_{R1,R2,R3}.txt + ARBITER_PROTOCOL_{R1,R2,R3}.txt")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["protocols", "lineage-freeze", "group-emit", "group-ingest",
                                    "eval-emit", "arb-emit", "pairwise-apply",
                                    "partition-promote", "score"])
    ap.add_argument("--task"); ap.add_argument("--level", default="R1")
    ap.add_argument("--resolution", type=float, default=1.0)
    ap.add_argument("--related-weight", type=float, default=0.0)
    ap.add_argument("--output-path")
    ap.add_argument("--partition-path")
    ap.add_argument("--replace-canonical", action="store_true",
                    help="partition-promote only: bank and replace a different canonical file")
    ap.add_argument("--arbiter-vote-path", action="append",
                    help="explicit isolated truth-vote JSONL (repeatable); bypasses canonical glob")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="diagnostic only: report a score despite missing votes/stale eval nodes")
    a = ap.parse_args()
    if a.cmd == "protocols":
        write_protocols()
    elif a.cmd == "lineage-freeze":
        _freeze_parent_for_new_build(a.task, a.level)
        print(json.dumps({"task": a.task, "level": a.level,
                          "manifest_path": _manifest_path(a.task, a.level)}, indent=1))
    elif a.cmd == "group-emit":
        emit_group_payloads(a.task, a.level)
    elif a.cmd == "group-ingest":
        if not a.output_path:
            ap.error("group-ingest requires --output-path with a non-canonical candidate name")
        ingest_groups(a.task, a.level, output_path=a.output_path)
    elif a.cmd == "eval-emit":
        emit_level_eval(a.task, a.level)
    elif a.cmd == "arb-emit":
        emit_arbiter_payloads(a.task, a.level)
    elif a.cmd == "pairwise-apply":
        if not a.output_path:
            ap.error("pairwise-apply requires --output-path with a non-canonical candidate name")
        print(json.dumps(apply_pairwise(a.task, a.level, resolution=a.resolution,
                                        related_weight=a.related_weight,
                                        output_path=a.output_path), indent=1))
    elif a.cmd == "partition-promote":
        if not a.partition_path:
            ap.error("partition-promote requires --partition-path pointing to a candidate")
        print(json.dumps(promote_partition(a.task, a.level, a.partition_path,
                                           replace=a.replace_canonical), indent=1))
    elif a.cmd == "score":
        print(json.dumps(score(a.task, a.level, require_complete=not a.allow_incomplete,
                               partition_path=a.partition_path,
                               arbiter_vote_paths=a.arbiter_vote_path), indent=1))
