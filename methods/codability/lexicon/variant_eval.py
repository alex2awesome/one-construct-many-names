"""Matched, versioned LLM evaluation for non-canonical hierarchy variants.

Pair selection is deterministic bookkeeping (half representation-neighbor, half random). Semantic
truth is supplied only by LLM scores. The same machinery can evaluate R1/R2/R3 without changing
measurement method at R3.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict

from .build_level import OUT, _file_sha256
from .judge import canon_map


ROOT = Path(OUT) / "variant_eval"


def _rep(row: dict) -> str:
    bits = [row.get("name") or "", row.get("gloss") or ""]
    bits.extend((row.get("member_examples") or [])[:4])
    return " | ".join(str(x).strip() for x in bits if str(x).strip())


def prepare_l0_nodes(task: str, tag: str, l0_partition_path: str,
                     l0_names_path: str) -> str:
    """Represent active L0 clusters as R1-evaluation nodes without semantic inference."""
    ROOT.mkdir(parents=True, exist_ok=True)
    raw = json.loads(Path(l0_partition_path).read_text())
    part = {str(k): str(v) for k, v in raw.get("partition", raw).items()}
    names = json.loads(Path(l0_names_path).read_text())
    members: Dict[str, list[str]] = defaultdict(list)
    for key, cluster in part.items():
        members[cluster].append(key)
    cmap = canon_map(task)
    path = ROOT / f"{task}_{tag}_R1_nodes.jsonl"
    with path.open("w") as out:
        for cluster in sorted(members):
            row = names.get(cluster) or {}
            examples = [cmap[k] for k in sorted(members[cluster]) if k in cmap][:4]
            out.write(json.dumps({"node_id": cluster, "name": row.get("name") or cluster,
                                  "gloss": row.get("gloss") or "",
                                  "member_examples": examples}, ensure_ascii=False) + "\n")
    return str(path)


def emit(task: str, tag: str, level: str, nodes_path: str, partition_path: str,
         protocol_path: str, n_pairs: int = 900, per_agent: int = 150) -> dict:
    ROOT.mkdir(parents=True, exist_ok=True)
    nodes = {row["node_id"]: row for row in
             (json.loads(x) for x in Path(nodes_path).read_text().splitlines()) if row.get("node_id")}
    raw = json.loads(Path(partition_path).read_text())
    partition = {str(k): str(v) for k, v in
                 (raw.get("assignment") or raw.get("partition") or raw).items()}
    if set(nodes) != set(partition):
        raise ValueError(f"[{task}/{tag}/{level}] node/partition coverage mismatch")
    ids = sorted(nodes)
    max_pairs = len(ids) * (len(ids) - 1) // 2
    n_pairs = min(n_pairs, max_pairs)
    reps = [_rep(nodes[x]) for x in ids]
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors
    X = TfidfVectorizer(min_df=1, max_features=40000, sublinear_tf=True).fit_transform(reps)
    k = min(11, len(ids))
    dist, idx = NearestNeighbors(n_neighbors=k, metric="cosine").fit(X).kneighbors(X)
    pool = {}

    def add(a: str, b: str, stratum: str, sim=None) -> None:
        if a == b:
            return
        pair = tuple(sorted((a, b)))
        if pair not in pool:
            pid = hashlib.sha1(f"variant-eval||{task}||{tag}||{level}||{pair[0]}||{pair[1]}".encode()).hexdigest()[:16]
            pool[pair] = {"pair_id": pid, "task": task, "tag": tag, "level": level,
                          "stratum": stratum, "node_a": pair[0], "node_b": pair[1],
                          "concept_a": _rep(nodes[pair[0]]), "concept_b": _rep(nodes[pair[1]]),
                          "tfidf_cos": sim}
    for i in range(len(ids)):
        for jp in range(1, k):
            j = int(idx[i][jp])
            if j == i:
                continue
            sim = 1.0 - float(dist[i][jp])
            # Keep the same fixed-k neighbor stratum at every level. A cosine cutoff would make
            # small/diffuse R3 inventories silently receive fewer neighbor pairs than R1/R2 and
            # reintroduce the measurement-method change this battery is designed to remove.
            add(ids[i], ids[j], "neighbor", round(sim, 3))
    high = sorted(pool.values(), key=lambda r: hashlib.sha256(
        f"{task}||{tag}||{level}||neighbor||{r['pair_id']}".encode()).hexdigest())[:n_pairs // 2]
    selected = {tuple(sorted((r["node_a"], r["node_b"]))): r for r in high}
    rng = random.Random(hashlib.sha256(f"{task}||{tag}||{level}".encode()).digest())
    while len(selected) < n_pairs:
        a, b = ids[rng.randrange(len(ids))], ids[rng.randrange(len(ids))]
        if a == b:
            continue
        pair = tuple(sorted((a, b)))
        # Random is disjoint from the full neighbor pool, not merely the selected top half.
        if pair not in selected and pair not in pool:
            add(a, b, "random")
            selected[pair] = pool[pair]
    rows = sorted(selected.values(), key=lambda r: hashlib.sha256(
        f"order||{r['pair_id']}".encode()).hexdigest())
    stem = f"{task}_{tag}_{level}"
    eval_path = ROOT / f"{stem}_eval.jsonl"
    with eval_path.open("w") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    payload_dir = ROOT / "arbiter"
    payload_dir.mkdir(exist_ok=True)
    for old in payload_dir.glob(f"{stem}_*.jsonl"):
        old.unlink()
    for start in range(0, len(rows), per_agent):
        with (payload_dir / f"{stem}_{start//per_agent:03d}.jsonl").open("w") as out:
            for row in rows[start:start + per_agent]:
                out.write(json.dumps({"pair_id": row["pair_id"],
                                      "concept_a": row["concept_a"],
                                      "concept_b": row["concept_b"]}, ensure_ascii=False) + "\n")
    manifest = {"task": task, "tag": tag, "level": level, "n_nodes": len(nodes),
                "n_pairs": len(rows), "n_shards": math.ceil(len(rows)/per_agent),
                "strata": dict(Counter(r["stratum"] for r in rows)),
                "nodes_path": nodes_path, "nodes_sha256": _file_sha256(nodes_path),
                "partition_path": partition_path, "partition_sha256": _file_sha256(partition_path),
                "protocol_path": protocol_path, "protocol_sha256": _file_sha256(protocol_path),
                "eval_path": str(eval_path), "eval_sha256": _file_sha256(str(eval_path)),
                "semantic_truth": "LLM arbiter votes only; similarity is sampling strata, never truth"}
    (ROOT / f"{stem}_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def score(task: str, tag: str, level: str, votes_glob: str) -> dict:
    stem = f"{task}_{tag}_{level}"
    manifest = json.loads((ROOT / f"{stem}_manifest.json").read_text())
    eval_rows = {row["pair_id"]: row for row in
                 (json.loads(x) for x in Path(manifest["eval_path"]).read_text().splitlines()) if row}
    votes = {}
    malformed = 0
    for path in sorted(Path().glob(votes_glob)):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1; continue
            pid, value = row.get("pair_id"), row.get("score")
            if (set(row) != {"pair_id", "score"} or pid not in eval_rows or pid in votes
                    or type(value) is not int or value not in (0, 1, 2)):
                malformed += 1; continue
            votes[pid] = value
    missing = set(eval_rows) - set(votes)
    if missing or malformed:
        raise ValueError(f"[{stem}] votes invalid: missing={len(missing)} malformed={malformed}")
    raw = json.loads(Path(manifest["partition_path"]).read_text())
    part = {str(k): str(v) for k, v in
            (raw.get("assignment") or raw.get("partition") or raw).items()}
    truth, pred = [], []
    by = defaultdict(list)
    for pid, row in eval_rows.items():
        same, co = votes[pid] == 2, part[row["node_a"]] == part[row["node_b"]]
        truth.append(same); pred.append(co); by[row["stratum"]].append((same, co))
    n_same, tp = sum(truth), sum(a and b for a, b in zip(truth, pred))
    n_co, co_same = sum(pred), sum(a and b for a, b in zip(truth, pred))
    recall = tp/n_same if n_same else None; precision = co_same/n_co if n_co else None
    sizes = Counter(part.values()); n = len(part)
    p0 = sum(s*(s-1)//2 for s in sizes.values())/(n*(n-1)/2) if n > 1 else None
    cc = (recall-p0)/(1-p0) if recall is not None and p0 != 1 else None
    po = sum(a == b for a, b in zip(truth,pred))/len(truth)
    pa, pb = sum(truth)/len(truth), sum(pred)/len(pred)
    pe = pa*pb + (1-pa)*(1-pb)
    kappa = (po-pe)/(1-pe) if pe < 1 else None
    result = {"task": task, "tag": tag, "level": level, "n_eval": len(truth),
              "n_truth_same": n_same, "recall": round(recall,3) if recall is not None else None,
              "n_colabeled": n_co, "precision": round(precision,3) if precision is not None else None,
              "n_groups": len(sizes), "p0": round(p0,4) if p0 is not None else None,
              "chance_corrected_recall": round(cc,3) if cc is not None else None,
              "cohen_kappa": round(kappa,3) if kappa is not None else None,
              "by_stratum": {s:{"n":len(v),"truth_same":sum(x for x,_ in v),
                                  "colabeled":sum(x for _,x in v),
                                  "recall":round(sum(a and b for a,b in v)/sum(a for a,_ in v),3)
                                  if sum(a for a,_ in v) else None,
                                  "precision":round(sum(a and b for a,b in v)/sum(b for _,b in v),3)
                                  if sum(b for _,b in v) else None} for s,v in by.items()},
              "complete": True}
    (ROOT / f"{stem}_report.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
