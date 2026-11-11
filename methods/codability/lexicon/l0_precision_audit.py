"""Cross-task, LLM-judged audit of coherence introduced after the v6 L0 partition.

Code only identifies v6 clusters that the current L0v3 partition co-clusters and creates balanced
head-tail / tail-tail samples.  Whether two source clusters express the same criterion is decided
exclusively by a blind LLM judge.  Frozen adjudicated eval pairs are not used for selection or repair.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict

from .audit import load_partition
from .build_level import OUT, _file_sha256, _load_partition
from .judge import canon_map


ROOT = Path(OUT) / "l0_precision_audit"
VERSION = "l0-v6-to-v3-coherence-v1"
PROTOCOL = """# L0 same-criterion precision audit

You see two clusters of evaluation criteria. Score whether they express the SAME operational
criterion—not merely related topics or constructs.

- 2 = SAME CRITERION: applying either cluster to realistic work would make essentially the same
  pass/fail or comparative distinction. Wording, examples, threshold severity, and level of detail
  may differ when the operational judgment is the same.
- 1 = RELATED BUT DISTINCT: they concern the same topic, construct, workflow, or broad quality but
  test meaningfully different conditions; subset/superset with different coverage is also 1.
- 0 = DIFFERENT: distinct evaluative judgments.

When uncertain between 1 and 2, use 1. Do not infer sameness from shared vocabulary. Judge the
semantic criteria directly from all examples supplied.
"""


def _pid(task: str, a: str, b: str) -> str:
    return hashlib.sha1(f"{VERSION}||{task}||{'||'.join(sorted((a,b)))}".encode()).hexdigest()[:16]


def _examples(task: str, members: Dict[str, list[str]], cluster: str, limit: int = 5) -> list[str]:
    cmap = canon_map(task)
    keys = sorted(members[cluster], key=lambda k: hashlib.sha1(
        f"{task}||{cluster}||{k}".encode()).hexdigest())[:limit]
    return [cmap[k] for k in keys if k in cmap]


def prepare(task: str, per_stratum: int = 60) -> dict:
    ROOT.mkdir(parents=True, exist_ok=True)
    protocol_path = ROOT / "L0_PRECISION_PROTOCOL.md"
    protocol_path.write_text(PROTOCOL)
    base = {str(k): str(v) for k, v in load_partition(task).items()}
    current_path = Path(OUT) / f"partition_{task}_L0v3.json"
    current = {str(k): str(v) for k, v in _load_partition(str(current_path)).items()}
    if set(base) != set(current):
        raise ValueError(f"[{task}] v6/current key coverage differs")

    base_members: Dict[str, list[str]] = defaultdict(list)
    current_of_base: Dict[str, set[str]] = defaultdict(set)
    for key, base_id in base.items():
        base_members[base_id].append(key)
        current_of_base[base_id].add(current[key])
    split = {b: values for b, values in current_of_base.items() if len(values) != 1}
    if split:
        raise ValueError(f"[{task}] current L0 is not merge-only over v6; split base clusters={len(split)}")
    sources_of_current: Dict[str, set[str]] = defaultdict(set)
    for base_id, values in current_of_base.items():
        sources_of_current[next(iter(values))].add(base_id)

    pools = {"head_tail": set(), "tail_tail": set()}
    for current_id, source_ids in sources_of_current.items():
        if len(source_ids) < 2:
            continue
        source_ids = sorted(source_ids)
        head = current_id if current_id in source_ids else source_ids[0]
        tails = [x for x in source_ids if x != head]
        pools["head_tail"].update(tuple(sorted((head, tail))) for tail in tails)
        for i, a in enumerate(tails):
            pools["tail_tail"].update((a, b) for b in tails[i + 1:])

    chosen = []
    for stratum, pairs in pools.items():
        ordered = sorted(pairs, key=lambda ab: hashlib.sha256(
            f"{VERSION}||{task}||{stratum}||{ab[0]}||{ab[1]}".encode()).hexdigest())
        chosen.extend((stratum, a, b) for a, b in ordered[:per_stratum])
    chosen.sort(key=lambda x: _pid(task, x[1], x[2]))

    blind_path = ROOT / f"{task}_blind.jsonl"
    key_path = ROOT / f"{task}_key.json"
    key = {}
    with blind_path.open("w") as out:
        for stratum, a, b in chosen:
            pid = _pid(task, a, b)
            out.write(json.dumps({"pair_id": pid, "task": task,
                                  "criterion_cluster_a": _examples(task, base_members, a),
                                  "criterion_cluster_b": _examples(task, base_members, b)},
                                 ensure_ascii=False) + "\n")
            key[pid] = {"stratum": stratum, "v6_cluster_a": a, "v6_cluster_b": b}
    key_path.write_text(json.dumps(key, indent=2) + "\n")
    manifest = {"task": task, "version": VERSION, "n_pairs": len(chosen),
                "counts": {s: sum(x[0] == s for x in chosen) for s in pools},
                "n_v6_clusters": len(base_members), "n_current_clusters": len(sources_of_current),
                "n_current_multi_v6_clusters": sum(len(x) > 1 for x in sources_of_current.values()),
                "protocol_path": str(protocol_path),
                "protocol_sha256": hashlib.sha256(PROTOCOL.encode()).hexdigest(),
                "current_partition_path": str(current_path),
                "current_partition_sha256": _file_sha256(str(current_path)),
                "blind_path": str(blind_path), "blind_sha256": _file_sha256(str(blind_path)),
                "key_path": str(key_path), "key_sha256": _file_sha256(str(key_path)),
                "selection_excludes_frozen_eval": True,
                "status": "not a semantic audit until an independent blind LLM supplies all votes"}
    (ROOT / f"{task}_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def summarize(task: str, votes_path: str, tag: str = "") -> dict:
    key = json.loads((ROOT / f"{task}_key.json").read_text())
    votes = {}
    malformed = 0
    for line in Path(votes_path).read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            continue
        pid, score = row.get("pair_id"), row.get("score")
        if (set(row) != {"pair_id", "score"} or pid not in key or pid in votes
                or type(score) is not int or score not in (0, 1, 2)):
            malformed += 1
            continue
        votes[pid] = score
    missing = set(key) - set(votes)
    if missing or malformed:
        raise ValueError(f"[{task}] invalid audit votes: missing={len(missing)} malformed={malformed}")
    by = {}
    for stratum in ("head_tail", "tail_tail"):
        scores = [votes[pid] for pid, meta in key.items() if meta["stratum"] == stratum]
        by[stratum] = {"n": len(scores), "same": sum(x == 2 for x in scores),
                       "same_rate": round(sum(x == 2 for x in scores) / len(scores), 3) if scores else None,
                       "related_rate": round(sum(x == 1 for x in scores) / len(scores), 3) if scores else None}
    scores = list(votes.values())
    result = {"task": task, "version": VERSION, "n": len(scores),
              "same": sum(x == 2 for x in scores),
              "same_rate": round(sum(x == 2 for x in scores) / len(scores), 3) if scores else None,
              "by_stratum": by,
              "interpretation": "LLM confirmation of v6-source-cluster coherence inside current L0v3"}
    suffix = f"_{tag}" if tag else ""
    (ROOT / f"{task}{suffix}_report.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def compare_judges(task: str, votes_a_path: str, votes_b_path: str) -> dict:
    """Agreement between two complete blind LLM audits; score-2 is SAME."""
    key = json.loads((ROOT / f"{task}_key.json").read_text())

    def load(path: str) -> Dict[str, int]:
        rows = {}
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line); pid, value = row.get("pair_id"), row.get("score")
            if (set(row) != {"pair_id", "score"} or pid not in key or pid in rows
                    or type(value) is not int or value not in (0, 1, 2)):
                raise ValueError(f"[{task}] invalid replicate vote row")
            rows[pid] = value
        if set(rows) != set(key):
            raise ValueError(f"[{task}] replicate coverage mismatch")
        return rows
    a, b = load(votes_a_path), load(votes_b_path)
    ids = list(key)
    aa, bb = [a[x] == 2 for x in ids], [b[x] == 2 for x in ids]
    po = sum(x == y for x, y in zip(aa, bb))/len(ids)
    pa, pb = sum(aa)/len(ids), sum(bb)/len(ids)
    pe = pa*pb + (1-pa)*(1-pb)
    result = {"task": task, "n": len(ids),
              "exact_3way_agreement": round(sum(a[x] == b[x] for x in ids)/len(ids), 3),
              "binary_same_agreement": round(po, 3),
              "binary_same_kappa": round((po-pe)/(1-pe), 3) if pe < 1 else None,
              "judge_a_same_rate": round(pa, 3), "judge_b_same_rate": round(pb, 3),
              "both_same": sum(x and y for x, y in zip(aa, bb)),
              "a_only_same": sum(x and not y for x, y in zip(aa, bb)),
              "b_only_same": sum(y and not x for x, y in zip(aa, bb)),
              "by_stratum": {s:{"n":sum(meta["stratum"]==s for meta in key.values()),
                                  "a_same":sum(a[pid]==2 and meta["stratum"]==s for pid,meta in key.items()),
                                  "b_same":sum(b[pid]==2 and meta["stratum"]==s for pid,meta in key.items())}
                              for s in ("head_tail","tail_tail")}}
    (ROOT / f"{task}_replication_agreement.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
