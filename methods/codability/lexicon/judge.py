"""Fresh pair judge — "beyond v6": same 0/1/2 protocol (scripts/judge_prompt.py, verbatim),
stronger model (GLM-4.7 via zai backend), 3 votes at temp 0.6 (the adopt_v2 protocol), majority
rule. Blinded anchor pairs (easy positives: old score=2 & cos>=0.97; easy negatives: old score=0
& cos<=0.5) ride in every payload; a batch whose anchors fail is flagged, not silently used.

Judgments PARTITION (feed audit.repaired_partition as fresh_pairs); they are never the census
score.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from . import audit
from .sources import CANON_PATH, ROOT

sys.path.insert(0, os.path.join(ROOT, "scripts"))
from judge_prompt import SYSTEM, build_user, salvage  # noqa: E402


def canon_map(task: str) -> Dict[str, str]:
    out = {}
    with open(CANON_PATH) as f:
        for line in f:
            r = json.loads(line)
            if r["task"] == task:
                out[r["key"]] = r["canonical"]
    return out


def _h(a: str, b: str) -> str:
    return hashlib.sha1(("||".join(sorted((a, b)))).encode()).hexdigest()[:16]


def build_payload(task: str, max_merge_member_pairs: int = 3,
                  n_anchors: int = 10) -> List[dict]:
    """T3 untrusted pairs + merge-candidate member pairs + blinded anchor pairs."""
    cmap = canon_map(task)
    meta = json.load(open(os.path.join(ROOT, "outputs", "lexicon",
                                       f"partition_{task}_meta.json")))
    part = json.load(open(os.path.join(ROOT, "outputs", "lexicon",
                                       f"partition_{task}.json")))
    rows: List[dict] = []
    seen = set()

    def add(ka, kb, kind, **extra):
        pid = _h(ka, kb)
        if pid in seen or ka not in cmap or kb not in cmap:
            return
        seen.add(pid)
        rows.append({"pair_id": pid, "task": task, "kind": kind, "key_a": ka, "key_b": kb,
                     "canonical_a": cmap[ka], "canonical_b": cmap[kb], **extra})

    for ka, kb in meta.get("fresh_judge_candidates", []):
        add(ka, kb, "t3")

    members: Dict[str, List[str]] = defaultdict(list)
    for k, c in part.items():
        members[c].append(k)
    for mc in meta.get("merge_candidates", []):
        ma = sorted(members.get(mc["a"], []), key=lambda k: _h(k, mc["b"]))
        mb = sorted(members.get(mc["b"], []), key=lambda k: _h(k, mc["a"]))
        for i in range(min(max_merge_member_pairs, len(ma), len(mb))):
            add(ma[i], mb[i], "merge", cluster_a=mc["a"], cluster_b=mc["b"])

    verd = audit.load_verdicts(task)
    pos = [r for r in verd if r.get("score") == 2 and (r.get("cos") or 0) >= 0.97]
    pos = sorted(pos, key=lambda r: _h(r["key_a"], r["key_b"]))[:n_anchors]
    neg = sorted((r for r in verd if r.get("score") == 0),
                 key=lambda r: (r.get("cos") or 1))[:n_anchors]
    for pool, kind in ((pos, "anchor_pos"), (neg, "anchor_neg")):
        for r in pool:
            add(r["key_a"], r["key_b"], kind)
    return rows


class FreshJudge:
    def __init__(self, model: str = "glm-4.7", backend: str = "zai_anthropic",
                 votes: int = 3, temperature: float = 0.6):
        from methods.metric_implementer import backends as _b, config as _c
        cfg = _c.ImplementerConfig(backend=backend)
        self.be = _b.LLMBackend(model, "fresh_pair_judge", cfg)
        self.model, self.votes, self.temp = model, votes, temperature

    def run(self, rows: List[dict], out_path: str, flush_every: int = 400) -> dict:
        done = set()
        if os.path.exists(out_path):
            with open(out_path) as f:
                done = {json.loads(l)["pair_id"] for l in f if l.strip()}
        todo = [r for r in rows if r["pair_id"] not in done]
        stats = {"judged": 0, "unparsed_votes": 0, "skipped": len(rows) - len(todo)}
        out = open(out_path, "a")
        for lo in range(0, len(todo), flush_every):
            chunk = todo[lo: lo + flush_every]
            prompts = [build_user(r["task"], r["canonical_a"], r["canonical_b"]) for r in chunk]
            all_votes: List[List[Optional[int]]] = [[] for _ in chunk]
            for _ in range(self.votes):
                outs = self.be.generate_batch(prompts, system=SYSTEM, max_tokens=200,
                                              temperature=self.temp)
                for i, o in enumerate(outs):
                    p = salvage(o or "")
                    s = None
                    if p is not None:
                        try:
                            s = int(p.get("score"))
                        except (TypeError, ValueError):
                            s = None
                    all_votes[i].append(s if s in (0, 1, 2) else None)
            for r, votes in zip(chunk, all_votes):
                valid = [v for v in votes if v is not None]
                stats["unparsed_votes"] += votes.count(None)
                same = sum(v == 2 for v in valid) * 2 > len(valid) if valid else None
                out.write(json.dumps({**r, "votes": votes, "n_valid": len(valid),
                                      "fresh_same": same, "model": self.model}) + "\n")
                stats["judged"] += 1
            out.flush()
            print(f"  [{self.model}] {min(lo + flush_every, len(todo))}/{len(todo)} pairs "
                  f"(x{self.votes} votes)", flush=True)
        out.close()
        return stats


def anchor_report(verdict_path: str) -> dict:
    rows = [json.loads(l) for l in open(verdict_path) if l.strip()]
    pos = [r for r in rows if r["kind"] == "anchor_pos" and r["fresh_same"] is not None]
    neg = [r for r in rows if r["kind"] == "anchor_neg" and r["fresh_same"] is not None]
    return {"n_pos": len(pos), "pos_pass": (sum(r["fresh_same"] for r in pos) / len(pos)) if pos else None,
            "n_neg": len(neg), "neg_pass": (sum(not r["fresh_same"] for r in neg) / len(neg)) if neg else None}


def fresh_pairs_from(verdict_path: str) -> Dict[frozenset, bool]:
    out: Dict[frozenset, bool] = {}
    with open(verdict_path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["kind"].startswith("anchor") or r.get("fresh_same") is None:
                continue
            out[frozenset((r["key_a"], r["key_b"]))] = bool(r["fresh_same"])
    return out


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", required=True)
    p.add_argument("--out-dir", default=os.path.join(ROOT, "outputs", "lexicon"))
    p.add_argument("--votes", type=int, default=3)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    j = FreshJudge(votes=a.votes)
    for task in a.tasks.split(","):
        task = task.strip()
        rows = build_payload(task)
        if a.limit:
            anchors = [r for r in rows if r["kind"].startswith("anchor")]
            rows = rows[: a.limit] + [r for r in anchors if r not in rows[: a.limit]]
        op = os.path.join(a.out_dir, f"fresh_verdicts_{task}.jsonl")
        print(f"[{task}] {len(rows)} pairs -> {op}")
        st = j.run(rows, op)
        print(f"[{task}] {st} anchors={anchor_report(op)}")


if __name__ == "__main__":
    main()
