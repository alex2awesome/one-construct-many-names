import json

import pytest

from methods.codability.lexicon import cross_family_validation as cf


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_frozen_manifest_and_strict_complete_votes(tmp_path, monkeypatch):
    monkeypatch.setattr(cf, "OUT", str(tmp_path))
    task, level = "toy", "R2"
    eval_rows, arb = [], []
    for score in (0, 1, 2):
        for i in range(2):
            pid = f"p{score}{i}"
            eval_rows.append({"pair_id": pid, "node_a": f"a{score}{i}",
                              "node_b": f"b{score}{i}", "canonical_a": "alpha",
                              "canonical_b": "beta"})
            arb.append({"pair_id": pid, "score": score})
    _write_jsonl(tmp_path / f"level_eval_{task}_{level}.jsonl", eval_rows)
    arb_path = tmp_path / "level_votes" / f"arb_{task}_{level}_000.jsonl"
    _write_jsonl(arb_path, arb)
    emitted = cf.emit(task, level, n_per_score=2)
    votes_path = tmp_path / "candidate.jsonl"
    _write_jsonl(votes_path, arb)
    rep = cf.report(task, str(votes_path), level, n_per_score=2, require_complete=True,
                    manifest_path=emitted["manifest_private"])
    assert rep["exact_3way_agreement"] == 1.0 and rep["n_scored"] == 6

    # Changing live reference files cannot change the already-frozen comparison.
    _write_jsonl(arb_path, [{**r, "score": (r["score"] + 1) % 3} for r in arb])
    assert cf.report(task, str(votes_path), level, n_per_score=2, require_complete=True,
                     manifest_path=emitted["manifest_private"])["exact_3way_agreement"] == 1.0

    # Duplicate/non-object rows never receive a last-write-wins interpretation.
    with open(votes_path, "a") as fh:
        fh.write(json.dumps(arb[0]) + "\n")
        fh.write("[]\n")
    with pytest.raises(ValueError):
        cf.report(task, str(votes_path), level, n_per_score=2, require_complete=True,
                  manifest_path=emitted["manifest_private"])

