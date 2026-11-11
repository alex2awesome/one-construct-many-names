import json
from pathlib import Path

from methods.codability.lexicon import versioned_level_eval as v


def _write_votes(path: Path, votes: dict[str, int]) -> None:
    path.mkdir(parents=True)
    (path / "votes.jsonl").write_text("".join(
        json.dumps({"pair_id": pair_id, "score": score}) + "\n"
        for pair_id, score in votes.items()))


def test_replicated_versioned_eval_uses_blind_median_truth(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(v, "ROOT", tmp_path / "eval")
    nodes = tmp_path / "nodes.jsonl"
    nodes.write_text("".join(json.dumps(row) + "\n" for row in (
        {"node_id": "x", "name": "alpha algebra", "gloss": "proof"},
        {"node_id": "y", "name": "alpha theorem", "gloss": "proof"},
        {"node_id": "z", "name": "visual design", "gloss": "figure"},
        {"node_id": "w", "name": "audience reach", "gloss": "distribution"})))
    partition = tmp_path / "partition.json"
    partition.write_text(json.dumps({"assignment": {"x": "g", "y": "g",
                                                       "z": "z", "w": "w"}}))
    protocol = tmp_path / "protocol.md"
    protocol.write_text("same top-level category")
    manifest = v.prepare("demo", "R3", "candidate", str(partition), str(nodes),
                         str(protocol), n_pairs=6, per_agent=2)
    key = json.loads(Path(manifest["key_path"]).read_text())
    xy = next(pair_id for pair_id, row in key.items()
              if {row["node_a"], row["node_b"]} == {"x", "y"})
    a = {pair_id: (2 if pair_id == xy else 0) for pair_id in key}
    b = {pair_id: (1 if pair_id == xy else 0) for pair_id in key}
    a_dir, b_dir, c_dir = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    _write_votes(a_dir, a)
    _write_votes(b_dir, b)
    staged = v.stage_adjudication("demo", "R3", "candidate", str(a_dir), str(b_dir))
    assert staged["pair_ids"] == [xy]
    _write_votes(c_dir, {xy: 2})
    report = v.summarize("demo", "R3", "candidate", str(a_dir), str(b_dir), str(c_dir))
    assert report["n_truth_same"] == 1
    assert report["n_adjudicated"] == 1
    assert report["recall"] == 1.0
    assert report["precision_mixture"] == 1.0


def test_vote_loader_rejects_non_integer_scores(tmp_path: Path):
    votes = tmp_path / "votes"
    votes.mkdir()
    (votes / "bad.jsonl").write_text(json.dumps({"pair_id": "p", "score": 2.0}) + "\n")
    try:
        v._load_votes(str(votes), {"p"})
    except ValueError:
        pass
    else:
        raise AssertionError("float score was accepted")
