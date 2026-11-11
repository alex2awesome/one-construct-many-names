import json
from pathlib import Path

from methods.codability.lexicon import r2_recluster as r2


def _write(path: Path, mapping: dict[str, str], field: str = "partition") -> Path:
    path.write_text(json.dumps({field: mapping}))
    return path


def test_rebased_key_comparison_conditions_on_distinct_r1(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(r2, "ROOT", tmp_path / "audit")
    keys = ["a", "b", "c", "d", "e"]
    monkeypatch.setattr(r2, "canon_map", lambda task: {k: f"concept {k}" for k in keys})
    current_l0 = _write(tmp_path / "current_l0.json", {k: f"cn{k}" for k in keys})
    current_r1 = _write(tmp_path / "current_r1.json", {f"cn{k}": f"cr{k}" for k in keys})
    codex_r2 = _write(tmp_path / "codex_r2.json",
                      {"cra": "cx", "crb": "cy", "crc": "cx",
                       "crd": "cy", "cre": "cz"}, "assignment")
    old_l0 = _write(tmp_path / "old_l0.json", {k: f"on{k}" for k in keys})
    old_r1 = _write(tmp_path / "old_r1.json",
                    {"ona": "or_ab", "onb": "or_ab", "onc": "or_c",
                     "ond": "or_d", "one": "or_e"})
    old_r2 = _write(tmp_path / "old_r2.json",
                    {"or_ab": "sx", "or_c": "sy", "or_d": "sy", "or_e": "sz"})
    protocol = tmp_path / "protocol.md"
    protocol.write_text("focused R2")

    report = r2.emit_rebased_key_comparison(
        "t", "rebased", codex_r2, current_l0, current_r1,
        old_l0, old_r1, old_r2, protocol, per_disagreement=20, per_agreement=20)

    assert report["n_pairs"] == 9  # ten key pairs minus a/b, co-labeled by historical R1
    key = json.loads(Path(report["key_path"]).read_text())
    assert not any({row["node_a"], row["node_b"]} == {"a", "b"} for row in key.values())
    assert sum(report["population_counts_excluding_pairs_colabeled_at_r1_by_either_hierarchy"].values()) == 9
    blind = [json.loads(line) for line in Path(report["blind_path"]).read_text().splitlines()]
    assert all(set(row) == {"pair_id", "task", "concept_a", "concept_b"} for row in blind)

    votes_a = tmp_path / "votes_a.jsonl"
    votes_b = tmp_path / "votes_b.jsonl"
    rows_a, rows_b = [], []
    for pid, row in key.items():
        score = 2 if row["stratum"] in ("both_same", "codex_only") else 0
        rows_a.append({"pair_id": pid, "score": score})
        rows_b.append({"pair_id": pid, "score": score})
    votes_a.write_text("".join(json.dumps(row) + "\n" for row in rows_a))
    votes_b.write_text("".join(json.dumps(row) + "\n" for row in rows_b))
    summary = r2.summarize_replicated_comparison("t", "rebased", votes_a, votes_b)
    assert summary["dual_confirmed"]["codex"]["precision"] == 1.0
    assert summary["dual_confirmed"]["codex"]["recall"] == 1.0
    assert summary["binary_same_agreement"] == 1.0
