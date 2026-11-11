import hashlib
import json

import pytest

from methods.codability.lexicon_distill.frontier_calibration import (
    assemble,
    prepare,
    prepare_disagreements,
    run_judge,
)
from methods.codability.lexicon_distill.hierarchy_contracts import PAIR_INPUT_SCHEMA, PAIR_OUTPUT_SCHEMA, pair_input_sha256


def _write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _files(tmp_path):
    adapter = hashlib.sha256(b"adapter").hexdigest()
    protocol = hashlib.sha256(b"protocol").hexdigest()
    inputs, outputs = [], []
    for index, label in enumerate(("DIFFERENT", "RELATED", "SAME")):
        row = {
            "schema_version": PAIR_INPUT_SCHEMA, "pair_id": f"p{index}",
            "task": "math-stackexchange", "level": "R2",
            "protocol_id": "r2-focused-operational-family-v2.1",
            "node_a": f"a{index}", "node_b": f"b{index}",
            "text_a": f"Concept A{index}", "text_b": f"Concept B{index}",
            "source_node_a_sha256": hashlib.sha256(f"a{index}".encode()).hexdigest(),
            "source_node_b_sha256": hashlib.sha256(f"b{index}".encode()).hexdigest(),
        }
        probabilities = {name: 0.05 for name in ("DIFFERENT", "RELATED", "SAME")}
        probabilities[label] = 0.9
        inputs.append(row)
        outputs.append({
            "schema_version": PAIR_OUTPUT_SCHEMA, "pair_id": row["pair_id"],
            "task": row["task"], "level": row["level"], "protocol_id": row["protocol_id"],
            "input_sha256": pair_input_sha256(row), "prediction": label,
            "probabilities": probabilities,
            "order_views": {view: {"prediction": label, "probabilities": probabilities}
                            for view in ("ab", "ba")},
            "order_consistent": True, "adapter_sha256": adapter, "protocol_sha256": protocol,
        })
    inputs_path, outputs_path = tmp_path / "inputs.jsonl", tmp_path / "outputs.jsonl"
    _write(inputs_path, inputs); _write(outputs_path, outputs)
    protocol_path = tmp_path / "protocol.txt"; protocol_path.write_text("protocol")
    return inputs_path, outputs_path, protocol_path


def test_frontier_panel_is_blind_and_tiebreak_is_disagreement_only(tmp_path):
    inputs, outputs, protocol = _files(tmp_path)
    out = tmp_path / "panel"
    manifest = prepare(pair_inputs_path=inputs, pair_outputs_path=outputs,
                       protocol_path=protocol, output_dir=out, per_predicted_class=2)
    blind = [json.loads(line) for line in (out / "audit.jsonl").read_text().splitlines()]
    assert manifest["n_pairs"] == 3
    assert all("prediction" not in row and "probabilities" not in row for row in blind)
    ids = [row["pair_id"] for row in blind]
    a, b, tie = tmp_path / "a.jsonl", tmp_path / "b.jsonl", tmp_path / "tie.jsonl"
    _write(a, [{"pair_id": pair_id, "score": index} for index, pair_id in enumerate(ids)])
    _write(b, [{"pair_id": pair_id, "score": (1 if index == 0 else index)}
               for index, pair_id in enumerate(ids)])
    _write(tie, [{"pair_id": ids[0], "score": 0}])
    report = assemble(manifest_path=out / "manifest.json", votes_a_path=a, votes_b_path=b,
                      tiebreak_votes_path=tie, predictions_path=out / "dev.jsonl",
                      report_path=out / "assembly.json")
    rows = [json.loads(line) for line in (out / "dev.jsonl").read_text().splitlines()]
    assert report["n_disagreements"] == 1
    assert all(row["split"] == "frontier_dev" for row in rows)
    assert {row["truth"] for row in rows} == {0, 1, 2}


def test_frontier_panel_rejects_same_judge_artifact(tmp_path):
    inputs, outputs, protocol = _files(tmp_path)
    out = tmp_path / "panel"
    prepare(pair_inputs_path=inputs, pair_outputs_path=outputs,
            protocol_path=protocol, output_dir=out)
    ids = [json.loads(line)["pair_id"] for line in (out / "audit.jsonl").read_text().splitlines()]
    votes = tmp_path / "votes.jsonl"
    _write(votes, [{"pair_id": pair_id, "score": 0} for pair_id in ids])
    with pytest.raises(ValueError, match="distinct artifacts"):
        assemble(manifest_path=out / "manifest.json", votes_a_path=votes, votes_b_path=votes,
                 tiebreak_votes_path=None, predictions_path=out / "dev.jsonl",
                 report_path=out / "assembly.json")


def test_openrouter_judge_resumes_under_cap_and_stages_only_disagreements(tmp_path, monkeypatch):
    inputs, outputs, protocol = _files(tmp_path)
    panel = tmp_path / "panel"
    prepare(pair_inputs_path=inputs, pair_outputs_path=outputs,
            protocol_path=protocol, output_dir=panel, per_shard=1)
    api_key = tmp_path / "key.txt"
    api_key.write_text("test-secret")
    calls = []

    def fake_completion(**kwargs):
        batch = json.loads(kwargs["messages"][1]["content"])
        assert all(set(row) == {"pair_id", "task", "level", "concept_a", "concept_b"}
                   for row in batch)
        calls.append(batch)
        return json.dumps({"decisions": [
            {"pair_id": row["pair_id"], "score": int(row["pair_id"][-1])}
            for row in batch
        ]})

    from scripts.tools.silver_match_v3 import adjudicate_gemma_api
    monkeypatch.setattr(adjudicate_gemma_api, "chat_completion", fake_completion)
    judge_dir = tmp_path / "sonnet"
    first = run_judge(
        manifest_path=panel / "manifest.json", output_dir=judge_dir,
        model="anthropic/claude-sonnet-test", api_key_file=api_key, request_cap=1)
    assert first["complete"] is False and first["n_votes"] == 1
    second = run_judge(
        manifest_path=panel / "manifest.json", output_dir=judge_dir,
        model="anthropic/claude-sonnet-test", api_key_file=api_key, request_cap=2)
    assert second["complete"] is True and second["requests_made_this_run"] == 2
    assert len(calls) == 3 and len(second["raw_transcript_sha256"]) == 3

    votes_a = [json.loads(line) for line in (judge_dir / "votes.jsonl").read_text().splitlines()]
    votes_b = tmp_path / "gpt5_votes.jsonl"
    changed = [{**row, "score": ((row["score"] + 1) % 3 if index == 0 else row["score"])}
               for index, row in enumerate(votes_a)]
    _write(votes_b, changed)
    tie_panel = tmp_path / "tiebreak"
    tie_manifest = prepare_disagreements(
        manifest_path=panel / "manifest.json", votes_a_path=judge_dir / "votes.jsonl",
        votes_b_path=votes_b, output_dir=tie_panel, per_shard=1)
    assert tie_manifest["n_pairs"] == 1
    tie_rows = [json.loads(line) for line in (tie_panel / "audit.jsonl").read_text().splitlines()]
    assert [row["pair_id"] for row in tie_rows] == [votes_a[0]["pair_id"]]
    assert all("prediction" not in row and "probabilities" not in row for row in tie_rows)

    tie_result = run_judge(
        manifest_path=tie_panel / "manifest.json", output_dir=tmp_path / "third",
        model="openai/gpt-5-test", api_key_file=api_key, request_cap=1)
    assert tie_result["complete"] is True and tie_result["n_votes"] == 1
