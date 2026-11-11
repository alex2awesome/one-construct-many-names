"""Isomorphism-first bank preserves names, source knowledge, and target-matched forms."""

import json

from methods.codability.experiments.compile_policy_isomorphism_bank import (
    compile_bank,
    validate_bank,
)


def test_bank_is_unique_name_anchored_and_form_matched(tmp_path):
    source = tmp_path / "messages.json"
    message = {"name": "Target quality", "rungs": {
            "definition": "A faithful definition.",
            "explanation": "A faithful explanation.",
            "full_rubric": "Apply this recognition rule.",
            "exemplars": "Judge by these examples ONLY.\nGood and bad cases.",
            "dossier": "A combined faithful dossier.",
        }}
    source.write_text(json.dumps({"23": message, "49": message}))
    bank = compile_bank({"humor": source})
    assert not validate_bank(bank)
    cell = bank["cells"][0]
    definition = next(arm for arm in cell["arms"] if arm["id"] == "iso_definition")
    prompts = {form["id"]: form["prompt"] for form in definition["forms"]}
    assert prompts["canonical"].startswith("Criterion: Target quality.")
    assert prompts["question"].startswith("Does the item meet this criterion?")
    assert prompts["question"].endswith("Answer YES or NO.")
    assert prompts["boilerplate"].startswith("You are an expert evaluator. Evaluate strictly.")
    assert len({form["prompt_sha256"] for arm in cell["arms"] for form in arm["forms"]}) \
        == 3 * len(cell["arms"])
