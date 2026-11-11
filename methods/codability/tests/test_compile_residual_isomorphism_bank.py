"""Residual teaching example selection is strong-first, diverse, and bounded."""

from methods.codability.experiments.compile_residual_isomorphism_bank import (
    _example_block,
    _eligible_teaching_item,
    select_diverse_examples,
)


def test_example_selection_truncates_and_avoids_duplicate_surface_cases():
    rows = [
        {"text": "same repeated phrase alpha beta gamma", "text_sha256": "a", "priority": 1.0},
        {"text": "same repeated phrase alpha beta delta", "text_sha256": "b", "priority": 0.99},
        {"text": "lexically distinct boundary example with many extra words here", "text_sha256": "c",
         "priority": 0.9},
    ]
    selected = select_diverse_examples(rows, n=2, max_words=5)
    assert selected[0]["text_sha256"] == "a"
    assert selected[1]["text_sha256"] == "c"
    assert all(len(row["text"].split()) <= 6 for row in selected)  # ellipsis is one token
    block = _example_block("quality", selected[:1], selected[1:], heading="Contrasts.")
    assert "SATISFY" in block and "NOT to satisfy" in block


def test_pr_extraction_failures_are_not_used_as_tacit_knowledge_examples():
    assert not _eligible_teaching_item(
        "pr", "The provided raw page content does not contain a press release.")
    assert _eligible_teaching_item("pr", "The company explained its product in short clear sentences.")
