import json

from methods.codability.experiments.compile_residual_policy_revision_bank import compile_bank
from methods.codability.experiments.synthesize_residual_policy_revisions import (
    choose_parents,
    residual_panel,
)


def _point(mae, rho, flip=0.1, bias=0.05):
    return {"candidate_robust": {"mae_tvd": mae, "spearman": rho,
                                  "binary_flip_rate": flip, "absolute_bias": bias},
            "target_self_robust": {"mae_tvd": 0.1, "spearman": 0.9,
                                    "binary_flip_rate": 0.1, "absolute_bias": 0.05}}


def test_parent_selection_keeps_incumbent_and_distinct_mae_rank_leaders():
    rows = [{"arm_id": "inc", "point": _point(0.2, 0.6)},
            {"arm_id": "mae", "point": _point(0.12, 0.61)},
            {"arm_id": "rank", "point": _point(0.14, 0.8)}]
    assert [row["arm_id"] for row in choose_parents(rows, incumbent_id="inc")] == [
        "inc", "mae", "rank"]


def test_parent_selection_fills_collapsed_roles_by_identity_loss():
    rows = [
        {"arm_id": "inc", "point": _point(0.11, 0.89)},
        {"arm_id": "next", "point": _point(0.14, 0.80)},
        {"arm_id": "last", "point": _point(0.18, 0.70)},
    ]
    assert [row["arm_id"] for row in choose_parents(
        rows, incumbent_id="inc", max_parents=3
    )] == ["inc", "next", "last"]


def test_parent_selection_fills_beyond_three_distinct_priority_roles():
    """The general fill path must remain reachable after all three primary roles win."""
    rows = [
        {"arm_id": "inc", "point": _point(0.20, 0.60)},
        {"arm_id": "mae", "point": _point(0.11, 0.61)},
        {"arm_id": "rank", "point": _point(0.13, 0.88)},
        {"arm_id": "joint", "point": _point(0.12, 0.84)},
        {"arm_id": "weak", "point": _point(0.30, 0.30)},
    ]
    selected = choose_parents(rows, incumbent_id="inc", max_parents=4)
    assert [row["arm_id"] for row in selected[:3]] == ["inc", "mae", "rank"]
    assert [row["arm_id"] for row in selected] == ["inc", "mae", "rank", "joint"]


def test_residual_panel_contains_both_directions_and_rank_reversal():
    rows = [
        {"text_sha256": "a", "text": "alpha", "target": 0.9, "executor": 0.1},
        {"text_sha256": "b", "text": "bravo", "target": 0.8, "executor": 0.2},
        {"text_sha256": "c", "text": "charlie", "target": 0.7, "executor": 0.3},
        {"text_sha256": "d", "text": "delta", "target": 0.1, "executor": 0.9},
        {"text_sha256": "e", "text": "echo", "target": 0.2, "executor": 0.8},
        {"text_sha256": "f", "text": "foxtrot", "target": 0.3, "executor": 0.7},
        {"text_sha256": "g", "text": "golf", "target": 0.75, "executor": 0.05},
        {"text_sha256": "h", "text": "hotel", "target": 0.05, "executor": 0.75},
    ]
    panel = residual_panel(rows, n_direction=2, n_pairs=1)
    assert len(panel["underpredicted"]) == 2
    assert len(panel["overpredicted"]) == 2
    assert len(panel["rank_inversions"]) == 1


def _forms(text):
    return [{"id": form, "prompt": text, "prompt_sha256": form,
             "total_word_count": len(text.split())}
            for form in ("canonical", "question", "boilerplate")]


def test_revision_compiler_retains_fold_and_parent_slot(tmp_path):
    best = {"N_humor_23": "source_definition", "N_humor_49": "source_explanation",
            "N_cw_27": "source_dossier_v2", "N_pr_8": "source_definition"}
    domains = {"N_humor_23": "humor", "N_humor_49": "humor",
               "N_cw_27": "cw", "N_pr_8": "pr"}
    cells, rows = [], []
    for cell_id, source_id in best.items():
        cells.append({"id": cell_id, "domain": domains[cell_id], "gi": 1,
                      "construct": cell_id,
                      "arms": [{"id": "name", "channel": "sparse",
                                "provenance": "construct_name", "control_for": None,
                                "semantic_content_word_count": 1, "forms": _forms(cell_id)},
                               {"id": source_id, "channel": "declarative",
                                "provenance": "source_telling", "control_for": None,
                                "semantic_content_word_count": 2, "forms": _forms("source")}]})
        rows.append({"cell_id": cell_id, "source_partition": "residual_prompt_selection",
                     "parent_slot": 1, "parent_arm_id": "parent",
                     "parent_provenance": "self", "view": "rank_repair", "variant": 0,
                     "articulation": "standalone revised criterion", "writer_model": "target",
                     "prompt_sha256": "p", "articulation_sha256": "a",
                     "teaching_item_sha256": []})
    source = tmp_path / "source.json"
    revisions = tmp_path / "revisions.json"
    source.write_text(json.dumps({"cells": cells}))
    revisions.write_text(json.dumps({"rows": rows}))
    bank = compile_bank(revision_path=revisions, source_bank_path=source)
    arms = {arm["id"]: arm for arm in bank["cells"][0]["arms"]}
    arm = arms["revision_rank_repair_v0_slot1_from_prompt_selection"]
    assert arm["source_partition"] == "residual_prompt_selection"
    assert arm["parent_slot"] == 1
