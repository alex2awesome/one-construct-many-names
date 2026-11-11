import json

import numpy as np

from methods.codability.experiments.compile_target_policy_rule_bank import compile_bank
from methods.codability.experiments.synthesize_target_policy_rules import (
    calibration_text,
    policy_calibration,
    select_teaching_panel,
)


def test_teaching_panel_is_deterministic_stratified_and_unique():
    rows = [{"text_sha256": f"h{i}", "text": f"distinct token{i}",
             "target": i / 11.0, "small": (11 - i) / 11.0}
            for i in range(12)]
    first = select_teaching_panel(rows, per_slice=2)
    second = select_teaching_panel(list(reversed(rows)), per_slice=2)
    assert [(r["slice"], r["text_sha256"]) for r in first] == [
        (r["slice"], r["text_sha256"]) for r in second]
    assert len({r["text_sha256"] for r in first}) == len(first)
    assert {r["slice"] for r in first} >= {"high_target", "low_target", "boundary"}


def test_calibration_is_descriptive_and_not_a_quota():
    value = policy_calibration(np.array([0.1, 0.3, 0.8, 0.9]))
    assert value["binary_positive_rate"] == 0.5
    assert value["mean_p_yes"] == 0.525
    assert "never as a quota" in calibration_text(value)


def _forms(text):
    return [{"id": key, "prompt": text, "prompt_sha256": key,
             "total_word_count": len(text.split())}
            for key in ("canonical", "question", "boilerplate")]


def test_compiler_preserves_fold_provenance_and_construct_only_rules(tmp_path):
    cells = []
    synth_rows = []
    panels = {}
    for cell_id, domain in (("N_humor_23", "humor"), ("N_humor_49", "humor"),
                            ("N_cw_27", "cw"), ("N_pr_8", "pr")):
        name = f"criterion {cell_id}"
        source_id = {"N_humor_23": "source_definition",
                     "N_humor_49": "source_explanation",
                     "N_cw_27": "source_dossier_v2",
                     "N_pr_8": "source_definition"}[cell_id]
        cells.append({"id": cell_id, "domain": domain, "gi": 1, "construct": name,
                      "arms": [{"id": "name", "channel": "sparse",
                                "provenance": "construct_name", "control_for": None,
                                "semantic_content_word_count": 2, "forms": _forms(name)},
                               {"id": source_id, "channel": "declarative",
                                "provenance": "source_telling", "control_for": None,
                                "semantic_content_word_count": 2,
                                "forms": _forms("source text")} ]})
        for partition in (None, "residual_prompt_selection", "residual_unit_certification"):
            synth_rows.append({"cell_id": cell_id, "source_partition": partition,
                               "view": "gestalt", "variant": 0,
                               "articulation": "a sufficiently explicit reusable policy rule",
                               "writer_model": "target", "prompt_sha256": "p",
                               "articulation_sha256": "a", "teaching_item_sha256": [],
                               "calibration": ({"binary_positive_rate": 0.25,
                                                "mean_p_yes": 0.3}
                                               if partition else None)})
        for partition in ("residual_prompt_selection", "residual_unit_certification"):
            panels[f"{cell_id}:{partition}"] = {
                "calibration": {"binary_positive_rate": 0.25, "mean_p_yes": 0.3}}
    source = tmp_path / "source.json"
    synthesis = tmp_path / "synthesis.json"
    source.write_text(json.dumps({"cells": cells}))
    synthesis.write_text(json.dumps({"rows": synth_rows, "panels": panels}))
    bank = compile_bank(synthesis_path=synthesis, source_bank_path=source)
    arms = {a["id"]: a for a in bank["cells"][0]["arms"]}
    assert arms["rule_gestalt_v0_from_self"]["source_partition"] is None
    assert (arms["rule_gestalt_v0_from_prompt_selection"]["source_partition"]
            == "residual_prompt_selection")
    assert arms["source_calibrated_from_unit_certification"]["recipe"] == (
        "source_plus_calibration")
    assert bank["anchor_policy"].startswith("no external ground truth")
