import json

from methods.codability.experiments.compile_hierarchical_policy_textbook_bank import compile_bank
from methods.codability.experiments.synthesize_hierarchical_policy_textbooks import (
    interleaved_chunks,
)


def test_interleaved_chunks_cover_every_item_and_each_target_range():
    rows = [{"text_sha256": str(index), "target": index / 31, "text": str(index)}
            for index in range(32)]
    chunks = interleaved_chunks(rows, n_chunks=4)
    assert sorted(row["text_sha256"] for chunk in chunks for row in chunk) == [
        str(index) for index in sorted(range(32), key=lambda value: str(value))]
    assert all(len(chunk) == 8 for chunk in chunks)
    assert all(min(row["target"] for row in chunk) < 0.2
               and max(row["target"] for row in chunk) > 0.8 for chunk in chunks)


def _forms(text):
    return [{"id": form, "prompt": text, "prompt_sha256": form,
             "total_word_count": len(text.split())}
            for form in ("canonical", "question", "boilerplate")]


def test_textbook_compiler_preserves_fold_provenance(tmp_path):
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
                                "semantic_content_word_count": 1, "forms": _forms("source")}]})
        rows.append({"cell_id": cell_id, "source_partition": "residual_unit_certification",
                     "mode": "textbook", "variant": 1,
                     "articulation": "comprehensive target policy textbook",
                     "writer_model": "target", "prompt_sha256": "p",
                     "articulation_sha256": "a", "teaching_item_sha256": ["x"]})
    source = tmp_path / "source.json"
    textbooks = tmp_path / "textbooks.json"
    source.write_text(json.dumps({"cells": cells}))
    textbooks.write_text(json.dumps({"rows": rows}))
    bank = compile_bank(textbook_path=textbooks, source_bank_path=source)
    arms = {arm["id"]: arm for arm in bank["cells"][0]["arms"]}
    arm = arms["textbook_textbook_v1_from_unit_certification"]
    assert arm["source_partition"] == "residual_unit_certification"
    assert arm["teaching_item_sha256"] == ["x"]
