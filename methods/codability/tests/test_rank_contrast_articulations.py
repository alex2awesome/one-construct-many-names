"""Rank-bearing articulation generation and compilation tests."""

import json

from methods.codability.experiments.build_fresh_item_partitions import text_sha256
from methods.codability.experiments.compile_rank_contrast_bank import compile_bank
from methods.codability.experiments.synthesize_rank_contrast_articulations import (
    select_rank_contrasts,
)


def _item(index, target, executor):
    text = f"item {index} with motif {index % 3}"
    return {"text": text, "text_sha256": text_sha256(text),
            "target": target, "executor": executor}


def _arm(arm_id, text):
    return {
        "id": arm_id, "channel": "sparse", "provenance": "construct_name",
        "control_for": None, "semantic_content_word_count": len(text.split()),
        "forms": [{"id": form, "prompt": value,
                   "prompt_sha256": text_sha256(value),
                   "total_word_count": len(value.split())}
                  for form, value in (
                      ("canonical", text), ("question", f"Question {text}"),
                      ("boilerplate", f"Boilerplate {text}"))],
    }


def test_rank_contrast_selection_is_disjoint_and_actually_reversed():
    rows = [
        _item(0, 0.9, 0.1), _item(1, 0.2, 0.8),
        _item(2, 0.8, 0.2), _item(3, 0.3, 0.7),
        _item(4, 0.7, 0.3), _item(5, 0.4, 0.6),
    ]
    pairs = select_rank_contrasts(rows, n_pairs=3, min_target_gap=0.1)
    hashes = [pair[side]["text_sha256"] for pair in pairs for side in ("high", "low")]
    assert len(hashes) == len(set(hashes))
    assert all(pair["high"]["target"] > pair["low"]["target"] for pair in pairs)
    assert all(pair["high"]["executor"] < pair["low"]["executor"] for pair in pairs)


def test_compiler_freezes_fold_suffixes_and_ostensive_curricula(tmp_path):
    name = _arm("name", "Wordplay quality and clarity")
    behavior = _arm("rule_contrastive_v1_from_prompt_selection", "Behavior rule")
    source_bank = {"cells": [{"id": "N_humor_49", "domain": "humor", "gi": 49,
                              "construct": "Wordplay quality and clarity", "arms": [name]}]}
    rule_bank = {"cells": [{"id": "N_humor_49", "domain": "humor", "gi": 49,
                            "construct": "Wordplay quality and clarity",
                            "arms": [name, behavior]}]}
    high, low = _item(0, 0.9, 0.1), _item(1, 0.2, 0.8)
    pair = {"high": high, "low": low, "target_gap": 0.7,
            "executor_reversal": 0.7, "priority": 0.49}
    context_key = "N_humor_49:residual_prompt_selection:behavior"
    synthesis = {
        "contexts": {context_key: {
            "cell_id": "N_humor_49", "domain": "humor",
            "construct": "Wordplay quality and clarity",
            "source_partition": "residual_prompt_selection", "parent_id": "behavior",
            "parent_arm_id": behavior["id"], "parent_text": "Behavior rule",
            "pairs": [pair, pair, pair, pair],
        }},
        "contrasts": [{"context_key": context_key, "pair_index": index,
                       "micro_rule": f"distinction {index}"} for index in range(4)],
        "rows": [{
            "context_key": context_key, "mode": "rank_patch", "variant": 0,
            "articulation": "A compact conditional ordering patch.",
            "articulation_sha256": text_sha256("A compact conditional ordering patch."),
            "writer_model": "writer", "prompt_sha256": "prompt-hash",
        }],
    }
    paths = {}
    for key, payload in (("synthesis", synthesis), ("source", source_bank),
                         ("rule", rule_bank)):
        path = tmp_path / f"{key}.json"
        path.write_text(json.dumps(payload))
        paths[key] = path
    bank = compile_bank(synthesis_path=paths["synthesis"],
                        source_bank_path=paths["source"], rule_bank_path=paths["rule"])
    arms = bank["cells"][0]["arms"]
    assert any(arm["channel"] == "ostensive" for arm in arms)
    assert all(arm["source_partition"] == "residual_prompt_selection"
               for arm in arms if arm["id"] != "name")
    assert all(arm["id"].endswith("_from_prompt_selection")
               for arm in arms if arm["id"] != "name")
