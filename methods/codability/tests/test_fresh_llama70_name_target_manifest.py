"""Closure checks for the clean Llama-70B name-only target job."""

import json
from pathlib import Path


PATH = Path(__file__).parents[1] / "experiments" / "fresh_llama70_name_target_manifest_v1.json"


def test_llama70_name_target_contains_only_priority_n_cells():
    manifest = json.loads(PATH.read_text())
    assert {cell["id"] for cell in manifest["cells"]} == {
        "N_humor_23", "N_humor_49", "N_pr_8"}
    assert all(cell["view"] == "N" and len(cell["forms"]) == 3
               for cell in manifest["cells"])
    referenced = {value for job in manifest["model_jobs"]
                  for domain in job["domains"] for value in domain["cells"]}
    assert referenced == {cell["id"] for cell in manifest["cells"]}
