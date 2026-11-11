"""Treatment-boundary checks for the clean non-name gestalt target."""

import json
from pathlib import Path


PATH = Path(__file__).parents[1] / "experiments" / "fresh_gestalt_target_manifest_v1.json"


def test_gestalt_target_has_no_construct_or_criterion_wrapper():
    manifest = json.loads(PATH.read_text())
    forbidden_names = ["wordplay quality", "laugh density", "plain language", "pitch/query"]
    assert "criterion" not in manifest["readout_template"].lower()
    assert len(manifest["cells"]) == 4
    for cell in manifest["cells"]:
        assert cell["view"] == "G" and cell["construct"] is None
        assert len(cell["forms"]) == 3
        prompts = " ".join(form["prompt"].lower() for form in cell["forms"])
        assert not any(name in prompts for name in forbidden_names)
