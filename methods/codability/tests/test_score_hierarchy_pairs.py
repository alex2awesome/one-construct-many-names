import hashlib

import pytest

from methods.codability.lexicon_distill.hierarchy_contracts import PAIR_INPUT_SCHEMA, validate_pair_output
from methods.codability.lexicon_distill.score_hierarchy_pairs import assemble_outputs


def test_assemble_outputs_averages_both_orders_and_binds_lineage():
    digest_a = hashlib.sha256(b"a").hexdigest()
    digest_b = hashlib.sha256(b"b").hexdigest()
    row = {
        "schema_version": PAIR_INPUT_SCHEMA,
        "pair_id": "pair-1",
        "task": "math-stackexchange",
        "level": "R2",
        "protocol_id": "r2-focused-operational-family-v2.1",
        "node_a": "a", "node_b": "b", "text_a": "First", "text_b": "Second",
        "source_node_a_sha256": digest_a, "source_node_b_sha256": digest_b,
    }
    outputs = assemble_outputs(
        [row],
        {"pair-1": {"ab": [0.1, 0.2, 0.7], "ba": [0.2, 0.2, 0.6]}},
        adapter_sha256=hashlib.sha256(b"adapter").hexdigest(),
        protocol_sha256=hashlib.sha256(b"protocol").hexdigest(),
    )
    output = validate_pair_output(outputs[0])
    assert output["prediction"] == "SAME"
    assert output["probabilities"] == pytest.approx(
        {"DIFFERENT": 0.15, "RELATED": 0.2, "SAME": 0.65})
    assert output["order_consistent"] is True


def test_assemble_outputs_resolves_exact_bf16_tie_conservatively():
    digest_a = hashlib.sha256(b"a").hexdigest()
    digest_b = hashlib.sha256(b"b").hexdigest()
    row = {
        "schema_version": PAIR_INPUT_SCHEMA, "pair_id": "tie", "task": "math-stackexchange",
        "level": "R2", "protocol_id": "r2-focused-operational-family-v2.1",
        "node_a": "a", "node_b": "b", "text_a": "First", "text_b": "Second",
        "source_node_a_sha256": digest_a, "source_node_b_sha256": digest_b,
    }
    output = assemble_outputs(
        [row], {"tie": {"ab": [0.5, 0.5, 0.0], "ba": [0.5, 0.5, 0.0]}},
        adapter_sha256=hashlib.sha256(b"adapter").hexdigest(),
        protocol_sha256=hashlib.sha256(b"protocol").hexdigest(),
    )[0]
    assert validate_pair_output(output)["prediction"] == "DIFFERENT"
