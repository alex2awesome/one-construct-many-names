"""Recall-aware acceptance policy for corrected hierarchy candidates."""
from __future__ import annotations


def passes_frontier(recall: float | None, precision: float | None) -> bool:
    """Accept only candidates with both corrected recall and audited precision above .50.

    Recall is the matched, corrected-LLM-truth diagnostic. Precision must come from a direct,
    uniform predicted-positive LLM audit, not the old neighbor/random mixture estimate.
    """
    return recall is not None and precision is not None and recall > .50 and precision > .50


def frontier_tier(recall: float | None, precision: float | None) -> str:
    if not passes_frontier(recall, precision):
        return "reject"
    return "balanced-above-50"
