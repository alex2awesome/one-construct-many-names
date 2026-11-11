from methods.codability.lexicon.frontier_gate import frontier_tier, passes_frontier


def test_frontier_policy():
    assert passes_frontier(.51, .51)
    assert passes_frontier(.60, .60)
    assert passes_frontier(.80, .65)
    assert not passes_frontier(.50, .70)
    assert not passes_frontier(.70, .50)
    assert frontier_tier(.60, .65) == "balanced-above-50"
