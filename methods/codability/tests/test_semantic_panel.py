from methods.codability.lexicon.semantic_panel import majority_same


def test_majority_same():
    assert majority_same(2, 2)
    assert majority_same(2, 1, 2)
    assert not majority_same(2, 1, 1)
    assert not majority_same(1, 1, 2)
