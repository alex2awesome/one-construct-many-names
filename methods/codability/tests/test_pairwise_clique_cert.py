from itertools import combinations

from methods.codability.lexicon.pairwise_clique_cert import _clique_cover


def test_clique_cover_never_uses_uncertified_transitivity():
    # a-b-c is connected, but a-c is not certified: it must not become one group.
    groups = _clique_cover(["a", "b", "c"], {("a", "b"), ("b", "c")})
    assert sorted(map(len, groups)) == [1, 2]
    for group in groups:
        for pair in combinations(group, 2):
            assert tuple(sorted(pair)) in {("a", "b"), ("b", "c")}


def test_clique_cover_keeps_complete_group():
    edges = {("a", "b"), ("a", "c"), ("b", "c")}
    assert _clique_cover(["a", "b", "c"], edges) == [["a", "b", "c"]]
