"""Unit tests for the iso-performance expansion chain (pure helpers; GPU phases smoke via --fake)."""
import numpy as np
import pytest

from methods.codability import run_expansion_chain as chain

from methods.codability.run_expansion_chain import (
    assemble_levels, bal_acc, build_planted, km_median_level, match_level, size_key,
    triangle_slack,
)


def test_assemble_levels_nesting_invariant():
    levels = assemble_levels("Surprise ending", ["adds twist.", "reader misled.", "payoff lands."])
    assert len(levels) == 4
    for i in range(1, len(levels)):
        assert levels[i].startswith(levels[i - 1])          # strict prefix property = nested chain


def test_expansion_mixed_level_checkpoint_collision_refused(tmp_path):
    np.savez(tmp_path / "t_R1_metric5_sigs.npz", x=1)
    np.savez(tmp_path / "t_R2_metric5_sigs.npz", x=1)
    with pytest.raises(ValueError):
        chain._ckpts(str(tmp_path), None)


def test_match_level_exact_delta_censored():
    weak = [0.50, 0.55, 0.61, 0.70]
    assert match_level(weak, 0.61) == 2
    assert match_level(weak, 0.63, delta=0.02) == 2         # tolerance widens the match
    assert match_level(weak, 0.90) is None                  # censored: never matches
    assert match_level(weak, float("nan")) is None


def test_km_median_with_censoring():
    med, frac = km_median_level([1, 2, 2, None, None], n_levels=8)
    assert med == 2 and frac == 0.6
    med, frac = km_median_level([1, None, None, None, None], n_levels=8)
    assert med is None and frac == 0.2                      # median censored -> undefined


def test_triangle_slack_additive_and_overshoot():
    # additive world: each hop costs 2 levels uniformly
    strong = [0.80] * 8
    mid = [0.60, 0.70, 0.80, 0.80, 0.80, 0.80, 0.80, 0.80]      # matches strong@0 at level 2
    weak = [0.40, 0.50, 0.60, 0.70, 0.80, 0.80, 0.80, 0.80]      # matches mid@2 (=.8) at 4; strong@0 at 4
    t = triangle_slack(weak, mid, strong)
    assert t["h_mid"] == 2 and t["composed"] == 4 and t["direct"] == 4 and t["slack"] == 0
    # overshoot world: composed pays for mid-specific scaffolding the weak didn't need
    mid2 = [0.55, 0.58, 0.62, 0.66, 0.70, 0.74, 0.80, 0.85]      # matches strong@0 late (level 6)
    weak2 = [0.40, 0.80, 0.80, 0.80, 0.80, 0.80, 0.80, 0.80]     # direct match at level 1
    t2 = triangle_slack(weak2, mid2, strong)
    assert t2["direct"] == 1 and t2["composed"] == 1 and t2["slack"] == 0
    weak3 = [0.40, 0.50, 0.60, 0.70, 0.75, 0.78, 0.80, 0.82]
    t3 = triangle_slack(weak3, mid2, strong)                     # composed goes via mid2@6=.80
    assert t3["h_mid"] == 6 and t3["composed"] == 6 and t3["direct"] == 6 and t3["slack"] == 0
    # TRUE overshoot: mid overshoots strong@0 at its match level (.85 > .80), so the composed
    # path demands more of the weak reader than the direct one -- strict sub-additivity slack
    mid3 = [0.60, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85]     # h_mid=1, but mid3[1]=.85
    weak4 = [0.40, 0.55, 0.70, 0.80, 0.82, 0.85, 0.85, 0.85]    # reaches .80@3 but .85 only @5
    t4 = triangle_slack(weak4, mid3, strong)
    assert t4["h_mid"] == 1 and t4["composed"] == 5 and t4["direct"] == 3 and t4["slack"] == 2


def test_planted_rules_gold():
    texts = (["Is this funny? " + "word " * 30] * 3
             + ['He said "hi" and left. ' + "aa " * 10] * 3
             + ["plain filler text. " * 20] * 4)
    planted = build_planted(texts, max_chars=4000)
    q = planted["planted_question"][2]
    assert q("Is this funny?") and not q("This is funny.")
    d = planted["planted_dialogue"][2]
    assert d('He said "hello" quietly.') and not d("No speech here.")
    sp = planted["planted_second_person"][2]
    assert sp("You will love this.") and not sp("Your youth is over.")   # word-boundary: 'you' only
    # median rules: the threshold baked into the rule TEXT is the one the lambda enforces
    _, rule_text, rule = planted["planted_length_median"]
    n = int(rule_text.split("longer than ")[1].split(" words")[0])
    assert rule("w " * (n + 1)) and not rule("w " * n)
    # balance by construction on the build corpus (median split can't be lopsided)
    for key in ("planted_length_median", "planted_sentences_median", "planted_wordlen_median"):
        g = np.array([bool(planted[key][2](t)) for t in texts])
        assert 0.2 <= g.mean() <= 0.8, key


def test_planted_gold_on_truncated_view():
    # v2 fix: a tail-sensitive rule must be evaluated on the reader's truncated view, not the
    # full text (v1 computed gold on the full text -> unreachable gold on long-text domains)
    long_text = ("no questions here. " * 300) + " Was that all?"
    planted = build_planted([long_text, "short? " * 5], max_chars=100)
    assert not planted["planted_question"][2](long_text)     # '?' lives past the 100-char view
    assert planted["planted_question"][2]("Really? " * 3)


def test_bal_acc_and_size_ordering():
    ref = np.array([True, True, False, False])
    assert bal_acc(np.array([True, False, False, True]), ref) == 0.5
    assert bal_acc(ref, ref) == 1.0
    tags = sorted(["Llama-3.1-8B", "Llama-3.2-1B", "Llama-3.3-70B", "Llama-3.2-3B"], key=size_key)
    assert tags == ["Llama-3.2-1B", "Llama-3.2-3B", "Llama-3.1-8B", "Llama-3.3-70B"]
