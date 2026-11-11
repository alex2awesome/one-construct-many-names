"""Offline tests for the lexicon census leg (no model calls)."""
import json

from methods.codability.lexicon import anchors as anch
from methods.codability.lexicon import audit, census
from methods.codability.lexicon.extract import parse_reply, validate
from methods.codability.lexicon.sources import best_window, norm_text, normalize_url


DOC = ("Copy desk memo. The last line of a story is called the kicker, and it is not a summary. "
       "A kicker is the detail you saved on purpose.")


def test_validate_accepts_verbatim():
    d = {"found": True, "quote": "the last line of a story is called the kicker",
         "key_terms": ["kicker"], "head_term": "kicker", "named_in_source": True}
    assert validate(d, DOC) == ""


def test_validate_rejects_paraphrase_and_foreign_terms():
    d = {"found": True, "quote": "the final line is known as the kicker",
         "key_terms": ["kicker"], "head_term": "kicker", "named_in_source": True}
    assert validate(d, DOC) == "quote_not_in_source"
    d2 = {"found": True, "quote": "A kicker is the detail you saved on purpose",
          "key_terms": ["resonant ending"], "head_term": None, "named_in_source": False}
    assert validate(d2, DOC) == "term_not_in_source"


def test_validate_notfound_must_be_empty():
    base = {"found": False, "quote": "", "key_terms": [], "head_term": None,
            "named_in_source": False}
    assert validate(base, DOC) == ""
    assert validate({**base, "quote": "x"}, DOC) == "notfound_nonempty"


def test_validate_preserves_word_boundaries_and_field_invariants():
    d = {"found": True, "quote": "therapist", "key_terms": ["the rapist"],
         "head_term": None, "named_in_source": False}
    assert validate(d, "The therapist answered.") == "term_not_in_source"
    assert validate({**d, "key_terms": [], "head_term": "therapist"},
                    "The therapist answered.") == "head_without_named"
    assert validate({**d, "found": "false"}, "The therapist answered.") == "flags_not_bool"


def test_parse_reply_fenced():
    txt = "```json\n{\"found\": true, \"quote\": \"q\", \"key_terms\": [], " \
          "\"head_term\": null, \"named_in_source\": false}\n```"
    assert parse_reply(txt)["found"] is True


def test_anchor_scoring_catches_substitution():
    key = "peer-review::raw::methods_seminar_handout_9.html::1"
    good = {"found": True, "named_in_source": False, "head_term": None,
            "key_terms": ["rerun", "the same numbers"]}
    bad = {"found": True, "named_in_source": True, "head_term": "reproducibility",
           "key_terms": ["reproducibility"]}
    assert anch.score_anchor(key, good)["pass"]
    assert not anch.score_anchor(key, bad)["pass"]


def test_concept_census_counts_lexicalizations():
    recs = [
        {"status": "ok", "found": True, "source_id": "s1", "key": "k1",
         "key_terms": ["punching up"], "head_term": "punching up", "named_in_source": True},
        {"status": "ok", "found": True, "source_id": "s2", "key": "k2",
         "key_terms": ["punch up", "targets"], "head_term": "punching up", "named_in_source": True},
        {"status": "ok", "found": True, "source_id": "s3", "key": "k3",
         "key_terms": ["aim at the powerful"], "head_term": None, "named_in_source": False},
    ]
    c = census.concept_census(recs)
    assert c["n_sources"] == 3 and c["n_named_sources"] == 2
    assert c["n_head_lexicalizations"] == 1 and c["naming_agreement"] == 1.0
    assert 0 < c["unnamed_rate"] < 0.5
    assert c["disjoint_pair_rate"] > 0  # s3 shares no lexeme with s1


def test_same_source_never_independent():
    recs = [{"status": "ok", "found": True, "source_id": "s1", "key": f"k{i}",
             "key_terms": [t], "head_term": t, "named_in_source": True}
            for i, t in enumerate(["kicker", "closer", "button"])]
    c = census.concept_census(recs)
    assert c["n_sources"] == 1 and c["n_pairs"] == 0


def test_trust_edges_triangle_vs_single():
    verd = [
        {"key_a": "a", "key_b": "b", "score": 2},
        {"key_a": "b", "key_b": "c", "score": 2},
        {"key_a": "a", "key_b": "c", "score": 2},   # triangle: all T2
        {"key_a": "x", "key_b": "y", "score": 2},   # lone edge: T3
        {"key_a": "x", "key_b": "z", "score": 0},
    ]
    T = audit.trust_edges(verd, {})
    assert frozenset(("a", "b")) in T["t2"] and frozenset(("x", "y")) in T["t3"]
    assert frozenset(("x", "z")) in T["score0"]


def test_window_contains_pointer():
    doc = ("filler " * 3000) + " the kicker is the saved detail " + ("filler " * 3000)
    w = best_window(doc, "kicker saved detail")
    assert "kicker" in w and len(w) < 12000


def test_norms():
    assert normalize_url("https://www.Site.com/a/b/?q=1") == "site.com/a/b"
    assert norm_text("A  B\n c") == "a b c"
    assert census.norm_term("Punching-Up!") == "punching-up"
    assert census.norm_term("kickers") == "kicker"
