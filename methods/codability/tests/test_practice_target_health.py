"""Health readouts for archival practice-target probabilities."""

from methods.codability.experiments.practice_target_health import summarize_target


def test_practice_target_summary_reports_information_and_balance():
    report = summarize_target([0, 0, 1, 1, 1, 0])
    assert report["mean"] == 0.5
    assert report["T_tvd"] == 0.5
    assert report["T_shannon"] == 1.0
    assert report["n_unique"] == 2
