"""Target-health summaries for fresh name and holistic views."""

import numpy as np

from methods.codability.experiments.fresh_target_score_report import target_summary


def test_target_summary_separates_information_from_form_instability():
    stable = np.array([[0.1, 0.2, 0.8, 0.9], [0.11, 0.19, 0.79, 0.91]])
    unstable = np.array([[0.1, 0.2, 0.8, 0.9], [0.9, 0.8, 0.2, 0.1]])

    stable_report = target_summary(stable)
    unstable_report = target_summary(unstable)

    assert stable_report["T_tvd"] > 0.2
    assert stable_report["mean_form_flip_rate"] == 0.0
    assert unstable_report["mean_form_flip_rate"] == 0.5
    assert unstable_report["T_tvd"] == 0.0
