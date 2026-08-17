"""Uncertainty summaries used by the Monte Carlo experiments."""

import pytest

from experiments.exp01_causal_weighting import _monte_carlo_mean_summary
from experiments.exp04_partial_id import _wilson_interval


def test_monte_carlo_mean_summary_reports_sd_se_and_normal_interval():
    out = _monte_carlo_mean_summary([1.0, 2.0, 3.0, 4.0], level=0.95)

    assert out["mean"] == pytest.approx(2.5)
    assert out["mc_sd"] == pytest.approx(1.2909944487358056)
    assert out["mc_se"] == pytest.approx(0.6454972243679028)
    assert out["mc_ci"]["method"] == "normal"
    assert out["mc_ci"]["level"] == pytest.approx(0.95)
    assert out["mc_ci"]["lower"] < out["mean"] < out["mc_ci"]["upper"]


def test_wilson_interval_is_non_degenerate_after_all_successes():
    out = _wilson_interval(500, 500, level=0.95)

    assert out["method"] == "wilson"
    assert out["level"] == pytest.approx(0.95)
    assert out["lower"] == pytest.approx(0.9923756595384479)
    assert out["upper"] == pytest.approx(1.0)
