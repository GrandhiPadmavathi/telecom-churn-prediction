import numpy as np
import pytest

from churn import campaign, metrics
from churn.config import Campaign


def test_top_share_precision_recall_lift():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 1])
    p = np.linspace(1, 0, 10)
    r = metrics.top_share(y, p, 0.2)
    assert r["precision"] == 1.0
    assert r["recall"] == pytest.approx(2 / 3)
    assert r["lift"] == pytest.approx(1 / 0.3)


def test_perfectly_calibrated_scores_have_small_ece():
    rng = np.random.default_rng(0)
    p = rng.random(50_000)
    y = rng.random(50_000) < p
    assert metrics.expected_calibration_error(y, p) < 0.01


def test_expected_value_is_zero_at_breakeven():
    c = Campaign()
    monthly = np.array([20.0, 70.0, 110.0])
    assert np.allclose(c.expected_value(c.breakeven_p(monthly), monthly), 0)


def test_cheap_plans_need_a_higher_risk_to_be_worth_calling():
    c = Campaign()
    cheap, pricey = c.breakeven_p(np.array([20.0, 110.0]))
    assert cheap == pytest.approx(30 / 72) and pricey == pytest.approx(120 / 396)
    assert c.expected_value(np.array([0.35]), np.array([110.0]))[0] > 0 > c.expected_value(np.array([0.35]), np.array([20.0]))[0]


def test_policy_value_counts_wasted_incentives():
    c = Campaign()
    churned, monthly = np.array([1, 0]), np.array([100.0, 100.0])
    r = campaign.outcome(np.array([True, True]), churned, monthly, c)
    assert r["value"] == pytest.approx(0.3 * 12 * 100 - 2 * (10 + 100))
    assert campaign.outcome(np.zeros(2, bool), churned, monthly, c)["value"] == 0


def test_value_curve_starts_at_zero_and_ends_at_everyone():
    rng = np.random.default_rng(1)
    p, y, m = rng.random(200), rng.integers(0, 2, 200), rng.uniform(20, 110, 200)
    rows = campaign.value_curve(p, y, m, Campaign(), steps=10)
    assert rows[0]["contacted"] == 0 and rows[-1]["contacted"] == 200
    assert rows[-1]["value"] == pytest.approx(campaign.policies(p, y, m, Campaign())["call_everyone"]["value"])
