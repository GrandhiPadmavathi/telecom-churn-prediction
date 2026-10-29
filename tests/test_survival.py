import numpy as np
import pytest

from churn import survival


def test_kaplan_meier_hand_example():
    # 6 customers: churned at 2, 3, 3, 5; still active at 3 and 6 (censored).
    curve = survival.kaplan_meier([2, 3, 3, 3, 5, 6], [1, 1, 1, 0, 1, 0])
    assert curve["time"] == [2, 3, 5]
    assert curve["at_risk"] == [6, 5, 2]
    s2, s3, s5 = 5 / 6, 5 / 6 * 3 / 5, 5 / 6 * 3 / 5 * 1 / 2
    assert curve["survival"] == pytest.approx([s2, s3, s5])
    assert survival.survival_at(curve, 4) == pytest.approx(s3)
    assert survival.median_survival(curve) == 3


def test_logrank_same_process_is_not_significant():
    rng = np.random.default_rng(0)
    t = rng.integers(1, 60, 2000)
    e = rng.random(2000) < 0.3
    g = rng.integers(0, 2, 2000)
    assert survival.logrank(t, e, g)["p_value"] > 0.01


def test_logrank_detects_different_hazards():
    rng = np.random.default_rng(0)
    fast = np.minimum(rng.exponential(10, 500), 72).round() + 1
    slow = np.minimum(rng.exponential(60, 500), 72).round() + 1
    t = np.concatenate([fast, slow])
    e = (t < 73).astype(int)
    g = np.repeat(["fast", "slow"], 500)
    result = survival.logrank(t, e, g)
    assert result["df"] == 1 and result["p_value"] < 1e-10
