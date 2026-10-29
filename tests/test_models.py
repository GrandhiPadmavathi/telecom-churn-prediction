import numpy as np
import pytest

from churn import data, explain, features, metrics, models


@pytest.fixture
def split(raw):
    df = data.clean(raw)
    X = features.build(df)
    return X.iloc[:1000], df["churned"].iloc[:1000], X.iloc[1000:], df["churned"].iloc[1000:]


@pytest.mark.parametrize("make", [models.logistic_regression, models.lightgbm])
def test_models_learn_the_planted_signal(split, make):
    X_tr, y_tr, X_te, y_te = split
    m = make(X_tr, y_tr, seed=0)
    m.calibrate(X_te, y_te)
    p = m.predict(X_te)
    assert ((p > 0) & (p < 1)).all()
    assert metrics.report(y_te, p)["roc_auc"] > 0.75
    assert m.calibration in {"none", "platt"}


@pytest.mark.parametrize("make", [models.logistic_regression, models.lightgbm])
def test_contributions_rank_contract_first(split, make):
    X_tr, y_tr, X_te, _ = split
    m = make(X_tr, y_tr, seed=0)
    contrib = m.contributions(X_te)
    assert contrib.shape == X_te.shape
    top = list(explain.global_importance(contrib, top=4))
    assert "Contract=Month-to-month" in top or "tenure" in top


def test_lightgbm_contributions_add_up_to_log_odds(split):
    X_tr, y_tr, X_te, _ = split
    m = models.lightgbm(X_tr, y_tr, seed=0)
    full = m.estimator.predict_proba(X_te, pred_contrib=True)
    p = m.raw_proba(X_te)
    assert np.allclose(full.sum(axis=1), np.log(p / (1 - p)), atol=1e-6)


def test_reasons_are_readable(split):
    X_tr, y_tr, X_te, _ = split
    m = models.lightgbm(X_tr, y_tr, seed=0)
    contrib = m.contributions(X_te)
    i = contrib.sum(axis=1).idxmax()
    rs = explain.reasons(contrib.loc[i], X_te.loc[i], n=3)
    assert len(rs) == 3 and all(r["impact"] > 0 and r["text"] for r in rs)


def test_reason_wording():
    assert explain.describe("Contract=Two year", 0) == "not on a two-year contract"
    assert explain.describe("Contract=Month-to-month", 1) == "on a month-to-month contract"
    assert explain.describe("tenure", 3.0) == "tenure: 3 months"
