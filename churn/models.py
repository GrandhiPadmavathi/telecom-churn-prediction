"""Logistic regression baseline and LightGBM challenger behind one interface.

Classes are not re-weighted: the campaign maths needs real probabilities, and re-weighting would inflate them.
Calibration (none or Platt scaling) is chosen by cross-validated calibration error on the validation set.
"""

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from churn.metrics import expected_calibration_error


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class PlattScaler:
    def fit(self, p, y):
        self.lr = LogisticRegression(C=1e4).fit(_logit(p).reshape(-1, 1), y)
        return self

    def predict(self, p):
        return self.lr.predict_proba(_logit(p).reshape(-1, 1))[:, 1]


@dataclass
class ChurnModel:
    name: str
    estimator: object
    columns: list[str]
    calibrator: PlattScaler | None = None
    calibration_cv_ece: dict | None = None

    def raw_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict_proba(X[self.columns])[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.raw_proba(X)
        return raw if self.calibrator is None else self.calibrator.predict(raw)

    def calibrate(self, X: pd.DataFrame, y, seed: int = 0, folds: int = 5) -> None:
        p, y = self.raw_proba(X), np.asarray(y)
        platt = np.empty_like(p)
        for fit_idx, eval_idx in StratifiedKFold(folds, shuffle=True, random_state=seed).split(p.reshape(-1, 1), y):
            platt[eval_idx] = PlattScaler().fit(p[fit_idx], y[fit_idx]).predict(p[eval_idx])
        self.calibration_cv_ece = {"none": expected_calibration_error(y, p),
                                   "platt": expected_calibration_error(y, platt)}
        if self.calibration_cv_ece["platt"] < self.calibration_cv_ece["none"]:
            self.calibrator = PlattScaler().fit(p, y)

    @property
    def calibration(self) -> str:
        return "none" if self.calibrator is None else "platt"

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Each input's additive push on the log-odds of churn (positive = more likely to leave)."""
        if self.name == "lightgbm":
            contrib = self.estimator.predict_proba(X[self.columns], pred_contrib=True)[:, :-1]
            return pd.DataFrame(contrib, columns=self.columns, index=X.index)
        scaler, clf = self.estimator.named_steps["scale"], self.estimator.named_steps["clf"]
        z = scaler.transform(X[self.columns])
        return pd.DataFrame((z - self.reference_mean_) * clf.coef_[0], columns=self.columns, index=X.index)


def logistic_regression(X: pd.DataFrame, y, seed: int) -> ChurnModel:
    pipe = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(C=0.3, max_iter=3000, random_state=seed))])
    pipe.fit(X, y)
    model = ChurnModel("logistic_regression", pipe, list(X.columns))
    model.reference_mean_ = pipe.named_steps["scale"].transform(X).mean(axis=0)
    return model


def lightgbm(X: pd.DataFrame, y, seed: int, early_stop_frac: float = 0.15) -> ChurnModel:
    y = np.asarray(y)
    holdout = np.random.default_rng(seed).random(len(X)) < early_stop_frac
    clf = lgb.LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.02,
        num_leaves=8,
        min_child_samples=40,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        random_state=seed,
        verbose=-1,
    )
    clf.fit(X[~holdout], y[~holdout], eval_X=X[holdout], eval_y=y[holdout],
            callbacks=[lgb.early_stopping(100, verbose=False)])
    return ChurnModel("lightgbm", clf, list(X.columns))
