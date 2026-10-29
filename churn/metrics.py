"""Metrics that match how a retention team works: they can only call the top of the list."""

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def expected_calibration_error(y, p, bins: int = 10) -> float:
    """Weighted gap between predicted and observed churn rate over quantile bins."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)[1:-1]))
    which = np.searchsorted(edges, p, side="right")
    return float(sum((which == b).mean() * abs(p[which == b].mean() - y[which == b].mean()) for b in np.unique(which)))


def top_share(y, p, share: float) -> dict:
    """Precision, recall and lift when calling the `share` of customers with the highest scores."""
    y, p = np.asarray(y), np.asarray(p)
    k = max(1, int(round(share * len(y))))
    top = np.argsort(-p, kind="stable")[:k]
    precision = y[top].mean()
    return {
        "share": share,
        "precision": float(precision),
        "recall": float(y[top].sum() / max(y.sum(), 1)),
        "lift": float(precision / y.mean()),
    }


def report(y, p) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p),
        "top10": top_share(y, p, 0.10),
        "top20": top_share(y, p, 0.20),
        "base_rate": float(np.mean(y)),
    }


def bootstrap_diff(y, p, p_other, n_boot: int = 500, seed: int = 0) -> dict:
    """Paired bootstrap 95% intervals for the ROC-AUC and PR-AUC gain of `p` over `p_other`."""
    y, p, p_other = np.asarray(y), np.asarray(p), np.asarray(p_other)
    rng = np.random.default_rng(seed)
    gains = {"roc_auc": [], "pr_auc": []}
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        gains["roc_auc"].append(roc_auc_score(y[i], p[i]) - roc_auc_score(y[i], p_other[i]))
        gains["pr_auc"].append(average_precision_score(y[i], p[i]) - average_precision_score(y[i], p_other[i]))
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in gains.items()}
