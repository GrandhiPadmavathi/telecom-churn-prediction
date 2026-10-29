"""Kaplan-Meier survival curves and the log-rank test, written out so every step can be explained.

Each customer has a duration (tenure in months) and an event flag (1 = churned, 0 = still a customer, which
is "censored": we only know they lasted at least that long).
"""

import numpy as np
from scipy.stats import chi2


def kaplan_meier(durations, events) -> dict:
    """S(t) = product over event times t_i <= t of (1 - d_i / n_i): d_i churned at t_i out of n_i still at risk."""
    durations, events = np.asarray(durations, float), np.asarray(events, int)
    times = np.unique(durations[events == 1])
    at_risk = np.array([(durations >= t).sum() for t in times])
    churned = np.array([((durations == t) & (events == 1)).sum() for t in times])
    survival = np.cumprod(1 - churned / at_risk)
    return {"time": times.tolist(), "survival": survival.tolist(), "at_risk": at_risk.tolist()}


def survival_at(curve: dict, t: float) -> float:
    s = 1.0
    for time, value in zip(curve["time"], curve["survival"]):
        if time > t:
            break
        s = value
    return float(s)


def median_survival(curve: dict) -> float | None:
    """First time the curve drops to 0.5 or below; None when more than half never churn in the data."""
    for time, value in zip(curve["time"], curve["survival"]):
        if value <= 0.5 + 1e-12:
            return float(time)
    return None


def logrank(durations, events, groups) -> dict:
    """k-group log-rank test: do the groups' survival curves differ by more than chance?"""
    durations, events, groups = np.asarray(durations, float), np.asarray(events, int), np.asarray(groups)
    labels = sorted(set(groups.tolist()))
    k = len(labels)
    observed_minus_expected = np.zeros(k)
    variance = np.zeros((k, k))
    for t in np.unique(durations[events == 1]):
        at_risk = durations >= t
        n = at_risk.sum()
        d = ((durations == t) & (events == 1)).sum()
        n_g = np.array([(at_risk & (groups == g)).sum() for g in labels])
        d_g = np.array([((durations == t) & (events == 1) & (groups == g)).sum() for g in labels])
        observed_minus_expected += d_g - d * n_g / n
        if n > 1:
            share = n_g / n
            variance += d * (n - d) / (n - 1) * (np.diag(share) - np.outer(share, share))
    stat = float(observed_minus_expected[:-1] @ np.linalg.pinv(variance[:-1, :-1]) @ observed_minus_expected[:-1])
    return {"chi2": stat, "df": k - 1, "p_value": float(chi2.sf(stat, k - 1))}
