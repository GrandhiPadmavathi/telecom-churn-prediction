"""End-to-end run: split, train both models, calibrate, pick a champion, evaluate, price the campaign, save.

python -m churn.train
"""

import json

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

from churn import campaign, data, explain, features, metrics, models, plots, survival
from churn.config import SETTINGS


def split(df, seed):
    """60% train, 20% validation (calibration and champion choice), 20% test (reported once)."""
    train, rest = train_test_split(df, test_size=0.4, stratify=df["churned"], random_state=seed)
    valid, test = train_test_split(rest, test_size=0.5, stratify=rest["churned"], random_state=seed)
    return train, valid, test


def survival_by_contract(df) -> dict:
    curves = {c: survival.kaplan_meier(g["tenure"], g["churned"]) for c, g in df.groupby("Contract")}
    test = survival.logrank(df["tenure"], df["churned"], df["Contract"])
    summary = {c: {"retained_12m": survival.survival_at(k, 12), "retained_24m": survival.survival_at(k, 24),
                   "median_months": survival.median_survival(k)} for c, k in curves.items()}
    return {"curves": curves, "logrank": test, "summary": summary}


def gender_audit(test, contact) -> dict:
    """Gender is not a model input; check the call list does not treat the groups very differently anyway."""
    out = {}
    for g, idx in test.groupby("gender").indices.items():
        churners = test["churned"].values[idx] == 1
        out[g] = {"contact_rate": float(contact[idx].mean()),
                  "churners_reached": float(contact[idx][churners].mean()) if churners.any() else None}
    return out


def main() -> dict:
    s = SETTINGS
    df = data.load()
    train, valid, test = split(df, s.seed)
    X_train = features.build(train)
    cols = list(X_train.columns)
    X_valid, X_test = features.build(valid, cols), features.build(test, cols)

    candidates = [models.logistic_regression(X_train, train["churned"], s.seed),
                  models.lightgbm(X_train, train["churned"], s.seed)]
    for m in candidates:
        m.calibrate(X_valid, valid["churned"], seed=s.seed)
    valid_scores = {m.name: metrics.report(valid["churned"], m.predict(X_valid)) for m in candidates}
    champion = max(candidates, key=lambda m: valid_scores[m.name]["pr_auc"])
    baseline = next(m for m in candidates if m.name == "logistic_regression")
    challenger = next(m for m in candidates if m.name == "lightgbm")

    y = test["churned"].values
    monthly = test["MonthlyCharges"].values
    test_p = {m.name: m.predict(X_test) for m in candidates}
    p = test_p[champion.name]

    rules = campaign.policies(p, y, monthly, s.campaign)
    curve = campaign.value_curve(p, y, monthly, s.campaign)
    best = max(curve, key=lambda r: r["value"])
    contact = s.campaign.expected_value(p, monthly) > 0

    contrib = champion.contributions(X_test)
    importance = explain.global_importance(contrib)
    surv = survival_by_contract(df)

    out = {
        "data": {"customers": len(df), "churn_rate": float(df["churned"].mean()),
                 "split": {"train": len(train), "valid": len(valid), "test": len(test)}, "features": len(cols)},
        "champion": champion.name,
        "calibration": {m.name: {"chosen": m.calibration, "cv_ece": m.calibration_cv_ece} for m in candidates},
        "validation": valid_scores,
        "test": {name: metrics.report(y, q) for name, q in test_p.items()},
        "lightgbm_minus_logreg_ci": metrics.bootstrap_diff(y, test_p[challenger.name], test_p[baseline.name],
                                                           seed=s.seed),
        "lightgbm_trees": int(challenger.estimator.best_iteration_),
        "campaign": {"assumptions": s.campaign.__dict__, "policies": rules,
                     "best_cutoff": {"contact_rate": best["contact_rate"], "value": best["value"]}},
        "importance": importance,
        "survival": {"logrank": surv["logrank"], "by_contract": surv["summary"]},
        "gender_audit": gender_audit(test, contact),
    }

    paths = s.paths
    paths.artifacts.mkdir(parents=True, exist_ok=True)
    paths.reports.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": champion, "columns": cols, "test_ids": test["customer_id"].tolist()}, paths.bundle)
    paths.metrics.write_text(json.dumps(out, indent=2))

    figs = paths.reports / "figures"
    figs.mkdir(exist_ok=True)
    plots.churn_rates(df, figs / "churn_rates.png")
    plots.curves(y, test_p, figs / "roc_pr.png")
    plots.calibration(y, p, figs / "calibration.png")
    plots.value_curve(curve, best, figs / "campaign_value.png")
    plots.importance(importance, figs / "importance.png")
    plots.survival(surv["curves"], surv["logrank"], figs / "survival_by_contract.png")
    return out


if __name__ == "__main__":
    result = main()
    t = result["test"][result["champion"]]
    print(f"champion: {result['champion']}  test PR-AUC {t['pr_auc']:.3f}  ROC-AUC {t['roc_auc']:.3f}  "
          f"top-10% lift {t['top10']['lift']:.2f}")
    for name, r in result["campaign"]["policies"].items():
        print(f"  {name:26s} call {r['contact_rate']:6.1%}  reach {r['churners_reached']:6.1%}  value ${r['value']:>10,.0f}")
