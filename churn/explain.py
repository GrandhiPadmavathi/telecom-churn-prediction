"""Plain-English reasons behind one customer's churn score, from the model's additive contributions."""

import pandas as pd

LABELS = {
    "Contract=Month-to-month": "on a month-to-month contract",
    "Contract=Two year": "on a two-year contract",
    "Contract=One year": "on a one-year contract",
    "tenure": "tenure",
    "new_customer": "joined in the last 6 months",
    "InternetService=Fiber optic": "has fiber internet",
    "InternetService=No": "has no internet service",
    "PaymentMethod=Electronic check": "pays by electronic check",
    "auto_pay": "pays automatically",
    "OnlineSecurity=No": "no online security add-on",
    "TechSupport=No": "no tech support add-on",
    "MonthlyCharges": "monthly charges",
    "TotalCharges": "total billed so far",
    "price_change": "current bill vs. their average",
    "PaperlessBilling": "paperless billing",
    "n_add_ons": "number of add-on services",
    "SeniorCitizen": "senior citizen",
    "Dependents": "has dependents",
    "Partner": "has a partner",
}


FLAGS = {"SeniorCitizen", "Partner", "Dependents", "PhoneService", "PaperlessBilling", "new_customer", "auto_pay",
         "has_support_bundle"}
MONEY = {"MonthlyCharges", "TotalCharges", "avg_monthly_billed", "price_change"}


def label(column: str) -> str:
    return LABELS.get(column, column.replace("=", ": "))


def describe(column: str, value: float) -> str:
    """'on a two-year contract' for a 1, 'not on a two-year contract' for a 0, 'tenure: 3 months' for numbers."""
    if "=" in column or column in FLAGS:
        return label(column) if value else f"not {label(column)}"
    if column in MONEY:
        return f"{label(column)}: ${value:,.2f}"
    unit = (" month" if value == 1 else " months") if column == "tenure" else ""
    return f"{label(column)}: {value:g}{unit}"


def reasons(contrib: pd.Series, row: pd.Series, n: int = 4) -> list[dict]:
    """The `n` inputs that push this customer's churn risk up the most (or down, if none push up)."""
    ups = contrib[contrib > 0].sort_values(ascending=False)
    picked = ups if len(ups) else contrib.sort_values()
    return [
        {"feature": col, "text": describe(col, row[col]), "value": float(row[col]), "impact": float(contrib[col])}
        for col in picked.index[:n]
    ]


def global_importance(contrib: pd.DataFrame, top: int = 15) -> dict:
    return contrib.abs().mean().sort_values(ascending=False).head(top).round(4).to_dict()
