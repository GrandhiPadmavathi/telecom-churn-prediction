"""Turn one customer row into model inputs: a few engineered signals plus one-hot service and contract columns.

Gender is kept out of the model; it is only used to audit outcomes.
"""

import numpy as np
import pandas as pd

ADD_ONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
CATEGORICAL = ["MultipleLines", "InternetService", "Contract", "PaymentMethod", *ADD_ONS]
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
           "PaperlessBilling"]
ENGINEERED = ["n_add_ons", "has_support_bundle", "auto_pay", "new_customer", "avg_monthly_billed", "price_change"]
NOT_INPUTS = ["gender"]


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["n_add_ons"] = sum((df[c] == "Yes").astype(int) for c in ADD_ONS)
    out["has_support_bundle"] = ((df["OnlineSecurity"] == "Yes") & (df["TechSupport"] == "Yes")).astype(int)
    out["auto_pay"] = df["PaymentMethod"].str.contains("automatic").astype(int)
    out["new_customer"] = (df["tenure"] <= 6).astype(int)
    # What the customer paid per month on average so far, and how today's bill compares with it.
    avg = np.where(df["tenure"] > 0, df["TotalCharges"] / df["tenure"].clip(lower=1), df["MonthlyCharges"])
    out["avg_monthly_billed"] = avg
    out["price_change"] = df["MonthlyCharges"] - avg
    return out


def build(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Model matrix. Pass the training `columns` when scoring new data so the one-hot columns line up."""
    onehot = pd.get_dummies(df[CATEGORICAL], prefix=CATEGORICAL, prefix_sep="=", dtype=int)
    X = pd.concat([df[NUMERIC].astype(float), engineer(df), onehot], axis=1)
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0)
    return X
