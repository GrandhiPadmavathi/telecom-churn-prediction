"""Load and clean the IBM Telco Customer Churn data (7,043 customers, one row each)."""

import pandas as pd

from churn.config import SETTINGS

YES_NO = ["Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    # TotalCharges is text and blank for the 11 customers who joined this month (tenure 0): nothing billed yet.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].str.strip().replace("", "0"), errors="raise")
    for col in YES_NO:
        df[col] = (df[col] == "Yes").astype(int)
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)
    return df.rename(columns={"customerID": "customer_id", "Churn": "churned"})


def load() -> pd.DataFrame:
    path = SETTINGS.paths.raw_csv
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python scripts/download_data.py` first")
    return clean(pd.read_csv(path, dtype={"TotalCharges": str}))
