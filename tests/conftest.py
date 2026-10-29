import numpy as np
import pandas as pd
import pytest


def synthetic(n: int = 1500, seed: int = 0) -> pd.DataFrame:
    """Raw-format customers where month-to-month, fiber and short tenure raise churn, like the real data."""
    rng = np.random.default_rng(seed)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.21, 0.24])
    internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])
    tenure = rng.integers(0, 73, n)
    monthly = np.round(rng.uniform(18, 118, n), 2)
    total = np.where(tenure == 0, " ", np.round(monthly * tenure, 2).astype(str))
    yes_no = lambda: rng.choice(["Yes", "No"], n)  # noqa: E731
    addon = lambda: np.where(internet == "No", "No internet service", yes_no())  # noqa: E731
    logit = (-1.2 + 1.6 * (contract == "Month-to-month") - 0.8 * (contract == "Two year")
             + 0.7 * (internet == "Fiber optic") - 0.04 * tenure)
    churn = rng.random(n) < 1 / (1 + np.exp(-logit))
    return pd.DataFrame({
        "customerID": [f"C{i:05d}" for i in range(n)],
        "gender": rng.choice(["Female", "Male"], n),
        "SeniorCitizen": rng.integers(0, 2, n),
        "Partner": yes_no(), "Dependents": yes_no(), "tenure": tenure, "PhoneService": yes_no(),
        "MultipleLines": yes_no(), "InternetService": internet,
        "OnlineSecurity": addon(), "OnlineBackup": addon(), "DeviceProtection": addon(), "TechSupport": addon(),
        "StreamingTV": addon(), "StreamingMovies": addon(), "Contract": contract, "PaperlessBilling": yes_no(),
        "PaymentMethod": rng.choice(["Electronic check", "Mailed check", "Bank transfer (automatic)",
                                     "Credit card (automatic)"], n),
        "MonthlyCharges": monthly, "TotalCharges": total, "Churn": np.where(churn, "Yes", "No"),
    })


@pytest.fixture
def raw():
    return synthetic()
