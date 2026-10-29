from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"


@dataclass(frozen=True)
class Paths:
    raw_csv: Path = ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
    artifacts: Path = ROOT / "artifacts"
    reports: Path = ROOT / "reports"

    @property
    def bundle(self) -> Path:
        return self.artifacts / "model_bundle.joblib"

    @property
    def metrics(self) -> Path:
        return self.reports / "metrics.json"


@dataclass(frozen=True)
class Campaign:
    """Economics of a retention call, per customer, in the dataset's currency (US$ per month of charges).

    Calling costs `contact_cost`, and everyone called is offered one free month (`incentive_months` x monthly
    charges). A customer who would have churned stays with probability `save_rate` and is then worth
    `horizon_months` of charges. Customers who would have stayed anyway still take the incentive.
    These are illustrative assumptions, documented in the README.
    """

    contact_cost: float = 10.0
    incentive_months: float = 1.0
    save_rate: float = 0.30
    horizon_months: float = 12.0

    def cost(self, monthly):
        return self.contact_cost + self.incentive_months * monthly

    def expected_value(self, p_churn, monthly):
        """Expected gain from calling one customer, given their churn probability."""
        return p_churn * self.save_rate * self.horizon_months * monthly - self.cost(monthly)

    def realised_value(self, churned, monthly):
        """Gain from calling a customer whose true outcome is known (used to score a policy on test data)."""
        return churned * self.save_rate * self.horizon_months * monthly - self.cost(monthly)

    def breakeven_p(self, monthly):
        return self.cost(monthly) / (self.save_rate * self.horizon_months * monthly)


@dataclass(frozen=True)
class Settings:
    seed: int = 42
    paths: Paths = field(default_factory=Paths)
    campaign: Campaign = field(default_factory=Campaign)


SETTINGS = Settings()
