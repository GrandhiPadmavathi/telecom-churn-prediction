"""Turn churn probabilities into a call list, and score call-list policies with known outcomes."""

import numpy as np

from churn.config import Campaign


def outcome(contact, churned, monthly, campaign: Campaign) -> dict:
    contact, churned, monthly = np.asarray(contact, bool), np.asarray(churned), np.asarray(monthly, float)
    value = campaign.realised_value(churned[contact], monthly[contact]).sum()
    return {
        "contacted": int(contact.sum()),
        "contact_rate": float(contact.mean()),
        "churners_reached": float(churned[contact].sum() / max(churned.sum(), 1)),
        "precision": float(churned[contact].mean()) if contact.any() else 0.0,
        "value": float(value),
        "value_per_customer": float(value / len(contact)),
    }


def top_share(p, share: float):
    k = int(round(share * len(p)))
    mask = np.zeros(len(p), bool)
    mask[np.argsort(-np.asarray(p), kind="stable")[:k]] = True
    return mask


def policies(p, churned, monthly, campaign: Campaign) -> dict:
    """Compare simple call-list rules with the expected-value rule on the same customers."""
    ev = campaign.expected_value(np.asarray(p), np.asarray(monthly, float))
    rules = {
        "call_nobody": np.zeros(len(p), bool),
        "call_everyone": np.ones(len(p), bool),
        "top_10pct_by_risk": top_share(p, 0.10),
        "top_20pct_by_risk": top_share(p, 0.20),
        "positive_expected_value": ev > 0,
    }
    return {name: outcome(mask, churned, monthly, campaign) for name, mask in rules.items()}


def value_curve(p, churned, monthly, campaign: Campaign, steps: int = 50) -> list[dict]:
    """Realised value when calling the top k customers ranked by expected value, for k from 0 to everyone."""
    ev = campaign.expected_value(np.asarray(p), np.asarray(monthly, float))
    order = np.argsort(-ev, kind="stable")
    rows = []
    for share in np.linspace(0, 1, steps + 1):
        mask = np.zeros(len(p), bool)
        mask[order[: int(round(share * len(p)))]] = True
        rows.append({"share": float(share), **outcome(mask, churned, monthly, campaign)})
    return rows
