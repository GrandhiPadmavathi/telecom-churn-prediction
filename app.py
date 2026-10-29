"""Retention desk: who to call this month, why they are at risk, and what the campaign is worth.

streamlit run app.py
"""

import json
from dataclasses import replace

import joblib
import pandas as pd
import streamlit as st

from churn import campaign, data, explain, features
from churn.config import SETTINGS

st.set_page_config(page_title="Churn retention desk", layout="wide")


@st.cache_resource
def load():
    bundle = joblib.load(SETTINGS.paths.bundle)
    df = data.load().set_index("customer_id")
    test = df.loc[bundle["test_ids"]]
    X = features.build(test, bundle["columns"])
    model = bundle["model"]
    return test, X, model.predict(X), model.contributions(X), model.name, json.loads(SETTINGS.paths.metrics.read_text())


if not SETTINGS.paths.bundle.exists():
    st.error("No trained model found. Run `python scripts/download_data.py` and `python -m churn.train` first.")
    st.stop()

customers, X, p, contrib, model_name, report = load()
monthly = customers["MonthlyCharges"].values

with st.sidebar:
    st.header("Campaign assumptions")
    base = SETTINGS.campaign
    c = replace(
        base,
        contact_cost=st.number_input("Cost per call ($)", 0.0, 100.0, base.contact_cost, 1.0),
        incentive_months=st.number_input("Free months offered", 0.0, 6.0, base.incentive_months, 0.5),
        save_rate=st.slider("Share of would-be churners saved", 0.05, 0.9, base.save_rate, 0.05),
        horizon_months=st.number_input("Months of revenue a save is worth", 1.0, 60.0, base.horizon_months, 1.0),
    )
    st.caption(f"Scores from the {model_name} model on {len(customers):,} held-out customers.")

ev = c.expected_value(p, monthly)
table = pd.DataFrame({"churn_risk": p, "monthly_charges": monthly, "expected_value": ev,
                      "contract": customers["Contract"], "tenure": customers["tenure"],
                      "actually_churned": customers["churned"]}, index=customers.index).sort_values(
    "expected_value", ascending=False)

calls, detail_tab, why_tab, results_tab = st.tabs(["Call list", "Customer detail", "Why", "Model results"])

with calls:
    worth = table[table["expected_value"] > 0]
    out = campaign.outcome(ev > 0, customers["churned"].values, monthly, c)
    a, b, d = st.columns(3)
    a.metric("Customers worth calling", f"{len(worth):,}", f"{len(worth) / len(table):.0%} of base")
    b.metric("Churners on the list", f"{out['churners_reached']:.0%}")
    d.metric("Realised value on this test month", f"${out['value']:,.0f}")
    st.dataframe(worth.drop(columns="actually_churned").style.format(
        {"churn_risk": "{:.0%}", "monthly_charges": "${:.2f}", "expected_value": "${:.2f}"}), height=420)
    st.download_button("Download call list (CSV)", worth.drop(columns="actually_churned").to_csv(), "call_list.csv")

with detail_tab:
    cid = st.selectbox("Customer", table.index, format_func=lambda i: f"{i}  ({table.loc[i, 'churn_risk']:.0%} risk)")
    row = customers.loc[cid]
    a, b, d = st.columns(3)
    a.metric("Churn risk", f"{table.loc[cid, 'churn_risk']:.0%}")
    b.metric("Break-even risk for this plan", f"{c.breakeven_p(row['MonthlyCharges']):.0%}")
    d.metric("Expected value of a call", f"${table.loc[cid, 'expected_value']:.2f}")
    st.subheader("Main reasons")
    for r in explain.reasons(contrib.loc[cid], X.loc[cid]):
        st.write(f"- **{r['text']}** (pushes the log-odds of churn by {r['impact']:+.2f})")
    st.json(row.drop("churned").to_dict(), expanded=False)

with why_tab:
    figs = SETTINGS.paths.reports / "figures"
    st.image(str(figs / "churn_rates.png"))
    left, right = st.columns(2)
    left.image(str(figs / "importance.png"))
    right.image(str(figs / "survival_by_contract.png"))

with results_tab:
    t = report["test"][report["champion"]]
    st.write(f"Champion **{report['champion']}**: PR-AUC {t['pr_auc']:.3f}, ROC-AUC {t['roc_auc']:.3f}, "
             f"top-10% lift {t['top10']['lift']:.2f}, ECE {t['ece']:.3f} (base churn rate {t['base_rate']:.1%}).")
    st.dataframe(pd.DataFrame(report["campaign"]["policies"]).T)
    left, right = st.columns(2)
    left.image(str(figs / "roc_pr.png"))
    right.image(str(figs / "campaign_value.png"))
