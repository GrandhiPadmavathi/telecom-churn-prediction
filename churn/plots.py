from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

COLORS = {"logistic_regression": "#7f8c8d", "lightgbm": "#2e86de"}


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def churn_rates(df: pd.DataFrame, path: Path) -> None:
    bands = pd.cut(df["tenure"], [-1, 6, 12, 24, 48, 72], labels=["0-6", "7-12", "13-24", "25-48", "49-72"])
    panels = [("Contract", df["Contract"]), ("Internet service", df["InternetService"]),
              ("Payment method", df["PaymentMethod"].str.replace(" (automatic)", " (auto)")), ("Tenure (months)", bands)]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    for ax, (title, key) in zip(axes, panels):
        rate = df.groupby(key, observed=True)["churned"].mean()
        ax.barh([str(i) for i in rate.index], rate.values, color="#e67e22")
        ax.axvline(df["churned"].mean(), color="black", ls="--", lw=1)
        for y, v in enumerate(rate.values):
            ax.text(v + 0.01, y, f"{v:.0%}", va="center", fontsize=9)
        ax.set_title(title)
        ax.set_xlim(0, 0.65)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    fig.suptitle("Churn rate by segment (dashed line = overall rate)")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def curves(y, scores: dict, path: Path) -> None:
    fig, (roc, pr) = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, p in scores.items():
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        roc.plot(fpr, tpr, label=name, color=COLORS.get(name))
        pr.plot(rec, prec, label=name, color=COLORS.get(name))
    roc.plot([0, 1], [0, 1], "k--", lw=1)
    pr.axhline(np.mean(y), color="black", ls="--", lw=1, label="random")
    roc.set(title="ROC curve (test)", xlabel="False positive rate", ylabel="True positive rate")
    pr.set(title="Precision-recall curve (test)", xlabel="Recall (churners found)", ylabel="Precision")
    roc.legend()
    pr.legend()
    _save(fig, path)


def calibration(y, p, path: Path, bins: int = 10) -> None:
    y, p = np.asarray(y), np.asarray(p)
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    which = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    pred = [p[which == b].mean() for b in range(bins) if (which == b).any()]
    obs = [y[which == b].mean() for b in range(bins) if (which == b).any()]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.plot(pred, obs, "o-", color="#2e86de")
    ax.set(title="Calibration (test): predicted vs actual", xlabel="Predicted churn probability",
           ylabel="Actual churn rate", xlim=(0, 1), ylim=(0, 1))
    _save(fig, path)


def value_curve(rows: list[dict], best: dict, path: Path) -> None:
    share = [r["share"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(share, [r["value"] for r in rows], color="#27ae60")
    ax.axhline(0, color="black", lw=1)
    ax.axvline(best["contact_rate"], color="#27ae60", ls="--", lw=1)
    ax.annotate(f"call {best['contact_rate']:.0%}: ${best['value']:,.0f}", (best["contact_rate"], best["value"]),
                xytext=(12, -30), textcoords="offset points")
    ax.set(title="Campaign value on test customers, calling the top k by expected value",
           xlabel="Share of customers called", ylabel="Realised value ($)")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    _save(fig, path)


def importance(imp: dict, path: Path) -> None:
    names, values = list(imp)[::-1], list(imp.values())[::-1]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(names, values, color="#2e86de")
    ax.set(title="What drives the churn score (mean |contribution| to log-odds)", xlabel="Mean absolute contribution")
    _save(fig, path)


def survival(curves_by_group: dict, test: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, c in curves_by_group.items():
        ax.step([0, *c["time"]], [1, *c["survival"]], where="post", label=name)
    p = "p < 1e-10" if test["p_value"] < 1e-10 else f"p = {test['p_value']:.1e}"
    ax.set(title=f"Kaplan-Meier: share still a customer by tenure (log-rank {p})",
           xlabel="Tenure (months)", ylabel="Still a customer", ylim=(0, 1.02))
    ax.legend(title="Contract")
    _save(fig, path)
