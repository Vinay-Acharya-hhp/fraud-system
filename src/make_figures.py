"""Make the paper figures (IEEE single-column size). Output goes to paper/figures/."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                     # no display needed (WSL)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve

from src.choose_threshold import THRESHOLD_PATH
from src.cost import savings
from src.data import load_data, split_xy, time_split
from src.predictor import FraudDetector

OUT = Path(__file__).resolve().parent.parent / "paper" / "figures"
plt.rcParams.update({"font.size": 8, "savefig.dpi": 300,
                     "axes.grid": True, "grid.alpha": 0.3})
SIZE = (3.5, 2.6)                         # inches, one IEEE column

# Copied from paper/test_results.txt and paper/leakage_control.txt - check them!
A_TEST_AUPRC = 0.7529
B = [0.7993, 0.7858, 0.7647]
C1 = [0.9998, 0.9997, 0.9999]
C2 = [0.9111, 0.8940, 0.9348]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.pdf")      # vector, best for LaTeX
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)


def fig_pr_curve(y, scores, points):
    prec, rec, _ = precision_recall_curve(y, scores)
    fig, ax = plt.subplots(figsize=SIZE)
    ax.plot(rec, prec, color="black",
            label=f"PR curve (AUPRC = {average_precision_score(y, scores):.3f})")
    for (name, t), marker in zip(points.items(), ["o", "s"]):
        pred = scores >= t
        tp = (pred & (y == 1)).sum()
        ax.plot(tp / y.sum(), tp / max(pred.sum(), 1), marker, color="black",
                markerfacecolor="white", label=f"{name} (t = {t:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left")
    save(fig, "pr_curve")


def fig_protocols():
    labels = ["A\ntime split,\ncorrect", "B\nrandom split,\nno SMOTE",
              "C2\nSMOTE first,\nreal rows", "C1\nSMOTE first,\nwhole split"]
    means = [A_TEST_AUPRC, np.mean(B), np.mean(C2), np.mean(C1)]
    errs = [0, np.std(B, ddof=1), np.std(C2, ddof=1), np.std(C1, ddof=1)]
    fig, ax = plt.subplots(figsize=SIZE)
    ax.bar(range(4), means, yerr=errs, capsize=3,
           color=["0.25", "0.55", "0.75", "0.9"], edgecolor="black")
    for i, (m, e) in enumerate(zip(means, errs)):
        ax.text(i, m + e + 0.02, f"{m:.3f}", ha="center", fontsize=7)
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylabel("AUPRC")
    ax.set_ylim(0, 1.1)
    save(fig, "protocols")


def fig_savings(y, scores, amount, by_fee):
    fees = list(by_fee.keys())
    cost_based = [savings(y, scores, amount, by_fee[f]["threshold"], float(f)) for f in fees]
    default = [savings(y, scores, amount, 0.5, float(f)) for f in fees]
    x = np.arange(len(fees))
    w = 0.38
    fig, ax = plt.subplots(figsize=SIZE)
    ax.bar(x - w / 2, cost_based, w, label="cost-based threshold",
           color="0.3", edgecolor="black")
    ax.bar(x + w / 2, default, w, label="default 0.5",
           color="white", edgecolor="black", hatch="//")
    ax.set_xticks(x)
    ax.set_xticklabels(fees)
    ax.set_xlabel("Review fee per flagged transaction")
    ax.set_ylabel("Savings vs. doing nothing")
    ax.set_ylim(0, 0.85)
    ax.legend(loc="upper right")
    save(fig, "savings_by_fee")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    det = FraudDetector()
    _, _, test = time_split(load_data())
    _, y_s, amount_s = split_xy(test)
    y, amount = y_s.to_numpy(), amount_s.to_numpy()
    scores = det.score(test)
    info = json.loads(THRESHOLD_PATH.read_text())

    fig_pr_curve(y, scores, {"cost-based, fee 5": info["threshold"], "default": 0.5})
    fig_protocols()
    fig_savings(y, scores, amount, info["by_fee"])
    print("Saved figures to", OUT)


if __name__ == "__main__":
    main()
