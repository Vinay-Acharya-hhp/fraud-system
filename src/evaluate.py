"""FINAL evaluation on the held-out test set. Run this ONCE."""
import json

from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score)

from src.choose_threshold import THRESHOLD_PATH
from src.cost import cost_of_doing_nothing, savings, total_cost
from src.data import load_data, split_xy, time_split
from src.predictor import FraudDetector


def report(label, y, scores, amount, threshold, fee):
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    cost = total_cost(y, scores, amount, threshold, fee)
    print(f"{label:<30} thr {threshold:.4f}  flagged {pred.sum():>4}  TP {tp:>3}  FP {fp:>4}  FN {fn:>3}")
    print(f"{'':<30} precision {precision_score(y, pred, zero_division=0):.3f}  "
          f"recall {recall_score(y, pred):.3f}  F1 {f1_score(y, pred):.3f}  "
          f"MCC {matthews_corrcoef(y, pred):.3f}  cost {cost:.2f}  "
          f"savings {savings(y, scores, amount, threshold, fee):.3f}")


def main():
    det = FraudDetector()
    _, _, test = time_split(load_data())
    _, y_test, amount_test = split_xy(test)
    y, amount = y_test.to_numpy(), amount_test.to_numpy()
    scores = det.score(test)

    print(f"Test set: {len(y)} rows, {int(y.sum())} frauds. "
          f"Cost of doing nothing: {cost_of_doing_nothing(y, amount):.2f}")
    print(f"AUPRC (threshold-free): {average_precision_score(y, scores):.4f}\n")

    info = json.loads(THRESHOLD_PATH.read_text())
    for fee, res in info["by_fee"].items():
        fee = float(fee)
        report(f"fee {fee:g} | cost-based", y, scores, amount, res["threshold"], fee)
        report(f"fee {fee:g} | default 0.5", y, scores, amount, 0.5, fee)
        print()


if __name__ == "__main__":
    main()
