"""Load the saved model + threshold and score transactions."""
import json

import joblib
import numpy as np
import pandas as pd

from src.choose_threshold import THRESHOLD_PATH
from src.train import MODEL_PATH


class FraudDetector:
    def __init__(self):
        bundle = joblib.load(MODEL_PATH)
        self.pipeline = bundle["pipeline"]
        self.features = bundle["features"]       # V1..V28 + Amount, in training order
        info = json.loads(THRESHOLD_PATH.read_text())
        self.threshold = info["threshold"]
        self.fee = info["main_fee"]

    def score(self, rows: pd.DataFrame) -> np.ndarray:
        """rows must contain V1..V28 and the RAW Amount (not the log)."""
        X = rows[self.features].copy()
        X["Amount"] = np.log1p(X["Amount"])      # same transform as in training
        return self.pipeline.predict_proba(X)[:, 1]

    def predict_one(self, transaction: dict) -> dict:
        p = float(self.score(pd.DataFrame([transaction]))[0])
        return {"fraud_probability": p,
                "is_fraud_alert": p >= self.threshold,
                "threshold": self.threshold}


if __name__ == "__main__":
    from sklearn.metrics import average_precision_score

    from src.data import load_data, time_split

    _, val, _ = time_split(load_data())
    det = FraudDetector()

    sample = pd.concat([val[val["Class"] == 1].head(3), val[val["Class"] == 0].head(3)])
    for label, amount, p in zip(sample["Class"], sample["Amount"], det.score(sample)):
        print(f"true class {label}  amount {amount:8.2f}  prob {p:.4f}  alert {p >= det.threshold}")

    print("validation AUPRC via FraudDetector:",
          round(average_precision_score(val["Class"], det.score(val)), 4))
