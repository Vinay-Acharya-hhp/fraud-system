"""Prediction API for the cost-aware fraud detector."""
from fastapi import FastAPI
from pydantic import Field, create_model

from src.predictor import FraudDetector

# Input: the 28 PCA features V1..V28 plus the raw Amount (must be >= 0).
fields = {f"V{i}": (float, ...) for i in range(1, 29)}
fields["Amount"] = (float, Field(..., ge=0))
Transaction = create_model("Transaction", **fields)

app = FastAPI(title="Cost-Aware Credit Card Fraud Detection API")
detector = FraudDetector()          # loaded once, when the server starts


@app.get("/health")
def health():
    return {"status": "ok", "threshold": detector.threshold, "review_fee": detector.fee}


@app.post("/predict")
def predict(tx: Transaction):
    return detector.predict_one(tx.model_dump())
