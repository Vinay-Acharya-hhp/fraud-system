"""Train the fraud model: split by time, balance only the training data,
choose the best setup by validation AUPRC, and save it."""
from pathlib import Path

import joblib
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from src.data import load_data, time_split, split_xy, summary

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "fraud_model.joblib"
RANDOM_STATE = 42


def make_pipeline(strategy):
    steps = [("scale", StandardScaler())]
    if strategy == "smote":
        steps.append(("balance", SMOTE(random_state=RANDOM_STATE)))
        model = HistGradientBoostingClassifier(random_state=RANDOM_STATE)
    elif strategy == "class_weight":
        model = HistGradientBoostingClassifier(class_weight="balanced",
                                               random_state=RANDOM_STATE)
    else:
        raise ValueError(f"unknown strategy: {strategy}")
    steps.append(("model", model))
    return Pipeline(steps)


def main():
    df = load_data()
    train, val, test = time_split(df)
    for name, part in zip(["train", "val", "test"], (train, val, test)):
        summary(name, part)

    X_train, y_train, _ = split_xy(train)
    X_val, y_val, _ = split_xy(val)
    # The test set is NOT used here. It is only touched once, in the final evaluation step.

    results = {}
    for strategy in ["class_weight", "smote"]:
        pipe = make_pipeline(strategy)
        pipe.fit(X_train, y_train)
        scores = pipe.predict_proba(X_val)[:, 1]
        ap = average_precision_score(y_val, scores)
        results[strategy] = (ap, pipe)
        print(f"{strategy:>12}: validation AUPRC = {ap:.4f}")

    best = max(results, key=lambda k: results[k][0])
    print("Chosen setup:", best)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"pipeline": results[best][1],
                 "strategy": best,
                 "features": list(X_train.columns)}, MODEL_PATH)
    print("Saved to", MODEL_PATH)


if __name__ == "__main__":
    main()
