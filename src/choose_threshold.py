"""Pick the alert threshold on the VALIDATION set (never on the test set)."""
import json

import joblib

from src.cost import best_threshold, cost_of_doing_nothing, savings, total_cost
from src.data import load_data, split_xy, time_split
from src.train import MODEL_PATH

THRESHOLD_PATH = MODEL_PATH.parent / "threshold.json"
MAIN_FEE = 5                      # review fee per flagged transaction (an assumption)
SENSITIVITY_FEES = [1, 5, 10, 25]


def main():
    pipe = joblib.load(MODEL_PATH)["pipeline"]

    _, val, _ = time_split(load_data())
    X_val, y_val, amount_val = split_xy(val)
    scores = pipe.predict_proba(X_val)[:, 1]

    print(f"Cost of doing nothing (validation): {cost_of_doing_nothing(y_val, amount_val):.2f}")
    print(f"{'fee':>5} {'threshold':>10} {'flagged':>8} {'cost':>10} {'savings':>8} | savings at 0.5")
    results = {}
    for fee in SENSITIVITY_FEES:
        t = best_threshold(y_val, scores, amount_val, fee)
        flagged = int((scores >= t).sum())
        cost = total_cost(y_val, scores, amount_val, t, fee)
        sav = savings(y_val, scores, amount_val, t, fee)
        sav_default = savings(y_val, scores, amount_val, 0.5, fee)
        print(f"{fee:>5} {t:>10.4f} {flagged:>8} {cost:>10.2f} {sav:>8.3f} | {sav_default:>8.3f}")
        results[str(fee)] = {"threshold": t, "flagged_on_val": flagged, "val_savings": float(sav)}

    out = {"main_fee": MAIN_FEE,
           "threshold": results[str(MAIN_FEE)]["threshold"],
           "by_fee": results}
    THRESHOLD_PATH.write_text(json.dumps(out, indent=2))
    print("Saved to", THRESHOLD_PATH)


if __name__ == "__main__":
    main()
