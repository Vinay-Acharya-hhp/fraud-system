"""Leakage control: compare the correct protocol with flawed ones (uses all the data)."""
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score as auprc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data import load_data, split_xy

SEEDS = [1, 2, 3]


def run(seed, X, X_scaled, y):
    n = len(y)

    # B: random split, no oversampling (class weights, training rows only)
    tr, te = train_test_split(np.arange(n), test_size=0.2, stratify=y, random_state=seed)
    model = HistGradientBoostingClassifier(class_weight="balanced", random_state=seed)
    model.fit(X[tr], y[tr])
    b = auprc(y[te], model.predict_proba(X[te])[:, 1])

    # C: FLAWED - oversample the WHOLE dataset first, then split
    X_res, y_res = SMOTE(random_state=seed).fit_resample(X_scaled, y)
    assert np.allclose(X_res[:n], X_scaled)      # original rows come first, synthetic after
    is_real = np.arange(len(y_res)) < n
    tr, te = train_test_split(np.arange(len(y_res)), test_size=0.2,
                              stratify=y_res, random_state=seed)
    model = HistGradientBoostingClassifier(random_state=seed).fit(X_res[tr], y_res[tr])
    p = model.predict_proba(X_res[te])[:, 1]
    real = is_real[te]
    return b, auprc(y_res[te], p), auprc(y_res[te][real], p[real])


def main():
    X_df, y_s, _ = split_xy(load_data())
    X, y = X_df.to_numpy(), y_s.to_numpy()
    X_scaled = StandardScaler().fit_transform(X)   # fitted on ALL rows (also part of the flawed way)

    rows = np.array([run(s, X, X_scaled, y) for s in SEEDS])
    names = ["B  random split, no oversampling",
             "C1 flawed: SMOTE first, whole test split",
             "C2 flawed: SMOTE first, real rows only"]
    for name, col in zip(names, rows.T):
        print(f"{name:<42} AUPRC mean {col.mean():.4f}   per seed: "
              + ", ".join(f"{v:.4f}" for v in col))


if __name__ == "__main__":
    main()
