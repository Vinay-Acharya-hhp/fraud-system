"""Load the credit card fraud data and split it by time."""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "creditcard.csv"
TARGET = "Class"


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    # Sort by time so "earlier" and "later" really mean that.
    return df.sort_values("Time").reset_index(drop=True)


def time_split(df, train_frac=0.6, val_frac=0.2):
    """Chronological split: first 60% train, next 20% validation, last 20% test."""
    n = len(df)
    i_train = int(n * train_frac)
    i_val = int(n * (train_frac + val_frac))
    return df.iloc[:i_train], df.iloc[i_train:i_val], df.iloc[i_val:]


def split_xy(part):
    """Return model inputs X, labels y, and the raw Amount (kept for the cost rule later)."""
    X = part.drop(columns=[TARGET, "Time"]).copy()  # Time is only used for ordering
    X["Amount"] = np.log1p(X["Amount"])             # Amount is very skewed, so use log
    return X, part[TARGET], part["Amount"]


def summary(name, part):
    n = len(part)
    frauds = int(part[TARGET].sum())
    print(f"{name:>5}: {n:>7} rows, {frauds:>4} frauds ({100 * frauds / n:.3f}%), "
          f"Time {part['Time'].min():.0f} to {part['Time'].max():.0f}")


if __name__ == "__main__":
    df = load_data()
    for name, part in zip(["train", "val", "test"], time_split(df)):
        summary(name, part)
