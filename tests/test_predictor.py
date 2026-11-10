import numpy as np

from src.data import load_data, split_xy, time_split
from src.predictor import FraudDetector


def test_detector_matches_pipeline_directly():
    _, val, _ = time_split(load_data())
    det = FraudDetector()
    X_val, _, _ = split_xy(val)
    direct = det.pipeline.predict_proba(X_val)[:, 1]
    assert np.allclose(direct, det.score(val))
