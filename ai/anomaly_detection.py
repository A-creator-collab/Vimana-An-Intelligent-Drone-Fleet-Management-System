"""anomaly_detection.py - IsolationForest (multivariate) + sliding-window z-score (streaming)."""
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score

from dsa.extras import SlidingWindowStats


def train_isolation_forest(tel: dict, contamination: float = 0.04, seed: int = 42):
    """Fit on all three channels; returns (model, precision, recall)."""
    X = np.column_stack([tel["drain"], tel["off_course"], tel["temp"]])
    model = IsolationForest(n_estimators=100, contamination=contamination, random_state=seed, n_jobs=1)
    pred = (model.fit_predict(X) == -1).astype(int)
    return model, precision_score(tel["label"], pred), recall_score(tel["label"], pred)


def sliding_window_detect(series, window: int = 30, z_thresh: float = 4.0) -> np.ndarray:
    """Streaming detector: flag x if |x - mean| > z * std of the previous `window` clean
    samples. Flagged points are NOT pushed, so anomalies don't pollute the baseline. O(1)/sample."""
    sw = SlidingWindowStats(window)
    flags = np.zeros(len(series), dtype=int)
    for i, x in enumerate(series):
        if len(sw.buf) >= 10 and sw.std > 0 and abs(x - sw.mean) > z_thresh * sw.std:
            flags[i] = 1
        else:
            sw.push(x)
    return flags


def run_stream_detection(tel: dict):
    """Evaluate the sliding-window detector on the battery-drain channel."""
    flags = sliding_window_detect(tel["drain"])
    return precision_score(tel["label_drain"], flags), recall_score(tel["label_drain"], flags)
