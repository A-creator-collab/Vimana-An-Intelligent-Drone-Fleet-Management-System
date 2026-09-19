"""demand_forecasting.py - LinearRegression on lag features to forecast hourly demand per zone,
then pre-position drones proportionally."""
from typing import Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LinearRegression


def _features(counts: np.ndarray, z: int, h: int, nz: int) -> List[float]:
    onehot = [1.0 if k == z else 0.0 for k in range(nz)]
    ang = 2 * np.pi * (h % 24) / 24
    return [counts[z, h - 1], counts[z, h - 2], counts[z, h - 24], np.sin(ang), np.cos(ang)] + onehot


def train_forecaster(counts: np.ndarray, test_hours: int = 96) -> Tuple[LinearRegression, Dict[str, float]]:
    """Train on all but the last `test_hours`; report MAE/RMSE vs naive baselines."""
    nz, T = counts.shape
    split = T - test_hours
    Xtr, ytr, Xte, yte = [], [], [], []
    for z in range(nz):
        for h in range(24, T):
            (Xtr if h < split else Xte).append(_features(counts, z, h, nz))
            (ytr if h < split else yte).append(counts[z, h])
    model = LinearRegression().fit(Xtr, ytr)
    Xte, yte = np.array(Xte), np.array(yte)
    pred = model.predict(Xte)
    mae = lambda p: float(np.mean(np.abs(yte - p)))
    rmse = lambda p: float(np.sqrt(np.mean((yte - p) ** 2)))
    return model, {"mae": mae(pred), "rmse": rmse(pred),
                   "naive_lag1_mae": mae(Xte[:, 0]), "naive_lag24_mae": mae(Xte[:, 2]),
                   "test_true": yte, "test_pred": pred}


def forecast_next_hour(model, counts: np.ndarray) -> np.ndarray:
    nz = counts.shape[0]
    return np.maximum(model.predict(np.array([_features_next(counts, z, nz) for z in range(nz)])), 0)


def _features_next(counts: np.ndarray, z: int, nz: int) -> List[float]:
    """Feature vector for the hour right after the last observed one."""
    T = counts.shape[1]
    ext = np.concatenate([counts, np.zeros((nz, 1))], axis=1)
    return _features(ext, z, T, nz)


def allocate_drones(pred: np.ndarray, n_drones: int) -> List[int]:
    """Largest-remainder apportionment of drones to zones by predicted demand."""
    share = pred / pred.sum() * n_drones
    alloc = [int(s) for s in share]
    order = sorted(range(len(pred)), key=lambda i: share[i] - alloc[i], reverse=True)
    for i in order[: n_drones - sum(alloc)]:
        alloc[i] += 1
    return alloc
