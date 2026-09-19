"""predictive_maintenance.py - RandomForest classifier: 'needs maintenance soon?'"""
from typing import Dict, List

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

FEATURES = ["flight_hours", "battery_cycles", "vibration", "motor_temp"]


def train_maintenance_model(X: np.ndarray, y: np.ndarray, seed: int = 42):
    """Trains in well under a second. Returns (model, metrics)."""
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    model = RandomForestClassifier(n_estimators=60, max_depth=8, random_state=seed, n_jobs=1)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    return model, {"accuracy": accuracy_score(yte, pred), "f1": f1_score(yte, pred),
                   "importances": {f: float(round(v, 3)) for f, v in zip(FEATURES, model.feature_importances_)}}


def predict_fleet(model, drones) -> Dict[int, float]:
    """Probability of 'needs maintenance' for each drone."""
    X = np.array([[d.flight_hours, d.battery_cycles, d.vibration, d.motor_temp] for d in drones])
    return {d.id: float(p) for d, p in zip(drones, model.predict_proba(X)[:, 1])}
