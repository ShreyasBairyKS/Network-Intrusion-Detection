"""Helpers for the demo app: raw-row preprocessing and the three deployed models.

The deployed ABC-RF / ACO-RF are the runs with the highest *validation* fitness in results.json
(never picked by test score), retrained on train+val exactly as in the experiments.
"""
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import BASELINE_PARAMS
from .fitness import build_rf
from .preprocess import CATEGORICAL, FEATURES, LABEL_COL, _find_file, _read

DEPLOYED = ["Baseline RF", "ABC-RF", "ACO-RF"]


@dataclass
class DeployedModel:
    name: str
    mask: np.ndarray
    params: dict
    clf: object
    seed: int
    val_fitness: float


def load_raw(data_dir="data", kind="Test"):
    """Raw (unencoded) rows of KDDTest+/KDDTrain+: features, 0/1 label, attack-type names."""
    return _read(_find_file(data_dir, kind))


def transform_raw(raw, data):
    """Raw NSL-KDD rows (41 feature columns) -> scaled model input with the training columns."""
    dummies = pd.get_dummies(raw[FEATURES], columns=CATEGORICAL, dtype=np.float32)
    dummies = dummies.reindex(columns=data.feature_names, fill_value=0.0)  # unseen categories -> all zero
    return data.scaler.transform(dummies.to_numpy(dtype=np.float32)).astype(np.float32)


def read_uploaded_csv(file):
    """Parse an uploaded NSL-KDD style CSV: 41 feature columns, optional label in column 42."""
    df = pd.read_csv(file, header=None, low_memory=False)
    if df.empty:
        raise ValueError("The file is empty.")
    if str(df.iloc[0, 0]).strip().lower() == "duration":  # header row
        df = df.iloc[1:].reset_index(drop=True)
    if df.shape[1] < len(FEATURES):
        raise ValueError(f"Expected at least {len(FEATURES)} columns (NSL-KDD features), got {df.shape[1]}.")

    X = df.iloc[:, :len(FEATURES)].copy()
    X.columns = FEATURES
    for col in FEATURES:
        if col not in CATEGORICAL:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    if X[FEATURES].isna().any().any():
        raise ValueError("Some numeric feature values could not be read. Check the column order matches NSL-KDD.")

    attack = None
    if df.shape[1] > LABEL_COL:
        attack = df.iloc[:, LABEL_COL].astype(str).str.strip().str.rstrip(".").to_numpy()
    return X, attack


def select_runs(results_path):
    """For each optimiser, the run with the best validation fitness."""
    payload = json.loads(Path(results_path).read_text())
    n_features = len(payload["meta"]["feature_names"])
    picks = {}
    for name in ("ABC-RF", "ACO-RF"):
        best = max((r for r in payload["runs"] if r["model"] == name), key=lambda r: r["best_fitness"])
        mask = np.zeros(n_features, dtype=bool)
        mask[best["selected"]] = True
        picks[name] = {"mask": mask, "params": best["params"], "seed": best["seed"], "fitness": best["best_fitness"]}
    return picks


def train_deployed(data, picks, n_jobs=-1):
    """Fit the baseline and the two optimised forests on train+val."""
    X = np.vstack([data.X_train, data.X_val])
    y = np.concatenate([data.y_train, data.y_val])
    specs = {"Baseline RF": {"mask": np.ones(data.n_features, dtype=bool), "params": BASELINE_PARAMS,
                             "seed": 0, "fitness": float("nan")}, **picks}
    models = {}
    for name in DEPLOYED:
        s = specs[name]
        clf = build_rf(s["params"], n_jobs, s["seed"]).fit(X[:, s["mask"]], y)
        models[name] = DeployedModel(name, s["mask"], s["params"], clf, s["seed"], s["fitness"])
    return models


def predict_attack_proba(models, X):
    """P(attack) per model for scaled rows X, plus the batch prediction time (seconds) per model."""
    proba, seconds = {}, {}
    for name, m in models.items():
        t0 = time.perf_counter()
        proba[name] = m.clf.predict_proba(X[:, m.mask])[:, 1]
        seconds[name] = time.perf_counter() - t0
    return pd.DataFrame(proba), seconds
