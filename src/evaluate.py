"""Final evaluation on the held-out test set, plus helpers to tabulate runs."""
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

BASELINE_PARAMS = {"n_estimators": 100, "max_depth": None, "min_samples_split": 2}
METRICS = ["accuracy", "precision", "recall", "f1", "auc"]


def evaluate_model(mask, params, data, n_jobs=-1, seed=0):
    """Train on train+val with the given features/params and score on the test set."""
    mask = np.asarray(mask, dtype=bool)
    X_fit = np.vstack([data.X_train, data.X_val])[:, mask]
    y_fit = np.concatenate([data.y_train, data.y_val])
    X_test = data.X_test[:, mask]

    clf = RandomForestClassifier(
        n_estimators=params["n_estimators"], max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"], n_jobs=n_jobs, random_state=seed,
    )
    t0 = time.perf_counter()
    clf.fit(X_fit, y_fit)
    train_time = time.perf_counter() - t0

    predict_times = []
    for _ in range(3):  # best of three smooths out timer noise
        t0 = time.perf_counter()
        pred = clf.predict(X_test)
        predict_times.append(time.perf_counter() - t0)
    proba = clf.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(data.y_test, pred),
        "precision": precision_score(data.y_test, pred, zero_division=0),
        "recall": recall_score(data.y_test, pred, zero_division=0),
        "f1": f1_score(data.y_test, pred),
        "auc": roc_auc_score(data.y_test, proba),
        "n_features": int(mask.sum()),
        "train_time": train_time,
        "predict_time": min(predict_times),
        "confusion": confusion_matrix(data.y_test, pred).tolist(),
    }


def runs_to_frame(runs):
    """Flatten the list of run records into one row per (seed, model)."""
    rows = []
    for r in runs:
        row = {"seed": r["seed"], "model": r["model"], **{k: v for k, v in r["metrics"].items() if k != "confusion"}}
        row["search_time"] = r.get("search_time", np.nan)
        row["best_fitness"] = r.get("best_fitness", np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(runs):
    """Mean and std of every metric per model."""
    df = runs_to_frame(runs).drop(columns="seed")
    summary = df.groupby("model", sort=False).agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    return summary
