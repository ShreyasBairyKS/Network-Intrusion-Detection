"""Reference models: RF on all features, and RF on a mutual-information top-k filter."""
import numpy as np
from sklearn.feature_selection import mutual_info_classif

from .evaluate import BASELINE_PARAMS, evaluate_model


def feature_mi(data, n_samples=10000, seed=0):
    """Mutual information of each feature with the label (also ACO's heuristic)."""
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(data.X_train), min(n_samples, len(data.X_train)), replace=False)
    return mutual_info_classif(data.X_train[idx], data.y_train[idx], random_state=seed)


def run_baseline(data, n_jobs=-1, seed=0):
    mask = np.ones(data.n_features, dtype=bool)
    return mask, evaluate_model(mask, BASELINE_PARAMS, data, n_jobs, seed)


def run_mi_topk(data, mi, k, n_jobs=-1, seed=0):
    mask = np.zeros(data.n_features, dtype=bool)
    mask[np.argsort(mi)[::-1][:k]] = True
    return mask, evaluate_model(mask, BASELINE_PARAMS, data, n_jobs, seed)
