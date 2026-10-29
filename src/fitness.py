"""Shared solution encoding and fitness function used by both ABC and ACO.

A solution is a binary feature mask plus three Random Forest hyper-parameters.
fitness = w_f1 * F1(val) + w_feat * (1 - n_selected / n_total) + w_cplx * (1 - n_estimators / max)
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

PARAM_RANGES = {
    "n_estimators": (10, 100),
    "max_depth": (3, 25),
    "min_samples_split": (2, 20),
}
PARAM_NAMES = list(PARAM_RANGES)


def decode_params(u):
    """Map genes in [0, 1] to integer hyper-parameters."""
    params = {}
    for gene, name in zip(u, PARAM_NAMES):
        lo, hi = PARAM_RANGES[name]
        params[name] = int(round(lo + float(np.clip(gene, 0, 1)) * (hi - lo)))
    return params


def param_grid(name, n=8):
    """Discrete values of one hyper-parameter (used by ACO's pheromone tables)."""
    lo, hi = PARAM_RANGES[name]
    return np.unique(np.linspace(lo, hi, n).round().astype(int))


def build_rf(params, n_jobs=-1, seed=0):
    return RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        n_jobs=n_jobs,
        random_state=seed,
    )


def _subsample(X, y, n, rng):
    if n is None or len(X) <= n:
        return X, y
    idx = rng.choice(len(X), n, replace=False)
    return X[idx], y[idx]


class FitnessEvaluator:
    """Scores (mask, params) candidates on a validation split; counts and caches evaluations."""

    def __init__(self, data, w_f1=0.8, w_feat=0.1, w_cplx=0.1,
                 train_subsample=15000, val_subsample=10000, n_jobs=-1, seed=0):
        rng = np.random.RandomState(seed)
        self.X_tr, self.y_tr = _subsample(data.X_train, data.y_train, train_subsample, rng)
        self.X_va, self.y_va = _subsample(data.X_val, data.y_val, val_subsample, rng)
        self.n_features = data.n_features
        self.w = (w_f1, w_feat, w_cplx)
        self.n_jobs = n_jobs
        self.seed = seed
        self.n_evals = 0  # real (non-cached) evaluations; this is the shared search budget
        self._cache = {}

    def evaluate(self, mask, params):
        mask = np.asarray(mask, dtype=bool)
        if not mask.any():
            return 0.0
        key = (mask.tobytes(), tuple(params[n] for n in PARAM_NAMES))
        if key in self._cache:
            return self._cache[key]

        clf = build_rf(params, self.n_jobs, self.seed)
        clf.fit(self.X_tr[:, mask], self.y_tr)
        f1 = f1_score(self.y_va, clf.predict(self.X_va[:, mask]))

        w_f1, w_feat, w_cplx = self.w
        feat_term = 1 - mask.sum() / self.n_features
        cplx_term = 1 - params["n_estimators"] / PARAM_RANGES["n_estimators"][1]
        fitness = w_f1 * f1 + w_feat * feat_term + w_cplx * cplx_term

        self.n_evals += 1
        self._cache[key] = fitness
        return fitness
