"""Load and preprocess NSL-KDD (plus a synthetic stand-in for quick tests)."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

FEATURES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]
CATEGORICAL = ["protocol_type", "service", "flag"]
LABEL_COL = 41  # label sits right after the 41 features; later columns are ignored

# Attack types removed from training and used only for validation, so the optimiser is scored
# on attacks the model has never seen (KDDTest+ has the same kind of novelty).
HOLDOUT_ATTACKS = ["back", "teardrop", "nmap", "portsweep", "warezclient"]


@dataclass
class Dataset:
    name: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list
    val_mode: str = "random"
    scaler: object = None  # fitted MinMaxScaler, kept so raw rows can be transformed later

    @property
    def n_features(self):
        return self.X_train.shape[1]


def _find_file(data_dir, kind):
    """Find KDDTrain+/KDDTest+ (.txt or .csv), skipping the 20-percent subsets."""
    for path in sorted(Path(data_dir).glob(f"KDD{kind}*")):
        if "20" not in path.name and path.suffix.lower() in {".txt", ".csv"}:
            return path
    raise FileNotFoundError(
        f"No KDD{kind}+ file found in '{data_dir}'. Download NSL-KDD and place "
        f"KDDTrain+.txt and KDDTest+.txt there (or run with --synthetic)."
    )


def _read(path):
    df = pd.read_csv(path, header=None, low_memory=False)
    if str(df.iloc[0, 0]).strip().lower() == "duration":  # some mirrors ship a header row
        df = df.iloc[1:].reset_index(drop=True)
    X = df.iloc[:, :len(FEATURES)].copy()
    X.columns = FEATURES
    for col in FEATURES:
        if col not in CATEGORICAL:
            X[col] = pd.to_numeric(X[col])
    label = df.iloc[:, LABEL_COL].astype(str).str.strip().str.rstrip(".")
    return X, (label != "normal").astype(int).to_numpy(), label.to_numpy()  # y: 1 = attack


def load_nsl_kdd(data_dir="data", val_mode="attack_holdout", val_size=None, seed=42):
    """Official KDDTrain+ -> train/val split; KDDTest+ is kept for the final evaluation only.

    val_mode="random":         stratified random validation split (val_size default 0.2).
    val_mode="attack_holdout": every row of HOLDOUT_ATTACKS goes to validation, plus a random
                               share (val_size default 0.1) of the remaining rows.
    """
    X_tr, y_tr, attack_tr = _read(_find_file(data_dir, "Train"))
    X_te, y_te, _ = _read(_find_file(data_dir, "Test"))

    keep = ~pd.concat([X_tr, pd.Series(y_tr, name="y")], axis=1).duplicated().to_numpy()
    X_tr, y_tr, attack_tr = X_tr[keep].reset_index(drop=True), y_tr[keep], attack_tr[keep]

    # One-hot encode on train+test together so both share the same category columns.
    full = pd.get_dummies(
        pd.concat([X_tr, X_te], ignore_index=True), columns=CATEGORICAL, dtype=np.float32
    )
    names = list(full.columns)
    full = full.to_numpy(dtype=np.float32)
    X_tr, X_te = full[:len(X_tr)], full[len(X_tr):]

    if val_mode == "random":
        train_idx, val_idx = train_test_split(
            np.arange(len(y_tr)), test_size=val_size or 0.2, stratify=y_tr, random_state=seed
        )
    elif val_mode == "attack_holdout":
        novel = np.isin(attack_tr, HOLDOUT_ATTACKS)
        rest = np.flatnonzero(~novel)
        train_idx, rest_val = train_test_split(
            rest, test_size=val_size or 0.1, stratify=y_tr[rest], random_state=seed
        )
        val_idx = np.concatenate([rest_val, np.flatnonzero(novel)])
    else:
        raise ValueError(f"unknown val_mode '{val_mode}'")

    scaler = MinMaxScaler().fit(X_tr[train_idx])  # fitted on train only
    f32 = lambda a: scaler.transform(a).astype(np.float32)
    return Dataset(
        "NSL-KDD", f32(X_tr[train_idx]), y_tr[train_idx], f32(X_tr[val_idx]), y_tr[val_idx],
        f32(X_te), y_te, names, val_mode, scaler,
    )


def make_synthetic(n_samples=12000, n_features=40, seed=42):
    """Small tabular stand-in (informative + redundant + noise features) for smoke tests."""
    X, y = make_classification(
        n_samples=n_samples, n_features=n_features, n_informative=8, n_redundant=6,
        flip_y=0.03, class_sep=0.8, random_state=seed,
    )
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tr, y_tr, test_size=0.25, stratify=y_tr, random_state=seed
    )
    scaler = MinMaxScaler().fit(X_train)
    f32 = lambda a: scaler.transform(a).astype(np.float32)
    return Dataset(
        "Synthetic", f32(X_train), y_train, f32(X_val), y_val, f32(X_te), y_te,
        [f"f{i}" for i in range(n_features)],
    )
