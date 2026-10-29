"""Fast checks of the fitness function and both optimisers on a small synthetic dataset.

    python -m pytest tests -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.abc import ArtificialBeeColony
from src.aco import AntColony
from src.fitness import PARAM_RANGES, FitnessEvaluator, decode_params
from src.preprocess import load_nsl_kdd, make_synthetic

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def data():
    return make_synthetic(n_samples=3000, n_features=20)


def new_evaluator(data, seed=0):
    return FitnessEvaluator(data, train_subsample=800, val_subsample=500, n_jobs=1, seed=seed)


def test_decode_params_stays_in_range():
    for u in (np.zeros(3), np.ones(3), np.array([-5.0, 0.5, 7.0])):  # out-of-range genes are clipped
        for name, value in decode_params(u).items():
            lo, hi = PARAM_RANGES[name]
            assert lo <= value <= hi


def test_fitness_caches_and_handles_empty_mask(data):
    ev = new_evaluator(data)
    params = decode_params(np.array([0.2, 0.5, 0.1]))
    mask = np.zeros(data.n_features, dtype=bool)
    mask[:5] = True

    assert ev.evaluate(np.zeros_like(mask), params) == 0.0  # no features -> worst fitness
    assert ev.n_evals == 0

    first = ev.evaluate(mask, params)
    assert ev.evaluate(mask, params) == first  # repeat is served from the cache
    assert ev.n_evals == 1
    assert 0.0 < first <= 1.0


def test_abc_respects_budget_and_improves(data):
    res = ArtificialBeeColony(new_evaluator(data), n_sources=4, limit=3, max_evals=24, seed=1).run()
    best = [b for _, b in res["history"]]
    assert 24 <= res["n_evals"] <= 24 + 8  # at most one extra cycle's worth past the budget check
    assert best == sorted(best)  # best-so-far never decreases
    assert res["best_mask"].any()
    assert res["best_fitness"] == best[-1]


def test_aco_respects_budget_and_pheromone_bounds(data):
    ev = new_evaluator(data)
    aco = AntColony(ev, heuristic=np.arange(data.n_features), n_ants=4, max_evals=24, seed=1)
    res = aco.run()
    assert res["n_evals"] <= 24 + 4
    assert res["best_mask"].any()
    for tau in (aco.tau_feat, aco.tau_size, *aco.tau_param.values()):
        assert tau.min() >= aco.tau_min - 1e-12 and tau.max() <= aco.tau_max + 1e-12


@pytest.mark.parametrize("make", [
    lambda ev, d: ArtificialBeeColony(ev, n_sources=4, limit=3, max_evals=12, seed=2, select_features=False),
    lambda ev, d: AntColony(ev, np.ones(d.n_features), n_ants=4, max_evals=12, seed=2, select_features=False),
])
def test_tune_only_keeps_all_features(data, make):
    res = make(new_evaluator(data), data).run()
    assert res["best_mask"].all()


@pytest.mark.parametrize("make", [
    lambda ev, d: ArtificialBeeColony(ev, n_sources=4, limit=3, max_evals=16, seed=3),
    lambda ev, d: AntColony(ev, np.arange(d.n_features), n_ants=4, max_evals=16, seed=3),
])
def test_same_seed_is_reproducible(data, make):
    a = make(new_evaluator(data), data).run()
    b = make(new_evaluator(data), data).run()
    assert np.array_equal(a["best_mask"], b["best_mask"])
    assert a["best_params"] == b["best_params"]
    assert a["history"] == b["history"]


@pytest.mark.skipif(not (DATA_DIR / "KDDTrain+.txt").exists(), reason="NSL-KDD not downloaded")
def test_raw_rows_transform_to_the_training_matrix():
    from src.inference import load_raw, transform_raw

    data = load_nsl_kdd(DATA_DIR)
    raw, y, _ = load_raw(DATA_DIR, "Test")
    assert len(raw) == len(data.y_test) and np.array_equal(y, data.y_test)
    assert np.allclose(transform_raw(raw, data), data.X_test, atol=1e-6)  # app preprocessing == training pipeline


@pytest.mark.skipif(not (DATA_DIR / "KDDTrain+.txt").exists(), reason="NSL-KDD not downloaded")
def test_uploaded_csv_matches_training_pipeline(tmp_path):
    import io

    import pandas as pd

    from src.inference import load_raw, read_uploaded_csv, transform_raw

    data = load_nsl_kdd(DATA_DIR)
    raw, _, attack = load_raw(DATA_DIR, "Test")
    rows = slice(100, 110)

    example = pd.concat([raw.iloc[rows].reset_index(drop=True), pd.Series(attack[rows])], axis=1)
    for header in (False, True):  # the app accepts files with or without a header row
        text = example.to_csv(header=header, index=False)
        if header:
            text = ",".join(raw.columns) + ",label\n" + text.split("\n", 1)[1]
        parsed, labels = read_uploaded_csv(io.StringIO(text))
        assert list(labels) == list(attack[rows])
        assert np.allclose(transform_raw(parsed, data), data.X_test[rows], atol=1e-6)

    with pytest.raises(ValueError):
        read_uploaded_csv(io.StringIO("1,2,3\n4,5,6\n"))  # too few columns


@pytest.mark.skipif(not (DATA_DIR / "KDDTrain+.txt").exists(), reason="NSL-KDD not downloaded")
def test_nsl_kdd_validation_modes():
    holdout = load_nsl_kdd(DATA_DIR, val_mode="attack_holdout")
    random = load_nsl_kdd(DATA_DIR, val_mode="random")
    assert holdout.n_features == random.n_features == 122
    assert len(holdout.y_train) + len(holdout.y_val) == len(random.y_train) + len(random.y_val)
    assert len(holdout.y_val) > len(random.y_val) * 0.5  # held-out attacks all land in validation
    assert np.array_equal(holdout.y_test, random.y_test)  # the test set never changes
    with pytest.raises(ValueError):
        load_nsl_kdd(DATA_DIR, val_mode="bogus")
