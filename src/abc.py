"""Artificial Bee Colony (ABC) for joint feature selection and RF hyper-parameter tuning.

Each food source holds a continuous mask position x in [-X_MAX, X_MAX] (binarised through a
sigmoid transfer function) and hyper-parameter genes u in [0, 1].
"""
import time
from dataclasses import dataclass

import numpy as np

from .fitness import PARAM_NAMES, decode_params

X_MAX = 4.0


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class Source:
    x: np.ndarray       # continuous position of the feature-mask dimensions
    mask: np.ndarray    # binary mask derived from x
    u: np.ndarray       # hyper-parameter genes
    fit: float = 0.0
    trials: int = 0     # consecutive failed improvement attempts


class ArtificialBeeColony:
    def __init__(self, evaluator, n_sources=10, limit=10, max_evals=200, mod_rate=0.1, seed=0,
                 select_features=True, callback=None):
        self.ev = evaluator
        self.callback = callback  # optional fn(n_evals, best_fitness), called once per cycle
        self.select_features = select_features  # False = tune hyper-parameters only (all features)
        self.n_sources = n_sources
        self.limit = limit
        self.max_evals = max_evals
        self.mod_rate = mod_rate  # fraction of mask dimensions perturbed per neighbour move
        self.rng = np.random.RandomState(seed)
        self.D = evaluator.n_features
        self.best = None
        self.history = []

    def _exhausted(self):
        return self.ev.n_evals >= self.max_evals

    def _random_source(self):
        mask = self.rng.rand(self.D) < self.rng.uniform(0.2, 0.8)
        if not self.select_features:
            mask[:] = True
        elif not mask.any():
            mask[self.rng.randint(self.D)] = True
        x = np.where(mask, self.rng.uniform(0, X_MAX, self.D), self.rng.uniform(-X_MAX, 0, self.D))
        return Source(x, mask, self.rng.rand(len(PARAM_NAMES)))

    def _score(self, s):
        s.fit = self.ev.evaluate(s.mask, decode_params(s.u))
        if self.best is None or s.fit > self.best.fit:
            self.best = Source(s.x.copy(), s.mask.copy(), s.u.copy(), s.fit)

    def _neighbor(self, i):
        """v_ij = x_ij + phi * (x_ij - x_kj) on a random subset of dimensions."""
        s = self.sources[i]
        k = self.rng.choice([j for j in range(self.n_sources) if j != i])
        other = self.sources[k]

        x, mask = s.x.copy(), s.mask.copy()
        if self.select_features:
            dims = self.rng.rand(self.D) < self.mod_rate
            if not dims.any():
                dims[self.rng.randint(self.D)] = True
            phi = self.rng.uniform(-1, 1, self.D)
            x[dims] = np.clip(s.x[dims] + phi[dims] * (s.x[dims] - other.x[dims]), -X_MAX, X_MAX)
            mask[dims] = self.rng.rand(dims.sum()) < _sigmoid(x[dims])  # sigmoid transfer function

        u = s.u.copy()
        j = self.rng.randint(len(PARAM_NAMES))
        u[j] = np.clip(s.u[j] + self.rng.uniform(-1, 1) * (s.u[j] - other.u[j]), 0, 1)
        return Source(x, mask, u)

    def _try_improve(self, i):
        """Greedy selection between a source and its neighbour."""
        cand = self._neighbor(i)
        self._score(cand)
        if cand.fit > self.sources[i].fit:
            self.sources[i] = cand
        else:
            self.sources[i].trials += 1

    def _log(self):
        self.history.append((self.ev.n_evals, self.best.fit))
        if self.callback:
            self.callback(*self.history[-1])

    def run(self):
        t0 = time.perf_counter()
        self.sources = [self._random_source() for _ in range(self.n_sources)]
        for s in self.sources:
            self._score(s)
        self._log()

        cycle, max_cycles = 0, self.max_evals * 5  # guard: cached duplicates don't use budget
        while not self._exhausted() and cycle < max_cycles:
            # Employed bees: one neighbour move per source.
            for i in range(self.n_sources):
                if self._exhausted():
                    break
                self._try_improve(i)

            # Onlooker bees: roulette-wheel choice proportional to fitness.
            fits = np.array([s.fit for s in self.sources]) + 1e-9
            probs = fits / fits.sum()
            for _ in range(self.n_sources):
                if self._exhausted():
                    break
                self._try_improve(self.rng.choice(self.n_sources, p=probs))

            # Scout bee: abandon the most stagnant source once it exceeds `limit`.
            worst = max(range(self.n_sources), key=lambda j: self.sources[j].trials)
            if self.sources[worst].trials > self.limit and not self._exhausted():
                self.sources[worst] = self._random_source()
                self._score(self.sources[worst])

            self._log()
            cycle += 1

        return {
            "best_mask": self.best.mask,
            "best_params": decode_params(self.best.u),
            "best_fitness": self.best.fit,
            "history": self.history,
            "n_evals": self.ev.n_evals,
            "search_time": time.perf_counter() - t0,
        }
