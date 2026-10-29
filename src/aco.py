"""Ant Colony Optimization (ACO) for joint feature selection and RF hyper-parameter tuning.

Pheromone lives on (a) every feature, (b) a grid of subset sizes, and (c) a grid of values for
each hyper-parameter. Pheromone is kept within [tau_min, tau_max] (MAX-MIN Ant System style) so
the colony cannot stagnate completely.
"""
import time

import numpy as np

from .fitness import PARAM_NAMES, param_grid


class AntColony:
    def __init__(self, evaluator, heuristic, n_ants=10, alpha=1.0, beta=1.0, rho=0.1, q=1.0,
                 tau_min=0.1, tau_max=3.0, max_evals=200, seed=0, select_features=True,
                 callback=None):
        self.ev = evaluator
        self.callback = callback  # optional fn(n_evals, best_fitness), called once per iteration
        self.select_features = select_features  # False = tune hyper-parameters only (all features)
        self.n_ants = n_ants
        self.alpha, self.beta, self.rho, self.q = alpha, beta, rho, q
        self.tau_min, self.tau_max = tau_min, tau_max
        self.max_evals = max_evals
        self.rng = np.random.RandomState(seed)
        self.D = evaluator.n_features

        # Heuristic desirability eta_i: normalised mutual information, never exactly zero.
        h = np.asarray(heuristic, dtype=float)
        self.eta = 0.1 + 0.9 * h / (h.max() if h.max() > 0 else 1.0)

        self.sizes = np.unique(np.linspace(max(2, self.D // 20), self.D, 12).astype(int))
        self.grids = {n: param_grid(n) for n in PARAM_NAMES}

        self.tau_feat = np.ones(self.D)
        self.tau_size = np.ones(len(self.sizes))
        self.tau_param = {n: np.ones(len(g)) for n, g in self.grids.items()}
        self.best = None
        self.history = []

    def _exhausted(self):
        return self.ev.n_evals >= self.max_evals

    def _pick(self, tau):
        return self.rng.choice(len(tau), p=tau / tau.sum())

    def _build_ant(self):
        """Probabilistic construction: p_i proportional to tau_i^alpha * eta_i^beta."""
        if self.select_features:
            size_idx = self._pick(self.tau_size)
            p = self.tau_feat ** self.alpha * self.eta ** self.beta
            feats = self.rng.choice(self.D, size=min(self.sizes[size_idx], self.D), replace=False, p=p / p.sum())
            mask = np.zeros(self.D, dtype=bool)
            mask[feats] = True
        else:
            size_idx, mask = len(self.sizes) - 1, np.ones(self.D, dtype=bool)  # largest size = all features
        param_idx = {n: self._pick(t) for n, t in self.tau_param.items()}
        params = {n: int(self.grids[n][param_idx[n]]) for n in PARAM_NAMES}
        return {"mask": mask, "size_idx": size_idx, "param_idx": param_idx, "params": params, "fit": 0.0}

    def _deposit(self, ant, weight=1.0):
        delta = self.q * ant["fit"] * weight
        self.tau_feat[ant["mask"]] += delta
        self.tau_size[ant["size_idx"]] += delta
        for n, i in ant["param_idx"].items():
            self.tau_param[n][i] += delta

    def _tables(self):
        return (self.tau_feat, self.tau_size, *self.tau_param.values())

    def _evaporate(self):
        for t in self._tables():
            t *= 1.0 - self.rho

    def _clip(self):
        for t in self._tables():
            np.clip(t, self.tau_min, self.tau_max, out=t)

    def run(self):
        t0 = time.perf_counter()
        iteration, max_iters = 0, self.max_evals * 5  # guard: cached duplicates don't use budget
        while not self._exhausted() and iteration < max_iters:
            ants = []
            for _ in range(self.n_ants):
                if self._exhausted():
                    break
                ant = self._build_ant()
                ant["fit"] = self.ev.evaluate(ant["mask"], ant["params"])
                ants.append(ant)
                if self.best is None or ant["fit"] > self.best["fit"]:
                    self.best = ant

            # Evaporate, then reinforce with the iteration-best and (weaker) global-best ant.
            self._evaporate()
            self._deposit(max(ants, key=lambda a: a["fit"]))
            self._deposit(self.best, weight=0.5)
            self._clip()  # after depositing, so sampling always sees pheromone within bounds
            self.history.append((self.ev.n_evals, self.best["fit"]))
            if self.callback:
                self.callback(*self.history[-1])
            iteration += 1

        return {
            "best_mask": self.best["mask"],
            "best_params": self.best["params"],
            "best_fitness": self.best["fit"],
            "history": self.history,
            "n_evals": self.ev.n_evals,
            "search_time": time.perf_counter() - t0,
        }
