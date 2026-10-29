"""Run baseline, MI top-k, ABC and ACO over several seeds and save results + figures.

    python run_experiments.py --data-dir data --seeds 5 --max-evals 300
    python run_experiments.py --synthetic --seeds 2 --max-evals 60      # quick smoke test
"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")  # noisy, harmless

from src import plots
from src.abc import ArtificialBeeColony
from src.aco import AntColony
from src.baselines import feature_mi, run_baseline, run_mi_topk
from src.evaluate import evaluate_model, summarize
from src.fitness import FitnessEvaluator
from src.preprocess import load_nsl_kdd, make_synthetic
from src.stats import significance_table


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--synthetic", action="store_true", help="use a synthetic dataset instead of NSL-KDD")
    p.add_argument("--val-mode", choices=["attack_holdout", "random"], default="attack_holdout",
                   help="how the optimiser's validation split is built (NSL-KDD only)")
    p.add_argument("--skip-tune-only", action="store_true",
                   help="skip the ABC/ACO variants that tune hyper-parameters on all features")
    p.add_argument("--seeds", type=int, default=3, help="number of independent runs")
    p.add_argument("--seed0", type=int, default=0)
    p.add_argument("--max-evals", type=int, default=200, help="fitness-evaluation budget per algorithm")
    p.add_argument("--pop", type=int, default=10, help="food sources (ABC) / ants per iteration (ACO)")
    p.add_argument("--train-subsample", type=int, default=15000, help="rows used to train inside the fitness")
    p.add_argument("--val-subsample", type=int, default=10000, help="rows used to score inside the fitness")
    p.add_argument("--topk", type=int, default=None, help="features kept by the MI top-k reference")
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--out", default="results")
    return p.parse_args()


def to_python(obj):
    """Make numpy types JSON-serialisable."""
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_python(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_python(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def record(seed, model, mask, metrics, search=None):
    rec = {"seed": seed, "model": model, "metrics": metrics, "selected": np.flatnonzero(mask)}
    if search is not None:
        rec.update(params=search["best_params"], history=search["history"],
                   search_time=search["search_time"], best_fitness=search["best_fitness"])
    return rec


def main():
    args = parse_args()
    data = make_synthetic() if args.synthetic else load_nsl_kdd(args.data_dir, val_mode=args.val_mode)
    topk = args.topk or max(1, data.n_features // 4)
    print(f"Dataset: {data.name} | features: {data.n_features} | validation: {data.val_mode} | "
          f"train/val/test: {len(data.y_train)}/{len(data.y_val)}/{len(data.y_test)}")

    mi = feature_mi(data)
    runs = []
    for seed in range(args.seed0, args.seed0 + args.seeds):
        print(f"\n=== seed {seed} ===")

        mask, m = run_baseline(data, args.n_jobs, seed)
        runs.append(record(seed, "Baseline RF", mask, m))
        print(f"Baseline RF   F1={m['f1']:.4f} acc={m['accuracy']:.4f} features={m['n_features']}")

        mask, m = run_mi_topk(data, mi, topk, args.n_jobs, seed)
        runs.append(record(seed, f"MI top-{topk}", mask, m))
        print(f"MI top-{topk:<6} F1={m['f1']:.4f} acc={m['accuracy']:.4f} features={m['n_features']}")

        def new_evaluator():  # same seed => same subsample => fair comparison, separate budget/cache
            return FitnessEvaluator(data, train_subsample=args.train_subsample,
                                    val_subsample=args.val_subsample, n_jobs=args.n_jobs, seed=seed)

        searches = {
            "ABC-RF": ArtificialBeeColony(new_evaluator(), n_sources=args.pop, limit=args.pop,
                                          max_evals=args.max_evals, seed=seed),
            "ACO-RF": AntColony(new_evaluator(), mi, n_ants=args.pop,
                                max_evals=args.max_evals, seed=seed),
        }
        if not args.skip_tune_only:  # same algorithms, but only the RF hyper-parameters are searched
            searches["ABC-RF (tune only)"] = ArtificialBeeColony(
                new_evaluator(), n_sources=args.pop, limit=args.pop, max_evals=args.max_evals,
                seed=seed, select_features=False)
            searches["ACO-RF (tune only)"] = AntColony(
                new_evaluator(), mi, n_ants=args.pop, max_evals=args.max_evals, seed=seed,
                select_features=False)
        for name, algo in searches.items():
            res = algo.run()
            m = evaluate_model(res["best_mask"], res["best_params"], data, args.n_jobs, seed)
            runs.append(record(seed, name, res["best_mask"], m, res))
            print(f"{name:<19} F1={m['f1']:.4f} acc={m['accuracy']:.4f} features={m['n_features']} "
                  f"fitness={res['best_fitness']:.4f} evals={res['n_evals']} search={res['search_time']:.0f}s",
                  flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {"meta": {"dataset": data.name, "val_mode": data.val_mode,
                        "feature_names": data.feature_names, "args": vars(args)},
               "runs": runs}
    (out / "results.json").write_text(json.dumps(to_python(payload), indent=1))

    summary = summarize(runs)
    summary.to_csv(out / "summary.csv")
    cols = [f"{c}_mean" for c in ("accuracy", "precision", "recall", "f1", "auc", "n_features", "train_time", "predict_time")]
    print("\n=== Summary (mean over seeds) ===")
    print(summary[cols].round(4).to_string())

    if args.seeds >= 3:  # paired tests need a few seeds to mean anything
        sig = significance_table(runs)
        sig.to_csv(out / "significance.csv", index=False)
        print("\n=== Paired tests vs baseline ===")
        print(sig[sig.metric == "f1"].round(4).to_string(index=False))

    plots.make_all(out / "results.json", out / "figures")
    print(f"\nSaved results to {out}/ (results.json, summary.csv, figures/)")


if __name__ == "__main__":
    main()
