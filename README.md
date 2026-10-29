# Network Intrusion Detection: ABC vs ACO feature selection + RF tuning

Bio-Inspired Algorithms AAT (23DS7BSBIO). Artificial Bee Colony (ABC) and Ant Colony Optimization
(ACO), both implemented from scratch, jointly choose a feature subset and Random Forest
hyper-parameters for detecting attacks in NSL-KDD traffic.

## Setup
```bash
pip install -r requirements.txt
```
Download NSL-KDD and place `KDDTrain+.txt` and `KDDTest+.txt` in `data/`.

## Run
```bash
# main experiment: 6 models x 15 seeds, validated on held-out attack types
python run_experiments.py --seeds 15 --max-evals 300 --pop 10 --out results_15seeds

# ablation: same, but with a random validation split
python run_experiments.py --seeds 15 --max-evals 300 --pop 10 --val-mode random --skip-tune-only --out results_random_val

python run_experiments.py --synthetic --seeds 2 --max-evals 60 --pop 6 --out results_smoke   # quick smoke test
python -m src.plots --results results_15seeds/results.json --out results_15seeds/figures   # regenerate figures
python -m src.stats --results results_15seeds/results.json                                 # significance tests
python -m pytest tests -q                                                                  # unit tests
```
`notebook.ipynb` walks through the pipeline and shows the final results (run it from this folder).

## Demo app
```bash
python -m streamlit run app.py
```
Needs `data/` and `results_15seeds/` (it trains the three deployed models on first load, about 30 s). Four tabs:
- **Results**: headline numbers, the full comparison table, all figures, significance tests and the validation ablation.
- **Detect traffic**: classify real KDDTest+ connections (filter by normal / attack / attack types never seen in training)
  or upload a CSV in NSL-KDD format, with the baseline, ABC-RF and ACO-RF side by side and a per-connection inspector.
- **Optimizer lab**: run ABC and/or ACO live with an adjustable budget and watch them converge.
- **How it works**: pipeline diagram, fitness function, algorithm summaries and limitations.

The deployed ABC-RF and ACO-RF are the runs with the best *validation* fitness (never chosen by test score).

## Models compared
| Model | Features | RF hyper-parameters |
|---|---|---|
| Baseline RF | all 122 | defaults |
| MI top-30 | 30 best by mutual information | defaults |
| ABC-RF / ACO-RF | chosen by the optimiser | tuned by the optimiser |
| ABC-RF / ACO-RF (tune only) | all 122 | tuned by the optimiser |

The tune-only rows separate the effect of tuning from the effect of feature selection.

## Design
| File | Role |
|---|---|
| `src/preprocess.py` | Load NSL-KDD, dedupe, one-hot encode, MinMax scale, train/validation split (official `KDDTest+` held out) |
| `src/fitness.py` | Shared solution encoding (feature mask + 3 RF params) and fitness function |
| `src/abc.py` | ABC: employed / onlooker / scout bees, sigmoid binarisation |
| `src/aco.py` | ACO: pheromone on features, subset size and each hyper-parameter; evaporation, elitist deposit, MAX-MIN bounds |
| `src/baselines.py` | RF on all features and mutual-information top-k RF |
| `src/evaluate.py` | Final test-set metrics and timing; tabulation helpers |
| `src/stats.py` | Paired t-test / Wilcoxon of every model vs the baseline |
| `src/plots.py` | Report figures |
| `src/inference.py` | Raw-row preprocessing and the deployed models used by the app |
| `run_experiments.py` | Multi-seed driver |
| `app.py` | Streamlit demo app |
| `tests/` | Unit tests for the fitness function and both optimisers |

**Fitness** = 0.8 x F1(validation) + 0.1 x (1 - features used / total) + 0.1 x (1 - n_estimators / 100).
Both algorithms get the same evaluation budget (`--max-evals`, counted as real model trainings; repeats of an
already-scored candidate come from a cache and are free), the same training subsample and the same seeds.
Training time is measured and reported but kept out of the fitness so that runs are reproducible.

**Validation split.** With a random validation split the optimiser sees almost no generalisation gap, because
`KDDTest+` contains attack types absent from training. The default `--val-mode attack_holdout` moves every row of
`back`, `teardrop`, `nmap`, `portsweep` and `warezclient` out of training and into validation, so the fitness
measures detection of attacks the model has not seen. The test set is used only for the final report.

## Outputs
Each results folder contains `results.json` (every run), `summary.csv` (mean/std per model),
`significance.csv` (paired tests) and `figures/` (convergence, features kept, metrics, timing,
confusion matrices, feature frequency). `archive/` holds earlier runs made before an ACO bug fix (pheromone
was clipped before, not after, the deposit) and is kept only for reference.

## Results (15 seeds, scored on KDDTest+)
| Model | F1 | Accuracy | Features | Training time (median) |
|---|---|---|---|---|
| Baseline RF | 0.754 ± 0.011 | 0.771 | 122 | 1.43 s |
| MI top-30 | 0.753 ± 0.007 | 0.770 | 30 | 1.21 s |
| ABC-RF | 0.751 ± 0.023 | 0.766 | 40.9 ± 11.4 | 0.18 s (8.0x faster) |
| **ACO-RF** | **0.764 ± 0.019** | **0.775** | **14.9 ± 7.8** | **0.12 s (11.9x faster)** |
| ABC-RF (tune only) | 0.754 ± 0.015 | 0.771 | 122 | 0.19 s (7.6x faster) |
| ACO-RF (tune only) | 0.759 ± 0.015 | 0.774 | 122 | 0.19 s (7.6x faster) |

- ACO-RF is +0.010 F1 above the baseline with about 8x fewer features. The evidence is borderline: paired t-test
  p = 0.046, Wilcoxon p = 0.055, and no multiple-comparison correction was applied. ABC-RF is not distinguishable
  from the baseline (p = 0.68).
- Inference is about 3.3x faster for ACO-RF. Roughly 7.6x of the training speed-up already comes from the tuned forest
  alone (the tune-only rows); feature selection adds the rest.
- Validation matters: with a random validation split ACO-RF reaches only F1 0.735 against a 0.763 baseline
  (`results_random_val/`), because validation F1 is about 0.998 and hides the generalisation gap.

## Notes and limitations
- NSL-KDD is an older benchmark derived from KDD'99, not modern traffic.
- Test F1 varies by about +/-0.02 between seeds, so differences smaller than that need paired tests over many seeds.
- Timing columns are medians; occasional system slowdowns make the mean unreliable.
- The held-out attack types are a design choice; other choices would give somewhat different optimiser behaviour.
