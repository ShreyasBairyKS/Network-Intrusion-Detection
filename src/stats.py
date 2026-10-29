"""Paired significance tests of every model against the baseline (same seeds = paired samples).

    python -m src.stats --results results/results.json
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from scipy import stats

from .evaluate import runs_to_frame

BASELINE = "Baseline RF"
TESTED = ["f1", "accuracy", "recall"]


def significance_table(runs, baseline=BASELINE):
    df = runs_to_frame(runs)
    rows = []
    for metric in TESTED:
        piv = df.pivot(index="seed", columns="model", values=metric)
        for model in piv.columns:
            if model == baseline:
                continue
            diff = piv[model] - piv[baseline]
            wilcoxon_p = stats.wilcoxon(diff).pvalue if (diff != 0).any() else 1.0
            rows.append({
                "metric": metric, "model": model, "n_seeds": len(diff),
                "mean_diff": diff.mean(), "wins": int((diff > 0).sum()),
                "ttest_p": stats.ttest_rel(piv[model], piv[baseline]).pvalue,
                "wilcoxon_p": wilcoxon_p,
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/results.json")
    ap.add_argument("--out", default=None, help="CSV path (default: next to results.json)")
    a = ap.parse_args()
    runs = json.loads(Path(a.results).read_text())["runs"]
    table = significance_table(runs)
    out = Path(a.out) if a.out else Path(a.results).with_name("significance.csv")
    table.to_csv(out, index=False)
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
