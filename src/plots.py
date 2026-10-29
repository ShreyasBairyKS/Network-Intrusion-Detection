"""Figures for the report/presentation, generated from results/results.json.

    python -m src.plots --results results/results.json
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .evaluate import METRICS, runs_to_frame

COLORS = {"Baseline RF": "#7f7f7f", "ABC-RF": "#1f77b4", "ACO-RF": "#d62728",
          "ABC-RF (tune only)": "#9ecae1", "ACO-RF (tune only)": "#f4a6a6"}
MI_COLOR = "#bcbd22"


def _color(model):
    return COLORS.get(model, MI_COLOR)


def _models(df):
    return list(dict.fromkeys(df["model"]))


def plot_convergence(runs, path):
    """Best fitness vs fitness evaluations, mean +/- std over seeds."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model in ("ABC-RF", "ACO-RF", "ABC-RF (tune only)", "ACO-RF (tune only)"):
        hists = [np.array(r["history"], dtype=float) for r in runs if r["model"] == model]
        if not hists:
            continue
        style = "--" if "tune only" in model else "-"
        lo = max(h[0, 0] for h in hists)
        hi = min(h[-1, 0] for h in hists)
        grid = np.linspace(lo, max(hi, lo + 1), 100)
        curves = np.array([h[np.searchsorted(h[:, 0], grid, side="right") - 1, 1] for h in hists])
        mean, std = curves.mean(0), curves.std(0)
        ax.plot(grid, mean, style, color=_color(model), label=model, lw=2)
        ax.fill_between(grid, mean - std, mean + std, color=_color(model), alpha=0.2)
    ax.set(xlabel="Fitness evaluations", ylabel="Best fitness so far")
    ax.set_title("Convergence: ABC vs ACO\n(tune-only runs get no feature-count bonus, so sit lower)", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_features(df, n_total, path):
    models = _models(df)
    means = [df[df.model == m].n_features.mean() for m in models]
    stds = [df[df.model == m].n_features.std(ddof=0) for m in models]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(models, means, yerr=stds, color=[_color(m) for m in models], capsize=4)
    ax.bar_label(bars, labels=[f"{v:.0f}" for v in means], padding=3)
    ax.set(ylabel="Features used", title=f"Features kept (of {n_total})")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_metrics(df, path):
    models = _models(df)
    x = np.arange(len(METRICS))
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for i, model in enumerate(models):
        sub = df[df.model == model]
        ax.bar(x + i * width, [sub[m].mean() for m in METRICS], width,
               yerr=[sub[m].std(ddof=0) for m in METRICS], capsize=3,
               label=model, color=_color(model))
    ax.set_xticks(x + width * (len(models) - 1) / 2, [m.capitalize() if m != "auc" else "AUC" for m in METRICS])
    lo = min(df[METRICS].min().min() - 0.03, 0.9)
    ax.set(ylim=(max(lo, 0), 1.0), ylabel="Score (test set)", title="Detection performance")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=len(models), frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_times(df, path):
    models = _models(df)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, col, title in zip(axes, ("train_time", "predict_time"), ("Training time (s)", "Inference time on test set (s)")):
        # Median with inter-quartile range: occasional system slowdowns would distort mean/std.
        groups = [df[df.model == m][col] for m in models]
        med = np.array([g.median() for g in groups])
        err = np.array([[g.median() - g.quantile(0.25) for g in groups], [g.quantile(0.75) - g.median() for g in groups]])
        ax.bar(models, med, yerr=err, color=[_color(m) for m in models], capsize=4)
        ax.set_title(title + " - median, IQR")
        ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_confusions(runs, path):
    first_seed = min(r["seed"] for r in runs)
    picks = [r for r in runs if r["seed"] == first_seed]
    ncols = min(3, len(picks))
    nrows = -(-len(picks) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.6 * nrows), squeeze=False)
    for ax in axes.flat[len(picks):]:
        ax.axis("off")
    for ax, r in zip(axes.flat, picks):
        cm = np.array(r["metrics"]["confusion"])
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v}", ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
        ax.set(title=r["model"], xticks=[0, 1], yticks=[0, 1], xticklabels=["Normal", "Attack"],
               yticklabels=["Normal", "Attack"], xlabel="Predicted", ylabel="Actual")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_feature_frequency(runs, names, path, top=20):
    algos = ("ABC-RF", "ACO-RF")
    freq = {}
    for algo in algos:
        sel = [r["selected"] for r in runs if r["model"] == algo]
        counts = np.zeros(len(names))
        for s in sel:
            counts[s] += 1
        freq[algo] = counts / max(len(sel), 1)
    order = np.argsort(freq["ABC-RF"] + freq["ACO-RF"])[::-1][:top][::-1]
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(7.5, 0.3 * top + 1.5))
    ax.barh(y - 0.2, freq["ABC-RF"][order], 0.4, label="ABC-RF", color=COLORS["ABC-RF"])
    ax.barh(y + 0.2, freq["ACO-RF"][order], 0.4, label="ACO-RF", color=COLORS["ACO-RF"])
    ax.set_yticks(y, [names[i] for i in order])
    ax.set(xlabel="Fraction of runs selecting the feature", xlim=(0, 1.05),
           title=f"Most frequently selected features (top {top})")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def make_all(results_path, out_dir):
    payload = json.loads(Path(results_path).read_text())
    runs, names = payload["runs"], payload["meta"]["feature_names"]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = runs_to_frame(runs)

    plot_convergence(runs, out / "convergence.png")
    plot_features(df, len(names), out / "features_kept.png")
    plot_metrics(df, out / "metrics_comparison.png")
    plot_times(df, out / "time_comparison.png")
    plot_confusions(runs, out / "confusion_matrices.png")
    plot_feature_frequency(runs, names, out / "feature_frequency.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/results.json")
    ap.add_argument("--out", default="results/figures")
    a = ap.parse_args()
    make_all(a.results, a.out)
