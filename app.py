"""Demo app for the project.

    python -m streamlit run app.py

Needs data/KDDTrain+.txt, data/KDDTest+.txt and results_15seeds/ (from run_experiments.py).
"""
import json
import os
import time
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src.abc import ArtificialBeeColony
from src.aco import AntColony
from src.baselines import feature_mi
from src.evaluate import BASELINE_PARAMS, evaluate_model, runs_to_frame
from src.fitness import FitnessEvaluator
from src.inference import (DEPLOYED, load_raw, predict_attack_proba, read_uploaded_csv,
                           select_runs, train_deployed, transform_raw)
from src.preprocess import FEATURES, HOLDOUT_ATTACKS, load_nsl_kdd

DATA_DIR = "data"
RESULTS = Path(os.environ.get("RESULTS_DIR", "results_15seeds"))
ABLATION = Path(os.environ.get("ABLATION_DIR", "results_random_val"))
FIGURES = {
    "Convergence": "convergence.png",
    "Detection metrics": "metrics_comparison.png",
    "Features kept": "features_kept.png",
    "Training / inference time": "time_comparison.png",
    "Most selected features": "feature_frequency.png",
    "Confusion matrices": "confusion_matrices.png",
}

_version = tuple(int(p) for p in st.__version__.split(".")[:2])
STRETCH = {"width": "stretch"} if _version >= (1, 50) else {"use_container_width": True}

st.set_page_config(page_title="Intrusion detection: ABC vs ACO", page_icon=":shield:", layout="wide")


# ----------------------------------------------------------------------------- data and models
@st.cache_resource(show_spinner="Loading NSL-KDD and training the three models (about 30 s the first time)...")
def load_assets():
    data = load_nsl_kdd(DATA_DIR)
    raw_test, _, attack_test = load_raw(DATA_DIR, "Test")
    train_attacks = set(load_raw(DATA_DIR, "Train")[2])
    picks = select_runs(RESULTS / "results.json")
    models = train_deployed(data, picks)
    return data, raw_test, attack_test, train_attacks, picks, models, feature_mi(data)


@st.cache_data
def load_results(folder):
    folder = Path(folder)
    summary = pd.read_csv(folder / "summary.csv", index_col=0)
    sig_path = folder / "significance.csv"  # absent in runs made with fewer than 3 seeds
    significance = pd.read_csv(sig_path) if sig_path.exists() else pd.DataFrame()
    runs = json.loads((folder / "results.json").read_text())["runs"]
    return summary, significance, runs


@st.cache_data
def baseline_metrics():
    data = load_assets()[0]
    return evaluate_model(np.ones(data.n_features, dtype=bool), BASELINE_PARAMS, data, seed=0)


def used_raw_features(mask, feature_names):
    """Raw NSL-KDD features that contribute at least one (one-hot) column to the mask."""
    chosen = [n for n, keep in zip(feature_names, mask) if keep]
    return {f for f in FEATURES if any(n == f or n.startswith(f + "_") for n in chosen)}


def verdict(p):
    return f"{'Attack' if p >= 0.5 else 'Normal'} ({p:.2f})"


# ----------------------------------------------------------------------------- guard rails
try:
    data, raw_test, attack_test, train_attacks, picks, models, mi = load_assets()
except FileNotFoundError as err:
    st.error(f"{err}\n\nRun `python run_experiments.py --seeds 15 --out results_15seeds` and put NSL-KDD in `data/`.")
    st.stop()

summary, significance, runs = load_results(RESULTS)
runs_df = runs_to_frame(runs)

# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Deployed models")
    st.caption("ABC-RF and ACO-RF are the runs with the best *validation* fitness, retrained on train+val.")
    st.dataframe(pd.DataFrame({
        "features": {n: int(m.mask.sum()) for n, m in models.items()},
        "trees": {n: m.params["n_estimators"] for n, m in models.items()},
        "depth": {n: str(m.params["max_depth"]) for n, m in models.items()},
    }), **STRETCH)
    st.caption(f"Dataset: NSL-KDD, {data.n_features} features after one-hot encoding. "
               f"{len(data.y_test):,} test connections.")

st.title("Network intrusion detection")
st.markdown("Nature-inspired feature selection and Random Forest tuning on **NSL-KDD**: "
            "**Artificial Bee Colony (ABC)** vs **Ant Colony Optimization (ACO)**.")
tab_results, tab_detect, tab_lab, tab_how = st.tabs(["Results", "Detect traffic", "Optimizer lab", "How it works"])

# ----------------------------------------------------------------------------- results
with tab_results:
    n_seeds = runs_df.seed.nunique()
    base = runs_df[runs_df.model == "Baseline RF"]
    aco = runs_df[runs_df.model == "ACO-RF"]
    f1_test = significance[(significance.metric == "f1") & (significance.model == "ACO-RF")].iloc[0]
    p_t, p_w = f1_test.ttest_p, f1_test.wilcoxon_p
    strength = ("significant" if max(p_t, p_w) < 0.05 else
                "borderline: only one of the two tests is below 0.05" if min(p_t, p_w) < 0.05 else "not significant")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ACO-RF features", f"{aco.n_features.mean():.0f} / {data.n_features}",
              f"-{(1 - aco.n_features.mean() / data.n_features):.0%}", delta_color="off")
    c2.metric("ACO-RF test F1", f"{aco.f1.mean():.3f}", f"{aco.f1.mean() - base.f1.mean():+.3f} vs base", delta_color="off")
    c3.metric("Training speed-up", f"{base.train_time.median() / aco.train_time.median():.1f}x", "median", delta_color="off")
    c4.metric("Inference speed-up", f"{base.predict_time.median() / aco.predict_time.median():.1f}x", "median", delta_color="off")
    st.caption(
        f"{n_seeds} independent seeds, scored on KDDTest+. ACO-RF's F1 difference to the baseline is {f1_test.mean_diff:+.3f} "
        f"(paired t-test p = {p_t:.3f}, Wilcoxon p = {p_w:.3f}; {strength}). "
        "The clearest gains are in model size and speed.")

    st.subheader("All models (mean ± std over seeds)")
    cols = ["accuracy", "precision", "recall", "f1", "auc", "n_features"]
    st.dataframe(pd.DataFrame({
        c.replace("n_features", "features"): summary[c + "_mean"].round(3).astype(str) + " ± " + summary[c + "_std"].round(3).astype(str)
        for c in cols}), **STRETCH)

    st.subheader("Figures")
    choice = st.selectbox("Figure", list(FIGURES), label_visibility="collapsed")
    st.image(str(RESULTS / "figures" / FIGURES[choice]), **STRETCH)

    with st.expander("Paired significance tests vs the baseline"):
        st.dataframe(significance.round(4), hide_index=True, **STRETCH)
    if ABLATION.exists():
        with st.expander("Why validation matters: random vs held-out-attack validation"):
            rand_summary, _, _ = load_results(ABLATION)
            keep = ["Baseline RF", "MI top-30", "ABC-RF", "ACO-RF"]
            st.dataframe(pd.concat({
                "random validation": rand_summary.loc[keep, ["f1_mean", "f1_std", "n_features_mean"]],
                "held-out-attack validation": summary.loc[keep, ["f1_mean", "f1_std", "n_features_mean"]],
            }).round(4), **STRETCH)
            st.caption("With a random split the optimiser cannot see the gap between seen and unseen attacks, "
                       "so it trades test F1 for smaller models. Validating on held-out attack types fixes this.")

# ----------------------------------------------------------------------------- detect
with tab_detect:
    st.markdown("Classify connections with the **baseline** (all 122 features) and the two **optimised** forests, side by side.")
    source = st.radio("Traffic source", ["Sample from KDDTest+", "Upload a CSV"], horizontal=True)

    X_rows, raw_rows, truth, attack_names = None, None, None, None
    if source == "Sample from KDDTest+":
        c1, c2, c3 = st.columns([2, 2, 1])
        kind = c1.selectbox("Show", ["Any traffic", "Normal only", "Attacks only", "Attack types never seen in training"])
        n = c2.slider("Connections", 5, 50, 12)
        st.session_state.setdefault("sample_seed", 0)
        if c3.button("New sample", **STRETCH):
            st.session_state.sample_seed = int(time.time())

        is_attack = data.y_test == 1
        pool = {
            "Any traffic": np.ones(len(is_attack), dtype=bool),
            "Normal only": ~is_attack,
            "Attacks only": is_attack,
            "Attack types never seen in training": is_attack & ~np.isin(attack_test, list(train_attacks)),
        }[kind]
        chosen = np.random.RandomState(st.session_state.sample_seed).choice(np.flatnonzero(pool), n, replace=False)
        X_rows, raw_rows = data.X_test[chosen], raw_test.iloc[chosen].reset_index(drop=True)
        truth = np.where(data.y_test[chosen] == 1, "Attack", "Normal")
        attack_names = attack_test[chosen]
        example = pd.concat([raw_test.iloc[chosen[:20]].reset_index(drop=True),
                             pd.Series(attack_test[chosen[:20]])], axis=1)
        st.download_button("Download 20 of these as an example CSV", example.to_csv(header=False, index=False),
                           "example_connections.csv", "text/csv")
    else:
        st.caption(f"CSV with the {len(FEATURES)} NSL-KDD feature columns in the original order (header optional). "
                   "A label in column 42, if present, is used to score the predictions.")
        upload = st.file_uploader("CSV file", type=["csv", "txt"], label_visibility="collapsed")
        if upload is not None:
            try:
                raw_rows, attack_names = read_uploaded_csv(upload)
                X_rows = transform_raw(raw_rows, data)
                if attack_names is not None:
                    truth = np.where(attack_names == "normal", "Normal", "Attack")
            except ValueError as err:
                st.error(str(err))

    if X_rows is not None and len(X_rows):
        proba, seconds = predict_attack_proba(models, X_rows)
        table = pd.DataFrame({name: proba[name].map(verdict) for name in DEPLOYED})
        if truth is not None:
            table.insert(0, "True label", truth)
            table.insert(1, "Attack type", attack_names)
        table.index = np.arange(1, len(table) + 1)

        def shade(col):
            ok = [(v.startswith("Attack")) == (t == "Attack") for v, t in zip(col, table["True label"])]
            return ["background-color: rgba(46,160,67,.28)" if good else "background-color: rgba(218,54,51,.32)" for good in ok]

        st.dataframe(table.style.apply(shade, subset=DEPLOYED) if truth is not None else table, **STRETCH)

        cols = st.columns(len(DEPLOYED))
        for col, name in zip(cols, DEPLOYED):
            m = models[name]
            flagged = proba[name] >= 0.5
            if truth is not None:
                label, headline = f"{name}: correct", f"{int((flagged == (truth == 'Attack')).sum())} / {len(table)}"
            else:
                label, headline = f"{name}: flagged", f"{int(flagged.sum())} / {len(table)}"
            col.metric(label, headline, f"{seconds[name] * 1000:.0f} ms, {int(m.mask.sum())} feat.", delta_color="off")
        if truth is not None:
            st.caption("Green = correct, red = wrong. The number in brackets is each model's probability of attack.")

        st.subheader("Inspect one connection")
        row = st.selectbox("Connection", list(table.index),
                           format_func=lambda i: f"#{i}" + (f" - {table.loc[i, 'True label']} ({table.loc[i, 'Attack type']})" if truth is not None else ""))
        left, right = st.columns([3, 2])
        with left:
            used = {n: used_raw_features(models[n].mask, data.feature_names) for n in ("ABC-RF", "ACO-RF")}
            record = raw_rows.iloc[row - 1]
            st.dataframe(pd.DataFrame({
                "feature": FEATURES,
                "value": [str(record[f]) for f in FEATURES],
                "used by ABC-RF": ["yes" if f in used["ABC-RF"] else "" for f in FEATURES],
                "used by ACO-RF": ["yes" if f in used["ACO-RF"] else "" for f in FEATURES],
            }), hide_index=True, height=360, **STRETCH)
        with right:
            st.markdown("**Probability of attack**")
            st.bar_chart(pd.DataFrame({"P(attack)": proba.iloc[row - 1]}), horizontal=True)

# ----------------------------------------------------------------------------- optimizer lab
with tab_lab:
    st.markdown("Run the two optimisers yourself and watch them converge. The budget here is small so it finishes in seconds; "
                "the reported results use 300 evaluations over 15 seeds.")
    c1, c2, c3, c4 = st.columns(4)
    algos = c1.multiselect("Algorithms", ["ABC", "ACO"], default=["ABC", "ACO"])
    budget = c2.slider("Fitness evaluations", 20, 150, 60, step=10)
    pop = c3.slider("Bees / ants", 4, 12, 6)
    seed = c4.number_input("Seed", 0, 999, 0)
    run = st.button("Run optimisation", type="primary", disabled=not algos)

    def draw_curves(slot, series):
        frame = pd.DataFrame({n: pd.Series(v) for n, v in series.items()}).sort_index().ffill()
        long = frame.reset_index(names="evals").melt("evals", var_name="algorithm", value_name="best fitness").dropna()
        chart = alt.Chart(long).mark_line(interpolate="step-after", strokeWidth=3).encode(
            x=alt.X("evals:Q", title="fitness evaluations"),
            y=alt.Y("best fitness:Q", scale=alt.Scale(zero=False)),
            color=alt.Color("algorithm:N", scale=alt.Scale(domain=["ABC", "ACO"], range=["#1f77b4", "#d62728"])),
        )
        slot.altair_chart(chart, **STRETCH)

    live = st.empty()
    if run:
        series, results = {}, {}
        bar = st.progress(0.0)
        for k, name in enumerate(algos):
            ev = FitnessEvaluator(data, train_subsample=8000, val_subsample=6000, seed=int(seed))

            def on_progress(n_evals, best, name=name, k=k):
                series.setdefault(name, {})[n_evals] = best
                draw_curves(live, series)
                bar.progress(min((k + n_evals / budget) / len(algos), 1.0))

            if name == "ABC":
                algo = ArtificialBeeColony(ev, n_sources=pop, limit=pop, max_evals=budget, seed=int(seed), callback=on_progress)
            else:
                algo = AntColony(ev, mi, n_ants=pop, max_evals=budget, seed=int(seed), callback=on_progress)
            res = algo.run()
            results[name] = {"res": res, "metrics": evaluate_model(res["best_mask"], res["best_params"], data, seed=int(seed))}
        bar.empty()
        st.session_state.lab = {"series": series, "results": results}

    lab = st.session_state.get("lab")
    if lab:
        draw_curves(live, lab["series"])
        base = baseline_metrics()

        def lab_row(features, trees, depth, m):
            return {"features": str(features), "trees": str(trees), "max depth": str(depth),
                    "test F1": f"{m['f1']:.3f}", "test accuracy": f"{m['accuracy']:.3f}"}

        rows = {"Baseline RF (defaults)": lab_row(data.n_features, 100, "none", base)}
        for name, out in lab["results"].items():
            p = out["res"]["best_params"]
            rows[f"{name}-RF (just found)"] = lab_row(out["metrics"]["n_features"], p["n_estimators"], p["max_depth"], out["metrics"])
        st.dataframe(pd.DataFrame(rows).T, **STRETCH)
        for name, out in lab["results"].items():
            chosen_names = [n for n, keep in zip(data.feature_names, out["res"]["best_mask"]) if keep]
            with st.expander(f"{name}: the {len(chosen_names)} features it kept"):
                st.write(", ".join(chosen_names))
        st.caption("Short runs rarely beat the 300-evaluation, 15-seed results, and F1 on KDDTest+ varies by about ±0.02 between seeds.")

# ----------------------------------------------------------------------------- how it works
with tab_how:
    st.subheader("Pipeline")
    st.graphviz_chart(r'''
    digraph G {
      rankdir=TB; bgcolor="transparent"; nodesep=0.5; ranksep=0.35;
      node [shape=box, style="rounded,filled", fillcolor="#e8eef7", color="#7b8aa5", fontname="Helvetica", fontcolor="#111111"];
      edge [color="#7b8aa5", fontcolor="#9aa5b8", fontname="Helvetica"];
      raw [label="NSL-KDD\nconnections"]; pre [label="Preprocess\nclean, one-hot, scale"];
      split [label="Train / validation\n(held-out attack types)"];
      opt [label="ABC or ACO\nsearch", fillcolor="#fde7d6"];
      fit [label="Fitness:\nRF on mask + params\nscored on validation", fillcolor="#fde7d6"];
      final [label="Optimised RF\n(features + params)"]; test [label="Evaluate on KDDTest+\nF1, accuracy, speed"];
      raw -> pre -> split -> opt; opt -> fit [label=" candidate"]; fit -> opt [label=" score"];
      opt -> final -> test;
    }''')
    st.subheader("What each solution looks like")
    st.markdown(
        "A candidate is a **binary mask over the 122 features** plus three Random Forest settings "
        "(`n_estimators`, `max_depth`, `min_samples_split`). Both algorithms use the same fitness function:")
    st.code("fitness = 0.8 * F1(validation) + 0.1 * (1 - features_used / 122) + 0.1 * (1 - n_estimators / 100)", language=None)
    a, b = st.columns(2)
    a.markdown("""**Artificial Bee Colony**
- *Employed bees* perturb their food source with a neighbour move `v = x + φ (x − x_k)`; a sigmoid turns the position into a 0/1 mask.
- *Onlooker bees* pick sources by roulette wheel (better sources get more visits).
- *Scout bees* abandon a source after `limit` failed improvements and restart at random.""")
    b.markdown("""**Ant Colony Optimization**
- Each ant builds a feature subset, choosing features with probability ∝ `τ^α · η^β` (η = mutual information).
- Pheromone also lives on the subset size and on each hyper-parameter value.
- After every iteration pheromone *evaporates*, then the best ants *deposit* more; values are kept within MAX-MIN bounds.""")
    st.subheader("Why validation uses held-out attacks")
    st.markdown(
        "`KDDTest+` contains attack types that never appear in training, so a random validation split looks almost perfect "
        "and gives the optimiser no signal about generalisation. Validation therefore contains every row of "
        f"`{'`, `'.join(HOLDOUT_ATTACKS)}`, which the model never sees during the search.")
    st.subheader("Limitations")
    st.markdown(
        "- NSL-KDD is an older benchmark derived from KDD'99, not modern traffic.\n"
        "- Test F1 varies by about ±0.02 between seeds; only paired tests over many seeds can separate models.\n"
        "- The demo trains on KDDTrain+ only and is not a production intrusion detection system.")
