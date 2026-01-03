import argparse
from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize AMR model outputs")
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--examples-dir", type=Path, default=Path("reports/examples"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/figures"))
    parser.add_argument("--format", default="png", help="Image format (png, pdf, svg)")
    return parser.parse_args()


def _load_csv(path):
    if not path.exists():
        print(f"Missing file: {path}")
        return None
    return pd.read_csv(path)


def _get_id_col(df):
    if "antibiotic" in df.columns:
        return "antibiotic"
    if "Antibiotic" in df.columns:
        return "Antibiotic"
    if "Unnamed: 0" in df.columns:
        df.rename(columns={"Unnamed: 0": "antibiotic"}, inplace=True)
        return "antibiotic"
    return None


def _save(fig, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _wrap_labels(labels, width=20):
    wrapped = []
    for label in labels:
        text = str(label)
        if "/" in text:
            text = text.replace("/", "/\n")
        wrapped.append(
            textwrap.fill(text, width=width, break_long_words=False, break_on_hyphens=False)
        )
    return wrapped


def _apply_wrapped_ylabels(ax, width=20, fontsize=9):
    labels = [label.get_text() for label in ax.get_yticklabels()]
    wrapped = _wrap_labels(labels, width=width)
    ticks = ax.get_yticks()
    ax.set_yticks(ticks)
    ax.set_yticklabels(wrapped, fontsize=fontsize)


def plot_antibiotic_stats(stats_df, output_dir, fmt):
    df = stats_df.copy()
    df["resistance_rate"] = df["sum"] / df["count"]

    df_sorted = df.sort_values("count", ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(5, 0.5 * len(df_sorted) + 1.5)))
    sns.barplot(data=df_sorted, x="count", y="Antibiotic", ax=ax, color="#2a6f97")
    ax.set_title("Antibiotic Coverage (count)")
    ax.set_xlabel("labelled samples")
    ax.set_ylabel("")
    _apply_wrapped_ylabels(ax, width=20, fontsize=9)
    _save(fig, output_dir / f"antibiotic_counts.{fmt}")

    rate_sorted = df.sort_values("resistance_rate", ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(5, 0.5 * len(rate_sorted) + 1.5)))
    sns.barplot(
        data=rate_sorted, x="resistance_rate", y="Antibiotic", ax=ax, color="#3a86ff"
    )
    ax.set_xlim(0, 1)
    ax.set_title("Resistance Rate by Antibiotic")
    ax.set_xlabel("resistant / labelled")
    ax.set_ylabel("")
    _apply_wrapped_ylabels(ax, width=20, fontsize=9)
    _save(fig, output_dir / f"antibiotic_resistance_rate.{fmt}")


def plot_missingness(missing_df, output_dir, fmt):
    df = missing_df.copy().sort_values("percent_missing", ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(5, 0.5 * len(df) + 1.5)))
    sns.barplot(data=df, x="percent_missing", y="Antibiotic", ax=ax, color="#ff8fa3")
    ax.set_title("Missingness per Antibiotic")
    ax.set_xlabel("percent missing")
    ax.set_ylabel("")
    _apply_wrapped_ylabels(ax, width=20, fontsize=9)
    _save(fig, output_dir / f"missingness_by_antibiotic.{fmt}")


def plot_metrics(metrics_df, output_dir, fmt, stem):
    df = metrics_df.copy()
    id_col = _get_id_col(df)
    if id_col is None:
        print(f"Could not infer id column for {stem}")
        return

    df = df[df[id_col] != "macro_avg"]

    metrics = [col for col in ["accuracy", "f1", "roc_auc"] if col in df.columns]
    if not metrics:
        print(f"No metric columns found for {stem}")
        return

    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(metrics),
        figsize=(5 * len(metrics), max(5, 0.5 * len(df) + 1.5)),
    )
    fig.subplots_adjust(wspace=0.5)
    if len(metrics) == 1:
        axes = [axes]

    for metric, ax in zip(metrics, axes):
        plot_df = df[[id_col, metric]].dropna().sort_values(metric, ascending=True)
        sns.barplot(data=plot_df, x=metric, y=id_col, ax=ax, color="#34a0a4")
        if metric in {"accuracy", "f1", "roc_auc"}:
            ax.set_xlim(0, 1)
        ax.set_title(metric)
        ax.set_xlabel(metric)
        ax.set_ylabel("")
        _apply_wrapped_ylabels(ax, width=20, fontsize=9)

    fig.suptitle(stem.replace("_", " ").title())
    _save(fig, output_dir / f"{stem}_metrics.{fmt}")

    if "n_train" in df.columns and "n_test" in df.columns:
        size_df = df[[id_col, "n_train", "n_test"]].melt(
            id_vars=id_col, var_name="split", value_name="count"
        )
        size_df = size_df.sort_values("count", ascending=True)
        fig, ax = plt.subplots(figsize=(8, max(5, 0.5 * len(df) + 1.5)))
        sns.barplot(data=size_df, x="count", y=id_col, hue="split", ax=ax)
        ax.set_title(f"{stem.replace('_', ' ').title()} sample sizes")
        ax.set_xlabel("samples")
        ax.set_ylabel("")
        _apply_wrapped_ylabels(ax, width=20, fontsize=9)
        _save(fig, output_dir / f"{stem}_sample_sizes.{fmt}")


def plot_experiment_summary(summary_df, output_dir, fmt):
    df = summary_df.copy()
    if "antibiotic" in df.columns:
        df = df[df["antibiotic"] != "macro_avg"]
    if not {"task", "model"}.issubset(df.columns):
        print("experiment_summary.csv missing required columns: task/model")
        return

    metrics = [col for col in ["accuracy", "f1", "roc_auc"] if col in df.columns]
    if not metrics:
        print("experiment_summary.csv missing metric columns")
        return

    agg = df.groupby(["task", "model"])[metrics].mean(numeric_only=True).reset_index()
    agg["task_model"] = agg["task"] + " / " + agg["model"]

    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(metrics),
        figsize=(6 * len(metrics), max(6, 0.6 * len(agg) + 2)),
    )
    fig.subplots_adjust(wspace=0.6)
    if len(metrics) == 1:
        axes = [axes]

    for metric, ax in zip(metrics, axes):
        plot_df = agg.sort_values(metric, ascending=True)
        sns.barplot(data=plot_df, x=metric, y="task_model", ax=ax, color="#577590")
        ax.set_xlim(0, 1)
        ax.set_title(metric)
        ax.set_xlabel(metric)
        ax.set_ylabel("")
        _apply_wrapped_ylabels(ax, width=24, fontsize=9)

    fig.suptitle("Macro Metrics by Task and Model")
    _save(fig, output_dir / f"task_model_macro_metrics.{fmt}")


def main():
    args = parse_args()
    sns.set_theme(style="whitegrid")

    stats_df = _load_csv(args.reports_dir / "antibiotic_stats.csv")
    if stats_df is not None:
        plot_antibiotic_stats(stats_df, args.output_dir, args.format)

    missing_df = _load_csv(args.examples_dir / "missingness_summary.csv")
    if missing_df is not None:
        plot_missingness(missing_df, args.output_dir, args.format)

    metrics_files = [
        ("task_a_multioutput", args.reports_dir / "task_a_multioutput_metrics.csv"),
        ("task_a_independent", args.reports_dir / "task_a_independent_metrics.csv"),
        ("task_b_missing_outcomes", args.reports_dir / "task_b_missing_outcomes_metrics.csv"),
    ]
    for stem, path in metrics_files:
        df = _load_csv(path)
        if df is None:
            continue
        plot_metrics(df, args.output_dir, args.format, stem)

    summary_df = _load_csv(args.reports_dir / "experiment_summary.csv")
    if summary_df is not None:
        plot_experiment_summary(summary_df, args.output_dir, args.format)

    print(f"Figures written to {args.output_dir}")


if __name__ == "__main__":
    main()
