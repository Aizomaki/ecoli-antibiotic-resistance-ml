import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

from src.config import (
    DATA_PATH,
    LABEL_MAP,
    MAX_ANTIBIOTICS,
    MIN_COMPLETE_CASES,
    MIN_LABELS_PER_ANTIBIOTIC,
    MIN_NEGATIVES_PER_ANTIBIOTIC,
    MIN_POSITIVES_PER_ANTIBIOTIC,
    OUTPUT_DIR,
    RANDOM_STATE,
    TASK_B_MIN_OTHER_TESTS,
    TEST_SIZE,
)
from src.data_prep import (
    build_feature_table,
    build_label_matrix,
    load_raw_data,
    map_labels,
    select_antibiotics,
    split_feature_columns,
)
from src.modeling import (
    train_independent_models,
    train_missing_outcome_models,
    train_multioutput,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run E. coli AMR modeling pipeline")
    parser.add_argument("--data-path", default=DATA_PATH, type=Path)
    parser.add_argument(
        "--task",
        default="all",
        choices=["all", "task-a", "task-b"],
        help="Which task(s) to run",
    )
    parser.add_argument(
        "--model",
        default="logreg",
        choices=["logreg", "rf", "hgb"],
        help="(Deprecated) single model to run. Use --models for multiple.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["logreg", "rf", "hgb"],
        help="List of models to run; overrides --model.",
    )
    parser.add_argument("--min-labels", type=int, default=MIN_LABELS_PER_ANTIBIOTIC)
    parser.add_argument("--min-pos", type=int, default=MIN_POSITIVES_PER_ANTIBIOTIC)
    parser.add_argument("--min-neg", type=int, default=MIN_NEGATIVES_PER_ANTIBIOTIC)
    parser.add_argument("--max-antibiotics", type=int, default=MAX_ANTIBIOTICS)
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help="Optional list of seeds; overrides --random-state if provided.",
    )
    parser.add_argument(
        "--split-strategy",
        choices=["random", "stratified"],
        default="stratified",
        help="Train/test split strategy for single-label tasks.",
    )
    parser.add_argument(
        "--min-other-tests",
        type=int,
        default=TASK_B_MIN_OTHER_TESTS,
        help="Minimum known other antibiotics for Task B",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--min-complete-cases",
        type=int,
        default=MIN_COMPLETE_CASES,
        help="Skip Task A multi-output if fewer complete-case samples than this.",
    )
    parser.add_argument(
        "--multi-strategy",
        choices=["complete", "impute", "both"],
        default="both",
        help="Multi-output handling: complete-case only, mode-imputed labels, or both.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_names = args.models or [args.model]
    seeds = args.seeds or [args.random_state]

    df_raw = load_raw_data(args.data_path)
    df_labels = map_labels(df_raw, LABEL_MAP)

    antibiotics, stats = select_antibiotics(
        df_labels,
        min_labels=args.min_labels,
        min_pos=args.min_pos,
        min_neg=args.min_neg,
        max_antibiotics=args.max_antibiotics,
    )
    if not antibiotics:
        raise SystemExit("No antibiotics matched the selection thresholds.")

    stats.to_csv(args.output_dir / "antibiotic_stats.csv")

    label_matrix = build_label_matrix(df_labels, antibiotics)
    feature_table = build_feature_table(df_raw)
    cat_cols, num_cols = split_feature_columns(feature_table)

    summary_rows = []
    first_run = True

    for model_name in model_names:
        for seed in seeds:
            suffix = "" if (len(model_names) == 1 and len(seeds) == 1) else f"_{model_name}_seed{seed}"

            if args.task in {"all", "task-a"}:
                strategies = ["complete", "impute"] if args.multi_strategy == "both" else [args.multi_strategy]
                for strat in strategies:
                    try:
                        _, multi_metrics = train_multioutput(
                            feature_table,
                            label_matrix,
                            cat_cols,
                            num_cols,
                            model_name,
                            args.test_size,
                            seed,
                            strat,
                            args.min_complete_cases,
                        )
                        path = args.output_dir / f"task_a_multioutput_{strat}_metrics{suffix}.csv"
                        multi_metrics.to_csv(path)
                        if first_run and suffix:
                            multi_metrics.to_csv(args.output_dir / f"task_a_multioutput_{strat}_metrics.csv")
                        summary_rows.append(
                            multi_metrics.assign(
                                task=f"task_a_multioutput_{strat}", model=model_name, seed=seed
                            ).reset_index()
                        )
                    except ValueError as exc:
                        print(
                            f"Task A multi-output ({strat}) skipped for {model_name} seed {seed}: {exc}"
                        )

                independent_metrics = train_independent_models(
                    feature_table,
                    label_matrix,
                    cat_cols,
                    num_cols,
                    model_name,
                    args.test_size,
                    seed,
                    args.split_strategy,
                )
                path = args.output_dir / f"task_a_independent_metrics{suffix}.csv"
                independent_metrics.to_csv(path)
                if first_run and suffix:
                    independent_metrics.to_csv(args.output_dir / "task_a_independent_metrics.csv")
                summary_rows.append(
                    independent_metrics.assign(task="task_a_independent", model=model_name, seed=seed).reset_index()
                )

            if args.task in {"all", "task-b"}:
                missing_metrics = train_missing_outcome_models(
                    feature_table,
                    label_matrix,
                    cat_cols,
                    num_cols,
                    model_name,
                    args.test_size,
                    seed,
                    min_other_tests=args.min_other_tests,
                    split_strategy=args.split_strategy,
                )
                path = args.output_dir / f"task_b_missing_outcomes_metrics{suffix}.csv"
                missing_metrics.to_csv(path)
                if first_run and suffix:
                    missing_metrics.to_csv(args.output_dir / "task_b_missing_outcomes_metrics.csv")
                summary_rows.append(
                    missing_metrics.assign(task="task_b_missing_outcomes", model=model_name, seed=seed).reset_index()
                )

            first_run = False

    if summary_rows:
        summary = (
            pd.concat(summary_rows, ignore_index=True)
            .rename(columns={"index": "antibiotic"})
        )
        summary.to_csv(args.output_dir / "experiment_summary.csv", index=False)

    print(f"Selected antibiotics: {', '.join(antibiotics)}")
    print(f"Outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
