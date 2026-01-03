import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config import (
    ANTIBIOTIC_COL,
    DATA_PATH,
    ID_COL,
    LABEL_COL,
    LABEL_MAP,
    MAX_ANTIBIOTICS,
    MIN_LABELS_PER_ANTIBIOTIC,
    MIN_NEGATIVES_PER_ANTIBIOTIC,
    MIN_POSITIVES_PER_ANTIBIOTIC,
    OUTPUT_DIR,
)
from src.data_prep import (
    build_feature_table,
    build_label_matrix,
    load_raw_data,
    map_labels,
    select_antibiotics,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate report example tables")
    parser.add_argument("--data-path", default=DATA_PATH, type=Path)
    parser.add_argument("--output-dir", default=OUTPUT_DIR / "examples", type=Path)
    parser.add_argument("--max-antibiotics", type=int, default=MAX_ANTIBIOTICS)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df_raw = load_raw_data(args.data_path)
    raw_cols = [
        ID_COL,
        ANTIBIOTIC_COL,
        LABEL_COL,
        "Laboratory Typing Method",
        "Testing Standard",
        "Testing Standard Year",
    ]
    df_raw[raw_cols].head(10).to_csv(args.output_dir / "raw_rows.csv", index=False)

    label_map_df = pd.DataFrame(
        sorted(LABEL_MAP.items(), key=lambda item: item[0].lower()),
        columns=["raw_label", "binary_label"],
    )
    label_map_df.to_csv(args.output_dir / "label_mapping.csv", index=False)

    df_labels = map_labels(df_raw, LABEL_MAP)
    antibiotics, _ = select_antibiotics(
        df_labels,
        min_labels=MIN_LABELS_PER_ANTIBIOTIC,
        min_pos=MIN_POSITIVES_PER_ANTIBIOTIC,
        min_neg=MIN_NEGATIVES_PER_ANTIBIOTIC,
        max_antibiotics=args.max_antibiotics,
    )
    if not antibiotics:
        raise SystemExit("No antibiotics matched the selection thresholds.")

    label_matrix = build_label_matrix(df_labels, antibiotics)
    features = build_feature_table(df_raw)
    sample_level = features.join(label_matrix, how="left")
    sample_level.head(5).to_csv(args.output_dir / "sample_level_preview.csv")

    missingness = label_matrix.isna().mean().mul(100).sort_values(ascending=False)
    missingness.to_frame("percent_missing").to_csv(
        args.output_dir / "missingness_summary.csv"
    )

    if len(label_matrix.columns) > 1:
        partial_mask = label_matrix.notna().sum(axis=1).between(
            1, len(label_matrix.columns) - 1
        )
        partial_mask = partial_mask.reindex(sample_level.index, fill_value=False)
        if partial_mask.any():
            sample_level.loc[partial_mask].head(1).to_csv(
                args.output_dir / "partial_sample_example.csv"
            )

    print(f"Examples written to {args.output_dir}")


if __name__ == "__main__":
    main()
