from pathlib import Path

import pandas as pd

from .config import (
    ANTIBIOTIC_COL,
    CATEGORICAL_FEATURES,
    DATA_URL,
    ID_COL,
    LABEL_COL,
    NUMERIC_FEATURES,
    RAW_COLUMNS,
)


def load_raw_data(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download it from {DATA_URL} and place it there."
        )
    return pd.read_csv(path, usecols=RAW_COLUMNS, dtype=str, low_memory=False)


def _normalize_label(series):
    return series.fillna("").str.strip().str.lower()


def map_labels(df, label_map):
    df = df.copy()
    normalized = _normalize_label(df[LABEL_COL])
    label_map_lower = {k.lower(): v for k, v in label_map.items()}
    df["label"] = normalized.map(label_map_lower)
    return df


def select_antibiotics(
    df,
    min_labels,
    min_pos,
    min_neg,
    max_antibiotics=None,
):
    stats = (
        df.dropna(subset=["label"])
        .groupby(ANTIBIOTIC_COL)["label"]
        .agg(["count", "sum"])
    )
    stats["neg"] = stats["count"] - stats["sum"]
    filtered = stats[
        (stats["count"] >= min_labels)
        & (stats["sum"] >= min_pos)
        & (stats["neg"] >= min_neg)
    ]
    if max_antibiotics:
        filtered = filtered.sort_values("count", ascending=False).head(max_antibiotics)
    antibiotics = filtered.index.tolist()
    return antibiotics, filtered


def build_label_matrix(df, antibiotics):
    subset = df[df[ANTIBIOTIC_COL].isin(antibiotics)].copy()
    labels = subset.pivot_table(
        index=ID_COL,
        columns=ANTIBIOTIC_COL,
        values="label",
        aggfunc="max",
    )
    return labels


def _mode(series):
    series = series.dropna()
    if series.empty:
        return None
    return series.value_counts().idxmax()


def build_feature_table(df):
    work = df.copy()
    work["Testing Standard Year"] = pd.to_numeric(
        work["Testing Standard Year"], errors="coerce"
    )
    work["has_pubmed"] = work["PubMed"].notna().astype(int)
    work["has_source"] = work["Source"].notna().astype(int)

    grouped = work.groupby(ID_COL)

    features = pd.DataFrame(index=grouped.size().index)
    features["num_tests"] = grouped.size()
    features["num_unique_antibiotics"] = grouped[ANTIBIOTIC_COL].nunique()

    for col in CATEGORICAL_FEATURES:
        features[col] = grouped[col].agg(_mode)

    features["Testing Standard Year"] = grouped["Testing Standard Year"].median()
    features["has_pubmed"] = grouped["has_pubmed"].max()
    features["has_source"] = grouped["has_source"].max()

    ordered_cols = [
        *CATEGORICAL_FEATURES,
        "Testing Standard Year",
        "num_tests",
        "num_unique_antibiotics",
        "has_pubmed",
        "has_source",
    ]
    return features[ordered_cols]


def split_feature_columns(feature_df):
    categorical = [col for col in CATEGORICAL_FEATURES if col in feature_df.columns]
    numeric = [col for col in NUMERIC_FEATURES if col in feature_df.columns]
    return categorical, numeric
