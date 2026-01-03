import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import LOGREG_MAX_ITER
from .evaluate import classification_metrics, evaluate_multioutput


def _train_test_split_safe(X, y, test_size, random_state, split_strategy):
    stratify = None
    if split_strategy == "stratified" and y.nunique() > 1:
        stratify = y
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)


def _fill_labels_with_mode(labels):
    filled = labels.copy()
    for col in filled.columns:
        mode = filled[col].dropna().mode()
        fallback = 0
        val = mode.iloc[0] if not mode.empty else fallback
        filled[col] = filled[col].fillna(val)
    # Keep only rows that had at least one observed label originally
    mask = labels.notna().any(axis=1)
    return filled.loc[mask]


def build_preprocessor(categorical_cols, numeric_cols, abx_cols=None):
    transformers = []
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        # Dense output keeps downstream models compatible (including tree/boosting).
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_cols,
            )
        )
    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            )
        )
    if abx_cols:
        transformers.append(
            (
                "abx",
                SimpleImputer(strategy="constant", fill_value=-1),
                abx_cols,
            )
        )
    return ColumnTransformer(transformers=transformers)


def build_estimator(model_name, random_state):
    if model_name == "logreg":
        return LogisticRegression(
            max_iter=LOGREG_MAX_ITER,
            class_weight="balanced",
            solver="saga",
        )
    if model_name == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
    if model_name == "hgb":
        # Histogram-based GBDT, robust to mixed feature scales.
        return HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_depth=8,
            max_iter=400,
            random_state=random_state,
        )
    raise ValueError(f"Unknown model name: {model_name}")


def train_multioutput(
    features,
    labels,
    categorical_cols,
    numeric_cols,
    model_name,
    test_size,
    random_state,
    strategy,
    min_complete_cases,
):
    if strategy == "complete":
        y = labels.dropna()
        if len(y) < min_complete_cases:
            raise ValueError(
                f"Only {len(y)} complete-case samples (< {min_complete_cases}); skipping multi-output."
            )
    elif strategy == "impute":
        y = _fill_labels_with_mode(labels)
        if len(y) < min_complete_cases:
            raise ValueError(
                f"Only {len(y)} samples after label imputation (< {min_complete_cases}); skipping multi-output."
            )
    else:
        raise ValueError(f"Unknown multi-output strategy: {strategy}")

    X = features.loc[y.index]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    preprocessor = build_preprocessor(categorical_cols, numeric_cols)
    estimator = build_estimator(model_name, random_state)
    model = Pipeline(
        [
            ("preprocess", preprocessor),
            ("clf", MultiOutputClassifier(estimator)),
        ]
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_df = pd.DataFrame(y_pred, columns=y.columns, index=y_test.index)
    try:
        y_prob = model.predict_proba(X_test)
    except AttributeError:
        y_prob = None

    metrics = evaluate_multioutput(y_test, y_pred_df, y_prob)
    return model, metrics


def train_independent_models(
    features,
    labels,
    categorical_cols,
    numeric_cols,
    model_name,
    test_size,
    random_state,
    split_strategy,
):
    results = []
    for target in labels.columns:
        y = labels[target].dropna()
        if y.nunique() < 2:
            continue
        X = features.loc[y.index]
        X_train, X_test, y_train, y_test = _train_test_split_safe(
            X, y, test_size, random_state, split_strategy
        )

        preprocessor = build_preprocessor(categorical_cols, numeric_cols)
        estimator = build_estimator(model_name, random_state)
        model = Pipeline(
            [
                ("preprocess", preprocessor),
                ("clf", estimator),
            ]
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_score = None
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]

        metrics = classification_metrics(y_test, y_pred, y_score)
        metrics.update({"antibiotic": target, "n_train": len(y_train), "n_test": len(y_test)})
        results.append(metrics)

    result_df = pd.DataFrame(results).set_index("antibiotic")
    return result_df


def train_missing_outcome_models(
    features,
    labels,
    categorical_cols,
    numeric_cols,
    model_name,
    test_size,
    random_state,
    min_other_tests=1,
    split_strategy="random",
):
    results = []
    for target in labels.columns:
        y = labels[target].dropna()
        if y.nunique() < 2:
            continue

        other = labels.drop(columns=[target])
        X = pd.concat([features, other], axis=1)
        X = X.loc[y.index]
        other_known = other.loc[y.index]
        mask = other_known.notna().sum(axis=1) >= min_other_tests
        X = X[mask]
        y = y[mask]
        if len(y) < 100:
            continue

        abx_cols = list(other.columns)
        all_cat = categorical_cols
        all_num = numeric_cols

        X_train, X_test, y_train, y_test = _train_test_split_safe(
            X, y, test_size, random_state, split_strategy
        )

        preprocessor = build_preprocessor(all_cat, all_num, abx_cols=abx_cols)
        estimator = build_estimator(model_name, random_state)
        model = Pipeline(
            [
                ("preprocess", preprocessor),
                ("clf", estimator),
            ]
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_score = None
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]

        metrics = classification_metrics(y_test, y_pred, y_score)
        metrics.update({"antibiotic": target, "n_train": len(y_train), "n_test": len(y_test)})
        results.append(metrics)

    result_df = pd.DataFrame(results).set_index("antibiotic")
    return result_df
