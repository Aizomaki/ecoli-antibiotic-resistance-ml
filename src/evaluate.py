import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def _safe_roc_auc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return roc_auc_score(y_true, y_score)


def classification_metrics(y_true, y_pred, y_score=None):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_score is not None:
        metrics["roc_auc"] = _safe_roc_auc(y_true, y_score)
    else:
        metrics["roc_auc"] = float("nan")
    return metrics


def evaluate_multioutput(y_true_df, y_pred_df, y_prob_list=None):
    rows = []
    columns = list(y_true_df.columns)
    for idx, col in enumerate(columns):
        y_true = y_true_df[col].values
        y_pred = y_pred_df[col].values
        y_score = None
        if y_prob_list is not None:
            probs = y_prob_list[idx]
            if probs.ndim == 2 and probs.shape[1] >= 2:
                y_score = probs[:, 1]
        metrics = classification_metrics(y_true, y_pred, y_score)
        rows.append({"antibiotic": col, **metrics})

    result = pd.DataFrame(rows).set_index("antibiotic")
    macro = result.mean(numeric_only=True)
    macro.name = "macro_avg"
    result = pd.concat([result, macro.to_frame().T])
    return result
