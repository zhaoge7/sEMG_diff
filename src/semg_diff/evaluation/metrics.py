from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def per_class_f1(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    scores = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return pd.DataFrame({"label": labels, "f1": scores})


def confusion_matrix_frame(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=[f"true_{x}" for x in labels], columns=[f"pred_{x}" for x in labels])


def subject_stability_metrics(results: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
        if metric not in results:
            continue
        values = results[metric].dropna().to_numpy(float)
        if values.size == 0:
            continue
        ordered = np.sort(values)
        worst_quartile_n = max(1, int(np.ceil(values.size * 0.25)))
        out[f"{metric}_mean"] = float(np.mean(values))
        out[f"{metric}_std"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        out[f"{metric}_worst"] = float(np.min(values))
        out[f"{metric}_worst25_mean"] = float(np.mean(ordered[:worst_quartile_n]))
    return out
