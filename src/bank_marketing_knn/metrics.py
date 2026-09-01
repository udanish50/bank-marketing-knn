"""Evaluation metrics for imbalanced binary classification."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_binary_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probability: np.ndarray,
) -> dict[str, float | int]:
    """Compute a broad metric set for a binary classifier."""

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    negative_predictive_value = tn / (tn + fn) if (tn + fn) else 0.0
    false_positive_rate = fp / (fp + tn) if (fp + tn) else 0.0
    false_negative_rate = fn / (fn + tp) if (fn + tp) else 0.0

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall_sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "fbeta_2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_probability),
        "pr_auc_average_precision": average_precision_score(y_true, y_probability),
        "log_loss": log_loss(y_true, np.column_stack([1 - y_probability, y_probability])),
        "brier_score": brier_score_loss(y_true, y_probability),
        "matthews_corrcoef": matthews_corrcoef(y_true, y_pred),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "negative_predictive_value": negative_predictive_value,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "top_10_percent_lift": top_percent_lift(y_true, y_probability, percent=0.10),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def top_percent_lift(y_true: np.ndarray, y_probability: np.ndarray, percent: float = 0.10) -> float:
    """Measure how enriched positives are in the highest-scored population slice."""

    if not 0 < percent <= 1:
        raise ValueError("percent must be in the interval (0, 1].")

    y_true = np.asarray(y_true)
    y_probability = np.asarray(y_probability)
    top_n = max(1, int(np.ceil(len(y_true) * percent)))
    top_indices = np.argsort(y_probability)[-top_n:]
    baseline_rate = y_true.mean()
    if baseline_rate == 0:
        return 0.0
    return float(y_true[top_indices].mean() / baseline_rate)


def save_metrics(metrics: dict[str, float | int], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {key: float(value) for key, value in metrics.items()}
    output_path.write_text(json.dumps(serializable, indent=2, sort_keys=True) + "\n")


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    frame = pd.DataFrame(
        matrix,
        index=["actual_no", "actual_yes"],
        columns=["predicted_no", "predicted_yes"],
    )
    frame.to_csv(output_path)
