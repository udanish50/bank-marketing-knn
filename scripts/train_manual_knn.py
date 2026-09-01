#!/usr/bin/env python3
"""Train and evaluate a manual KNN model on the UCI Bank Marketing dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bank_marketing_knn.data import load_bank_marketing_csv
from bank_marketing_knn.knn import ManualKNNClassifier
from bank_marketing_knn.metrics import (
    evaluate_binary_classifier,
    save_confusion_matrix,
    save_metrics,
)
from bank_marketing_knn.preprocessing import BankMarketingPreprocessor, split_features_target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-zip", default="data/raw/bank+marketing.zip")
    parser.add_argument(
        "--dataset",
        default="bank",
        choices=["bank", "bank-full", "bank-additional", "bank-additional-full"],
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--metrics-output", default="reports/metrics.json")
    parser.add_argument("--confusion-output", default="reports/confusion_matrix.csv")
    parser.add_argument("--figure-output", default="reports/figures/confusion_matrix.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_zip = PROJECT_ROOT / args.dataset_zip

    frame = load_bank_marketing_csv(dataset_zip, dataset=args.dataset)
    raw_features, target = split_features_target(frame)

    x_train_raw, x_test_raw, y_train, y_test = train_test_split(
        raw_features,
        target,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=target,
    )

    preprocessor = BankMarketingPreprocessor.default()
    x_train = preprocessor.fit_transform(x_train_raw)
    x_test = preprocessor.transform(x_test_raw)

    model = ManualKNNClassifier(k=args.k, batch_size=args.batch_size).fit(x_train, y_train)
    y_probability = model.predict_proba(x_test)[:, 1]
    y_pred = model.predict(x_test)

    metrics = evaluate_binary_classifier(y_test, y_pred, y_probability)
    save_metrics(metrics, PROJECT_ROOT / args.metrics_output)
    save_confusion_matrix(y_test, y_pred, PROJECT_ROOT / args.confusion_output)
    save_confusion_matrix_plot(y_test, y_pred, PROJECT_ROOT / args.figure_output)

    print(f"Dataset: {args.dataset} ({frame.shape[0]:,} rows, {frame.shape[1]} columns)")
    print(f"Manual KNN: k={args.k}, test_size={args.test_size}, random_state={args.random_state}")
    print("\nKey metrics")
    for key in [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall_sensitivity",
        "specificity",
        "f1",
        "roc_auc",
        "pr_auc_average_precision",
        "matthews_corrcoef",
        "top_10_percent_lift",
    ]:
        print(f"  {key:28s} {metrics[key]:.4f}")


def save_confusion_matrix_plot(y_true, y_pred, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

    plt.figure(figsize=(6, 4.5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["no", "yes"],
        yticklabels=["no", "yes"],
        cbar=False,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Manual KNN Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


if __name__ == "__main__":
    main()
