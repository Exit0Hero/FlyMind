"""Evaluation metrics and reporting."""

import numpy as np
import json
import pathlib
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix,
)


def evaluate_predictions(y_true, y_pred, class_names):
    """Compute all evaluation metrics."""
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0,
    )
    report_str = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "report": report,
        "report_str": report_str,
        "confusion_matrix": cm.tolist(),
        "num_classes": len(class_names),
    }


def save_results(results: dict, path: pathlib.Path):
    """Save results to JSON."""
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
