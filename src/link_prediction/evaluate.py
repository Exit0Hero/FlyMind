"""FlyMind Link Prediction — Evaluation metrics."""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve


def evaluate_link_prediction(y_true, y_score, k_values=None):
    """Compute comprehensive link prediction metrics."""
    if k_values is None:
        k_values = [100, 1000, 5000]

    results = {}

    # ROC-AUC
    results["roc_auc"] = float(roc_auc_score(y_true, y_score))

    # PR-AUC
    results["pr_auc"] = float(average_precision_score(y_true, y_score))

    # Precision@K and Recall@K
    n_pos = int(y_true.sum())
    sorted_indices = np.argsort(-y_score)
    sorted_true = y_true[sorted_indices]

    for k in k_values:
        if k > len(y_true):
            continue
        top_k = sorted_true[:k]
        prec_k = float(top_k.sum() / k)
        rec_k = float(top_k.sum() / max(n_pos, 1))
        results[f"precision_at_{k}"] = prec_k
        results[f"recall_at_{k}"] = rec_k
        results[f"hits_at_{k}"] = int(top_k.sum())

    # F1 at best threshold
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    f1_scores = 2 * precisions * recalls / np.maximum(precisions + recalls, 1e-10)
    best_idx = np.argmax(f1_scores)
    results["best_f1"] = float(f1_scores[best_idx])
    results["best_threshold"] = float(thresholds[min(best_idx, len(thresholds) - 1)])

    return results


def compute_all_metrics(y_true, y_score, label=""):
    """Print and return all metrics."""
    metrics = evaluate_link_prediction(y_true, y_score)
    print(f"  [{label}] ROC-AUC: {metrics['roc_auc']:.4f} | PR-AUC: {metrics['pr_auc']:.4f}")
    for k in [100, 1000, 5000]:
        pk = f"precision_at_{k}"
        rk = f"recall_at_{k}"
        if pk in metrics:
            print(f"    P@{k}: {metrics[pk]:.4f} | R@{k}: {metrics[rk]:.4f} | Hits@{k}: {metrics[f'hits_at_{k}']}")
    return metrics
