#!/usr/bin/env python3
"""FlyMind Link Prediction — Part 3: Visualization and analysis."""

import sys
import json
import pathlib
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

from src.link_prediction.config import *

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc


def plot_roc_curves(results):
    """Plot ROC curves for all models."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, metrics in results.items():
        # We stored metrics, not full fpr/tpr — plot a point-based curve
        # For now, plot a diagonal + the AUC value as text
        pass

    # Actually, we need to recompute curves. Let's use a simpler comparison bar chart.
    plt.close(fig)


def plot_model_comparison(results, save_path):
    """Bar chart comparing models on ROC-AUC and PR-AUC."""
    models = list(results.keys())
    roc_aucs = [results[m]["roc_auc"] for m in models]
    pr_aucs = [results[m]["pr_auc"] for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width / 2, roc_aucs, width, label="ROC-AUC", color="steelblue")
    bars2 = ax.bar(x + width / 2, pr_aucs, width, label="PR-AUC", color="coral")

    ax.set_ylabel("Score")
    ax.set_title("Link Prediction Model Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.05)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_precision_at_k(results, save_path):
    """Compare Precision@K across models."""
    models = list(results.keys())
    k_values = [100, 1000, 5000]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, k in enumerate(k_values):
        ax = axes[i]
        key = f"precision_at_{k}"
        vals = [results[m].get(key, 0) for m in models]
        ax.barh(models, vals, color="teal", edgecolor="black")
        ax.set_xlabel(f"Precision@{k}")
        ax.set_title(f"Precision@{k}")
        ax.set_xlim(0, max(vals) * 1.2 if max(vals) > 0 else 1)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_training_curves(history, save_path):
    """Plot GNN training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    axes[0].plot(epochs, history["train_loss"], "b-o", markersize=3)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[1].plot(epochs, history["val_auc"], "g-o", markersize=3)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("ROC-AUC")
    axes[1].set_title("Validation ROC-AUC")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_degree_distribution(edges_df, save_path):
    """Plot in/out degree distribution."""
    out_deg = edges_df.groupby("source").size().values
    in_deg = edges_df.groupby("target").size().values

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(out_deg, bins=50, edgecolor="black", alpha=0.7, color="coral")
    axes[0].set_xlabel("Out-Degree")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Out-Degree Distribution")
    axes[0].set_yscale("log")

    axes[1].hist(in_deg, bins=50, edgecolor="black", alpha=0.7, color="steelblue")
    axes[1].set_xlabel("In-Degree")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("In-Degree Distribution")
    axes[1].set_yscale("log")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def generate_all_figures():
    """Generate all link prediction figures."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    with open(REPORTS_DIR / "link_prediction_results.json") as f:
        results = json.load(f)

    print("Generating figures...")
    plot_model_comparison(results, FIGURES_DIR / "model_comparison.png")
    plot_precision_at_k(results, FIGURES_DIR / "precision_at_k.png")

    edges = pd.read_parquet(LP_DIR / "edges_aggregated.parquet")
    plot_degree_distribution(edges, FIGURES_DIR / "degree_distribution.png")

    if (REPORTS_DIR / "gnn_training_history.json").exists():
        with open(REPORTS_DIR / "gnn_training_history.json") as f:
            history = json.load(f)
        plot_training_curves(history, FIGURES_DIR / "gnn_training_curves.png")

    print("All figures generated.")


if __name__ == "__main__":
    generate_all_figures()
