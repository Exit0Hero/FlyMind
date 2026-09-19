#!/usr/bin/env python3
"""Main ML pipeline: load, preprocess, baseline, GNN, evaluate, report."""

import sys
import time
import json
import pathlib
import numpy as np
import pandas as pd
import torch

# Setup paths
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.config import (
    SEED, TARGET_COL, ALL_FEATURES, SAFE_FEATURES_NUMERIC,
    SAFE_FEATURES_CATEGORICAL, PROCESSED_DIR, MODELS_DIR,
    RESULTS_DIR, REPORTS_DIR, FIGURES_DIR,
)
from src.utils import set_seed, get_device
from src.data_loader import load_neuron_table, load_and_aggregate_edges, get_edge_stats
from src.preprocessing import Preprocessor, create_split
from src.graph_builder import build_graph, compute_graph_features
from src.baselines import run_baselines, run_graph_feature_baselines
from src.train_gnn import train_graphsage
from src.evaluate import evaluate_predictions, save_results
from src.explain import analyze_neuron

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from collections import Counter


def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, values_format="d")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_class_distribution(y, class_names, title, save_path):
    counts = Counter(y)
    labels = [class_names[i] for i in sorted(counts.keys())]
    values = [counts[i] for i in sorted(counts.keys())]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(labels, values, color="steelblue", edgecolor="black")
    ax.set_xlabel("Count")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_training_curves(history, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    axes[0].plot(epochs, history["train_loss"], "b-o", markersize=3)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[1].plot(epochs, history["val_acc"], "g-o", markersize=3)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Validation Accuracy")
    axes[2].plot(epochs, history["val_f1"], "r-o", markersize=3)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Macro F1")
    axes[2].set_title("Validation Macro F1")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def main():
    set_seed(SEED)
    device = get_device()
    print("=" * 70)
    print("FLYMIND ML PIPELINE")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Seed: {SEED}")

    # =========================================================================
    # 1. Load data
    # =========================================================================
    print("\n[1] Loading data...")
    df = load_neuron_table()
    edges = load_and_aggregate_edges()
    edge_stats = get_edge_stats(edges)
    print(f"  Nodes: {len(df)}")
    print(f"  Edges (aggregated): {len(edges)}")
    print(f"  Edge stats: {json.dumps(edge_stats, indent=2)}")

    # =========================================================================
    # 2. Split data
    # =========================================================================
    print("\n[2] Creating train/val/test split...")
    idx_train, idx_val, idx_test = create_split(df, seed=SEED)
    print(f"  Train: {len(idx_train)} ({100*len(idx_train)/len(df):.1f}%)")
    print(f"  Val:   {len(idx_val)} ({100*len(idx_val)/len(df):.1f}%)")
    print(f"  Test:  {len(idx_test)} ({100*len(idx_test)/len(df):.1f}%)")

    # =========================================================================
    # 3. Class distribution
    # =========================================================================
    print("\n[3] Target distribution (super_class)...")
    target_classes = sorted(df[TARGET_COL].unique())
    num_classes = len(target_classes)
    print(f"  Classes: {num_classes}")
    for cls in target_classes:
        count = (df[TARGET_COL] == cls).sum()
        print(f"    {cls:<25} {count:>8} ({100*count/len(df):.1f}%)")

    # Class weights for imbalanced data
    train_targets = df[TARGET_COL].values[idx_train]
    class_counts = np.array([np.sum(train_targets == c) for c in range(num_classes)])
    class_weights = 1.0 / (class_counts + 1)
    class_weights = class_weights / class_weights.sum() * num_classes
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    # =========================================================================
    # 4. Preprocess features
    # =========================================================================
    print("\n[4] Preprocessing features...")
    train_mask = np.zeros(len(df), dtype=bool)
    train_mask[idx_train] = True

    pp = Preprocessor()
    X, y = pp.fit_transform(df, train_mask)
    print(f"  Feature matrix: {X.shape}")
    print(f"  Features: {pp.feature_names}")

    # Save preprocessor
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    pp.save(MODELS_DIR / "preprocessor_config.json")
    print(f"  Saved preprocessor config to {MODELS_DIR / 'preprocessor_config.json'}")

    # =========================================================================
    # 5. Baselines (biological features only)
    # =========================================================================
    print("\n[5] Running baselines (biological features only)...")
    X_train = X[idx_train]
    X_val = X[idx_val]
    X_test = X[idx_test]
    y_train = y[idx_train]
    y_val = y[idx_val]
    y_test = y[idx_test]

    baseline_results = run_baselines(
        X_train, y_train, X_val, y_val, X_test, y_test, target_classes,
    )
    for name, metrics in baseline_results.items():
        print(f"  {name:<25} Acc: {metrics['accuracy']:.4f}  Macro F1: {metrics['macro_f1']:.4f}")

    # =========================================================================
    # 6. Graph features
    # =========================================================================
    print("\n[6] Computing graph features...")
    node_ids = df["root_id"].values
    graph_feats = compute_graph_features(edges, node_ids)
    print(f"  Graph features: {graph_feats.shape}")

    graph_feat_names = [
        "in_degree", "out_degree", "total_degree",
        "weighted_in_degree", "weighted_out_degree",
        "num_upstream", "num_downstream",
    ]

    X_graph_train = graph_feats[idx_train]
    X_graph_test = graph_feats[idx_test]

    graph_baseline_results = run_graph_feature_baselines(
        X_train, X_graph_train, y_train,
        X_test, X_graph_test, y_test,
    )
    for name, metrics in graph_baseline_results.items():
        print(f"  {name:<25} Acc: {metrics['accuracy']:.4f}  Macro F1: {metrics['macro_f1']:.4f}")

    # =========================================================================
    # 7. Build PyG graph
    # =========================================================================
    print("\n[7] Building PyG graph...")
    train_mask_arr = np.zeros(len(df), dtype=bool)
    val_mask_arr = np.zeros(len(df), dtype=bool)
    test_mask_arr = np.zeros(len(df), dtype=bool)
    train_mask_arr[idx_train] = True
    val_mask_arr[idx_val] = True
    test_mask_arr[idx_test] = True

    edges_arr = edges[["source", "target"]].values
    edge_weights = edges["weight"].values

    data = build_graph(
        X, y, edges_arr, node_ids, edge_weights,
        train_mask_arr, val_mask_arr, test_mask_arr, add_reverse=True,
    )
    data = data.to(device)
    print(f"  Nodes: {data.num_nodes}")
    print(f"  Edges: {data.edge_index.shape[1]}")
    print(f"  Features: {data.x.shape[1]}")

    # =========================================================================
    # 8. Train GraphSAGE
    # =========================================================================
    print("\n[8] Training GraphSAGE...")
    gnn_model, history, train_time, best_epoch, best_val_f1 = train_graphsage(
        data, num_classes,
        data.train_mask, data.val_mask, data.test_mask,
        device,
        class_weights=class_weights_tensor,
        epochs=50,
        lr=0.005,
        hidden_dim=64,
        num_layers=2,
        dropout=0.3,
        batch_size=1024,
        neighbor_sizes=(15, 10),
    )

    # =========================================================================
    # 9. Evaluate GNN on test set
    # =========================================================================
    print("\n[9] Evaluating GNN on test set...")
    from src.train_gnn import evaluate_full
    gnn_preds, gnn_labels = evaluate_full(gnn_model, data, device, data.test_mask)

    gnn_results = evaluate_predictions(gnn_labels, gnn_preds, target_classes)
    print(f"  Accuracy:   {gnn_results['accuracy']:.4f}")
    print(f"  Macro F1:   {gnn_results['macro_f1']:.4f}")
    print(f"  Weighted F1: {gnn_results['weighted_f1']:.4f}")
    print(f"\n{gnn_results['report_str']}")

    # =========================================================================
    # 10. Comparison table
    # =========================================================================
    print("\n[10] Ablation comparison...")
    all_results = {}
    all_results.update(baseline_results)
    all_results.update(graph_baseline_results)
    all_results["graphsage"] = {
        "accuracy": gnn_results["accuracy"],
        "macro_f1": gnn_results["macro_f1"],
        "weighted_f1": gnn_results["weighted_f1"],
    }

    print(f"\n{'Model':<30} {'Features':<25} {'Graph':<6} {'Accuracy':>8} {'Macro F1':>8} {'Wt F1':>8}")
    print("-" * 90)
    model_info = {
        "majority_class": ("Baseline (all)", "No"),
        "logistic_regression": ("Bio features", "No"),
        "random_forest": ("Bio features", "No"),
        "rf_graph_only": ("Graph features", "No"),
        "rf_bio_only": ("Bio features", "No"),
        "rf_bio_plus_graph": ("Bio+Graph", "No"),
        "graphsage": ("Node features", "Yes"),
    }
    for name, metrics in all_results.items():
        feat, graph = model_info.get(name, ("?", "?"))
        print(f"  {name:<28} {feat:<25} {graph:<6} {metrics['accuracy']:>8.4f} {metrics['macro_f1']:>8.4f} {metrics['weighted_f1']:>8.4f}")

    # =========================================================================
    # 11. Local explanation for one test neuron
    # =========================================================================
    print("\n[11] Local neighborhood analysis...")
    id_to_idx = {int(rid): i for i, rid in enumerate(node_ids)}
    test_indices = idx_test[:5]  # First 5 test neurons
    explanations = []
    for tidx in test_indices:
        report = analyze_neuron(
            gnn_model, data, tidx, node_ids, df,
            id_to_idx, target_classes, device,
        )
        explanations.append(report)
        print(f"\n{report}\n")

    # Save explanation
    with open(REPORTS_DIR / "local_explanations.txt", "w") as f:
        f.write("\n\n---\n\n".join(explanations))

    # =========================================================================
    # 12. Generate plots
    # =========================================================================
    print("\n[12] Generating plots...")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Class distribution
    plot_class_distribution(
        y, target_classes,
        "Target Class Distribution (super_class)",
        FIGURES_DIR / "class_distribution.png",
    )

    # Training curves
    plot_training_curves(history, FIGURES_DIR / "training_curves.png")

    # Confusion matrix
    plot_confusion_matrix(
        gnn_labels, gnn_preds, target_classes,
        "GraphSAGE Confusion Matrix (Test Set)",
        FIGURES_DIR / "confusion_matrix_gnn.png",
    )

    # RF confusion matrix
    rf = run_baselines.__wrapped__ if hasattr(run_baselines, "__wrapped__") else None

    # =========================================================================
    # 13. Save all results
    # =========================================================================
    print("\n[13] Saving results...")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    save_results({
        "all_results": all_results,
        "edge_stats": edge_stats,
        "class_distribution": {c: int((df[TARGET_COL] == c).sum()) for c in target_classes},
        "train_size": len(idx_train),
        "val_size": len(idx_val),
        "test_size": len(idx_test),
        "num_classes": num_classes,
        "num_features": X.shape[1],
        "feature_names": pp.feature_names,
        "graph_nodes": data.num_nodes,
        "graph_edges": int(data.edge_index.shape[1]),
        "gnn_results": gnn_results,
        "train_time_seconds": train_time,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
    }, REPORTS_DIR / "ml_baseline_results.json")

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
