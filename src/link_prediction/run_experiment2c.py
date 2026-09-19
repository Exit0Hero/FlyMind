#!/usr/bin/env python3
"""Experiment 2C: Memory-safe main runner."""

import gc
import os
import sys
import json
import time
import resource
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.link_prediction.config import LP_DIR, REPORTS_DIR, MODELS_DIR, SEED
from src.link_prediction.evaluate import compute_all_metrics
from src.link_prediction.experiment2c_models import (
    GraphSAGEEncoder, FeatureOnlyEncoder, AsymmetricDecoder,
    train_encoder_decoder, evaluate_model, train_mlp_model,
    directionality_test, embedding_diagnostics, make_pair_indices,
    log_mem, check_mem_limit,
)
from src.link_prediction.experiment2c_data import (
    load_random_split, load_cold_start_split,
)

BATCH_SIZE = 4096
HIDDEN_DIM = 32
NUM_LAYERS = 2
DROPOUT = 0.3
EPOCHS = 50


def _rss_mb():
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * 4096 / (1024 * 1024)
    except Exception:
        return 0.0


def make_pyg_data(X, edges, id_to_idx):
    from torch_geometric.data import Data
    src = np.array([id_to_idx[int(s)] for s in edges[:, 0]], dtype=np.int64)
    dst = np.array([id_to_idx[int(d)] for d in edges[:, 1]], dtype=np.int64)
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    x = torch.tensor(X, dtype=torch.float32)
    return Data(x=x, edge_index=edge_index)


def ids_to_idx_array(ids, id_to_idx):
    return np.array([id_to_idx[int(i)] for i in ids], dtype=np.int64)


def run_protocol(X, node_ids, id_to_idx, splits, protocol_name,
                 rf_results=None, seed=42):
    print("\n" + "=" * 70)
    print(f"EXPERIMENT 2C: {protocol_name}")
    print("=" * 70)
    log_mem("protocol start")

    train_pos, train_neg = splits["train"]
    val_pos, val_neg = splits["val"]
    test_pos, test_neg = splits["test"]
    n_feat = X.shape[1]
    results = {}

    # -- Model A: RF (reuse from 2B if available) --
    if rf_results and "roc_auc" in rf_results:
        print("\n--- Model A: RF (reusing Experiment 2B results) ---")
        results["A_rf"] = rf_results
        print(f"  ROC-AUC: {rf_results['roc_auc']:.4f} | PR-AUC: {rf_results['pr_auc']:.4f}")
    else:
        print("\n--- Model A: RF (node-only) ---")
        from src.link_prediction.features import build_pair_features
        from src.link_prediction.models import train_rf
        tr_pairs = np.vstack([train_pos, train_neg])
        tr_labels = np.concatenate([
            np.ones(len(train_pos), dtype=np.float32),
            np.zeros(len(train_neg), dtype=np.float32),
        ])
        te_pairs = np.vstack([test_pos, test_neg])
        te_labels = np.concatenate([
            np.ones(len(test_pos), dtype=np.float32),
            np.zeros(len(test_neg), dtype=np.float32),
        ])
        X_tr = build_pair_features(X, tr_pairs, id_to_idx)
        X_te = build_pair_features(X, te_pairs, id_to_idx)
        t0 = time.time()
        rf = train_rf(X_tr, tr_labels, n_estimators=100)
        rf_scores = rf.predict_proba(X_te)[:, 1]
        rf_time = time.time() - t0
        rf_metrics = compute_all_metrics(te_labels, rf_scores, "RF")
        rf_metrics["train_time"] = rf_time
        results["A_rf"] = rf_metrics
        del X_tr, X_te, rf, rf_scores, tr_pairs, te_pairs, tr_labels, te_labels
        gc.collect()
        log_mem("after RF")

    # -- Model B: MLP (streaming, no full pair-feature matrix) --
    print("\n--- Model B: MLP (streaming pair features) ---")
    check_mem_limit("before MLP")
    mlp_metrics, _, mlp_model = train_mlp_model(
        X, id_to_idx, train_pos, train_neg, val_pos, val_neg,
        test_pos, test_neg, n_feat, epochs=EPOCHS, lr=0.001, label="MLP",
    )
    results["B_mlp"] = mlp_metrics
    del mlp_model
    gc.collect()
    log_mem("after MLP")

    # -- Build PyG data for GNN models --
    print("\n--- Building PyG Data ---")
    all_edges = np.vstack([train_pos, val_pos, test_pos])
    data = make_pyg_data(X, all_edges, id_to_idx)
    del all_edges
    gc.collect()
    log_mem("pyg data")

    train_s = torch.tensor(ids_to_idx_array(train_pos[:, 0], id_to_idx), dtype=torch.long)
    train_t = torch.tensor(ids_to_idx_array(train_pos[:, 1], id_to_idx), dtype=torch.long)
    train_ns = torch.tensor(ids_to_idx_array(train_neg[:, 0], id_to_idx), dtype=torch.long)
    train_nt = torch.tensor(ids_to_idx_array(train_neg[:, 1], id_to_idx), dtype=torch.long)
    val_s = torch.tensor(ids_to_idx_array(val_pos[:, 0], id_to_idx), dtype=torch.long)
    val_t = torch.tensor(ids_to_idx_array(val_pos[:, 1], id_to_idx), dtype=torch.long)
    val_ns = torch.tensor(ids_to_idx_array(val_neg[:, 0], id_to_idx), dtype=torch.long)
    val_nt = torch.tensor(ids_to_idx_array(val_neg[:, 1], id_to_idx), dtype=torch.long)
    test_s = torch.tensor(ids_to_idx_array(test_pos[:, 0], id_to_idx), dtype=torch.long)
    test_t = torch.tensor(ids_to_idx_array(test_pos[:, 1], id_to_idx), dtype=torch.long)
    test_ns = torch.tensor(ids_to_idx_array(test_neg[:, 0], id_to_idx), dtype=torch.long)
    test_nt = torch.tensor(ids_to_idx_array(test_neg[:, 1], id_to_idx), dtype=torch.long)
    log_mem("index tensors")

    # -- Model C: GraphSAGE --
    print("\n--- Model C: GraphSAGE (features + graph) ---")
    check_mem_limit("before GraphSAGE")
    enc_c = GraphSAGEEncoder(n_feat, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
    dec_c = AsymmetricDecoder(HIDDEN_DIM)
    enc_c, dec_c, sage_hist = train_encoder_decoder(
        enc_c, dec_c, data, train_s, train_t, train_ns, train_nt,
        val_s, val_t, val_ns, val_nt, epochs=EPOCHS, lr=0.001, label="GraphSAGE",
    )
    sage_metrics, _, _ = evaluate_model(
        enc_c, dec_c, data, test_s, test_t, test_ns, test_nt, "GraphSAGE"
    )
    sage_metrics["train_time"] = sage_hist.get("train_time", 0)
    sage_metrics["directionality"] = directionality_test(
        enc_c, dec_c, data, test_pos, id_to_idx
    )
    sage_metrics["embedding"] = embedding_diagnostics(enc_c, data)
    results["C_graphsage"] = sage_metrics
    del enc_c, dec_c, sage_hist
    gc.collect()
    log_mem("after GraphSAGE")

    # -- Model D: FeatureOnly --
    print("\n--- Model D: FeatureOnly (no graph edges) ---")
    check_mem_limit("before FeatureOnly")
    enc_d = FeatureOnlyEncoder(n_feat, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
    dec_d = AsymmetricDecoder(HIDDEN_DIM)
    enc_d, dec_d, fo_hist = train_encoder_decoder(
        enc_d, dec_d, data, train_s, train_t, train_ns, train_nt,
        val_s, val_t, val_ns, val_nt, epochs=EPOCHS, lr=0.001, label="FeatureOnly",
    )
    fo_metrics, _, _ = evaluate_model(
        enc_d, dec_d, data, test_s, test_t, test_ns, test_nt, "FeatureOnly"
    )
    fo_metrics["train_time"] = fo_hist.get("train_time", 0)
    fo_metrics["directionality"] = directionality_test(
        enc_d, dec_d, data, test_pos, id_to_idx
    )
    fo_metrics["embedding"] = embedding_diagnostics(enc_d, data)
    results["D_featureonly"] = fo_metrics
    del enc_d, dec_d, fo_hist
    gc.collect()
    log_mem("after FeatureOnly")

    # Cleanup all tensors
    del train_s, train_t, train_ns, train_nt
    del val_s, val_t, val_ns, val_nt
    del test_s, test_t, test_ns, test_nt
    del data
    gc.collect()

    # Summary
    print(f"\n--- {protocol_name} SUMMARY ---")
    for name in ["A_rf", "B_mlp", "C_graphsage", "D_featureonly"]:
        m = results[name]
        print(f"  {name:15s} | ROC-AUC: {m['roc_auc']:.4f} | PR-AUC: {m['pr_auc']:.4f}")

    c_pr = results["C_graphsage"]["pr_auc"]
    d_pr = results["D_featureonly"]["pr_auc"]
    delta = c_pr - d_pr
    pct = (delta / d_pr * 100) if d_pr > 0 else 0
    print(f"\n  Graph contribution (C-D): {delta:+.4f} ({pct:+.1f}%)")

    return results


def generate_report(all_results):
    lines = [
        "# Experiment 2C: Node-Only Baseline vs GraphSAGE\n",
        "## Setup\n",
        "**Four models:** A-RF, B-MLP, C-GraphSAGE (32d, 2 layers), D-FeatureOnly\n",
        "**Question:** Does graph message passing improve beyond features alone?\n",
    ]

    for proto, key in [("Random Edge Holdout", "random"),
                       ("Cold-Start Node Holdout", "cold_start")]:
        lines.append(f"## {proto}\n")
        lines.append("| Model | ROC-AUC | PR-AUC | Time |")
        lines.append("|-------|---------|--------|------|")
        data = all_results.get(key, {})
        for mk, mn in [("A_rf", "A-RF"), ("B_mlp", "B-MLP"),
                       ("C_graphsage", "C-GraphSAGE"), ("D_featureonly", "D-FeatureOnly")]:
            m = data.get(mk, {})
            lines.append(
                f"| {mn} | {m.get('roc_auc',0):.4f} | "
                f"{m.get('pr_auc',0):.4f} | {m.get('train_time',0):.1f}s |"
            )

        c = data.get("C_graphsage", {})
        d = data.get("D_featureonly", {})
        if c and d:
            delta = c["pr_auc"] - d["pr_auc"]
            pct = (delta / d["pr_auc"] * 100) if d["pr_auc"] > 0 else 0
            lines.append(f"\n**Graph contribution (C-D):** {delta:+.4f} ({pct:+.1f}%)\n")

            dir_info = c.get("directionality", {})
            if dir_info:
                lines.append("### Directionality (Model C)")
                lines.append(
                    f"- Forward: {dir_info.get('forward_mean',0):.4f}, "
                    f"Reverse: {dir_info.get('reverse_mean',0):.4f}"
                )
                lines.append(
                    f"- |fwd-rev|: {dir_info.get('mean_abs_diff',0):.4f}, "
                    f"Corr: {dir_info.get('correlation',0):.4f}\n"
                )

            emb_c = c.get("embedding", {})
            emb_d = d.get("embedding", {})
            if emb_c:
                lines.append("### Embedding Diagnostics")
                lines.append("| Metric | GraphSAGE | FeatureOnly |")
                lines.append("|--------|-----------|-------------|")
                for k in ["norm_mean", "variance", "cosine_sim_mean"]:
                    lines.append(
                        f"| {k} | {emb_c.get(k,0):.4f} | {emb_d.get(k,0):.4f} |"
                    )
                lines.append("")

    return "\n".join(lines)


def smoke_test():
    """Quick smoke test with 10K pairs to verify all models work."""
    print("=" * 70)
    print("SMOKE TEST (10K pairs)")
    print("=" * 70)
    data_dir = str(LP_DIR)
    X = np.load(os.path.join(data_dir, "X_features.npy"))
    with open(os.path.join(data_dir, "id_to_idx.json")) as f:
        id_to_idx = {int(k): v for k, v in json.load(f).items()}
    node_ids = np.array(sorted(id_to_idx.keys()), dtype=np.int64)
    n_feat = X.shape[1]

    train_pos = np.load(os.path.join(data_dir, "split_train.npy"))[:5000]
    train_neg = np.load(os.path.join(data_dir, "train_neg.npy"))[:5000]
    test_pos = np.load(os.path.join(data_dir, "split_test.npy"))[:5000]
    test_neg = np.load(os.path.join(data_dir, "test_neg.npy"))[:5000]
    val_pos = train_pos[:1000]
    val_neg = train_neg[:1000]

    print("\n--- Smoke: MLP ---")
    from src.link_prediction.experiment2c_models import (
        MLPNodeClassifier, batch_pair_features, log_mem as _lm
    )
    pair_dim = n_feat * 4
    mlp = MLPNodeClassifier(pair_dim)
    buf = np.empty((100, pair_dim), dtype=np.float32)
    src_idx_100 = np.array([id_to_idx[int(i)] for i in train_pos[:100, 0]], dtype=np.int64)
    dst_idx_100 = np.array([id_to_idx[int(i)] for i in train_pos[:100, 1]], dtype=np.int64)
    batch_pair_features(X, src_idx_100, dst_idx_100, buf)
    assert buf.shape == (100, pair_dim), f"Bad shape: {buf.shape}"
    assert not np.isnan(buf).any(), "NaN in pair features"
    xb = torch.from_numpy(buf)
    out = mlp(xb)
    assert out.shape == (100,), f"Bad output shape: {out.shape}"
    assert not torch.isnan(out).any(), "NaN in MLP output"
    del mlp, buf, xb, out
    gc.collect()
    print("  PASS: MLP forward pass OK")

    print("\n--- Smoke: GraphSAGE ---")
    from src.link_prediction.experiment2c_models import (
        GraphSAGEEncoder, AsymmetricDecoder, train_encoder_decoder, evaluate_model
    )
    all_edges = np.vstack([train_pos, test_pos])
    data = make_pyg_data(X, all_edges, id_to_idx)
    src_idx = np.array([id_to_idx[int(s)] for s in train_pos[:, 0]], dtype=np.int64)
    tgt_idx = np.array([id_to_idx[int(s)] for s in train_pos[:, 1]], dtype=np.int64)
    ns_idx = np.array([id_to_idx[int(s)] for s in train_neg[:, 0]], dtype=np.int64)
    nt_idx = np.array([id_to_idx[int(s)] for s in train_neg[:, 1]], dtype=np.int64)

    enc = GraphSAGEEncoder(n_feat, 32, 2, 0.3)
    dec = AsymmetricDecoder(32)
    ts = torch.tensor(src_idx, dtype=torch.long)
    tt = torch.tensor(tgt_idx, dtype=torch.long)
    tns = torch.tensor(ns_idx, dtype=torch.long)
    tnt = torch.tensor(nt_idx, dtype=torch.long)
    enc, dec, hist = train_encoder_decoder(
        enc, dec, data, ts, tt, tns, tnt, ts[:1000], tt[:1000], tns[:1000], tnt[:1000],
        epochs=3, lr=0.001, label="Smoke-GNN"
    )
    metrics, _, _ = evaluate_model(enc, dec, data, ts, tt, tns, tnt, "Smoke")
    assert metrics["roc_auc"] > 0, f"Bad AUC: {metrics['roc_auc']}"
    del enc, dec, data, ts, tt, tns, tnt, hist, metrics
    gc.collect()
    print("  PASS: GraphSAGE train+eval OK")

    print("\n--- Smoke: FeatureOnly ---")
    all_edges2 = np.vstack([train_pos, test_pos])
    data2 = make_pyg_data(X, all_edges2, id_to_idx)
    enc2 = FeatureOnlyEncoder(n_feat, 32, 2, 0.3)
    dec2 = AsymmetricDecoder(32)
    ts2 = torch.tensor(src_idx, dtype=torch.long)
    tt2 = torch.tensor(tgt_idx, dtype=torch.long)
    tns2 = torch.tensor(ns_idx, dtype=torch.long)
    tnt2 = torch.tensor(nt_idx, dtype=torch.long)
    enc2, dec2, hist2 = train_encoder_decoder(
        enc2, dec2, data2, ts2, tt2, tns2, tnt2, ts2[:1000], tt2[:1000], tns2[:1000], tnt2[:1000],
        epochs=3, lr=0.001, label="Smoke-FO"
    )
    metrics2, _, _ = evaluate_model(enc2, dec2, data2, ts2, tt2, tns2, tnt2, "Smoke")
    assert metrics2["roc_auc"] > 0, f"Bad AUC: {metrics2['roc_auc']}"
    del enc2, dec2, data2, ts2, tt2, tns2, tnt2, hist2, metrics2
    gc.collect()
    print("  PASS: FeatureOnly train+eval OK")

    print("\nSMOKE TEST PASSED")
    return True


def main():
    print("=" * 70)
    print("EXPERIMENT 2C: Node-Only Baseline vs GraphSAGE")
    print("=" * 70)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    log_mem("program start")

    # Smoke test first
    smoke_test()
    log_mem("after smoke test")

    # Load existing RF results
    rf_results = None
    rf_path = REPORTS_DIR / "link_prediction_results.json"
    if rf_path.exists():
        with open(rf_path) as f:
            data = json.load(f)
        rf_node = data.get("rf_node_only", {})
        if "roc_auc" in rf_node:
            rf_results = rf_node
            print(f"\nReusing RF results: ROC-AUC={rf_results['roc_auc']:.4f}")

    cold_rf_results = None
    cold_rf_path = REPORTS_DIR / "link_prediction_cold_start_results.json"
    if cold_rf_path.exists():
        with open(cold_rf_path) as f:
            data = json.load(f)
        cold_rf = data.get("rf_node_only", {})
        if "roc_auc" in cold_rf:
            cold_rf_results = cold_rf
            print(f"Reusing cold-start RF: ROC-AUC={cold_rf_results['roc_auc']:.4f}")

    # Load features once
    data_dir = str(LP_DIR)
    cold_dir = str(LP_DIR / "cold_start")
    X, node_ids, id_to_idx, n_feat, random_splits = load_random_split(data_dir)
    log_mem("after loading features")

    all_results = {}

    # COLD-START FIRST (scientifically important)
    print("\n[1] Loading cold-start split...")
    cold_splits = load_cold_start_split(cold_dir, node_ids)
    cold_results = run_protocol(
        X, node_ids, id_to_idx, cold_splits,
        "COLD-START NODE HOLDOUT",
        rf_results=cold_rf_results,
    )
    all_results["cold_start"] = cold_results
    del cold_splits
    gc.collect()
    log_mem("after cold-start")

    # RANDOM EDGE HOLDOUT
    print("\n[2] Running random edge holdout...")
    random_results = run_protocol(
        X, node_ids, id_to_idx, random_splits,
        "RANDOM EDGE HOLDOUT",
        rf_results=rf_results,
    )
    all_results["random"] = random_results
    del random_splits
    gc.collect()
    log_mem("after random")

    # Save
    report = generate_report(all_results)
    report_path = REPORTS_DIR / "experiment2c_results.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved: {report_path}")

    json_path = REPORTS_DIR / "experiment2c_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"JSON saved: {json_path}")

    log_mem("final")
    print("\nDone!")


if __name__ == "__main__":
    main()
