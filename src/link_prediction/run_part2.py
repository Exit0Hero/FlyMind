#!/usr/bin/env python3
"""FlyMind Link Prediction — Part 2: GNN (mini-batch, memory-safe)."""

import gc
import sys
import json
import time
import pathlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import pickle

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

from src.link_prediction.config import *
from src.link_prediction.data_prep import log_mem
from src.link_prediction.features import build_pair_features, build_edge_dicts, compute_heuristics
from src.link_prediction.evaluate import compute_all_metrics
from torch_geometric.data import Data


# ---------------------------------------------------------------------------
# Models (inline to avoid import issues)
# ---------------------------------------------------------------------------
class GraphSAGEEncoder(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        from torch_geometric.nn import SAGEConv
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class LinkPredictor(torch.nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embed_dim * 4, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )

    def forward(self, z_a, z_b):
        combined = torch.cat([z_a, z_b, z_a * z_b, torch.abs(z_a - z_b)], dim=-1)
        return self.mlp(combined).squeeze(-1)


def train_rf(X_train, y_train, n_estimators=100):
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=None,
        random_state=SEED, n_jobs=-1, class_weight="balanced",
    )
    rf.fit(X_train, y_train)
    return rf


# ---------------------------------------------------------------------------
# GNN training with NeighborLoader (mini-batch)
# ---------------------------------------------------------------------------
def run_gnn():
    print("=" * 70)
    print("FLYMIND LINK PREDICTION — GNN (mini-batch)")
    print("=" * 70)

    log_mem("start")

    # Load saved intermediates
    print("\n[1] Loading saved data...")
    X = np.load(LP_DIR / "X_features.npy")
    train_neg = np.load(LP_DIR / "train_neg.npy")
    val_neg = np.load(LP_DIR / "val_neg.npy")
    test_neg = np.load(LP_DIR / "test_neg.npy")
    train_edges = np.load(LP_DIR / "split_train.npy")
    val_edges = np.load(LP_DIR / "split_val.npy")
    test_edges = np.load(LP_DIR / "split_test.npy")

    with open(LP_DIR / "id_to_idx.json") as f:
        id_to_idx = {int(k): v for k, v in json.load(f).items()}

    n_nodes = X.shape[0]
    n_feat = X.shape[1]
    print(f"  Nodes: {n_nodes}, Features: {n_feat}, Train edges: {len(train_edges)}")
    log_mem("after load")

    # Build PyG graph from training edges (with reverse edges for undirected message passing)
    print("\n[2] Building PyG graph...")
    src_idx = np.array([id_to_idx[int(s)] for s in train_edges[:, 0]], dtype=np.int64)
    tgt_idx = np.array([id_to_idx[int(t)] for t in train_edges[:, 1]], dtype=np.int64)
    ei = np.stack([src_idx, tgt_idx])
    rev = np.stack([tgt_idx, src_idx])
    edge_index = np.concatenate([ei, rev], axis=1)

    data = Data(
        x=torch.tensor(X, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
    )
    del ei, rev, edge_index, X; gc.collect()
    log_mem("after pyg data")

    # Build training pair index tensors
    tr_pos_src = torch.tensor([id_to_idx[int(s)] for s in train_edges[:, 0]], dtype=torch.long)
    tr_pos_tgt = torch.tensor([id_to_idx[int(t)] for t in train_edges[:, 1]], dtype=torch.long)
    tr_neg_src = torch.tensor([id_to_idx[int(s)] for s in train_neg[:, 0]], dtype=torch.long)
    tr_neg_tgt = torch.tensor([id_to_idx[int(t)] for t in train_neg[:, 1]], dtype=torch.long)
    val_src = torch.tensor([id_to_idx[int(s)] for s in val_edges[:, 0]], dtype=torch.long)
    val_tgt = torch.tensor([id_to_idx[int(t)] for t in val_edges[:, 1]], dtype=torch.long)
    val_neg_src = torch.tensor([id_to_idx[int(s)] for s in val_neg[:, 0]], dtype=torch.long)
    val_neg_tgt = torch.tensor([id_to_idx[int(t)] for t in val_neg[:, 1]], dtype=torch.long)

    del train_neg, val_neg; gc.collect()
    log_mem("after pair tensors")

    # Initialize models
    encoder = GraphSAGEEncoder(n_feat, GNN_HIDDEN_DIM, GNN_NUM_LAYERS, GNN_DROPOUT)
    decoder = LinkPredictor(GNN_HIDDEN_DIM)

    # Xavier init for stability
    for p in encoder.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)
    for p in decoder.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)

    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=0.0005, weight_decay=1e-5)

    best_val_auc = 0
    best_state = None
    history = {"train_loss": [], "val_auc": []}

    BATCH_PAIRS = 8192
    print(f"\n[4] Training GraphSAGE (hidden={GNN_HIDDEN_DIM}, layers={GNN_NUM_LAYERS}, epochs={GNN_EPOCHS}, batch_pairs={BATCH_PAIRS})...")
    t0 = time.time()

    for epoch in range(1, GNN_EPOCHS + 1):
        encoder.train()
        decoder.train()

        optimizer.zero_grad()
        z = encoder(data.x, data.edge_index)

        n_pos = len(tr_pos_src)
        n_neg = len(tr_neg_src)
        batch_idx = torch.randperm(n_pos)[:BATCH_PAIRS]
        neg_batch_idx = torch.randperm(n_neg)[:BATCH_PAIRS]

        pos_score = decoder(z[tr_pos_src[batch_idx]], z[tr_pos_tgt[batch_idx]])
        neg_score = decoder(z[tr_neg_src[neg_batch_idx]], z[tr_neg_tgt[neg_batch_idx]])

        labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
        scores = torch.cat([pos_score, neg_score])
        scores = torch.nan_to_num(scores, nan=0.0)
        loss = F.binary_cross_entropy_with_logits(scores, labels)

        if torch.isnan(loss) or loss.item() > 100:
            optimizer.zero_grad()
            history["train_loss"].append(float("nan"))
            history["val_auc"].append(0.5)
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()

        # Validate
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            z_val = encoder(data.x, data.edge_index)
            vps = decoder(z_val[val_src], z_val[val_tgt]).cpu().numpy()
            vns = decoder(z_val[val_neg_src], z_val[val_neg_tgt]).cpu().numpy()

        from sklearn.metrics import roc_auc_score
        val_preds = np.concatenate([vps, vns])
        val_preds = np.nan_to_num(val_preds, nan=0.0)
        val_labels_arr = np.concatenate([np.ones(len(vps)), np.zeros(len(vns))])
        val_auc = roc_auc_score(val_labels_arr, val_preds)

        history["train_loss"].append(float(loss.item()))
        history["val_auc"].append(float(val_auc))

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {
                "encoder": {k: v.clone() for k, v in encoder.state_dict().items()},
                "decoder": {k: v.clone() for k, v in decoder.state_dict().items()},
            }

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | Loss: {loss.item():.4f} | Val AUC: {val_auc:.4f} | Best: {best_val_auc:.4f}")

    elapsed = time.time() - t0
    print(f"  Training: {elapsed:.1f}s")
    log_mem("after training")

    # Load best and evaluate test
    if best_state:
        encoder.load_state_dict(best_state["encoder"])
        decoder.load_state_dict(best_state["decoder"])

    print("\n[5] Evaluating on test set...")
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        z = encoder(data.x, data.edge_index)

        test_src = torch.tensor([id_to_idx[int(s)] for s in test_edges[:, 0]], dtype=torch.long)
        test_tgt = torch.tensor([id_to_idx[int(t)] for t in test_edges[:, 1]], dtype=torch.long)
        test_neg_src = torch.tensor([id_to_idx[int(s)] for s in test_neg[:, 0]], dtype=torch.long)
        test_neg_tgt = torch.tensor([id_to_idx[int(t)] for t in test_neg[:, 1]], dtype=torch.long)

        pos_scores = decoder(z[test_src], z[test_tgt]).cpu().numpy()
        neg_scores = decoder(z[test_neg_src], z[test_neg_tgt]).cpu().numpy()

    test_scores = np.concatenate([pos_scores, neg_scores])
    test_scores = np.nan_to_num(test_scores, nan=0.0)
    test_labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    gnn_metrics = compute_all_metrics(test_labels, test_scores, "GraphSAGE")
    gnn_metrics["train_time_seconds"] = elapsed

    # Save
    torch.save({
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "feature_dim": n_feat,
        "hidden_dim": GNN_HIDDEN_DIM,
        "num_layers": GNN_NUM_LAYERS,
    }, MODELS_DIR / "link_prediction_gnn.pt")

    with open(REPORTS_DIR / "link_prediction_results.json", "r") as f:
        all_results = json.load(f)
    all_results["gnn"] = gnn_metrics
    with open(REPORTS_DIR / "link_prediction_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    with open(REPORTS_DIR / "gnn_training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 70)
    print("GNN COMPLETE")
    print("=" * 70)
    log_mem("final")

    return gnn_metrics, history


if __name__ == "__main__":
    run_gnn()
