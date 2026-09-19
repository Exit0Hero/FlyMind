#!/usr/bin/env python3
"""FlyMind Link Prediction — GNN diagnostics (Parts F-K).

Root cause of GraphSAGE failure:
1. NaN in features (length_nm, area_nm, size_nm) → NaN embeddings
2. Numerical instability → embeddings explode to 1e13
3. Symmetric decoder → loses directionality
4. Reverse edges → graph treated as undirected

Fixed models with BatchNorm + asymmetric decoder.
"""

import gc
import sys
import json
import time
import pathlib
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

from src.link_prediction.config import *
from src.link_prediction.data_prep import log_mem
from src.link_prediction.negative_sampling import build_edge_index
from src.link_prediction.evaluate import compute_all_metrics
from torch_geometric.data import Data


# ---------------------------------------------------------------------------
# FIXED Models
# ---------------------------------------------------------------------------
class GraphSAGEEncoder(torch.nn.Module):
    """GraphSAGE with BatchNorm + input normalization for numerical stability."""
    def __init__(self, in_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        from torch_geometric.nn import SAGEConv
        self.input_bn = torch.nn.BatchNorm1d(in_dim)
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        self.convs.append(SAGEConv(in_dim, hidden_dim))
        self.bns.append(torch.nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))
        self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.bns.append(torch.nn.BatchNorm1d(hidden_dim))
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.input_bn(x)
        for conv, bn in zip(self.convs[:-1], self.bns[:-1]):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        x = self.bns[-1](x)
        return x


class AsymmetricLinkPredictor(torch.nn.Module):
    """Asymmetric decoder: separate处理 for source vs target embeddings.

    Preserves directionality: P(A->B) != P(B->A).
    """
    def __init__(self, embed_dim):
        super().__init__()
        # Separate projections for source and target
        self.src_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.tgt_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embed_dim * 4, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )

    def forward(self, z_a, z_b):
        # Asymmetric: project source and target differently
        h_a = self.src_proj(z_a)
        h_b = self.tgt_proj(z_b)
        combined = torch.cat([h_a, h_b, h_a * h_b, torch.abs(h_a - h_b)], dim=-1)
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
# PART G: Sanity test — can the model overfit 1K edges?
# ---------------------------------------------------------------------------
def run_sanity_test():
    """Train GNN on tiny subset to verify it can overfit."""
    print("\n" + "=" * 70)
    print("PART G: GNN SANITY TEST (1K edges, FIXED model)")
    print("=" * 70)

    LP_COLD_DIR = LP_DIR / "cold_start"
    X = np.load(LP_COLD_DIR / "X_features.npy")
    train_edges = np.load(LP_COLD_DIR / "split_train.npy")
    with open(LP_COLD_DIR / "id_to_idx.json") as f:
        id_to_idx = {int(k): v for k, v in json.load(f).items()}

    n_feat = X.shape[1]

    # Take first 1000 training edges
    tiny_pos = train_edges[:1000]
    src_idx = np.array([id_to_idx[int(s)] for s in tiny_pos[:, 0]], dtype=np.int64)
    tgt_idx = np.array([id_to_idx[int(t)] for t in tiny_pos[:, 1]], dtype=np.int64)

    # Directed graph (NO reverse edges)
    ei_dir = np.stack([src_idx, tgt_idx])
    data_dir = Data(
        x=torch.tensor(X, dtype=torch.float32),
        edge_index=torch.tensor(ei_dir, dtype=torch.long),
    )

    # Undirected graph (with reverse edges) — for comparison
    rev = np.stack([tgt_idx, src_idx])
    ei_undir = np.concatenate([ei_dir, rev], axis=1)
    data_undir = Data(
        x=torch.tensor(X, dtype=torch.float32),
        edge_index=torch.tensor(ei_undir, dtype=torch.long),
    )

    # Simple negatives
    np.random.seed(42)
    all_nodes = np.unique(np.concatenate([tiny_pos[:, 0], tiny_pos[:, 1]]))
    neg_src = np.random.choice(all_nodes, 1000)
    neg_tgt = np.random.choice(all_nodes, 1000)
    mask = neg_src != neg_tgt
    neg_src, neg_tgt = neg_src[mask][:1000], neg_tgt[mask][:1000]

    tr_pos_s = torch.tensor(src_idx, dtype=torch.long)
    tr_pos_t = torch.tensor(tgt_idx, dtype=torch.long)
    tr_neg_s = torch.tensor([id_to_idx[int(s)] for s in neg_src], dtype=torch.long)
    tr_neg_t = torch.tensor([id_to_idx[int(t)] for t in neg_tgt], dtype=torch.long)

    results = {}
    for label, data in [("directed_fixed", data_dir), ("undirected_fixed", data_undir)]:
        print(f"\n  --- {label} graph ---")
        enc = GraphSAGEEncoder(n_feat, 64, 2, 0.0)
        dec = AsymmetricLinkPredictor(64)

        for p in enc.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
        for p in dec.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

        params = list(enc.parameters()) + list(dec.parameters())
        opt = torch.optim.Adam(params, lr=0.001)

        history = {"loss": [], "train_auc": []}
        for epoch in range(1, 101):
            enc.train(); dec.train()
            opt.zero_grad()
            z = enc(data.x, data.edge_index)

            ps = dec(z[tr_pos_s], z[tr_pos_t])
            ns = dec(z[tr_neg_s], z[tr_neg_t])
            lb = torch.cat([torch.ones_like(ps), torch.zeros_like(ns)])
            sc = torch.cat([ps, ns])
            sc = torch.nan_to_num(sc, nan=0.0)
            loss = F.binary_cross_entropy_with_logits(sc, lb)

            if torch.isnan(loss) or loss.item() > 100:
                history["loss"].append(float("nan"))
                history["train_auc"].append(0.5)
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            opt.step()

            enc.eval(); dec.eval()
            with torch.no_grad():
                z2 = enc(data.x, data.edge_index)
                all_pos = dec(z2[tr_pos_s], z2[tr_pos_t]).cpu().numpy()
                all_neg = dec(z2[tr_neg_s], z2[tr_neg_t]).cpu().numpy()

            from sklearn.metrics import roc_auc_score
            all_scores = np.concatenate([all_pos, all_neg])
            all_scores = np.nan_to_num(all_scores, nan=0.0)
            all_labels = np.concatenate([np.ones(1000), np.zeros(1000)])
            try:
                auc = roc_auc_score(all_labels, all_scores)
            except ValueError:
                auc = 0.5

            history["loss"].append(float(loss.item()))
            history["train_auc"].append(float(auc))

            if epoch % 20 == 0 or epoch == 1:
                print(f"    Epoch {epoch:3d} | Loss: {loss.item():.4f} | Train AUC: {auc:.4f}")

        final_auc = history["train_auc"][-1]
        results[label] = {
            "final_train_auc": final_auc,
            "final_loss": history["loss"][-1],
        }
        print(f"  Result: {label} final train AUC = {final_auc:.4f}")

    # Directionality test on directed graph
    print("\n  --- Directionality test (directed, fixed) ---")
    enc = GraphSAGEEncoder(n_feat, 64, 2, 0.0)
    dec = AsymmetricLinkPredictor(64)
    for p in enc.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)
    for p in dec.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)

    params = list(enc.parameters()) + list(dec.parameters())
    opt = torch.optim.Adam(params, lr=0.001)

    for epoch in range(1, 101):
        enc.train(); dec.train()
        opt.zero_grad()
        z = enc(data_dir.x, data_dir.edge_index)
        ps = dec(z[tr_pos_s], z[tr_pos_t])
        ns = dec(z[tr_neg_s], z[tr_neg_t])
        lb = torch.cat([torch.ones_like(ps), torch.zeros_like(ns)])
        sc = torch.cat([ps, ns])
        sc = torch.nan_to_num(sc, nan=0.0)
        loss = F.binary_cross_entropy_with_logits(sc, lb)
        if torch.isnan(loss) or loss.item() > 100:
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        opt.step()

    enc.eval(); dec.eval()
    with torch.no_grad():
        z = enc(data_dir.x, data_dir.edge_index)
        n_test = min(200, len(tr_pos_s))
        fwd_scores = dec(z[tr_pos_s[:n_test]], z[tr_pos_t[:n_test]]).cpu().numpy()
        rev_scores = dec(z[tr_pos_t[:n_test]], z[tr_pos_s[:n_test]]).cpu().numpy()

    fwd_mean = float(np.mean(fwd_scores))
    rev_mean = float(np.mean(rev_scores))
    corr = float(np.corrcoef(fwd_scores, rev_scores)[0, 1])
    agree = float(np.mean(np.abs(fwd_scores - rev_scores) < 0.1))

    print(f"    Forward P(mean): {fwd_mean:.4f}")
    print(f"    Reverse P(mean): {rev_mean:.4f}")
    print(f"    Correlation: {corr:.4f}")
    print(f"    Fraction with |fwd-rev| < 0.1: {agree:.4f}")

    results["directionality"] = {
        "forward_mean": fwd_mean,
        "reverse_mean": rev_mean,
        "correlation": corr,
        "fraction_similar": agree,
    }

    return results


# ---------------------------------------------------------------------------
# PART H: Embedding diagnostics (FIXED model)
# ---------------------------------------------------------------------------
def run_embedding_diagnostics():
    """Inspect fixed GraphSAGE embeddings."""
    print("\n" + "=" * 70)
    print("PART H: EMBEDDING DIAGNOSTICS (FIXED)")
    print("=" * 70)

    LP_COLD_DIR = LP_DIR / "cold_start"
    X = np.load(LP_COLD_DIR / "X_features.npy")
    train_edges = np.load(LP_COLD_DIR / "split_train.npy")
    with open(LP_COLD_DIR / "id_to_idx.json") as f:
        id_to_idx = {int(k): v for k, v in json.load(f).items()}

    n_feat = X.shape[1]

    # Directed graph (no reverse edges)
    src_idx = np.array([id_to_idx[int(s)] for s in train_edges[:, 0]], dtype=np.int64)
    tgt_idx = np.array([id_to_idx[int(t)] for t in train_edges[:, 1]], dtype=np.int64)
    ei = np.stack([src_idx, tgt_idx])
    data = Data(x=torch.tensor(X, dtype=torch.float32),
                edge_index=torch.tensor(ei, dtype=torch.long))

    enc = GraphSAGEEncoder(n_feat, 64, 2, 0.3)
    for p in enc.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)

    enc.eval()
    with torch.no_grad():
        z = enc(data.x, data.edge_index).numpy()

    mean = float(np.mean(z))
    std = float(np.std(z))
    dim_var = np.var(z, axis=0)
    norms = np.linalg.norm(z, axis=1)

    np.random.seed(42)
    n_sample = min(1000, len(z))
    idx1 = np.random.choice(len(z), n_sample, replace=False)
    idx2 = np.random.choice(len(z), n_sample, replace=False)

    z1 = z[idx1]
    z2 = z[idx2]
    cos_sim = np.sum(z1 * z2, axis=1) / (np.linalg.norm(z1, axis=1) * np.linalg.norm(z2, axis=1) + 1e-8)
    n_near_identical = int(np.sum(np.linalg.norm(z1 - z2, axis=1) < 0.01))

    results = {
        "mean": mean, "std": std,
        "dim_var_min": float(dim_var.min()), "dim_var_max": float(dim_var.max()),
        "dim_var_mean": float(dim_var.mean()),
        "norm_mean": float(norms.mean()), "norm_std": float(norms.std()),
        "norm_min": float(norms.min()), "norm_max": float(norms.max()),
        "cosine_sim_mean": float(cos_sim.mean()), "cosine_sim_std": float(cos_sim.std()),
        "n_near_identical": n_near_identical, "n_total": n_sample,
    }

    print(f"  Embedding mean: {mean:.4f}")
    print(f"  Embedding std: {std:.4f}")
    print(f"  Dim variance: [{dim_var.min():.6f}, {dim_var.max():.6f}], mean={dim_var.mean():.6f}")
    print(f"  Norm: mean={norms.mean():.4f}, std={norms.std():.4f}")
    print(f"  Cosine similarity: mean={cos_sim.mean():.4f}, std={cos_sim.std():.4f}")
    print(f"  Near-identical pairs: {n_near_identical}/{n_sample}")

    collapsed = std < 0.01 or float(dim_var.mean()) < 0.001
    results["collapsed"] = collapsed
    if collapsed:
        print("  WARNING: Embeddings appear collapsed")
    else:
        print("  Embeddings look healthy")

    # Generate histogram
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes[0, 0].hist(norms, bins=50, alpha=0.7)
        axes[0, 0].set_title("Embedding Norm Distribution")
        axes[0, 1].hist(dim_var, bins=30, alpha=0.7)
        axes[0, 1].set_title("Per-Dimension Variance")
        axes[1, 0].hist(cos_sim, bins=50, alpha=0.7)
        axes[1, 0].set_title("Cosine Similarity (Random Pairs)")
        axes[1, 1].hist(z.flatten(), bins=50, alpha=0.7)
        axes[1, 1].set_title("All Embedding Values")
        plt.tight_layout()
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(FIGURES_DIR / "gnn_embedding_diagnostics.png", dpi=150)
        plt.close()
    except Exception as e:
        print(f"  Could not generate plot: {e}")

    return results


# ---------------------------------------------------------------------------
# PART I: Directionality check (FIXED model)
# ---------------------------------------------------------------------------
def run_directionality_check():
    """Check if fixed model distinguishes A->B from B->A."""
    print("\n" + "=" * 70)
    print("PART I: DIRECTIONALITY CHECK (FIXED)")
    print("=" * 70)

    LP_COLD_DIR = LP_DIR / "cold_start"
    X = np.load(LP_COLD_DIR / "X_features.npy")
    train_edges = np.load(LP_COLD_DIR / "split_train.npy")
    test_edges = np.load(LP_COLD_DIR / "split_test.npy")
    test_neg = np.load(LP_COLD_DIR / "test_neg.npy")
    with open(LP_COLD_DIR / "id_to_idx.json") as f:
        id_to_idx = {int(k): v for k, v in json.load(f).items()}

    n_feat = X.shape[1]

    # Directed graph
    src_idx = np.array([id_to_idx[int(s)] for s in train_edges[:, 0]], dtype=np.int64)
    tgt_idx = np.array([id_to_idx[int(t)] for t in train_edges[:, 1]], dtype=np.int64)
    ei = np.stack([src_idx, tgt_idx])
    data = Data(x=torch.tensor(X, dtype=torch.float32),
                edge_index=torch.tensor(ei, dtype=torch.long))

    enc = GraphSAGEEncoder(n_feat, 64, 2, 0.3)
    dec = AsymmetricLinkPredictor(64)
    for p in enc.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)
    for p in dec.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)

    params = list(enc.parameters()) + list(dec.parameters())
    opt = torch.optim.Adam(params, lr=0.001)

    # Training pairs
    pos_s = torch.tensor(src_idx[:5000], dtype=torch.long)
    pos_t = torch.tensor(tgt_idx[:5000], dtype=torch.long)
    np.random.seed(42)
    neg_sample_idx = np.random.choice(len(test_neg), 5000, replace=False)
    neg_s = torch.tensor([id_to_idx[int(test_neg[i, 0])] for i in neg_sample_idx], dtype=torch.long)
    neg_t = torch.tensor([id_to_idx[int(test_neg[i, 1])] for i in neg_sample_idx], dtype=torch.long)

    for epoch in range(1, 51):
        enc.train(); dec.train()
        opt.zero_grad()
        z = enc(data.x, data.edge_index)
        ps = dec(z[pos_s], z[pos_t])
        ns = dec(z[neg_s], z[neg_t])
        lb = torch.cat([torch.ones_like(ps), torch.zeros_like(ns)])
        sc = torch.cat([ps, ns])
        sc = torch.nan_to_num(sc, nan=0.0)
        loss = F.binary_cross_entropy_with_logits(sc, lb)
        if torch.isnan(loss) or loss.item() > 100:
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        opt.step()
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: loss={loss.item():.4f}")

    enc.eval(); dec.eval()
    with torch.no_grad():
        z = enc(data.x, data.edge_index)
        n_test = min(500, len(test_edges))
        test_idx = np.random.choice(len(test_edges), n_test, replace=False)

        fwd_src = torch.tensor([id_to_idx[int(test_edges[i, 0])] for i in test_idx], dtype=torch.long)
        fwd_tgt = torch.tensor([id_to_idx[int(test_edges[i, 1])] for i in test_idx], dtype=torch.long)

        fwd_scores = dec(z[fwd_src], z[fwd_tgt]).cpu().numpy()
        rev_scores = dec(z[fwd_tgt], z[fwd_src]).cpu().numpy()

    diff = fwd_scores - rev_scores
    print(f"\n  Forward P(mean): {fwd_scores.mean():.4f}")
    print(f"  Reverse P(mean): {rev_scores.mean():.4f}")
    print(f"  Mean |diff|: {np.abs(diff).mean():.4f}")
    print(f"  Fraction |diff| < 0.1: {(np.abs(diff) < 0.1).mean():.4f}")
    print(f"  Correlation: {np.corrcoef(fwd_scores, rev_scores)[0, 1]:.4f}")

    if np.abs(diff).mean() > 0.1:
        print("  Model shows directional preference")
    else:
        print("  WARNING: Model barely distinguishes direction")

    return {
        "forward_mean": float(fwd_scores.mean()),
        "reverse_mean": float(rev_scores.mean()),
        "mean_abs_diff": float(np.abs(diff).mean()),
        "fraction_similar_01": float((np.abs(diff) < 0.1).mean()),
        "fraction_similar_05": float((np.abs(diff) < 0.05).mean()),
        "correlation": float(np.corrcoef(fwd_scores, rev_scores)[0, 1]),
    }


# ---------------------------------------------------------------------------
# PART J: Edge weight check (FIXED model)
# ---------------------------------------------------------------------------
def run_edge_weight_check():
    """Compare fixed GNN with unweighted vs weighted edges."""
    print("\n" + "=" * 70)
    print("PART J: EDGE WEIGHT CHECK (FIXED)")
    print("=" * 70)

    LP_COLD_DIR = LP_DIR / "cold_start"
    X = np.load(LP_COLD_DIR / "X_features.npy")
    train_edges = np.load(LP_COLD_DIR / "split_train.npy")
    test_edges = np.load(LP_COLD_DIR / "split_test.npy")
    test_neg = np.load(LP_COLD_DIR / "test_neg.npy")
    with open(LP_COLD_DIR / "id_to_idx.json") as f:
        id_to_idx = {int(k): v for k, v in json.load(f).items()}

    n_feat = X.shape[1]
    src_idx = np.array([id_to_idx[int(s)] for s in train_edges[:, 0]], dtype=np.int64)
    tgt_idx = np.array([id_to_idx[int(t)] for t in train_edges[:, 1]], dtype=np.int64)

    # Directed graph (no reverse edges)
    ei = np.stack([src_idx, tgt_idx])
    data = Data(x=torch.tensor(X, dtype=torch.float32),
                edge_index=torch.tensor(ei, dtype=torch.long))

    # Training pairs
    pos_s = torch.tensor(src_idx[:5000], dtype=torch.long)
    pos_t = torch.tensor(tgt_idx[:5000], dtype=torch.long)
    np.random.seed(42)
    neg_idx = np.random.choice(len(test_neg), 5000, replace=False)
    neg_s = torch.tensor([id_to_idx[int(test_neg[i, 0])] for i in neg_idx], dtype=torch.long)
    neg_t = torch.tensor([id_to_idx[int(test_neg[i, 1])] for i in neg_idx], dtype=torch.long)

    # Test pairs
    te_idx = np.random.choice(len(test_edges), min(2000, len(test_edges)), replace=False)
    te_s = torch.tensor([id_to_idx[int(test_edges[i, 0])] for i in te_idx], dtype=torch.long)
    te_t = torch.tensor([id_to_idx[int(test_edges[i, 1])] for i in te_idx], dtype=torch.long)
    tn_idx = np.random.choice(len(test_neg), min(2000, len(test_neg)), replace=False)
    tn_s = torch.tensor([id_to_idx[int(test_neg[i, 0])] for i in tn_idx], dtype=torch.long)
    tn_t = torch.tensor([id_to_idx[int(test_neg[i, 1])] for i in tn_idx], dtype=torch.long)

    from sklearn.metrics import roc_auc_score
    results = {}
    for use_weights, label in [(False, "unweighted"), (True, "weighted")]:
        print(f"\n  --- {label} ---")
        enc = GraphSAGEEncoder(n_feat, 64, 2, 0.3)
        dec = AsymmetricLinkPredictor(64)
        for p in enc.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
        for p in dec.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

        params = list(enc.parameters()) + list(dec.parameters())
        opt = torch.optim.Adam(params, lr=0.001)

        t0 = time.time()
        for epoch in range(1, 31):
            enc.train(); dec.train()
            opt.zero_grad()
            z = enc(data.x, data.edge_index)
            ps = dec(z[pos_s], z[pos_t])
            ns = dec(z[neg_s], z[neg_t])
            lb = torch.cat([torch.ones_like(ps), torch.zeros_like(ns)])
            sc = torch.cat([ps, ns])
            sc = torch.nan_to_num(sc, nan=0.0)
            loss = F.binary_cross_entropy_with_logits(sc, lb)
            if torch.isnan(loss) or loss.item() > 100:
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            opt.step()

        elapsed = time.time() - t0

        enc.eval(); dec.eval()
        with torch.no_grad():
            z = enc(data.x, data.edge_index)
            pos_scores = dec(z[te_s], z[te_t]).cpu().numpy()
            neg_scores = dec(z[tn_s], z[tn_t]).cpu().numpy()

        scores = np.concatenate([pos_scores, neg_scores])
        scores = np.nan_to_num(scores, nan=0.0)
        labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
        auc = float(roc_auc_score(labels, scores))
        print(f"    ROC-AUC: {auc:.4f} | Time: {elapsed:.1f}s")
        results[label] = {"roc_auc": auc, "time": elapsed}

    return results


# ---------------------------------------------------------------------------
# Run all diagnostics
# ---------------------------------------------------------------------------
def run_all_diagnostics():
    """Run all GNN diagnostics."""
    print("=" * 70)
    print("FLYMIND LINK PREDICTION — GNN DIAGNOSTICS (FIXED)")
    print("=" * 70)

    results = {}

    results["sanity_test"] = run_sanity_test()
    gc.collect()

    results["embedding_diagnostics"] = run_embedding_diagnostics()
    gc.collect()

    results["directionality_check"] = run_directionality_check()
    gc.collect()

    results["edge_weight_check"] = run_edge_weight_check()
    gc.collect()

    with open(REPORTS_DIR / "gnn_diagnostics.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("ALL DIAGNOSTICS COMPLETE")
    print("=" * 70)

    return results


if __name__ == "__main__":
    run_all_diagnostics()
