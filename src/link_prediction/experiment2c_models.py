#!/usr/bin/env python3
"""Experiment 2C: Memory-safe models and training."""

import gc
import time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from src.link_prediction.evaluate import compute_all_metrics


BATCH_SIZE = 4096


def _rss_mb():
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * 4096 / (1024 * 1024)
    except Exception:
        return 0.0


def log_mem(tag=""):
    print(f"  [MEM:{tag}] RSS: {_rss_mb():.0f} MB")


def check_mem_limit(tag="", limit_mb=2500):
    rss = _rss_mb()
    if rss > limit_mb:
        raise MemoryError(f"RSS {rss:.0f} MB exceeds {limit_mb} MB limit at {tag}")
    return rss


def batch_pair_features(X, src_idx, dst_idx, out):
    """Compute [xa, xb, |xa-xb|, xa*xb] for given index arrays. Writes into out.

    src_idx, dst_idx must be integer arrays of row indices into X.
    len(src_idx) must equal out.shape[0].
    """
    xa = X[src_idx]
    xb = X[dst_idx]
    out[:] = np.hstack([xa, xb, np.abs(xa - xb), xa * xb])


class MLPNodeClassifier(torch.nn.Module):
    def __init__(self, pair_dim):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(pair_dim, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.mlp(x).squeeze(-1)


class GraphSAGEEncoder(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim=32, num_layers=2, dropout=0.3):
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

    def forward(self, x, edge_index=None):
        x = self.input_bn(x)
        for conv, bn in zip(self.convs[:-1], self.bns[:-1]):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        x = self.bns[-1](x)
        return x


class FeatureOnlyEncoder(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim=32, num_layers=2, dropout=0.3):
        super().__init__()
        layers = []
        dims = [in_dim] + [hidden_dim] * num_layers
        for i in range(num_layers):
            layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
            layers.append(torch.nn.BatchNorm1d(dims[i + 1]))
            layers.append(torch.nn.ReLU())
            if i < num_layers - 1:
                layers.append(torch.nn.Dropout(dropout))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x, edge_index=None):
        return self.mlp(x)


class AsymmetricDecoder(torch.nn.Module):
    def __init__(self, embed_dim=32):
        super().__init__()
        self.src_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.tgt_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embed_dim * 4, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(64, 1),
        )

    def forward(self, z_a, z_b):
        h_a = self.src_proj(z_a)
        h_b = self.tgt_proj(z_b)
        combined = torch.cat([h_a, h_b, h_a * h_b, torch.abs(h_a - h_b)], dim=-1)
        return self.mlp(combined).squeeze(-1)


def make_pair_indices(pairs, id_to_idx):
    s = torch.tensor([id_to_idx[int(p)] for p in pairs[:, 0]], dtype=torch.long)
    t = torch.tensor([id_to_idx[int(p)] for p in pairs[:, 1]], dtype=torch.long)
    return s, t


def _forward(encoder, data):
    if hasattr(data, 'edge_index') and data.edge_index is not None:
        return encoder(data.x, data.edge_index)
    return encoder(data.x)


def train_mlp_model(X, id_to_idx, train_pos, train_neg, val_pos, val_neg,
                    test_pos, test_neg, n_feat, epochs=50, lr=0.001, label="MLP"):
    """Memory-safe MLP: streaming pair features, no full-matrix materialization."""
    pair_dim = n_feat * 4

    tr_src = np.array([id_to_idx[int(i)] for i in np.concatenate([train_pos[:, 0], train_neg[:, 0]])], dtype=np.int64)
    tr_dst = np.array([id_to_idx[int(i)] for i in np.concatenate([train_pos[:, 1], train_neg[:, 1]])], dtype=np.int64)
    n_pos = len(train_pos)
    n_neg = len(train_neg)
    tr_labels = np.concatenate([
        np.ones(n_pos, dtype=np.float32),
        np.zeros(n_neg, dtype=np.float32),
    ])
    n_tr = len(tr_src)

    log_mem(f"{label} before norm stats")
    # Two-pass normalization: first compute mean, then variance
    sum_x = np.zeros(pair_dim, dtype=np.float64)
    count = 0
    for start in range(0, n_tr, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n_tr)
        buf = np.empty((end - start, pair_dim), dtype=np.float32)
        batch_pair_features(X, tr_src[start:end], tr_dst[start:end], buf)
        sum_x += buf.sum(axis=0)
        count += end - start
        del buf
    mean = (sum_x / count).astype(np.float32)
    del sum_x
    gc.collect()

    sum_sq = np.zeros(pair_dim, dtype=np.float64)
    for start in range(0, n_tr, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n_tr)
        buf = np.empty((end - start, pair_dim), dtype=np.float32)
        batch_pair_features(X, tr_src[start:end], tr_dst[start:end], buf)
        diff = buf.astype(np.float64) - mean.astype(np.float64)
        sum_sq += (diff * diff).sum(axis=0)
        del buf, diff
    std = np.sqrt(sum_sq / count).astype(np.float32) + 1e-8
    del sum_sq
    gc.collect()
    log_mem(f"{label} after norm stats")

    model = MLPNodeClassifier(pair_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    ytr_t = torch.tensor(tr_labels, dtype=torch.float32)
    rng = np.random.default_rng(42)
    order = np.arange(n_tr, dtype=np.int64)

    best_val_auc = 0
    best_state = None
    history = {"train_loss": [], "val_auc": [], "val_pr_auc": []}
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        rng.shuffle(order)
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_tr, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n_tr)
            idx = order[start:end]
            buf = np.empty((len(idx), pair_dim), dtype=np.float32)
            batch_pair_features(X, tr_src[idx], tr_dst[idx], buf)
            buf = (buf - mean) / std
            xb = torch.from_numpy(buf)
            yb = ytr_t[idx]
            del buf
            pred = model(xb)
            pred = torch.nan_to_num(pred, nan=0.0)
            loss = F.binary_cross_entropy_with_logits(pred, yb)
            if torch.isnan(loss) or loss.item() > 100:
                del xb, yb, pred, loss
                continue
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
            del xb, yb, pred, loss
        gc.collect()

        model.eval()
        val_scores = _eval_mlp_streaming(model, X, val_pos, val_neg, mean, std, id_to_idx)
        val_labels = np.concatenate([
            np.ones(len(val_pos), dtype=np.float32),
            np.zeros(len(val_neg), dtype=np.float32),
        ])
        val_auc = float(roc_auc_score(val_labels, val_scores))
        val_pr = float(average_precision_score(val_labels, val_scores))
        del val_scores, val_labels

        avg_loss = epoch_loss / max(n_batches, 1)
        history["train_loss"].append(avg_loss)
        history["val_auc"].append(val_auc)
        history["val_pr_auc"].append(val_pr)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(f"  [{label}] Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Val AUC: {val_auc:.4f}")

    elapsed = time.time() - t0
    if best_state:
        model.load_state_dict(best_state)
        del best_state

    test_scores = _eval_mlp_streaming(model, X, test_pos, test_neg, mean, std, id_to_idx)
    test_labels = np.concatenate([
        np.ones(len(test_pos), dtype=np.float32),
        np.zeros(len(test_neg), dtype=np.float32),
    ])
    metrics = compute_all_metrics(test_labels, test_scores, label)
    metrics["train_time"] = elapsed
    metrics["best_val_auc"] = best_val_auc
    print(f"  [{label}] Best val AUC: {best_val_auc:.4f} | Test PR-AUC: {metrics['pr_auc']:.4f}")

    del tr_src, tr_dst, tr_labels, ytr_t, order, mean, std, test_scores, test_labels
    gc.collect()
    return metrics, history, model


def _eval_mlp_streaming(model, X, pos, neg, mean, std, id_to_idx):
    """Evaluate MLP in chunks — never materializes full pair-feature matrix."""
    model.eval()
    n_pos = len(pos)
    n_neg = len(neg)
    n_total = n_pos + n_neg
    scores = np.empty(n_total, dtype=np.float32)

    all_src = np.array([id_to_idx[int(i)] for i in np.concatenate([pos[:, 0], neg[:, 0]])], dtype=np.int64)
    all_dst = np.array([id_to_idx[int(i)] for i in np.concatenate([pos[:, 1], neg[:, 1]])], dtype=np.int64)

    with torch.no_grad():
        for start in range(0, n_total, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n_total)
            buf = np.empty((end - start, len(mean)), dtype=np.float32)
            batch_pair_features(X, all_src[start:end], all_dst[start:end], buf)
            buf = (buf - mean) / std
            xb = torch.from_numpy(buf)
            scores[start:end] = model(xb).numpy()
            del xb, buf

    del all_src, all_dst
    return scores


def train_encoder_decoder(encoder, decoder, data, pos_s, pos_t, neg_s, neg_t,
                          val_pos_s, val_pos_t, val_neg_s, val_neg_t,
                          epochs=50, lr=0.001, label="model"):
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=1e-5)
    best_val_auc = 0
    best_state = None
    history = {"train_loss": [], "val_auc": [], "val_pr_auc": []}
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        encoder.train()
        decoder.train()
        optimizer.zero_grad()
        z = _forward(encoder, data)
        ps = decoder(z[pos_s], z[pos_t])
        ns = decoder(z[neg_s], z[neg_t])
        lb = torch.cat([torch.ones_like(ps), torch.zeros_like(ns)])
        sc = torch.cat([ps, ns])
        sc = torch.nan_to_num(sc, nan=0.0)
        loss = F.binary_cross_entropy_with_logits(sc, lb)
        del ps, ns, lb, sc
        if torch.isnan(loss) or loss.item() > 100:
            del loss
            history["train_loss"].append(float("nan"))
            history["val_auc"].append(0.5)
            history["val_pr_auc"].append(0.5)
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()
        del loss

        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            zv = _forward(encoder, data)
            vps = decoder(zv[val_pos_s], zv[val_pos_t]).cpu().numpy()
            vns = decoder(zv[val_neg_s], zv[val_neg_t]).cpu().numpy()
            del zv

        vp = np.concatenate([vps, vns])
        vp = np.nan_to_num(vp, nan=0.0)
        vl = np.concatenate([np.ones(len(vps)), np.zeros(len(vns))])
        val_auc = float(roc_auc_score(vl, vp))
        val_pr = float(average_precision_score(vl, vp))
        del vp, vl, vps, vns

        history["train_loss"].append(0.0)
        history["val_auc"].append(val_auc)
        history["val_pr_auc"].append(val_pr)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {
                "encoder": {k: v.clone() for k, v in encoder.state_dict().items()},
                "decoder": {k: v.clone() for k, v in decoder.state_dict().items()},
            }

        if epoch % 10 == 0 or epoch == 1:
            print(f"  [{label}] Epoch {epoch:3d} | Val AUC: {val_auc:.4f} | Val PR: {val_pr:.4f}")

    elapsed = time.time() - t0
    if best_state:
        encoder.load_state_dict(best_state["encoder"])
        decoder.load_state_dict(best_state["decoder"])
        del best_state
    history["train_time"] = elapsed
    history["best_val_auc"] = best_val_auc
    print(f"  [{label}] Best val AUC: {best_val_auc:.4f} | Time: {elapsed:.1f}s")
    return encoder, decoder, history


def evaluate_model(encoder, decoder, data, te_s, te_t, tn_s, tn_t, label=""):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        z = _forward(encoder, data)
        pos_scores = decoder(z[te_s], z[te_t]).cpu().numpy()
        neg_scores = decoder(z[tn_s], z[tn_t]).cpu().numpy()
        del z
    scores = np.concatenate([pos_scores, neg_scores])
    scores = np.nan_to_num(scores, nan=0.0)
    labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    del pos_scores, neg_scores
    metrics = compute_all_metrics(labels, scores, label)
    return metrics, scores, labels


def directionality_test(encoder, decoder, data, test_edges, id_to_idx, n_pairs=1000):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        z = _forward(encoder, data)
    idx = np.random.choice(len(test_edges), min(n_pairs, len(test_edges)), replace=False)
    fwd_s = torch.tensor([id_to_idx[int(test_edges[i, 0])] for i in idx], dtype=torch.long)
    fwd_t = torch.tensor([id_to_idx[int(test_edges[i, 1])] for i in idx], dtype=torch.long)
    with torch.no_grad():
        fwd_scores = decoder(z[fwd_s], z[fwd_t]).cpu().numpy()
        rev_scores = decoder(z[fwd_t], z[fwd_s]).cpu().numpy()
        del z
    diff = fwd_scores - rev_scores
    result = {
        "forward_mean": float(np.mean(fwd_scores)),
        "reverse_mean": float(np.mean(rev_scores)),
        "mean_abs_diff": float(np.abs(diff).mean()),
        "correlation": float(np.corrcoef(fwd_scores, rev_scores)[0, 1]),
    }
    del fwd_scores, rev_scores, diff, z
    return result


def embedding_diagnostics(encoder, data):
    encoder.eval()
    with torch.no_grad():
        z = _forward(encoder, data).numpy()
    norms = np.linalg.norm(z, axis=1)
    result = {
        "mean": float(np.mean(z)), "std": float(np.std(z)),
        "norm_mean": float(norms.mean()), "norm_std": float(norms.std()),
        "variance": float(np.var(z)),
        "has_nan": bool(np.isnan(z).any()), "has_inf": bool(np.isinf(z).any()),
    }
    np.random.seed(42)
    n = min(1000, len(z))
    i1 = np.random.choice(len(z), n, replace=False)
    i2 = np.random.choice(len(z), n, replace=False)
    z1, z2 = z[i1], z[i2]
    cos = np.sum(z1 * z2, axis=1) / (np.linalg.norm(z1, axis=1) * np.linalg.norm(z2, axis=1) + 1e-8)
    result["cosine_sim_mean"] = float(cos.mean())
    del z, norms, z1, z2, cos
    return result
