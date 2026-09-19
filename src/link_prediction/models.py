"""FlyMind Link Prediction — Models."""

import time
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from sklearn.ensemble import RandomForestClassifier

from src.link_prediction.config import (
    SEED, GNN_HIDDEN_DIM, GNN_NUM_LAYERS, GNN_DROPOUT,
    GNN_LR, GNN_EPOCHS, GNN_BATCH_SIZE,
)


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------
def train_rf(X_train, y_train, n_estimators=200):
    """Train a Random Forest classifier."""
    rf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=None,
        random_state=SEED, n_jobs=-1, class_weight="balanced",
    )
    rf.fit(X_train, y_train)
    return rf


# ---------------------------------------------------------------------------
# GraphSAGE Link Predictor
# ---------------------------------------------------------------------------
class GraphSAGEEncoder(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, dropout):
        super().__init__()
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
    """Asymmetric MLP decoder: takes [z_A, z_B, z_A*z_B, |z_A-z_B|]."""
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


def build_pyg_data(X, train_edges, reverse_edges=True):
    """Build PyG Data object from training edges."""
    src = train_edges[:, 0].astype(np.int64)
    tgt = train_edges[:, 1].astype(np.int64)
    edge_index = np.stack([src, tgt], axis=0)

    if reverse_edges:
        rev = np.stack([tgt, src], axis=0)
        edge_index = np.concatenate([edge_index, rev], axis=1)

    return Data(
        x=torch.tensor(X, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
    )


def train_gnn_link_predictor(
    data, node_id_to_idx, train_pos, train_neg,
    val_pos, val_neg,
    feature_dim, device,
    hidden_dim=GNN_HIDDEN_DIM, num_layers=GNN_NUM_LAYERS,
    dropout=GNN_DROPOUT, lr=GNN_LR, epochs=GNN_EPOCHS,
):
    """Train GraphSAGE + asymmetric link predictor."""
    encoder = GraphSAGEEncoder(feature_dim, hidden_dim, num_layers, dropout).to(device)
    decoder = LinkPredictor(hidden_dim).to(device)
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=1e-5)

    # Build training pairs
    train_src_idx = torch.tensor([node_id_to_idx[int(s)] for s in train_pos[:, 0]], dtype=torch.long)
    train_tgt_idx = torch.tensor([node_id_to_idx[int(t)] for t in train_pos[:, 1]], dtype=torch.long)
    train_neg_src_idx = torch.tensor([node_id_to_idx[int(s)] for s in train_neg[:, 0]], dtype=torch.long)
    train_neg_tgt_idx = torch.tensor([node_id_to_idx[int(t)] for t in train_neg[:, 1]], dtype=torch.long)

    val_src_idx = torch.tensor([node_id_to_idx[int(s)] for s in val_pos[:, 0]], dtype=torch.long)
    val_tgt_idx = torch.tensor([node_id_to_idx[int(t)] for t in val_pos[:, 1]], dtype=torch.long)
    val_neg_src_idx = torch.tensor([node_id_to_idx[int(s)] for s in val_neg[:, 0]], dtype=torch.long)
    val_neg_tgt_idx = torch.tensor([node_id_to_idx[int(t)] for t in val_neg[:, 1]], dtype=torch.long)

    best_val_auc = 0
    best_state = None
    history = {"train_loss": [], "val_auc": []}

    data = data.to(device)

    for epoch in range(1, epochs + 1):
        encoder.train()
        decoder.train()
        optimizer.zero_grad()

        z = encoder(data.x, data.edge_index)

        pos_score = decoder(z[train_src_idx], z[train_tgt_idx])
        neg_score = decoder(z[train_neg_src_idx], z[train_neg_tgt_idx])

        labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
        scores = torch.cat([pos_score, neg_score])
        loss = F.binary_cross_entropy_with_logits(scores, labels)
        loss.backward()
        optimizer.step()

        # Validate
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            z_val = encoder(data.x, data.edge_index)
            val_pos_s = decoder(z_val[val_src_idx], z_val[val_tgt_idx]).cpu().numpy()
            val_neg_s = decoder(z_val[val_neg_src_idx], z_val[val_neg_tgt_idx]).cpu().numpy()

        from sklearn.metrics import roc_auc_score
        val_preds = np.concatenate([val_pos_s, val_neg_s])
        val_labels = np.concatenate([np.ones(len(val_pos_s)), np.zeros(len(val_neg_s))])
        val_auc = roc_auc_score(val_labels, val_preds)

        history["train_loss"].append(loss.item())
        history["val_auc"].append(val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {
                "encoder": encoder.state_dict().copy(),
                "decoder": decoder.state_dict().copy(),
            }

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | Loss: {loss.item():.4f} | Val AUC: {val_auc:.4f}")

    if best_state:
        encoder.load_state_dict(best_state["encoder"])
        decoder.load_state_dict(best_state["decoder"])

    return encoder, decoder, history
