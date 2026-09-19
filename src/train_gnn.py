"""GraphSAGE training — full-batch approach (no neighbor sampler required)."""

import time
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

from src.config import SEED, MODELS_DIR
from src.utils import set_seed, get_device


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.3):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.norms.append(torch.nn.LayerNorm(hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.norms.append(torch.nn.LayerNorm(hidden_channels))

        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index)
        return x


class GraphSAGEClassifier(torch.nn.Module):
    def __init__(self, sage_model, num_classes):
        super().__init__()
        self.sage = sage_model
        self.classifier = torch.nn.Linear(
            sage_model.convs[-1].out_channels, num_classes
        )

    def forward(self, x, edge_index):
        z = self.sage(x, edge_index)
        return self.classifier(z)


def train_epoch_full(model, data, optimizer, device, class_weights, train_mask):
    model.train()
    optimizer.zero_grad()

    out = model(data.x.to(device), data.edge_index.to(device))
    out = out[train_mask]
    y = data.y[train_mask].to(device)

    loss = F.cross_entropy(out, y, weight=class_weights.to(device))
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate_full(model, data, device, mask):
    model.eval()
    out = model(data.x.to(device), data.edge_index.to(device))
    pred = out[mask].argmax(dim=-1).cpu().numpy()
    label = data.y[mask].cpu().numpy()
    return pred, label


def train_graphsage(data, num_classes, train_mask, val_mask, test_mask,
                    device, class_weights=None, epochs=50, lr=0.005,
                    hidden_dim=64, num_layers=2, dropout=0.3,
                    batch_size=1024, neighbor_sizes=None):
    """Train GraphSAGE (full-batch)."""
    set_seed(SEED)

    in_channels = data.x.shape[1]
    sage = GraphSAGE(in_channels, hidden_dim, hidden_dim, num_layers, dropout)
    model = GraphSAGEClassifier(sage, num_classes).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_f1 = 0
    best_epoch = 0
    patience = 10
    patience_counter = 0
    history = {"train_loss": [], "val_acc": [], "val_f1": []}

    print(f"  Training GraphSAGE (full-batch): {in_channels} -> {hidden_dim} -> {num_classes}")
    print(f"  Device: {device}")

    t_start = time.time()

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch_full(model, data, optimizer, device, class_weights, train_mask)

        val_preds, val_labels = evaluate_full(model, data, device, val_mask)
        from sklearn.metrics import accuracy_score, f1_score
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)

        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_f1": val_f1,
                "val_acc": val_acc,
                "in_channels": in_channels,
                "hidden_dim": hidden_dim,
                "num_classes": num_classes,
                "num_layers": num_layers,
                "dropout": dropout,
            }, MODELS_DIR / "graphsage_superclass_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch} (best: {best_epoch})")
                break

    elapsed = time.time() - t_start
    print(f"  Training complete in {elapsed:.1f}s (best epoch: {best_epoch}, val F1: {best_val_f1:.4f})")

    checkpoint = torch.load(MODELS_DIR / "graphsage_superclass_best.pt", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    return model, history, elapsed, best_epoch, best_val_f1
