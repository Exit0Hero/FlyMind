"""Build PyTorch Geometric graph from neuron data."""

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops

from src.config import SEED


def build_graph(
    X: np.ndarray,
    y: np.ndarray,
    edges: np.ndarray,
    node_ids: np.ndarray,
    edge_weights: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    add_reverse: bool = True,
) -> Data:
    """Build a PyG Data object.

    Args:
        X: node features (num_nodes, num_features)
        y: node labels (num_nodes,)
        edges: edge indices (2, num_edges) — source/target root_ids
        node_ids: root_id for each node index
        edge_weights: (num_edges,) edge weights
        train_mask/val_mask/test_mask: boolean masks
        add_reverse: if True, add reverse edges for message passing
    """
    # Create mapping from root_id to node index
    id_to_idx = {int(rid): i for i, rid in enumerate(node_ids)}

    # Map edges to indices
    src = edges[:, 0]
    tgt = edges[:, 1]
    src_idx = np.array([id_to_idx[int(s)] for s in src], dtype=np.int64)
    tgt_idx = np.array([id_to_idx[int(t)] for t in tgt], dtype=np.int64)

    edge_index = np.stack([src_idx, tgt_idx], axis=0)

    if add_reverse:
        reverse_index = np.stack([tgt_idx, src_idx], axis=0)
        edge_index = np.concatenate([edge_index, reverse_index], axis=1)
        edge_weights_full = np.concatenate([edge_weights, edge_weights], axis=0)
    else:
        edge_weights_full = edge_weights

    # Convert to tensors
    x = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)
    edge_weight_tensor = torch.tensor(edge_weights_full, dtype=torch.float32)

    data = Data(
        x=x,
        y=y_tensor,
        edge_index=edge_index_tensor,
        edge_attr=edge_weight_tensor.unsqueeze(1),
        train_mask=torch.tensor(train_mask, dtype=torch.bool),
        val_mask=torch.tensor(val_mask, dtype=torch.bool),
        test_mask=torch.tensor(test_mask, dtype=torch.bool),
    )

    data.num_nodes = len(X)
    return data


def compute_graph_features(edges_df, node_ids: np.ndarray) -> np.ndarray:
    """Compute simple graph statistics for each node.

    Returns: (num_nodes, 7) array with features:
        in_degree, out_degree, total_degree,
        weighted_in_degree, weighted_out_degree,
        num_upstream, num_downstream
    """
    import pandas as pd

    node_set = set(node_ids.tolist())
    id_to_idx = {int(rid): i for i, rid in enumerate(node_ids)}

    in_degree = np.zeros(len(node_ids), dtype=np.float32)
    out_degree = np.zeros(len(node_ids), dtype=np.float32)
    weighted_in = np.zeros(len(node_ids), dtype=np.float32)
    weighted_out = np.zeros(len(node_ids), dtype=np.float32)
    upstream_count = np.zeros(len(node_ids), dtype=np.float32)
    downstream_count = np.zeros(len(node_ids), dtype=np.float32)

    for _, row in edges_df.iterrows():
        src = int(row["source"])
        tgt = int(row["target"])
        w = float(row["weight"])

        if tgt in id_to_idx:
            idx = id_to_idx[tgt]
            in_degree[idx] += 1
            weighted_in[idx] += w
            if src in node_set:
                upstream_count[idx] += 1

        if src in id_to_idx:
            idx = id_to_idx[src]
            out_degree[idx] += 1
            weighted_out[idx] += w
            if tgt in node_set:
                downstream_count[idx] += 1

    total_degree = in_degree + out_degree

    features = np.stack([
        in_degree, out_degree, total_degree,
        weighted_in, weighted_out,
        upstream_count, downstream_count,
    ], axis=1)

    return features
