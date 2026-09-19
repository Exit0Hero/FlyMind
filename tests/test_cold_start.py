"""Tests for Cold-Start + GNN diagnostics (Experiment 2B)."""

import sys
import pathlib
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


def test_no_training_edge_touches_test_node():
    """No training edge should have both endpoints in test set."""
    from src.link_prediction.cold_start import create_cold_start_node_split
    from src.link_prediction.data_prep import load_neuron_table, aggregate_edges, load_raw_edges

    nt = load_neuron_table()
    raw = load_raw_edges()
    edges = aggregate_edges(raw)
    cold = create_cold_start_node_split(edges, nt["root_id"].values)

    train_set = cold["train_nodes"]
    test_set = cold["test_nodes"]
    for s, t in cold["train"]:
        assert int(s) in train_set and int(t) in train_set, \
            f"Training edge ({s},{t}) touches non-train node!"
    print("  PASS: no training edge touches test/val nodes")


def test_cold_start_positives_absent_from_training():
    """Cold-start test positives should not be in training edges."""
    from src.link_prediction.cold_start import create_cold_start_node_split
    from src.link_prediction.data_prep import load_neuron_table, aggregate_edges, load_raw_edges
    from src.link_prediction.negative_sampling import build_sorted_pairs, verify_not_in_edges

    nt = load_neuron_table()
    raw = load_raw_edges()
    edges = aggregate_edges(raw)
    cold = create_cold_start_node_split(edges, nt["root_id"].values)

    train_pairs = build_sorted_pairs(cold["train"][:, 0], cold["train"][:, 1])
    keep = verify_not_in_edges(cold["test"][:, 0], cold["test"][:, 1], train_pairs)
    n_leak = (~keep).sum()
    assert n_leak == 0, f"{n_leak} test positives found in training!"
    print("  PASS: cold-start test positives absent from training")


def test_cold_start_negatives_absent_from_full_graph():
    """Cold-start negatives should not be any known positive edge."""
    from src.link_prediction.data_prep import load_neuron_table, aggregate_edges, load_raw_edges
    from src.link_prediction.negative_sampling import build_sorted_pairs, verify_not_in_edges

    nt = load_neuron_table()
    raw = load_raw_edges()
    edges = aggregate_edges(raw)
    all_pos = build_sorted_pairs(edges["source"].values, edges["target"].values)

    from src.link_prediction.cold_start import create_cold_start_node_split
    cold = create_cold_start_node_split(edges, nt["root_id"].values)
    from src.link_prediction.negative_sampling import build_edge_index, sample_negatives

    all_node_ids = np.unique(np.concatenate([cold["train"][:, 0], cold["train"][:, 1],
                                              cold["test"][:, 0], cold["test"][:, 1]]))
    full_idx = build_edge_index(edges[["source", "target"]].values)
    test_neg = sample_negatives(all_node_ids, 1000, full_idx, seed=999)

    keep = verify_not_in_edges(test_neg[:, 0], test_neg[:, 1], all_pos)
    n_leak = (~keep).sum()
    assert n_leak == 0, f"{n_leak} negatives are known positives!"
    print("  PASS: cold-start negatives absent from full positive graph")


def test_training_only_graph_stats():
    """Training graph should have no test-node edges."""
    from src.link_prediction.cold_start import create_cold_start_node_split
    from src.link_prediction.data_prep import load_neuron_table, aggregate_edges, load_raw_edges

    nt = load_neuron_table()
    raw = load_raw_edges()
    edges = aggregate_edges(raw)
    cold = create_cold_start_node_split(edges, nt["root_id"].values)

    train_nodes = cold["train_nodes"]
    test_nodes = cold["test_nodes"]
    val_nodes = cold["val_nodes"]

    for s, t in cold["train"]:
        assert int(s) not in test_nodes, f"Train edge has test src: {s}"
        assert int(t) not in test_nodes, f"Train edge has test tgt: {t}"
        assert int(s) not in val_nodes, f"Train edge has val src: {s}"
        assert int(t) not in val_nodes, f"Train edge has val tgt: {t}"
    print("  PASS: training graph has no test/val node edges")


def test_directionality():
    """Test that root_id to node-index mapping is correct."""
    from src.link_prediction.data_prep import load_neuron_table, build_feature_matrix
    from src.link_prediction.config import NUMERIC_FEATURES, CATEGORICAL_FEATURES

    nt = load_neuron_table()
    X, root_ids, id_to_idx, _, _ = build_feature_matrix(nt, NUMERIC_FEATURES, CATEGORICAL_FEATURES)

    for i, rid in enumerate(root_ids):
        assert id_to_idx[int(rid)] == i, f"Mapping mismatch for {rid}"
    print("  PASS: root_id to node-index mapping correct")


def test_cold_start_node_splits_disjoint():
    """Train/val/test node sets should be disjoint."""
    from src.link_prediction.cold_start import create_cold_start_node_split
    from src.link_prediction.data_prep import load_neuron_table, aggregate_edges, load_raw_edges

    nt = load_neuron_table()
    raw = load_raw_edges()
    edges = aggregate_edges(raw)
    cold = create_cold_start_node_split(edges, nt["root_id"].values)

    train = cold["train_nodes"]
    val = cold["val_nodes"]
    test = cold["test_nodes"]
    assert len(train & val) == 0, "Train/val overlap!"
    assert len(train & test) == 0, "Train/test overlap!"
    assert len(val & test) == 0, "Val/test overlap!"
    print("  PASS: node splits are disjoint")


def test_fixed_model_overfits():
    """Fixed GNN should overfit 1K edges."""
    import torch
    import torch.nn.functional as F
    from src.link_prediction.gnn_diagnostics import GraphSAGEEncoder, AsymmetricLinkPredictor
    from torch_geometric.data import Data

    LP_COLD_DIR = __import__('pathlib').Path(__file__).resolve().parent.parent / "data" / "processed" / "link_prediction" / "cold_start"
    X = np.load(LP_COLD_DIR / "X_features.npy")
    train_edges = np.load(LP_COLD_DIR / "split_train.npy")
    with open(LP_COLD_DIR / "id_to_idx.json") as f:
        id_to_idx = {int(k): v for k, v in __import__('json').load(f).items()}

    n_feat = X.shape[1]
    tiny = train_edges[:1000]
    src_idx = np.array([id_to_idx[int(s)] for s in tiny[:, 0]], dtype=np.int64)
    tgt_idx = np.array([id_to_idx[int(t)] for t in tiny[:, 1]], dtype=np.int64)

    ei = np.stack([src_idx, tgt_idx])
    data = Data(x=torch.tensor(X, dtype=torch.float32),
                edge_index=torch.tensor(ei, dtype=torch.long))

    np.random.seed(42)
    all_nodes = np.unique(np.concatenate([tiny[:, 0], tiny[:, 1]]))
    neg_src = np.random.choice(all_nodes, 1000)
    neg_tgt = np.random.choice(all_nodes, 1000)
    mask = neg_src != neg_tgt
    neg_src, neg_tgt = neg_src[mask][:1000], neg_tgt[mask][:1000]

    pos_s = torch.tensor(src_idx, dtype=torch.long)
    pos_t = torch.tensor(tgt_idx, dtype=torch.long)
    neg_s = torch.tensor([id_to_idx[int(s)] for s in neg_src], dtype=torch.long)
    neg_t = torch.tensor([id_to_idx[int(t)] for t in neg_tgt], dtype=torch.long)

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

    enc.eval(); dec.eval()
    with torch.no_grad():
        z = enc(data.x, data.edge_index)
        pos_scores = dec(z[pos_s], z[pos_t]).cpu().numpy()
        neg_scores = dec(z[neg_s], z[neg_t]).cpu().numpy()

    from sklearn.metrics import roc_auc_score
    scores = np.concatenate([pos_scores, neg_scores])
    scores = np.nan_to_num(scores, nan=0.0)
    labels = np.concatenate([np.ones(1000), np.zeros(1000)])
    auc = roc_auc_score(labels, scores)
    assert auc > 0.9, f"Fixed model failed to overfit: AUC = {auc:.4f}"
    print(f"  PASS: fixed model overfits (AUC = {auc:.4f})")


if __name__ == __name__:
    tests = [
        test_no_training_edge_touches_test_node,
        test_cold_start_positives_absent_from_training,
        test_cold_start_negatives_absent_from_full_graph,
        test_training_only_graph_stats,
        test_directionality,
        test_cold_start_node_splits_disjoint,
        test_fixed_model_overfits,
    ]

    print("=" * 60)
    print("COLD-START + GNN DIAGNOSTICS TESTS")
    print("=" * 60)
    passed = 0
    failed = 0
    for test in tests:
        try:
            print(f"\n{test.__name__}...")
            test()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")
