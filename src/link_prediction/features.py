"""FlyMind Link Prediction — Pair features and graph heuristics (memory-safe)."""

import numpy as np
from collections import defaultdict


# ---------------------------------------------------------------------------
# Pair features — chunked to control peak memory
# ---------------------------------------------------------------------------
def build_pair_features(X, pairs, id_to_idx, chunk_size=50000):
    """Build pair features: [x_A, x_B, |x_A-x_B|, x_A*x_B].

    Processes in chunks to avoid allocating 4× full arrays simultaneously.
    Returns (n_pairs, 4*n_feat) float32 array.
    """
    n_feat = X.shape[1]
    n_pairs = len(pairs)
    result = np.empty((n_pairs, 4 * n_feat), dtype=np.float32)

    idx_a = np.array([id_to_idx.get(int(p), -1) for p in pairs[:, 0]], dtype=np.int64)
    idx_b = np.array([id_to_idx.get(int(p), -1) for p in pairs[:, 1]], dtype=np.int64)

    for start in range(0, n_pairs, chunk_size):
        end = min(start + chunk_size, n_pairs)
        ia = idx_a[start:end]
        ib = idx_b[start:end]
        valid = (ia >= 0) & (ib >= 0)

        xa = np.zeros((end - start, n_feat), dtype=np.float32)
        xb = np.zeros((end - start, n_feat), dtype=np.float32)
        xa[valid] = X[ia[valid]]
        xb[valid] = X[ib[valid]]

        result[start:end] = np.hstack([xa, xb, np.abs(xa - xb), xa * xb])

    return result


# ---------------------------------------------------------------------------
# Edge adjacency dicts (needed for heuristics)
# ---------------------------------------------------------------------------
def build_edge_dicts(train_edges):
    """Build adjacency dicts from training edges only."""
    out_edges = defaultdict(set)
    in_edges = defaultdict(set)
    for s, t in train_edges:
        s, t = int(s), int(t)
        out_edges[s].add(t)
        in_edges[t].add(s)
    return out_edges, in_edges


# ---------------------------------------------------------------------------
# Graph heuristics — vectorized where possible
# ---------------------------------------------------------------------------
def compute_heuristics(pairs, out_edges, in_edges, chunk_size=50000):
    """Compute directed graph heuristics for candidate pairs.

    Returns: (n_pairs, 7) array with:
      common_out, common_in, total_common,
      jaccard_out, jaccard_in,
      pa_out, pa_in
    """
    n = len(pairs)
    result = np.zeros((n, 7), dtype=np.float32)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = pairs[start:end]
        m = end - start

        common_out = np.zeros(m, dtype=np.float32)
        common_in = np.zeros(m, dtype=np.float32)
        union_out = np.ones(m, dtype=np.float32)
        union_in = np.ones(m, dtype=np.float32)
        out_deg_a = np.zeros(m, dtype=np.float32)
        out_deg_b = np.zeros(m, dtype=np.float32)
        in_deg_a = np.zeros(m, dtype=np.float32)
        in_deg_b = np.zeros(m, dtype=np.float32)

        for i in range(m):
            a, b = int(chunk[i, 0]), int(chunk[i, 1])

            o_a = out_edges.get(a, set())
            o_b = out_edges.get(b, set())
            i_a = in_edges.get(a, set())
            i_b = in_edges.get(b, set())

            common_out[i] = len(o_a & o_b)
            common_in[i] = len(i_a & i_b)

            u_out = len(o_a | o_b) if (o_a or o_b) else 0
            union_out[i] = u_out if u_out > 0 else 1
            u_in = len(i_a | i_b) if (i_a or i_b) else 0
            union_in[i] = u_in if u_in > 0 else 1

            out_deg_a[i] = len(o_a)
            out_deg_b[i] = len(o_b)
            in_deg_a[i] = len(i_a)
            in_deg_b[i] = len(i_b)

        total_common = common_out + common_in
        jaccard_out = common_out / np.maximum(union_out, 1)
        jaccard_in = common_in / np.maximum(union_in, 1)
        pa_out = out_deg_a * out_deg_b
        pa_in = in_deg_a * in_deg_b

        result[start:end] = np.stack([
            common_out, common_in, total_common,
            jaccard_out, jaccard_in,
            pa_out, pa_in,
        ], axis=1)

    return result
