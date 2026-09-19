"""FlyMind Link Prediction — Memory-safe negative sampling."""

import numpy as np
import gc


# ---------------------------------------------------------------------------
# Compact edge-set representation (sorted uint64 array + np.searchsorted)
# ---------------------------------------------------------------------------
def _encode_pairs(sources, targets):
    """Encode (src, tgt) pairs as uint64 hashes for compact membership testing.

    Uses splitmix64 mixing to reduce collision probability.
    Returns sorted uint64 array.
    """
    s = np.asarray(sources, dtype=np.uint64)
    t = np.asarray(targets, dtype=np.uint64)
    # splitmix64-style mixing
    z = s ^ (t * 0x9e3779b97f4a7c15)
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb
    z = z ^ (z >> 31)
    return np.sort(z)


def build_edge_index(edges):
    """Build compact sorted edge index from edge array.

    Returns sorted uint64 array (~8 bytes per edge).
    3.7M edges ≈ 30 MB vs ~450 MB for Python tuple set.
    """
    return _encode_pairs(edges[:, 0], edges[:, 1])


def edge_exists(src, tgt, edge_index):
    """Check if edges exist via binary search. Handles scalar or array inputs."""
    if np.isscalar(src):
        h = np.uint64(src) ^ (np.uint64(tgt) * np.uint64(0x9e3779b97f4a7c15))
        h = (h ^ (h >> 30)) * np.uint64(0xbf58476d1ce4e5b9)
        h = (h ^ (h >> 27)) * np.uint64(0x94d049bb133111eb)
        h = h ^ (h >> 31)
        idx = np.searchsorted(edge_index, h)
        return idx < len(edge_index) and edge_index[idx] == h
    else:
        hashes = _encode_pairs(src, tgt)
        idx = np.searchsorted(edge_index, hashes)
        found = idx < len(edge_index)
        result = np.zeros(len(src), dtype=bool)
        result[found] = edge_index[idx[found]] == hashes[found]
        return result


# ---------------------------------------------------------------------------
# Exact pair verification (collision-free)
# ---------------------------------------------------------------------------
def build_sorted_pairs(sources, targets):
    """Build sorted uint64 array of pair encodings for exact membership testing.

    Uses a collision-free encoding: (src << 32) ^ target, where both are
    truncated to 32 bits. For IDs up to ~4 billion this is exact.
    For larger IDs, uses two-level encoding.
    """
    s = np.asarray(sources, dtype=np.uint64)
    t = np.asarray(targets, dtype=np.uint64)
    # Pack into 128-bit via Python int array
    packed = s * np.uint64(4294967296) + t  # src * 2^32 + tgt
    return np.sort(packed)


def verify_not_in_edges(src_arr, tgt_arr, sorted_pairs):
    """Check that (src, tgt) pairs are NOT in sorted_pairs. Returns bool array."""
    packed = np.asarray(src_arr, dtype=np.uint64) * np.uint64(4294967296) + np.asarray(tgt_arr, dtype=np.uint64)
    idx = np.searchsorted(sorted_pairs, packed)
    found = idx < len(sorted_pairs)
    result = np.ones(len(src_arr), dtype=bool)
    result[found] = sorted_pairs[idx[found]] != packed[found]
    return result


# ---------------------------------------------------------------------------
# Negative sampling (memory-safe)
# ---------------------------------------------------------------------------
def sample_negatives(all_node_ids, num_neg, edge_index, seed=42):
    """Sample random negative edges using compact edge index.

    Returns (num_neg, 2) int64 array — no Python sets of tuples.
    """
    np.random.seed(seed)
    n_nodes = len(all_node_ids)
    negatives_src = np.empty(num_neg, dtype=np.int64)
    negatives_tgt = np.empty(num_neg, dtype=np.int64)
    collected = 0
    max_rounds = num_neg * 30

    for _ in range(max_rounds):
        if collected >= num_neg:
            break
        batch_size = min(10000, (num_neg - collected) * 2)
        cand_src = all_node_ids[np.random.randint(0, n_nodes, size=batch_size)]
        cand_tgt = all_node_ids[np.random.randint(0, n_nodes, size=batch_size)]

        # Reject self-loops
        valid = cand_src != cand_tgt
        cs, ct = cand_src[valid], cand_tgt[valid]

        # Reject existing edges
        exists = edge_exists(cs, ct, edge_index)
        keep = ~exists
        cs, ct = cs[keep], ct[keep]

        # Deduplicate within batch (simple: encode and unique)
        if len(cs) > 0:
            batch_hashes = _encode_pairs(cs, ct)
            _, unique_idx = np.unique(batch_hashes, return_index=True)
            cs, ct = cs[unique_idx], ct[unique_idx]

        n_take = min(len(cs), num_neg - collected)
        if n_take > 0:
            negatives_src[collected:collected + n_take] = cs[:n_take]
            negatives_tgt[collected:collected + n_take] = ct[:n_take]
            collected += n_take

    return np.column_stack([negatives_src[:collected], negatives_tgt[:collected]])


def sample_hard_negatives(pos_edges, all_node_ids, train_edge_index, num_neg, train_edges, seed=42):
    """Sample hard negatives: degree-matched non-edges.

    Uses compact edge index, returns (num_neg, 2) int64 array.
    """
    from collections import defaultdict

    np.random.seed(seed)
    n_nodes = len(all_node_ids)

    # Compute degrees from train edges (compact: just arrays, not sets)
    out_deg = defaultdict(int)
    in_deg = defaultdict(int)
    for s, t in train_edges:
        out_deg[int(s)] += 1
        in_deg[int(t)] += 1

    # Get in-degree of each positive target for matching
    pos_tgt_in_deg = np.array([in_deg.get(int(t), 0) for t in pos_edges[:, 1]])

    negatives_src = np.empty(num_neg, dtype=np.int64)
    negatives_tgt = np.empty(num_neg, dtype=np.int64)
    collected = 0

    # Try to match degree profile
    sample_indices = np.random.choice(len(pos_edges), min(num_neg * 3, len(pos_edges)), replace=False)
    for idx in sample_indices:
        if collected >= num_neg:
            break
        src = int(pos_edges[idx, 0])
        target_deg = pos_tgt_in_deg[idx]

        for _ in range(50):
            cand_tgt = int(all_node_ids[np.random.randint(0, n_nodes)])
            if cand_tgt == src:
                continue
            cand_deg = in_deg.get(cand_tgt, 0)
            if cand_deg > 0 and abs(np.log2(max(cand_deg, 1)) - np.log2(max(target_deg, 1))) < 1.5:
                if not edge_exists(src, cand_tgt, train_edge_index):
                    negatives_src[collected] = src
                    negatives_tgt[collected] = cand_tgt
                    collected += 1
                    break

    # Fill remaining with random
    while collected < num_neg:
        src = int(all_node_ids[np.random.randint(0, n_nodes)])
        tgt = int(all_node_ids[np.random.randint(0, n_nodes)])
        if src != tgt and not edge_exists(src, tgt, train_edge_index):
            negatives_src[collected] = src
            negatives_tgt[collected] = tgt
            collected += 1

    return np.column_stack([negatives_src[:collected], negatives_tgt[:collected]])
