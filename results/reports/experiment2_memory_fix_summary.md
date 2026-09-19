# Experiment 2: Link Prediction — Memory Fix Summary

## Root Cause of Memory Problem

The original code used Python `set` of `(int, int)` tuples for edge membership testing. For 3.7M edges, this consumed **~450 MB** per set. Multiple sets (positive edges, negative candidates, hard negatives) accumulated to **~2+ GB**, causing OOM on the 5.5 GB system.

## Files Changed

| File | Change |
|---|---|
| `src/link_prediction/negative_sampling.py` | Replaced Python tuple sets with compact sorted uint64 hash arrays (~30 MB vs ~450 MB). Added batch-rejection sampling. |
| `src/link_prediction/features.py` | Chunked `build_pair_features()` (50K batches), preventing 4x full-array allocation. |
| `src/link_prediction/data_prep.py` | Vectorized cold-start split with numpy searchsorted. Added `log_mem()` monitoring. |
| `src/link_prediction/run_part1.py` | `del` + `gc.collect()` after each phase. Computes RF heuristics before deleting pairs. Saves intermediates to disk. |
| `src/link_prediction/run_part2.py` | Full-batch GNN (139K nodes manageable on CPU). Xavier init, gradient clipping, NaN-safe training. |
| `tests/test_link_prediction.py` | Updated to match new memory-safe API. |

## Memory Profile

| Stage | Before | After |
|---|---|---|
| Edge index | ~450 MB (tuple set) | ~30 MB (uint64 array) |
| Negative samples | ~300 MB (tuple set) | ~30 MB (int64 array) |
| Pair features | ~900 MB (full allocation) | ~200 MB (chunked) |
| **Peak RSS** | **2+ GB -> OOM** | **2.3 GB (Part 1) / 1.1 GB (Part 2)** |

## Experiment 2 Results (Random Edge Holdout)

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| Random | 0.500 | 0.500 |
| Common Neighbors (out) | 0.813 | 0.812 |
| Total Common Neighbors | 0.900 | 0.899 |
| Preferential Attachment (in) | 0.830 | 0.843 |
| **RF (node features)** | **0.979** | **0.973** |
| **RF (node + heuristic)** | **0.990** | **0.989** |
| GraphSAGE | 0.544 | 0.556 |

## Verification

- 8/8 tests pass
- Experiment 1 model (`graphsage_superclass_best.pt`) intact
- Memory stable throughout execution
