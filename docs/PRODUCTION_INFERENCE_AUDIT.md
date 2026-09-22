# FlyMind Production Inference Audit

**Date:** 2026-09-20
**Branch:** `feature/production-ml-inference`

## Current Model Loading

| Property | Value |
|----------|-------|
| Load mechanism | `pickle.load()` on first use (lazy) |
| Singleton | Module-level `_inference_instance` via `get_inference()` |
| Streamlit caching | `@st.cache_resource` on `get_datastore()` |
| Validation | `model_guard.validate_loaded_model` — model class, 15/60 feature contract, matrix shape |
| Hash verification | `model_guard.verify_artifact_integrity` — SHA-256 of the pickle vs metadata (`FLYMIND_ENABLE_MODEL_INTEGRITY_CHECK`) |

## Feature Construction

| Property | Value |
|----------|-------|
| Node features | 15 (13 numeric + 2 categorical label-encoded) |
| Pair features | `[x_A, x_B, |x_A-x_B|, x_A*x_B]` = 60 dimensions |
| NaN handling | `np.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)` on load |
| Missing neuron | Zero vector substituted |

## Feature Order (Verified from Trained Model)

```
['nt_type_score', 'da_avg', 'ser_avg', 'gaba_avg', 'glut_avg', 'ach_avg', 'oct_avg',
 'length_nm', 'area_nm', 'size_nm', 'coord_x', 'coord_y', 'coord_z', 'flow_enc', 'side_x_enc']
```

## UI Feature Name Mismatch (BUG FOUND)

`connection_predictor.py:104-106` and `candidate_ranking.py:153-155` use a hardcoded list:
```python
["length_nm", "area_nm", "size_nm", "coord_x", "coord_y", "coord_z",
 "nt_type_score", "ach_avg", "gaba_avg", "glut_avg", "da_avg", "ser_avg", "oct_avg",
 "synapse_count", "total_nt_score"]
```

This includes phantom features `synapse_count` and `total_nt_score` that don't exist in the 15-dim feature vector. **This is a display-only bug** — the actual model inference uses the correct features from the pickle's `feature_cols`.

## Observed Edge Checking

- Sorted int64 edge array built from `edges_aggregated.parquet`
- Binary search for membership testing
- Vectorized batch checking for candidate ranking

## Candidate Ranking

- Random sample 3x pool, filter known edges, top-k by score
- Chunked at 10,000 with explicit `del` for memory safety
- Self-loops excluded

## Neuron ID Validation

- Checked against `_id_to_idx` dict
- `KeyError` raised if missing
- UI pages catch `ValueError` for non-integer input

## Memory Concerns

| Issue | Status |
|-------|--------|
| Feature matrix | ~18 MB, acceptable |
| Sorted edges | Compact int64 array |
| Neuron table | Kept in memory after dict built (waste) |
| Duplicate edge loading | Edges loaded in both inference and presentation |
| Presentation singleton | Separate from inference singleton |

## Exception Handling

- `KeyError` for missing neurons — propagates raw
- No pickle validation
- No feature dimension assertion
- No structured exceptions

## Recommendations

1. Add model artifact validation (hash, type, dimension)
2. Add structured exceptions
3. Add feature contract validation
4. Add health check
5. Add logging
6. Fix UI feature name display
7. Eliminate duplicate edge loading
8. Release `_neuron_table` after dict built
