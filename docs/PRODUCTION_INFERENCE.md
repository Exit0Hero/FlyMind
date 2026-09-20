# FlyMind Production Inference

**Version:** FlyMind-RF v1.0.0
**Status:** Production-hardened, not a new scientific validation

---

## 1. Architecture

```
Streamlit / Application
        ↓
FlyMindInferenceService
        ↓
FeatureContract (60D)
        ↓
ModelLoader (hash-verified)
        ↓
Frozen RF Artifact (link_prediction_rf.pkl)
```

One source of truth for inference. All paths go through `FlyMindInferenceService`.

## 2. Model Artifact

| Property | Value |
|----------|-------|
| File | `models/link_prediction_rf.pkl` |
| Type | RandomForestClassifier |
| Estimators | 100 |
| Feature dimension | 60 (pair features) |
| Node features | 15 |
| SHA-256 | `c6cdf6ca49d9216b8a7eff8e2b90cf9c0bfa8f0619fc1b5bfed74c7b65180316` |
| Version | FlyMind-RF v1.0.0 |

The artifact is trusted, version-controlled project content. Python pickle is inherently unsafe for untrusted files — never load user-uploaded models.

## 3. Model Contract

Defined in `src/inference/model_contract.py`:

- Model type: RandomForestClassifier
- Estimators: 100
- Node features: 15
- Pair features: 60
- Random seed: 42
- Feature contract version: 1.0.0

## 4. Feature Contract

**Node features (15):**

```
['nt_type_score', 'da_avg', 'ser_avg', 'gaba_avg', 'glut_avg', 'ach_avg', 'oct_avg',
 'length_nm', 'area_nm', 'size_nm', 'coord_x', 'coord_y', 'coord_z', 'flow_enc', 'side_x_enc']
```

**Pair representation (60):**

```
[x_source(15), x_target(15), |x_source-x_target|(15), x_source*x_target(15)]
```

Production inference asserts `n_features == 60` before calling the model. If not, it fails.

## 5. Input Validation

| Check | Behavior |
|-------|----------|
| Neuron ID not found | `InvalidNeuronIDError` |
| NaN in features | `FeatureContractError` |
| Inf in features | `FeatureContractError` |
| Wrong feature count | `FeatureContractError` |
| Candidate count > max | `CandidateLimitError` |
| Model artifact mismatch | `ModelArtifactMismatchError` |

## 6. Prediction Flow

1. Validate source and target neuron IDs exist
2. Look up 15-dimensional feature vectors
3. Construct 60-dimensional pair features
4. Validate finite numeric values
5. Run frozen Random Forest `predict_proba`
6. Return model score (0-1)
7. Check whether pair is an observed edge

**Response fields:** `source_root_id`, `target_root_id`, `model_score`, `observed_in_dataset`, `model_version`

## 7. Candidate Ranking

1. Validate source neuron
2. Validate requested count (max 1000)
3. Sample 3x pool from all nodes
4. Exclude self and known observed edges
5. Score in bounded chunks (10,000)
6. Sort by descending score
7. Return top-k with metadata

## 8. Error Handling

Structured exceptions in `src/inference/exceptions.py`:

- `FlyMindError` (base)
- `ModelLoadError`
- `ModelArtifactMismatchError`
- `FeatureContractError`
- `InvalidNeuronIDError`
- `InferenceError`
- `CandidateLimitError`

No stack traces exposed to users. Internal logs retain diagnostic information.

## 9. Memory Safety

- No all-pairs matrix
- Bounded chunks (10,000)
- Explicit cleanup of temporary arrays
- Feature matrix loaded once
- Sorted edge array built once
- Neuron table released after dict built

## 10. Caching

- Model loaded once via lazy initialization
- Feature matrix cached in memory
- Edge index cached in memory
- Streamlit `@st.cache_resource` for DataStore singleton

## 11. Health Checks

`check_model_health()` verifies:

- Model exists and loads
- Metadata matches contract
- Feature dimension is correct
- Artifact hash is valid
- Required processed data files exist
- Smoke test prediction succeeds

## 12. Logging

Structured logging via Python `logging` module:

- Model loaded with timing
- Inference requests with source/target IDs
- Candidate ranking with count and duration
- Errors with context

## 13. Security Considerations

- Model artifact is trusted, version-controlled content
- No user-uploaded pickle loading
- No secrets or credentials
- No path traversal
- Bounded candidate enumeration
- No traceback leakage to users

## 14. Deployment Considerations

- Single-process Streamlit app
- No external API service required
- Environment variables for configuration:
  - `FLYMIND_DATA_ROOT`
  - `FLYMIND_MODEL_DIR`
  - `FLYMIND_MODEL_VERSION`
  - `FLYMIND_MAX_CANDIDATES`
  - `FLYMIND_LOG_LEVEL`

## 15. Known Limitations

- Dataset-specific to FlyWire FAFB
- 14.1% of neurons lack neurotransmitter annotations
- Scores are model rankings, not biological probabilities
- Candidate connections require experimental validation
- Production hardening does not constitute a new scientific validation
