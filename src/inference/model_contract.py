"""FlyMind Model Contract — Immutable production model specification."""

from dataclasses import dataclass
from typing import List

MODEL_VERSION = "FlyMind-RF v1.0.0"

# Verified against models/link_prediction_rf.pkl
MODEL_TYPE = "RandomForestClassifier"
EXPECTED_ESTIMATORS = 100
NODE_FEATURE_COUNT = 15
PAIR_FEATURE_COUNT = 60
RANDOM_SEED = 42
N_CLASSES = 2

# Canonical feature order — verified from trained model feature_cols
# DO NOT reorder. Production inference MUST use this exact ordering.
CANONICAL_NODE_FEATURES: List[str] = [
    "nt_type_score",
    "da_avg",
    "ser_avg",
    "gaba_avg",
    "glut_avg",
    "ach_avg",
    "oct_avg",
    "length_nm",
    "area_nm",
    "size_nm",
    "coord_x",
    "coord_y",
    "coord_z",
    "flow_enc",
    "side_x_enc",
]

# Categorical features (label-encoded in training)
CATEGORICAL_FEATURES = ["flow", "side_x"]

# Numeric features (used as-is)
NUMERIC_FEATURES = [f for f in CANONICAL_NODE_FEATURES if f not in CATEGORICAL_FEATURES]

# Pair feature dimensions
PAIR_FEATURE_NAMES = [
    *[f"src_{f}" for f in CANONICAL_NODE_FEATURES],
    *[f"tgt_{f}" for f in CANONICAL_NODE_FEATURES],
    *[f"abs_diff_{f}" for f in CANONICAL_NODE_FEATURES],
    *[f"product_{f}" for f in CANONICAL_NODE_FEATURES],
]

# Candidate ranking limits
DEFAULT_CANDIDATE_COUNT = 100
MAX_CANDIDATE_COUNT = 1000
CANDIDATE_POOL_MULTIPLIER = 3

# Artifact SHA-256 hash (verified)
EXPECTED_ARTIFACT_HASH = "c6cdf6ca49d9216b8a7eff8e2b90cf9c0bfa8f0619fc1b5bfed74c7b65180316"

# Feature contract version
FEATURE_CONTRACT_VERSION = "1.0.0"


@dataclass(frozen=True)
class ModelContract:
    """Immutable specification of the frozen production model."""
    model_version: str = MODEL_VERSION
    model_type: str = MODEL_TYPE
    estimators: int = EXPECTED_ESTIMATORS
    node_feature_count: int = NODE_FEATURE_COUNT
    pair_feature_count: int = PAIR_FEATURE_COUNT
    random_seed: int = RANDOM_SEED
    n_classes: int = N_CLASSES
    canonical_features: tuple = tuple(CANONICAL_NODE_FEATURES)
    expected_hash: str = EXPECTED_ARTIFACT_HASH
    feature_contract_version: str = FEATURE_CONTRACT_VERSION
