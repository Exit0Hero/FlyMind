"""FlyMind Link Prediction — Configuration."""

import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
LP_DIR = PROJECT_ROOT / "data" / "processed" / "link_prediction"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures" / "link_prediction"
REPORTS_DIR = PROJECT_ROOT / "results" / "reports"
MODELS_DIR = PROJECT_ROOT / "models"

SEED = 42

# Feature columns (safe, non-leaking)
NUMERIC_FEATURES = [
    "nt_type_score",
    "da_avg", "ser_avg", "gaba_avg", "glut_avg", "ach_avg", "oct_avg",
    "length_nm", "area_nm", "size_nm",
    "coord_x", "coord_y", "coord_z",
]
CATEGORICAL_FEATURES = ["flow", "side_x"]

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Negative sampling
NEG_RATIO_RANDOM = 1  # 1:1 for random
NEG_RATIO_HARD = 5    # 1:5 for hard negatives

# Cold-start
COLD_TRAIN_RATIO = 0.70
COLD_VAL_RATIO = 0.15
COLD_TEST_RATIO = 0.15

# Evaluation
EVAL_K_VALUES = [100, 1000, 5000]

# GNN
GNN_HIDDEN_DIM = 64
GNN_NUM_LAYERS = 2
GNN_DROPOUT = 0.3
GNN_LR = 0.001
GNN_EPOCHS = 50
GNN_BATCH_SIZE = 2048
