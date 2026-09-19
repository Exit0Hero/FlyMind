"""FlyMind configuration constants."""

import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT.parent / "dataset"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
REPORTS_DIR = RESULTS_DIR / "reports"
MODELS_DIR = PROJECT_ROOT / "models"

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

TARGET_COL = "super_class"

SAFE_FEATURES_NUMERIC = [
    "nt_type_score",
    "da_avg", "ser_avg", "gaba_avg", "glut_avg", "ach_avg", "oct_avg",
    "length_nm", "area_nm", "size_nm",
    "coord_x", "coord_y", "coord_z",
]

SAFE_FEATURES_CATEGORICAL = [
    "flow",
    "side_x",
]

EXCLUDED_FEATURES = [
    "root_id",
    "primary_type", "additional_type(s)",
    "class", "sub_class",
    "hemilineage", "nerve",
    "name", "group_x", "group_y",
    "connectivity_tags", "processed_labels",
    "type_x", "family", "subsystem", "category", "side_y",
    "hemisphere", "type_y", "column_id", "x", "y", "p", "q",
    "nt_type",
]

ALL_FEATURES = SAFE_FEATURES_NUMERIC + SAFE_FEATURES_CATEGORICAL
