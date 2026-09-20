"""FlyMind Production Inference — Configuration with environment overrides."""

import os
import pathlib
from typing import Optional

# Project root
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

# Data paths (environment overrides)
DATA_ROOT = os.environ.get(
    "FLYMIND_DATA_ROOT",
    str(PROJECT_ROOT / "data" / "processed")
)
PROCESSED_DIR = pathlib.Path(DATA_ROOT)
LP_DIR = PROCESSED_DIR / "link_prediction"
MODELS_DIR = pathlib.Path(os.environ.get(
    "FLYMIND_MODEL_DIR",
    str(PROJECT_ROOT / "models")
))

# Model settings
MODEL_VERSION = os.environ.get("FLYMIND_MODEL_VERSION", "FlyMind-RF v1.0.0")
MODEL_FILENAME = os.environ.get("FLYMIND_MODEL_FILENAME", "link_prediction_rf.pkl")

# Candidate ranking limits
DEFAULT_CANDIDATE_COUNT = int(os.environ.get("FLYMIND_DEFAULT_CANDIDATES", "100"))
MAX_CANDIDATE_COUNT = int(os.environ.get("FLYMIND_MAX_CANDIDATES", "1000"))

# Logging
LOG_LEVEL = os.environ.get("FLYMIND_LOG_LEVEL", "INFO")

# Feature contract version
FEATURE_CONTRACT_VERSION = "1.0.0"

# Application version
APP_VERSION = "1.0.0"
