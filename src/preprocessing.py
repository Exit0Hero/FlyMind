"""Feature preprocessing and train/val/test splitting."""

import json
import pathlib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

from src.config import (
    SAFE_FEATURES_NUMERIC, SAFE_FEATURES_CATEGORICAL,
    TARGET_COL, SEED, ALL_FEATURES, PROCESSED_DIR,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
)


class Preprocessor:
    def __init__(self):
        self.num_imputer = SimpleImputer(strategy="median")
        self.cat_imputer = SimpleImputer(strategy="most_frequent")
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.target_encoder = LabelEncoder()
        self.feature_names = []
        self.is_fitted = False

    def fit_transform(self, df: pd.DataFrame, train_mask: np.ndarray):
        """Fit on training data, transform all data.

        train_mask: boolean array, True for training rows.
        """
        train_df = df[train_mask]

        # Impute numeric
        num_train = train_df[SAFE_FEATURES_NUMERIC].values.astype(np.float32)
        num_all = df[SAFE_FEATURES_NUMERIC].values.astype(np.float32)
        self.num_imputer.fit(num_train)
        num_all = self.num_imputer.transform(num_all)

        # Scale numeric (fit on train only)
        self.scaler.fit(num_train)
        num_all = self.scaler.transform(num_all)

        # Impute and encode categorical
        cat_parts = []
        for col in SAFE_FEATURES_CATEGORICAL:
            values = df[col].fillna("__MISSING__").values.astype(str)
            train_vals = train_df[col].fillna("__MISSING__").values.astype(str)

            le = LabelEncoder()
            le.fit(train_vals)

            # Handle unseen categories
            known = set(le.classes_)
            values = np.array([v if v in known else "__MISSING__" for v in values])
            encoded = le.transform(values).astype(np.float32).reshape(-1, 1)
            cat_parts.append(encoded)
            self.label_encoders[col] = le

        cat_all = np.hstack(cat_parts) if cat_parts else np.empty((len(df), 0), dtype=np.float32)

        # Combine features
        X = np.hstack([num_all, cat_all]).astype(np.float32)
        self.feature_names = SAFE_FEATURES_NUMERIC.copy()
        for col in SAFE_FEATURES_CATEGORICAL:
            self.feature_names.append(f"cat_{col}")

        # Target encoding
        target_values = df[TARGET_COL].values.astype(str)
        self.target_encoder.fit(target_values[train_mask])
        y = self.target_encoder.transform(target_values).astype(np.int64)

        self.is_fitted = True
        return X, y

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform using fitted preprocessor."""
        assert self.is_fitted, "Must call fit_transform first"

        num_all = df[SAFE_FEATURES_NUMERIC].values.astype(np.float32)
        num_all = self.num_imputer.transform(num_all)
        num_all = self.scaler.transform(num_all)

        cat_parts = []
        for col in SAFE_FEATURES_CATEGORICAL:
            values = df[col].fillna("__MISSING__").values.astype(str)
            known = set(self.label_encoders[col].classes_)
            values = np.array([v if v in known else "__MISSING__" for v in values])
            encoded = self.label_encoders[col].transform(values).astype(np.float32).reshape(-1, 1)
            cat_parts.append(encoded)

        cat_all = np.hstack(cat_parts) if cat_parts else np.empty((len(df), 0), dtype=np.float32)
        X = np.hstack([num_all, cat_all]).astype(np.float32)
        return X

    def save(self, path: pathlib.Path):
        """Save preprocessor configuration."""
        config = {
            "feature_names": self.feature_names,
            "target_classes": self.target_encoder.classes_.tolist(),
            "num_features": SAFE_FEATURES_NUMERIC,
            "cat_features": SAFE_FEATURES_CATEGORICAL,
            "num_imputer_statistics": self.num_imputer.statistics_.tolist(),
            "scaler_mean": self.scaler.mean_.tolist(),
            "scaler_scale": self.scaler.scale_.tolist(),
        }
        for col, le in self.label_encoders.items():
            config[f"cat_encoder_{col}"] = le.classes_.tolist()

        with open(path, "w") as f:
            json.dump(config, f, indent=2)


def create_split(df: pd.DataFrame, seed: int = SEED):
    """Create stratified train/val/test split (70/15/15) by super_class."""
    indices = np.arange(len(df))
    targets = df[TARGET_COL].values

    # First split: 85% train+val, 15% test
    idx_trainval, idx_test = train_test_split(
        indices, test_size=TEST_RATIO, random_state=seed, stratify=targets
    )

    # Second split: from trainval, 70/15 of total = 70/85 of trainval
    targets_trainval = targets[idx_trainval]
    relative_val = TEST_RATIO / (TRAIN_RATIO + TEST_RATIO)
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=relative_val, random_state=seed, stratify=targets_trainval
    )

    return idx_train, idx_val, idx_test
