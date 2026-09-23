"""Tests for the model artifact guard (integrity + feature contract)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.core.errors import ModelUnavailableError
from app.services.model_guard import (
    read_metadata,
    sha256_file,
    validate_loaded_model,
    verify_artifact_integrity,
)


class _RandomForestClassifierStub:
    """Fake estimator whose class name passes the RF classifier check."""

    def __init__(self):
        self.n_estimators = 100
        self.classes_ = [0, 1]


def _rf():
    return _RandomForestClassifierStub()


def _write(tmp_path: Path) -> Path:
    p = tmp_path / "artifact.bin"
    p.write_bytes(b"some-bytes-123")
    return p


def test_sha256_matches_hashlib(tmp_path):
    p = _write(tmp_path)
    assert sha256_file(p) == hashlib.sha256(b"some-bytes-123").hexdigest()


def test_missing_model_raises(tmp_path):
    with pytest.raises(ModelUnavailableError):
        verify_artifact_integrity(
            model_path=tmp_path / "nope.pkl",
            metadata_path=tmp_path / "nope.metadata.json",
        )


def test_checksum_mismatch_raises(tmp_path):
    p = _write(tmp_path)
    meta = tmp_path / "m.json"
    meta.write_text('{"artifact_hash_sha256": "' + "0" * 64 + '"}')
    with pytest.raises(ModelUnavailableError):
        verify_artifact_integrity(model_path=p, metadata_path=meta)


def test_checksum_match_ok(tmp_path):
    p = _write(tmp_path)
    meta = tmp_path / "m.json"
    meta.write_text(f'{{"artifact_hash_sha256": "{sha256_file(p)}"}}')
    summary = verify_artifact_integrity(model_path=p, metadata_path=meta)
    assert summary["checksum_verified"] is True
    assert summary["ok"] is True


def test_integrity_disabled_skips_checksum(tmp_path):
    p = _write(tmp_path)
    meta = tmp_path / "m.json"
    meta.write_text('{"artifact_hash_sha256": "' + "0" * 64 + '"}')
    summary = verify_artifact_integrity(
        model_path=p, metadata_path=meta, integrity_check_enabled=False
    )
    assert summary["checksum_verified"] is False
    assert "integrity check disabled" in " ".join(summary["notes"])


def test_missing_metadata_skips_checksum(tmp_path):
    p = _write(tmp_path)
    summary = verify_artifact_integrity(model_path=p, metadata_path=tmp_path / "no.json")
    assert summary["checksum_verified"] is False


def test_wrong_model_class_raises():
    inference = SimpleNamespace(_rf=object(), _feature_cols=list(range(15)), _X=None, feature_dim=60)
    with pytest.raises(ModelUnavailableError, match="not a RandomForest"):
        validate_loaded_model(inference)


def test_feature_count_mismatch_raises():
    rf = _rf()
    inference = SimpleNamespace(_rf=rf, _feature_cols=list(range(14)), _X=None, feature_dim=56)
    with pytest.raises(ModelUnavailableError, match="expected 15 node features, got 14"):
        validate_loaded_model(inference)


def test_feature_matrix_shape_mismatch_raises(tmp_path):
    import numpy as np

    rf = _rf()
    inference = SimpleNamespace(
        _rf=rf, _feature_cols=list(range(15)), _X=np.zeros((3, 14)), feature_dim=60,
    )
    with pytest.raises(ModelUnavailableError, match="expected 15 columns, got 14"):
        validate_loaded_model(inference)


def test_validation_ok():
    import numpy as np

    rf = _rf()
    cols = [f"f{i}" for i in range(15)]
    inference = SimpleNamespace(
        _rf=rf, _feature_cols=cols, _X=np.zeros((3, 15)), feature_dim=60,
    )
    report = validate_loaded_model(inference)
    assert report["ok"] is True
    assert report["node_features_loaded"] == 15


def test_read_metadata_defensive(tmp_path):
    assert read_metadata(tmp_path / "missing.json") == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert read_metadata(bad) == {}


def test_real_artifact_matches_metadata():
    """The shipped pickle must match the recorded SHA-256 (no model load)."""
    root = Path(__file__).resolve().parent.parent.parent
    model_path = root / "models" / "link_prediction_rf.pkl"
    metadata_path = root / "models" / "link_prediction_rf.metadata.json"
    if not model_path.exists():
        pytest.skip("model artifact not present")
    metadata = read_metadata(metadata_path)
    recorded = metadata.get("artifact_hash_sha256")
    if not recorded:
        pytest.skip("no recorded checksum in metadata")
    actual = sha256_file(model_path)
    assert actual == recorded.lower()