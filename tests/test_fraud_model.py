"""Unit and integration tests for PayGuard Phase 5 fraud model service."""

from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from fastapi.testclient import TestClient
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

from payguard.config import Settings
from payguard.gateway.main import create_app
from payguard.ml.features import (
    ALLOWED_FEATURES,
    FORBIDDEN_FEATURES,
    compute_features_batch,
    compute_features_single,
    validate_feature_columns,
)
from payguard.ml.pipeline import (
    compute_cost,
    compute_deterministic_version,
    find_optimal_threshold,
    load_dataset_from_db,
    prepare_dataset_splits,
)
from payguard.ml.service import (
    FraudModelService,
    ModelIntegrityError,
    TransactionNotFoundError,
)


def test_forbidden_feature_guard() -> None:
    """Validate that forbidden columns and target leakage columns raise ValueError."""
    for forbidden in FORBIDDEN_FEATURES:
        with pytest.raises(ValueError, match="Data leakage detected"):
            validate_feature_columns([forbidden, "amount", "txn_hour"])

    with pytest.raises(ValueError, match="Unapproved column"):
        validate_feature_columns(["unapproved_custom_column"])

    # Allowed features should pass cleanly
    validate_feature_columns(ALLOWED_FEATURES)


def test_time_split_ordering() -> None:
    """Time-based split must be strictly chronological: train < val_a < val_b < test."""
    n = 1000
    base_time = datetime.datetime(2026, 8, 1, 0, 0, 0, tzinfo=datetime.UTC)
    timestamps = [base_time + datetime.timedelta(minutes=i) for i in range(n)]

    synthetic_df = pd.DataFrame(
        {
            "transaction_id": [f"txn_{i:04d}" for i in range(n)],
            "card_token": [f"tok_{i % 50}" for i in range(n)],
            "merchant_id": [f"merch_{i % 10}" for i in range(n)],
            "amount": [50.0 + (i % 20) for i in range(n)],
            "timestamp": timestamps,
            "txn_hour": [ts.hour for ts in timestamps],
            "channel": ["POS" if i % 2 == 0 else "ECOMMERCE" for i in range(n)],
            "mcc": ["5411" for _ in range(n)],
            "merchant_risk_tier": ["LOW" if i % 3 == 0 else "HIGH" for i in range(n)],
            "is_international": [0 for _ in range(n)],
            "country_mismatch": [0 for _ in range(n)],
            "is_fraud": [1 if i % 100 == 0 else 0 for i in range(n)],
        }
    )

    splits = prepare_dataset_splits(synthetic_df)

    assert len(splits.train_df) == 700
    assert len(splits.val_a_df) == 75
    assert len(splits.val_b_df) == 75
    assert len(splits.test_df) == 150

    # Ensure strict chronological ordering with no overlaps
    assert splits.train_df["timestamp"].max() <= splits.val_a_df["timestamp"].min()
    assert splits.val_a_df["timestamp"].max() <= splits.val_b_df["timestamp"].min()
    assert splits.val_b_df["timestamp"].max() <= splits.test_df["timestamp"].min()


def test_feature_leakage_point_in_time_vs_batch() -> None:
    """Batch features MUST equal serving features for >= 200 transactions."""
    try:
        raw_df = load_dataset_from_db(limit=2000)
    except Exception:
        pytest.skip("Postgres database not accessible for feature leakage test.")

    batch_df = compute_features_batch(raw_df)

    # Pick 200 random transactions to evaluate against point-in-time extraction
    rng = np.random.default_rng(1234)
    sample_indices = rng.choice(len(raw_df), size=min(250, len(raw_df)), replace=False)

    for idx in sample_indices:
        target = raw_df.iloc[idx].to_dict()
        card = target["card_token"]

        # Only rows strictly before this transaction in the sorted dataset
        hist = raw_df.iloc[:idx]
        hist_card = hist[hist["card_token"] == card].to_dict(orient="records")

        single_feat = compute_features_single(target, hist_card)
        batch_feat = batch_df.iloc[[idx]][ALLOWED_FEATURES]

        for col in ALLOWED_FEATURES:
            b_val = batch_feat[col].values[0]
            s_val = single_feat[col].values[0]
            if isinstance(b_val, int | float | np.floating | np.integer):
                msg = f"Mismatch in '{col}' at {idx}: batch={b_val}, single={s_val}"
                assert np.isclose(b_val, s_val, atol=1e-5), msg
            else:
                msg = f"Mismatch in '{col}' at {idx}: batch={b_val}, single={s_val}"
                assert b_val == s_val, msg


def test_threshold_selection_toy_cost_matrix() -> None:
    """Optimal threshold selection minimizes FN cost (amount) and FP cost ($25)."""
    # 3 samples:
    # Sample 0: prob=0.1, y=0, amount=$100
    # Sample 1: prob=0.4, y=1, amount=$300
    # Sample 2: prob=0.8, y=1, amount=$1000
    y_true = np.array([0, 1, 1])
    y_probs = np.array([0.1, 0.4, 0.8])
    amounts = np.array([100.0, 300.0, 1000.0])
    false_alarm_cost = 25.0

    # If threshold = 0.9 (Flag nothing):
    # FNs: sample 1 ($300) + sample 2 ($1000) = $1300 cost
    # If threshold = 0.05 (Flag everything):
    # FPs: sample 0 (1 * $25 = $25), FNs: 0 -> Cost: $25
    # If threshold = 0.3:
    # Flagged: sample 1 (TP), sample 2 (TP). Sample 0: TN.
    # Total cost = 0 FNs + 0 FPs = $0.00!
    best_th = find_optimal_threshold(y_true, y_probs, amounts, false_alarm_cost)
    cost_m = compute_cost(y_true, y_probs, amounts, best_th, false_alarm_cost)

    assert cost_m.total_cost == 0.0
    assert 0.1 < best_th <= 0.4


def test_calibration_improves_or_maintains_brier() -> None:
    """Isotonic regression calibration improves or maintains Brier score on fixture."""
    rng = np.random.default_rng(42)
    y_true = rng.binomial(1, 0.1, size=500)
    # Poorly calibrated / overconfident predictions
    uncalibrated_probs = np.clip(
        y_true * 0.7 + rng.uniform(0.1, 0.5, size=500), 0.01, 0.99
    )

    brier_before = float(brier_score_loss(y_true, uncalibrated_probs))

    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(uncalibrated_probs, y_true)
    calibrated_probs = calibrator.predict(uncalibrated_probs)

    brier_after = float(brier_score_loss(y_true, calibrated_probs))

    assert brier_after <= brier_before


def test_model_version_deterministic() -> None:
    """Model version hash is completely deterministic for identical inputs."""
    min_ts = "2026-08-01 00:00:00+00:00"
    max_ts = "2026-10-29 23:59:59+00:00"
    n_rows = 500000
    features = ALLOWED_FEATURES
    params = {"n_estimators": 120, "max_depth": 5, "learning_rate": 0.08}

    v1 = compute_deterministic_version(min_ts, max_ts, n_rows, features, params)
    v2 = compute_deterministic_version(min_ts, max_ts, n_rows, features, params)
    assert v1 == v2
    assert len(v1) == 16

    # Change one param -> different version
    params2 = {"n_estimators": 121, "max_depth": 5, "learning_rate": 0.08}
    v3 = compute_deterministic_version(min_ts, max_ts, n_rows, features, params2)
    assert v1 != v3


def test_model_integrity_sha256_mismatch(tmp_path: Path) -> None:
    """Tampering with the model artifact triggers ModelIntegrityError."""
    # Create a tiny toy model and save it
    model = xgb.XGBClassifier(
        n_estimators=2, max_depth=2, enable_categorical=True, random_state=42
    )
    df = pd.DataFrame(
        {
            "txn_hour": [1, 2],
            "day_of_week": [0, 1],
            "amount": [10.0, 20.0],
            "log_amount": [2.3, 3.0],
            "channel": pd.Categorical(["POS", "ATM"]),
            "mcc": pd.Categorical(["5411", "5411"]),
            "merchant_risk_tier": pd.Categorical(["LOW", "HIGH"]),
            "is_international": [0, 1],
            "country_mismatch": [0, 1],
            "txn_count_1h": [0, 1],
            "txn_count_24h": [0, 1],
            "distinct_merchants_24h": [0, 1],
            "is_new_merchant_for_card": [1, 0],
            "amount_past_mean_zscore": [0.0, 1.2],
        }
    )
    y = np.array([0, 1])
    model.fit(df, y)

    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit([0.2, 0.8], y)

    bundle = {
        "model": model,
        "calibrator": calibrator,
        "feature_names": ALLOWED_FEATURES,
        "threshold": 0.5,
    }
    artifact_path = tmp_path / "fraud_model.joblib"
    joblib.dump(bundle, artifact_path)

    sha256_hash = hashlib.sha256(artifact_path.read_bytes()).hexdigest()

    metadata = {
        "model_version": "test_version_1234",
        "artifact_sha256": sha256_hash,
        "threshold": 0.5,
        "features": ALLOWED_FEATURES,
    }
    with open(tmp_path / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f)

    settings = Settings(model_dir=tmp_path)
    service = FraudModelService(settings)
    service.load_model()
    assert service.is_loaded

    # Tamper with the artifact: append arbitrary byte
    with open(artifact_path, "ab") as f:
        f.write(b"\x00")

    service2 = FraudModelService(settings)
    with pytest.raises(ModelIntegrityError, match="Model integrity violation"):
        service2.load_model()


def test_ml_score_endpoint_and_guards() -> None:
    """Test /ml/score responses: 400 for PAN, 503 uninitialized, 404 unknown, 200."""
    app = create_app()
    client = TestClient(app)

    # 1. Test 400 rejection for card number (Luhn valid)
    resp_luhn = client.post("/ml/score", json={"txn_id": "4111111111111111"})
    assert resp_luhn.status_code == 400
    assert "potential card number" in resp_luhn.json()["detail"]

    # 2. Test 503 when model is not loaded
    app.state.fraud_model_service = None
    # Provide a non-existent model dir to ensure load_model fails
    app.state.settings = Settings(model_dir=Path("/non_existent_path_xyz"))
    resp_503 = client.post("/ml/score", json={"txn_id": "txn_0000000001_test"})
    assert resp_503.status_code == 503

    # 3. Test 404 for unknown transaction ID with a loaded service mock
    class MockService:
        is_loaded = True
        model_version = "v_mock"

        def score_transaction(self, txn_id: str) -> dict[str, Any]:
            if txn_id == "txn_unknown":
                raise TransactionNotFoundError("Not found")
            return {
                "txn_id": txn_id,
                "score": 0.1234,
                "label": 0,
                "threshold": 0.35,
                "top_features": [
                    {"feature": "amount", "value": 45.0, "shap": 0.5},
                    {"feature": "channel", "value": "POS", "shap": -0.3},
                ],
                "model_version": "v_mock",
            }

    app.state.fraud_model_service = MockService()

    resp_404 = client.post("/ml/score", json={"txn_id": "txn_unknown"})
    assert resp_404.status_code == 404

    # 4. Test 200 for known transaction
    resp_200 = client.post("/ml/score", json={"txn_id": "txn_known_001"})
    assert resp_200.status_code == 200
    data = resp_200.json()
    assert data["txn_id"] == "txn_known_001"
    assert data["score"] == 0.1234
    assert data["label"] == 0
    assert data["threshold"] == 0.35
    assert len(data["top_features"]) == 2
    assert data["top_features"][0]["feature"] == "amount"
    assert data["model_version"] == "v_mock"
