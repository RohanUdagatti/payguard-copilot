"""Fraud model serving service.

Verifies model artifact SHA-256 integrity on startup, extracts point-in-time features
for a transaction from Postgres, scores using the calibrated pipeline, and explains
predictions using TreeSHAP in log-odds.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import shap

from payguard.config import Settings, get_settings
from payguard.ml.features import (
    ALLOWED_FEATURES,
    CATEGORICAL_FEATURES,
    compute_features_single,
    validate_feature_columns,
)

logger = logging.getLogger(__name__)


class ModelNotLoadedError(RuntimeError):
    """Raised when scoring is requested but no model is loaded."""


class ModelIntegrityError(ValueError):
    """Raised when model artifact SHA-256 hash does not match metadata."""


class TransactionNotFoundError(KeyError):
    """Raised when a requested transaction_id is not found in the database."""


class FraudModelService:
    """Service for loading and scoring transactions with the PayGuard fraud model."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.model_dir: Path = self.settings.model_dir
        self.model: Any = None
        self.calibrator: Any = None
        self.explainer: shap.TreeExplainer | None = None
        self.metadata: dict[str, Any] = {}
        self.threshold: float = 0.5
        self.model_version: str = "unknown"
        self.feature_names: list[str] = ALLOWED_FEATURES
        self.is_loaded: bool = False
        self._categories: dict[str, list[Any]] = {}

    def load_model(self) -> None:
        """Load the model artifact and verify SHA-256 integrity.

        Raises:
            FileNotFoundError: If the model file or metadata file is not found.
            ModelIntegrityError: If artifact SHA-256 hash mismatches metadata.
        """
        artifact_path = self.model_dir / "fraud_model.joblib"
        metadata_path = self.model_dir / "model_metadata.json"

        if not artifact_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                f"Model artifact or metadata missing in {self.model_dir}."
            )

        with open(metadata_path, encoding="utf-8") as f:
            self.metadata = json.load(f)

        expected_sha256 = self.metadata.get("artifact_sha256")
        if not expected_sha256:
            raise ModelIntegrityError("Metadata missing artifact_sha256 field.")

        # Compute actual SHA-256
        hasher = hashlib.sha256()
        with open(artifact_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        actual_sha256 = hasher.hexdigest()

        if actual_sha256 != expected_sha256:
            raise ModelIntegrityError(
                f"Model integrity violation! Expected SHA-256 {expected_sha256}, "
                f"got {actual_sha256}. Refusing to load corrupted or tampered model."
            )

        bundle = joblib.load(artifact_path)
        self.model = bundle["model"]
        self.calibrator = bundle["calibrator"]
        self.feature_names = bundle.get("feature_names", ALLOWED_FEATURES)
        self.threshold = float(
            self.metadata.get("threshold", bundle.get("threshold", 0.5))
        )
        self.model_version = str(self.metadata.get("model_version", "unknown"))

        # Initialize TreeExplainer
        self.explainer = shap.TreeExplainer(self.model)

        self.is_loaded = True
        logger.info(
            "fraud_model_loaded_successfully",
            extra={
                "model_version": self.model_version,
                "threshold": self.threshold,
                "artifact_sha256": actual_sha256,
            },
        )

    def _get_db_connection(self) -> Any:
        import psycopg

        s = self.settings
        return psycopg.connect(
            f"host={s.postgres_host} port={s.postgres_port} "
            f"dbname={s.postgres_db} user={s.postgres_user} "
            f"password={s.postgres_password}"
        )

    def score_transaction(
        self,
        txn_id: str,
        conn: Any = None,
    ) -> dict[str, Any]:
        """Score a transaction by loading it and strictly past history from PostgreSQL.

        Logs txn_id, score, and model_version only. No raw feature dumps, no secrets.
        """
        if not self.is_loaded or self.model is None or self.calibrator is None:
            raise ModelNotLoadedError("Fraud model is not loaded.")

        validate_feature_columns(self.feature_names)

        should_close_conn = False
        if conn is None:
            conn = self._get_db_connection()
            should_close_conn = True

        try:
            with conn.cursor() as cur:
                # 1. Fetch transaction details
                cur.execute(
                    """
                    SELECT 
                        t.transaction_id,
                        t.card_token,
                        t.merchant_id,
                        t.amount::float AS amount,
                        t.timestamp,
                        t.txn_hour,
                        t.channel,
                        t.mcc,
                        t.risk_tier AS merchant_risk_tier,
                        t.is_international,
                        (c.issuer_country != m.country) AS country_mismatch
                    FROM transactions t
                    JOIN cards c ON t.card_token = c.card_token
                    JOIN merchants m ON t.merchant_id = m.merchant_id
                    WHERE t.transaction_id = %s;
                    """,
                    (txn_id,),
                )
                row = cur.fetchone()
                if row is None:
                    raise TransactionNotFoundError(
                        f"Transaction {txn_id} not found in database."
                    )

                target_dict = {
                    "transaction_id": row[0],
                    "card_token": row[1],
                    "merchant_id": row[2],
                    "amount": row[3],
                    "timestamp": row[4],
                    "txn_hour": row[5],
                    "channel": row[6],
                    "mcc": row[7],
                    "merchant_risk_tier": row[8],
                    "is_international": row[9],
                    "country_mismatch": row[10],
                }

                # 2. Fetch strictly earlier history of same card using indexed query
                card_token = target_dict["card_token"]
                ts = target_dict["timestamp"]
                cur.execute(
                    """
                    SELECT
                        transaction_id,
                        merchant_id,
                        amount::float AS amount,
                        timestamp
                    FROM transactions
                    WHERE card_token = %s
                      AND timestamp <= %s
                      AND (timestamp < %s OR transaction_id < %s)
                    ORDER BY timestamp ASC, transaction_id ASC;
                    """,
                    (card_token, ts, ts, txn_id),
                )
                hist_rows = cur.fetchall()
                history_txns = [
                    {
                        "transaction_id": r[0],
                        "merchant_id": r[1],
                        "amount": r[2],
                        "timestamp": r[3],
                    }
                    for r in hist_rows
                ]
        finally:
            if should_close_conn:
                conn.close()

        # 3. Compute point-in-time features
        df_feat = compute_features_single(target_dict, history_txns)

        # Retain original raw values for top_features response
        raw_values: dict[str, Any] = df_feat.iloc[0].to_dict()

        # Cast categoricals to match model expectation
        for col in CATEGORICAL_FEATURES:
            df_feat[col] = df_feat[col].astype("category")

        # 4. Score
        uncalibrated_prob = float(self.model.predict_proba(df_feat)[:, 1][0])
        calibrated_score = float(self.calibrator.predict([uncalibrated_prob])[0])
        # Clip to [0, 1]
        calibrated_score = max(0.0, min(1.0, calibrated_score))
        label = 1 if calibrated_score >= self.threshold else 0

        # 5. SHAP values in log-odds of the base model
        assert self.explainer is not None
        shap_explanation = self.explainer(df_feat)
        shap_values_log_odds = shap_explanation.values[0]

        # Top-5 features by magnitude of SHAP contribution
        shap_contribs = []
        for feat_name, shap_val in zip(
            self.feature_names, shap_values_log_odds, strict=False
        ):
            shap_contribs.append(
                {
                    "feature": feat_name,
                    "value": raw_values[feat_name],
                    "shap": float(shap_val),
                    "abs_shap": abs(float(shap_val)),
                }
            )

        shap_contribs.sort(key=lambda x: x["abs_shap"], reverse=True)
        top_5_features = [
            {
                "feature": item["feature"],
                "value": item["value"],
                "shap": round(item["shap"], 4),
            }
            for item in shap_contribs[:5]
        ]

        # Audit logging: txn_id, score, model_version ONLY. No raw features, no secrets.
        logger.info(
            "ml_score_evaluated",
            extra={
                "txn_id": txn_id,
                "score": round(calibrated_score, 4),
                "model_version": self.model_version,
            },
        )

        return {
            "txn_id": txn_id,
            "score": round(calibrated_score, 4),
            "label": label,
            "threshold": round(self.threshold, 4),
            "top_features": top_5_features,
            "model_version": self.model_version,
        }
