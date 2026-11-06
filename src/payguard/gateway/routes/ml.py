"""Machine learning scoring endpoints for fraud detection."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from payguard.ml.service import (
    FraudModelService,
    ModelIntegrityError,
    ModelNotLoadedError,
    TransactionNotFoundError,
)
from payguard.rag.security_checks import contains_luhn_pan

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ml", tags=["ml"])


class MLScoreRequest(BaseModel):
    """Transaction scoring request payload."""

    txn_id: str = Field(..., description="Unique transaction ID to score.")


class TopFeatureExplanation(BaseModel):
    """Top feature explanation with original raw value and log-odds SHAP value."""

    feature: str
    value: Any
    shap: float


class MLScoreResponse(BaseModel):
    """Fraud score response with calibrated probability and top SHAP explanations."""

    txn_id: str
    score: float
    label: int
    threshold: float
    top_features: list[TopFeatureExplanation]
    model_version: str


def _get_ml_service(request: Request) -> FraudModelService:
    """Retrieve the FraudModelService singleton from the FastAPI application state."""
    service = getattr(request.app.state, "fraud_model_service", None)
    if service is None or not service.is_loaded:
        # Attempt to load if not initialized
        try:
            service = FraudModelService(request.app.state.settings)
            service.load_model()
            request.app.state.fraud_model_service = service
        except FileNotFoundError as fnf:
            detail = (
                "Fraud model service unavailable: model artifact not found. "
                "Run 'make train' first."
            )
            raise HTTPException(status_code=503, detail=detail) from fnf
        except ModelIntegrityError as mie:
            detail = (
                "Fraud model service unavailable: "
                f"model integrity verification failed ({mie})."
            )
            raise HTTPException(status_code=503, detail=detail) from mie
        except Exception as exc:
            detail = f"Fraud model service unavailable: {exc}"
            raise HTTPException(status_code=503, detail=detail) from exc

    return service


@router.post("/score", response_model=MLScoreResponse)
def score_transaction(
    body: MLScoreRequest,
    request: Request,
) -> MLScoreResponse:
    """Score a transaction for fraud risk.

    Loads the transaction and strictly earlier history from PostgreSQL,
    applies the calibrated XGBoost model, and explains predictions using SHAP.

    Note:
        SHAP values are reported in log-odds of the base model.
        This endpoint currently has no authentication (Phase 8).
    """
    # 1. Luhn check: reject card numbers in request payload
    if contains_luhn_pan(body.txn_id):
        logger.warning("ml_score_rejected_pan")
        raise HTTPException(
            status_code=400,
            detail="Request rejected: potential card number detected.",
        )

    # 2. Get ML service
    ml_service = _get_ml_service(request)

    # 3. Score transaction
    try:
        result = ml_service.score_transaction(body.txn_id)
        return MLScoreResponse(**result)
    except TransactionNotFoundError as tne:
        raise HTTPException(
            status_code=404,
            detail=f"Transaction '{body.txn_id}' not found.",
        ) from tne
    except ModelNotLoadedError as mne:
        raise HTTPException(
            status_code=503,
            detail=f"Fraud model service unavailable: {mne}",
        ) from mne
    except Exception as exc:
        logger.exception("Unexpected error during fraud scoring")
        raise HTTPException(
            status_code=500,
            detail=f"Error scoring transaction: {exc}",
        ) from exc
