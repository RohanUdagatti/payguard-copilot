"""RAG HTTP endpoints."""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from payguard.config import Settings
from payguard.rag.pipeline import RAGPipeline
from payguard.rag.security_checks import contains_luhn_pan

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rag", tags=["rag"])


class RAGQueryRequest(BaseModel):
    """Request body for POST /rag/query."""

    question: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=50)
    as_of_date: str | None = Field(
        default=None,
        description="Optional ISO date string (YYYY-MM-DD) for temporal filtering.",
    )


class RAGQueryResponse(BaseModel):
    """Response body for POST /rag/query."""

    answer: str
    citations: list[dict[str, Any]]
    retrieved_chunks: list[dict[str, Any]]
    latency_ms: float
    cached: bool = False
    has_unversioned_chunks: bool = False
    timings: dict[str, float] = Field(default_factory=dict)
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0
    retry_count: int = 0
    backoff_ms: float = 0.0


def _get_pipeline(request: Request) -> RAGPipeline:
    pipeline = getattr(request.app.state, "rag_pipeline", None)
    if pipeline is None:
        settings: Settings = request.app.state.settings
        pipeline = RAGPipeline(settings)
        request.app.state.rag_pipeline = pipeline
    return pipeline


@router.post("/query", response_model=RAGQueryResponse)
def rag_query(body: RAGQueryRequest, request: Request) -> RAGQueryResponse:
    """Answer a question from ingested documents with citations."""
    t_route_0 = time.perf_counter()
    if contains_luhn_pan(body.question):
        logger.warning("rag_query_rejected_pan")
        raise HTTPException(
            status_code=400,
            detail="Question rejected: potential card number detected.",
        )

    try:
        pipeline = _get_pipeline(request)
        result = pipeline.query(
            body.question,
            top_k=body.top_k,
            as_of_date=body.as_of_date,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("rag_query_failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    route_total_ms = (time.perf_counter() - t_route_0) * 1000.0

    return RAGQueryResponse(
        answer=result.answer,
        citations=result.citations,
        retrieved_chunks=result.retrieved_chunks,
        latency_ms=result.latency_ms,
        cached=result.cached,
        has_unversioned_chunks=result.has_unversioned_chunks,
        timings=result.timings,
        retrieval_ms=result.retrieval_ms,
        generation_ms=result.generation_ms,
        total_ms=round(route_total_ms, 2),
        retry_count=result.retry_count,
        backoff_ms=result.backoff_ms,
    )
