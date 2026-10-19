"""Naive dense RAG pipeline: embed -> retrieve -> generate."""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Any, Protocol

from payguard.config import Settings
from payguard.rag.embedders import build_embedder
from payguard.rag.generator import NOT_FOUND_MESSAGE, AnswerGenerator, GenerationResult
from payguard.rag.interfaces import Embedder, RetrievedChunk, VectorStore
from payguard.rag.vectorstore import QdrantVectorStore

logger = logging.getLogger(__name__)


class SupportsGenerate(Protocol):
    """Minimal generator protocol for tests and production."""

    def generate(
        self, question: str, chunks: list[RetrievedChunk]
    ) -> GenerationResult: ...


def hash_question(question: str) -> str:
    """SHA-256 hash of the question (never log raw text)."""
    return hashlib.sha256(question.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RAGResponse:
    """API-facing RAG result."""

    answer: str
    citations: list[dict[str, Any]]
    retrieved_chunks: list[dict[str, Any]]
    latency_ms: float
    model: str
    question_hash: str


class RAGPipeline:
    """Dense-only retrieval + citation-enforced generation."""

    def __init__(
        self,
        settings: Settings,
        *,
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
        generator: SupportsGenerate | None = None,
    ) -> None:
        self.settings = settings
        self.embedder = embedder or build_embedder(settings)
        self.store = store or QdrantVectorStore(
            url=settings.qdrant_url,
            collection=settings.qdrant_collection,
            api_key=settings.qdrant_api_key or None,
        )
        self.generator = generator or AnswerGenerator(settings)

    def query(self, question: str, *, top_k: int | None = None) -> RAGResponse:
        """Run the naive RAG baseline for a single question."""
        started = time.perf_counter()
        q_hash = hash_question(question)
        k = top_k or self.settings.rag_top_k

        vector = self.embedder.embed_query(question)
        retrieved: list[RetrievedChunk] = self.store.search(
            vector,
            top_k=k,
            score_threshold=self.settings.rag_score_threshold,
        )
        generation: GenerationResult = self.generator.generate(question, retrieved)
        latency_ms = (time.perf_counter() - started) * 1000.0

        logger.info(
            "rag_query",
            extra={
                "question_hash": q_hash,
                "chunk_ids": [c.chunk_id for c in retrieved],
                "latency_ms": round(latency_ms, 2),
                "model": generation.model,
                "prompt_tokens": generation.prompt_tokens,
                "completion_tokens": generation.completion_tokens,
                "top_k": k,
                "not_found": generation.answer == NOT_FOUND_MESSAGE,
            },
        )

        return RAGResponse(
            answer=generation.answer,
            citations=generation.citations,
            retrieved_chunks=[{"id": c.chunk_id, "score": c.score} for c in retrieved],
            latency_ms=round(latency_ms, 2),
            model=generation.model,
            question_hash=q_hash,
        )
