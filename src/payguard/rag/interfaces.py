"""Abstract interfaces for embedding and vector storage (Phase 3-ready)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ChunkRecord:
    """A text chunk with retrieval metadata."""

    chunk_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned from vector search."""

    chunk_id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class Embedder(ABC):
    """Embedding model interface. Swap implementations via config."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Vector dimensionality produced by this embedder."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""


class VectorStore(ABC):
    """Vector store interface. Dense-only in Phase 2; hybrid later."""

    @abstractmethod
    def ensure_collection(self, dimension: int, *, recreate: bool = False) -> None:
        """Create the collection if needed, or recreate when requested."""

    @abstractmethod
    def upsert(self, chunks: list[ChunkRecord], vectors: list[list[float]]) -> int:
        """Upsert chunks with deterministic IDs. Returns number written."""

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        *,
        top_k: int,
        score_threshold: float | None = None,
    ) -> list[RetrievedChunk]:
        """Dense similarity search."""

    @abstractmethod
    def delete_by_doc_id(self, doc_id: str) -> None:
        """Remove all points for a document (used before re-ingest)."""
