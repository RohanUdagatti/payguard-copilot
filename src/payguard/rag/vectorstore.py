"""Qdrant vector store with deterministic point IDs and dimension guard."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from payguard.rag.interfaces import ChunkRecord, RetrievedChunk, VectorStore

logger = logging.getLogger(__name__)

_POINT_NS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


class DimensionMismatchError(RuntimeError):
    """Raised when an existing collection has a different vector size."""


def deterministic_point_id(chunk_id: str) -> str:
    """Map a chunk_id to a stable UUID string for Qdrant points."""
    return str(uuid.uuid5(_POINT_NS, chunk_id))


class QdrantVectorStore(VectorStore):
    """Dense Qdrant store used by the naive RAG baseline."""

    def __init__(
        self,
        *,
        url: str,
        collection: str,
        api_key: str | None = None,
    ) -> None:
        self.collection = collection
        kwargs: dict[str, Any] = {
            "url": url,
            "prefer_grpc": False,
            "timeout": 120,
        }
        if api_key:
            kwargs["api_key"] = api_key
        self._client = QdrantClient(**kwargs)
        self._upsert_batch_size = 64

    def _collection_exists(self) -> bool:
        names = {c.name for c in self._client.get_collections().collections}
        return self.collection in names

    def _existing_dimension(self) -> int | None:
        if not self._collection_exists():
            return None
        info = self._client.get_collection(self.collection)
        vectors = info.config.params.vectors
        if isinstance(vectors, qm.VectorParams):
            return int(vectors.size)
        if isinstance(vectors, dict) and vectors:
            first = next(iter(vectors.values()))
            return int(first.size)
        return None

    def ensure_collection(self, dimension: int, *, recreate: bool = False) -> None:
        """Create collection or recreate; refuse on silent dimension mismatch."""
        existing = self._existing_dimension()
        if existing is not None and existing != dimension and not recreate:
            raise DimensionMismatchError(
                f"Collection {self.collection!r} has dimension {existing}, "
                f"but the active embedder uses {dimension}. "
                "Re-run ingest with --recreate to rebuild the collection."
            )

        if recreate and self._collection_exists():
            self._client.delete_collection(self.collection)
            existing = None

        if existing is None:
            self._client.create_collection(
                collection_name=self.collection,
                vectors_config=qm.VectorParams(
                    size=dimension,
                    distance=qm.Distance.COSINE,
                ),
            )
            for field_name, schema in (
                ("doc_id", qm.PayloadSchemaType.KEYWORD),
                ("region", qm.PayloadSchemaType.KEYWORD),
                ("effective_date", qm.PayloadSchemaType.KEYWORD),
            ):
                self._client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field_name,
                    field_schema=schema,
                )
            logger.info(
                "qdrant_collection_created",
                extra={"collection": self.collection, "dimension": dimension},
            )

    def delete_by_doc_id(self, doc_id: str) -> None:
        if not self._collection_exists():
            return
        self._client.delete(
            collection_name=self.collection,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(
                    must=[
                        qm.FieldCondition(
                            key="doc_id",
                            match=qm.MatchValue(value=doc_id),
                        )
                    ]
                )
            ),
        )

    def upsert(self, chunks: list[ChunkRecord], vectors: list[list[float]]) -> int:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        if not chunks:
            return 0

        points = [
            qm.PointStruct(
                id=deterministic_point_id(chunk.chunk_id),
                vector=vector,
                payload={
                    **chunk.metadata,
                    "text": chunk.text,
                    "chunk_id": chunk.chunk_id,
                },
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        for i in range(0, len(points), self._upsert_batch_size):
            batch = points[i : i + self._upsert_batch_size]
            self._client.upsert(collection_name=self.collection, points=batch)
        return len(points)

    def search(
        self,
        query_vector: list[float],
        *,
        top_k: int,
        score_threshold: float | None = None,
    ) -> list[RetrievedChunk]:
        results = self._client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
        )
        retrieved: list[RetrievedChunk] = []
        for hit in results:
            payload = hit.payload or {}
            chunk_id = str(payload.get("chunk_id") or hit.id)
            text = str(payload.get("text") or "")
            metadata = {k: v for k, v in payload.items() if k != "text"}
            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=text,
                    score=float(hit.score),
                    metadata=metadata,
                )
            )
        return retrieved
