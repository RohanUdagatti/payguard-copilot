"""Qdrant vector store supporting dense and hybrid (dense + BM25) retrieval."""

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


class CollectionSchemaMismatchError(RuntimeError):
    """Raised when an existing collection schema does not match RETRIEVAL_MODE."""


def deterministic_point_id(chunk_id: str) -> str:
    """Map a chunk_id to a stable UUID string for Qdrant points."""
    return str(uuid.uuid5(_POINT_NS, chunk_id))


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievedChunk]],
    k: int = 60,
    top_n: int = 20,
) -> list[RetrievedChunk]:
    """Fuse multiple ranked lists using Reciprocal Rank Fusion (RRF)."""
    scores: dict[str, float] = {}
    chunk_map: dict[str, RetrievedChunk] = {}

    for ranked_list in ranked_lists:
        for rank, chunk in enumerate(ranked_list, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + (
                1.0 / (k + rank)
            )
            if chunk.chunk_id not in chunk_map:
                chunk_map[chunk.chunk_id] = chunk

    # Sort descending by fused score, deterministic tie-breaking on chunk_id
    sorted_ids = sorted(
        scores.keys(),
        key=lambda cid: (scores[cid], cid),
        reverse=True,
    )

    fused: list[RetrievedChunk] = []
    for cid in sorted_ids[:top_n]:
        orig = chunk_map[cid]
        fused.append(
            RetrievedChunk(
                chunk_id=cid,
                text=orig.text,
                score=scores[cid],
                metadata=orig.metadata,
            )
        )
    return fused


class QdrantVectorStore(VectorStore):
    """Qdrant store supporting dense and hybrid (dense + fastembed BM25) search."""

    def __init__(
        self,
        *,
        url: str,
        collection: str,
        api_key: str | None = None,
        retrieval_mode: str = "dense",
    ) -> None:
        self.collection = collection
        self.retrieval_mode = retrieval_mode
        kwargs: dict[str, Any] = {
            "url": url,
            "prefer_grpc": False,
            "timeout": 120,
        }
        if api_key:
            kwargs["api_key"] = api_key
        self._client = QdrantClient(**kwargs)
        self._upsert_batch_size = 64
        self._bm25_model: Any = None

    def _get_bm25_model(self) -> Any:
        if self._bm25_model is None:
            from fastembed import SparseTextEmbedding

            self._bm25_model = SparseTextEmbedding(model_name="Qdrant/bm25")
        return self._bm25_model

    def _collection_exists(self) -> bool:
        names = {c.name for c in self._client.get_collections().collections}
        return self.collection in names

    def is_hybrid_collection(self) -> bool:
        """Return True if existing collection has sparse vectors configured."""
        if not self._collection_exists():
            return False
        info = self._client.get_collection(self.collection)
        sparse_cfg = getattr(info.config.params, "sparse_vectors", None)
        return bool(sparse_cfg and "sparse" in sparse_cfg)

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

    def check_schema_guard(self) -> None:
        """Validate collection schema against RETRIEVAL_MODE."""
        if not self._collection_exists():
            return
        is_hybrid = self.is_hybrid_collection()
        if self.retrieval_mode == "hybrid" and not is_hybrid:
            raise CollectionSchemaMismatchError(
                f"Collection {self.collection!r} schema does not match "
                f"RETRIEVAL_MODE='hybrid' (missing sparse vectors). "
                "Re-run ingest with --recreate to rebuild the collection."
            )
        if self.retrieval_mode == "dense" and is_hybrid:
            # Dense mode can still read hybrid collections by searching the dense vector
            pass

    def ensure_collection(
        self,
        dimension: int,
        *,
        recreate: bool = False,
        hybrid: bool = False,
    ) -> None:
        """Create or recreate collection; guard against dimension/schema mismatch."""
        existing = self._existing_dimension()
        if existing is not None and existing != dimension and not recreate:
            raise DimensionMismatchError(
                f"Collection {self.collection!r} has dimension {existing}, "
                f"but the active embedder uses {dimension}. "
                "Re-run ingest with --recreate to rebuild the collection."
            )

        if self._collection_exists() and not recreate:
            is_hybrid = self.is_hybrid_collection()
            if hybrid and not is_hybrid:
                raise CollectionSchemaMismatchError(
                    f"Collection {self.collection!r} schema does not match "
                    "RETRIEVAL_MODE='hybrid'. "
                    "Re-run ingest with --recreate to rebuild the collection."
                )

        if recreate and self._collection_exists():
            self._client.delete_collection(self.collection)
            existing = None

        if existing is None:
            if hybrid:
                self._client.create_collection(
                    collection_name=self.collection,
                    vectors_config={
                        "dense": qm.VectorParams(
                            size=dimension,
                            distance=qm.Distance.COSINE,
                        )
                    },
                    sparse_vectors_config={
                        "sparse": qm.SparseVectorParams(
                            index=qm.SparseIndexParams(on_disk=False)
                        )
                    },
                )
            else:
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
                ("is_child", qm.PayloadSchemaType.KEYWORD),
                ("parent_id", qm.PayloadSchemaType.KEYWORD),
            ):
                self._client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field_name,
                    field_schema=schema,
                )
            logger.info(
                "qdrant_collection_created",
                extra={
                    "collection": self.collection,
                    "dimension": dimension,
                    "hybrid": hybrid,
                },
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

    def upsert(
        self,
        chunks: list[ChunkRecord],
        vectors: list[list[float]],
        *,
        hybrid: bool = False,
    ) -> int:
        """Upsert chunks with dense (or dense + BM25 sparse) vectors."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        if not chunks:
            return 0

        sparse_vectors: list[Any] = []
        if hybrid:
            bm25 = self._get_bm25_model()
            texts = [c.text for c in chunks]
            sparse_vectors = list(bm25.embed(texts))

        points: list[qm.PointStruct] = []
        for idx, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True)):
            if hybrid:
                s_vec = sparse_vectors[idx]
                s_indices = (
                    s_vec.indices.tolist()
                    if hasattr(s_vec.indices, "tolist")
                    else list(s_vec.indices)
                )
                s_values = (
                    s_vec.values.tolist()
                    if hasattr(s_vec.values, "tolist")
                    else list(s_vec.values)
                )
                pt_vector: Any = {
                    "dense": vector,
                    "sparse": qm.SparseVector(
                        indices=s_indices,
                        values=s_values,
                    ),
                }
            else:
                pt_vector = vector

            points.append(
                qm.PointStruct(
                    id=deterministic_point_id(chunk.chunk_id),
                    vector=pt_vector,
                    payload={
                        **chunk.metadata,
                        "text": chunk.text,
                        "chunk_id": chunk.chunk_id,
                    },
                )
            )

        for i in range(0, len(points), self._upsert_batch_size):
            batch = points[i : i + self._upsert_batch_size]
            self._client.upsert(collection_name=self.collection, points=batch)
        return len(points)

    def _build_filter(
        self,
        *,
        filter_is_child: bool | None = None,
    ) -> qm.Filter | None:
        if filter_is_child is True:
            return qm.Filter(
                must=[
                    qm.FieldCondition(
                        key="is_child",
                        match=qm.MatchValue(value=True),
                    )
                ]
            )
        if filter_is_child is False:
            # Exclude child chunks; include chunks where is_child is False or not set
            return qm.Filter(
                must_not=[
                    qm.FieldCondition(
                        key="is_child",
                        match=qm.MatchValue(value=True),
                    )
                ]
            )
        return None

    def search(
        self,
        query_vector: list[float],
        *,
        top_k: int,
        score_threshold: float | None = None,
        filter_is_child: bool | None = None,
        query_text: str | None = None,
    ) -> list[RetrievedChunk]:
        """Search collection via dense search or hybrid search with RRF."""
        self.check_schema_guard()
        is_hybrid = self.is_hybrid_collection() and self.retrieval_mode == "hybrid"
        q_filter = self._build_filter(filter_is_child=filter_is_child)

        if not is_hybrid:
            # Dense search
            dense_target: Any
            info = self._client.get_collection(self.collection)
            if (
                isinstance(info.config.params.vectors, dict)
                and "dense" in info.config.params.vectors
            ):
                dense_target = qm.NamedVector(name="dense", vector=query_vector)
            else:
                dense_target = query_vector

            results = self._client.search(
                collection_name=self.collection,
                query_vector=dense_target,
                limit=top_k,
                score_threshold=score_threshold,
                query_filter=q_filter,
                with_payload=True,
            )
            return self._to_retrieved_chunks(results)

        # Hybrid Search: Dense + BM25 Sparse fused via RRF
        dense_results = self._client.search(
            collection_name=self.collection,
            query_vector=qm.NamedVector(name="dense", vector=query_vector),
            limit=max(top_k * 2, 20),
            score_threshold=score_threshold,
            query_filter=q_filter,
            with_payload=True,
        )
        dense_hits = self._to_retrieved_chunks(dense_results)

        sparse_hits: list[RetrievedChunk] = []
        if query_text:
            bm25 = self._get_bm25_model()
            sparse_out = list(bm25.embed([query_text]))[0]
            s_indices = (
                sparse_out.indices.tolist()
                if hasattr(sparse_out.indices, "tolist")
                else list(sparse_out.indices)
            )
            s_values = (
                sparse_out.values.tolist()
                if hasattr(sparse_out.values, "tolist")
                else list(sparse_out.values)
            )
            sparse_results = self._client.search(
                collection_name=self.collection,
                query_vector=qm.NamedSparseVector(
                    name="sparse",
                    vector=qm.SparseVector(indices=s_indices, values=s_values),
                ),
                limit=max(top_k * 2, 20),
                query_filter=q_filter,
                with_payload=True,
            )
            sparse_hits = self._to_retrieved_chunks(sparse_results)

        fused = reciprocal_rank_fusion([dense_hits, sparse_hits], k=60, top_n=top_k)
        return fused

    def _to_retrieved_chunks(self, results: list[Any]) -> list[RetrievedChunk]:
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
