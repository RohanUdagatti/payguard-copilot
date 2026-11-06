"""Advanced RAG pipeline: retrieval, reranking, caching, and generation."""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Protocol

from payguard.config import Settings
from payguard.rag.cache import SemanticCache
from payguard.rag.embedders import build_embedder
from payguard.rag.generator import NOT_FOUND_MESSAGE, AnswerGenerator, GenerationResult
from payguard.rag.ingest.chunker import deduplicate_to_parents
from payguard.rag.interfaces import Embedder, RetrievedChunk, VectorStore
from payguard.rag.reranker import BGEReranker
from payguard.rag.vectorstore import QdrantVectorStore, reciprocal_rank_fusion

logger = logging.getLogger(__name__)


class SupportsGenerate(Protocol):
    """Minimal generator protocol for tests and production."""

    def generate(
        self, question: str, chunks: list[RetrievedChunk]
    ) -> GenerationResult: ...


class SupportsRerank(Protocol):
    """Minimal reranker protocol for tests and production."""

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], *, top_k: int = 5
    ) -> list[RetrievedChunk]: ...


def hash_question(question: str) -> str:
    """SHA-256 hash of the question (never log raw text)."""
    return hashlib.sha256(question.encode("utf-8")).hexdigest()


def apply_temporal_filter(
    chunks: list[RetrievedChunk],
    as_of_date_str: str | None,
) -> tuple[list[RetrievedChunk], bool]:
    """Filter chunks by effective_date <= as_of_date, keeping latest version per doc."""
    if not as_of_date_str:
        return chunks, any(c.metadata.get("effective_date") is None for c in chunks)

    try:
        as_of = date.fromisoformat(as_of_date_str)
    except ValueError:
        return chunks, any(c.metadata.get("effective_date") is None for c in chunks)

    kept: list[RetrievedChunk] = []
    has_null = False
    doc_versions: dict[str, dict[str, list[RetrievedChunk]]] = {}

    for c in chunks:
        eff_str = c.metadata.get("effective_date")
        if eff_str is None:
            has_null = True
            kept.append(c)
            continue

        try:
            eff_date = date.fromisoformat(str(eff_str))
        except ValueError:
            has_null = True
            kept.append(c)
            continue

        if eff_date <= as_of:
            doc_id = str(c.metadata.get("doc_id", "default"))
            ver = str(c.metadata.get("version") or eff_str)
            if doc_id not in doc_versions:
                doc_versions[doc_id] = {}
            if ver not in doc_versions[doc_id]:
                doc_versions[doc_id][ver] = []
            doc_versions[doc_id][ver].append(c)

    for _doc_id, ver_dict in doc_versions.items():
        sorted_versions = sorted(ver_dict.keys(), reverse=True)
        if sorted_versions:
            latest_ver = sorted_versions[0]
            kept.extend(ver_dict[latest_ver])

    chunk_order = {c.chunk_id: idx for idx, c in enumerate(chunks)}
    kept.sort(key=lambda c: chunk_order.get(c.chunk_id, 999))
    return kept, has_null


@dataclass(frozen=True)
class RAGResponse:
    """API-facing RAG result."""

    answer: str
    citations: list[dict[str, Any]]
    retrieved_chunks: list[dict[str, Any]]
    latency_ms: float
    model: str
    question_hash: str
    timings: dict[str, float] = field(default_factory=dict)
    cached: bool = False
    has_unversioned_chunks: bool = False
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0
    retry_count: int = 0
    backoff_ms: float = 0.0


class RAGPipeline:
    """Config-driven RAG pipeline supporting Phase 2 and Phase 3 advanced features."""

    def __init__(
        self,
        settings: Settings,
        *,
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
        generator: SupportsGenerate | None = None,
        reranker: SupportsRerank | None = None,
        cache: SemanticCache | None = None,
    ) -> None:
        t0 = time.perf_counter()
        self.settings = settings
        self.embedder = embedder or build_embedder(settings)
        self.store = store or QdrantVectorStore(
            url=settings.qdrant_url,
            collection=settings.qdrant_collection,
            api_key=settings.qdrant_api_key or None,
            retrieval_mode=settings.retrieval_mode,
        )
        self.generator = generator or AnswerGenerator(settings)
        self.reranker: SupportsRerank | None = reranker
        if self.reranker is None and settings.rerank_enabled:
            self.reranker = BGEReranker(settings.rerank_model)

        self.cache: SemanticCache | None = cache
        if self.cache is None and settings.cache_enabled:
            try:
                self.cache = SemanticCache(settings)
            except Exception as exc:  # noqa: BLE001
                logger.warning("cache_init_failed", extra={"error": str(exc)})
                self.cache = None

        self._init_load_time_ms = (time.perf_counter() - t0) * 1000.0

    def _expand_query(self, question: str) -> list[str]:
        """Generate up to 3 query variations with free-tier safety."""
        prompt = (
            f"Generate {self.settings.multi_query_count} diverse search queries "
            "for vector retrieval over payment compliance policies.\n"
            "Return only the queries, one per line, without numbers or bullets.\n\n"
            f"Query: {question}"
        )
        try:
            llm = getattr(self.generator, "_llm", None)
            if llm is None:
                return []
            response = llm.invoke(prompt)
            content = getattr(response, "content", str(response)).strip()
            variants = [
                v.strip().lstrip("0123456789.- ")
                for v in content.splitlines()
                if v.strip()
            ]
            return variants[: self.settings.multi_query_count]
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "multi_query_expansion_failed_fallback",
                extra={"error": str(exc)},
            )
            return []

    def query(
        self,
        question: str,
        *,
        top_k: int | None = None,
        as_of_date: str | None = None,
    ) -> RAGResponse:
        """Execute RAG pipeline with profiling and configurable Phase 3 features."""
        started = time.perf_counter()
        q_hash = hash_question(question)
        k = top_k or self.settings.rag_top_k
        timings: dict[str, float] = {
            "model_load_ms": round(self._init_load_time_ms, 2),
            "embed_ms": 0.0,
            "qdrant_ms": 0.0,
            "llm_ms": 0.0,
            "rerank_ms": 0.0,
            "retry_count": 0.0,
            "backoff_ms": 0.0,
        }

        # Step 0: Fast path exact question hash cache lookup
        if self.settings.cache_enabled and self.cache is not None:
            cached_data = self.cache.get_exact(q_hash)
            if cached_data is not None:
                elapsed = (time.perf_counter() - started) * 1000.0
                return RAGResponse(
                    answer=cached_data.get("answer", NOT_FOUND_MESSAGE),
                    citations=cached_data.get("citations", []),
                    retrieved_chunks=cached_data.get("retrieved_chunks", []),
                    latency_ms=round(elapsed, 2),
                    model=cached_data.get("model", "cached"),
                    question_hash=q_hash,
                    timings=timings,
                    cached=True,
                    retrieval_ms=0.0,
                    generation_ms=0.0,
                    total_ms=round(elapsed, 2),
                    retry_count=0,
                    backoff_ms=0.0,
                )

        # Step 1: Embed question
        t_embed_0 = time.perf_counter()
        query_vector = self.embedder.embed_query(question)
        timings["embed_ms"] = round((time.perf_counter() - t_embed_0) * 1000.0, 2)

        # Step 2: Redis Semantic Cache lookup (only when cache_enabled is True)
        if self.settings.cache_enabled and self.cache is not None:
            cached_data, _sim = self.cache.get_semantic(
                query_vector, question_hash=q_hash
            )
            if cached_data is not None:
                elapsed = (time.perf_counter() - started) * 1000.0
                return RAGResponse(
                    answer=cached_data.get("answer", NOT_FOUND_MESSAGE),
                    citations=cached_data.get("citations", []),
                    retrieved_chunks=cached_data.get("retrieved_chunks", []),
                    latency_ms=round(elapsed, 2),
                    model=cached_data.get("model", "cached"),
                    question_hash=q_hash,
                    timings=timings,
                    cached=True,
                    retrieval_ms=timings["embed_ms"],
                    generation_ms=0.0,
                    total_ms=round(elapsed, 2),
                    retry_count=0,
                    backoff_ms=0.0,
                )

        # Step 3: Multi-query expansion (if enabled)
        queries = [question]
        if self.settings.multi_query_enabled:
            variants = self._expand_query(question)
            if variants:
                queries.extend(variants)

        # Step 4: Retrieval (Dense / Hybrid, Child chunks if parent_child_enabled)
        t_qdrant_0 = time.perf_counter()
        filter_child = bool(self.settings.parent_child_enabled)
        search_k = (
            self.settings.rerank_top_n
            if (self.settings.rerank_enabled and self.reranker)
            else k
        )

        all_ranked: list[list[RetrievedChunk]] = []
        for q_text in queries:
            q_vec = (
                query_vector
                if q_text == question
                else self.embedder.embed_query(q_text)
            )
            kwargs: dict[str, Any] = {
                "top_k": search_k,
                "score_threshold": self.settings.rag_score_threshold,
            }
            if hasattr(self.store, "search"):
                import inspect

                sig = inspect.signature(self.store.search)
                if "filter_is_child" in sig.parameters:
                    kwargs["filter_is_child"] = filter_child
                if "query_text" in sig.parameters:
                    kwargs["query_text"] = q_text

            hits = self.store.search(q_vec, **kwargs)
            all_ranked.append(hits)

        if len(all_ranked) == 1:
            retrieved = all_ranked[0]
        else:
            retrieved = reciprocal_rank_fusion(all_ranked, k=60, top_n=search_k)

        # Step 5: Parent deduplication
        if self.settings.parent_child_enabled:
            retrieved = deduplicate_to_parents(retrieved)

        timings["qdrant_ms"] = round((time.perf_counter() - t_qdrant_0) * 1000.0, 2)

        # Step 6: Temporal Filter
        has_unversioned = False
        if self.settings.temporal_filter_enabled:
            retrieved, has_unversioned = apply_temporal_filter(retrieved, as_of_date)

        # Step 7: Reranking
        if self.settings.rerank_enabled and self.reranker is not None:
            t_rerank_0 = time.perf_counter()
            retrieved = self.reranker.rerank(question, retrieved, top_k=k)
            timings["rerank_ms"] = round((time.perf_counter() - t_rerank_0) * 1000.0, 2)
        else:
            retrieved = retrieved[:k]

        t_retrieval_done = time.perf_counter()
        retrieval_ms = (t_retrieval_done - started) * 1000.0
        self._last_contexts = [c.text for c in retrieved]
        self._last_retrieved = retrieved

        # Step 8: Generation
        t_llm_0 = time.perf_counter()
        generation: GenerationResult = self.generator.generate(question, retrieved)
        generation_ms = (time.perf_counter() - t_llm_0) * 1000.0
        generation_retry_count = int(getattr(generation, "retry_count", 0))
        generation_backoff_ms = float(getattr(generation, "backoff_ms", 0.0))

        timings["llm_ms"] = round(generation_ms, 2)
        timings["retry_count"] = float(generation_retry_count)
        timings["backoff_ms"] = round(generation_backoff_ms, 2)
        total_ms = (time.perf_counter() - started) * 1000.0

        # Step 9: Store in cache (only when cache_enabled is True)
        if (
            self.settings.cache_enabled
            and self.cache is not None
            and generation.answer != NOT_FOUND_MESSAGE
        ):
            self.cache.set(
                q_hash,
                query_vector,
                {
                    "answer": generation.answer,
                    "citations": generation.citations,
                    "retrieved_chunks": [
                        {"id": c.chunk_id, "score": c.score} for c in retrieved
                    ],
                    "model": generation.model,
                },
            )

        logger.info(
            "rag_query",
            extra={
                "question_hash": q_hash,
                "chunk_ids": [c.chunk_id for c in retrieved],
                "latency_ms": round(total_ms, 2),
                "model": generation.model,
                "top_k": k,
                "timings": timings,
                "cached": False,
                "not_found": generation.answer == NOT_FOUND_MESSAGE,
                "retry_count": generation_retry_count,
                "backoff_ms": round(generation_backoff_ms, 2),
            },
        )

        return RAGResponse(
            answer=generation.answer,
            citations=generation.citations,
            retrieved_chunks=[{"id": c.chunk_id, "score": c.score} for c in retrieved],
            latency_ms=round(total_ms, 2),
            model=generation.model,
            question_hash=q_hash,
            timings=timings,
            cached=False,
            has_unversioned_chunks=has_unversioned,
            retrieval_ms=round(retrieval_ms, 2),
            generation_ms=round(generation_ms, 2),
            total_ms=round(total_ms, 2),
            retry_count=generation_retry_count,
            backoff_ms=round(generation_backoff_ms, 2),
        )

    def retrieve_contexts(
        self,
        question: str,
        *,
        top_k: int | None = None,
        as_of_date: str | None = None,
    ) -> tuple[list[str], list[RetrievedChunk], dict[str, float]]:
        """Thin retrieval hook returning contexts, chunks, and timings."""
        started = time.perf_counter()
        k = top_k or self.settings.rag_top_k
        timings: dict[str, float] = {
            "embed_ms": 0.0,
            "qdrant_ms": 0.0,
            "rerank_ms": 0.0,
            "total_retrieval_ms": 0.0,
        }

        t_embed_0 = time.perf_counter()
        query_vector = self.embedder.embed_query(question)
        timings["embed_ms"] = round((time.perf_counter() - t_embed_0) * 1000.0, 2)

        queries = [question]
        if self.settings.multi_query_enabled:
            variants = self._expand_query(question)
            if variants:
                queries.extend(variants)

        t_qdrant_0 = time.perf_counter()
        filter_child = bool(self.settings.parent_child_enabled)
        search_k = (
            self.settings.rerank_top_n
            if (self.settings.rerank_enabled and self.reranker)
            else k
        )

        all_ranked: list[list[RetrievedChunk]] = []
        for q_text in queries:
            q_vec = (
                query_vector
                if q_text == question
                else self.embedder.embed_query(q_text)
            )
            kwargs: dict[str, Any] = {
                "top_k": search_k,
                "score_threshold": self.settings.rag_score_threshold,
            }
            if hasattr(self.store, "search"):
                import inspect

                sig = inspect.signature(self.store.search)
                if "filter_is_child" in sig.parameters:
                    kwargs["filter_is_child"] = filter_child
                if "query_text" in sig.parameters:
                    kwargs["query_text"] = q_text

            hits = self.store.search(q_vec, **kwargs)
            all_ranked.append(hits)

        if len(all_ranked) == 1:
            retrieved = all_ranked[0]
        else:
            retrieved = reciprocal_rank_fusion(all_ranked, k=60, top_n=search_k)

        if self.settings.parent_child_enabled:
            retrieved = deduplicate_to_parents(retrieved)

        timings["qdrant_ms"] = round((time.perf_counter() - t_qdrant_0) * 1000.0, 2)

        if self.settings.temporal_filter_enabled:
            retrieved, _ = apply_temporal_filter(retrieved, as_of_date)

        if self.settings.rerank_enabled and self.reranker is not None:
            t_rerank_0 = time.perf_counter()
            retrieved = self.reranker.rerank(question, retrieved, top_k=k)
            timings["rerank_ms"] = round((time.perf_counter() - t_rerank_0) * 1000.0, 2)
        else:
            retrieved = retrieved[:k]

        timings["total_retrieval_ms"] = round(
            (time.perf_counter() - started) * 1000.0, 2
        )
        contexts = [c.text for c in retrieved]
        return contexts, retrieved, timings
