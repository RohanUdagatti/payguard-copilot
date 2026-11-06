"""Redis cache for RAG queries: fast exact-match hash and strict semantic matching."""

from __future__ import annotations

import json
import logging
import math
from typing import Any

import redis

from payguard.config import Settings

logger = logging.getLogger(__name__)


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two float vectors."""
    if len(vec_a) != len(vec_b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for a, b in zip(vec_a, vec_b, strict=True):
        dot += a * b
        norm_a += a * a
        norm_b += b * b
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


class SemanticCache:
    """Question-answer cache powered by Redis with exact-match and semantic matching."""

    def __init__(
        self,
        settings: Settings,
        *,
        client: redis.Redis | None = None,
    ) -> None:
        self.settings = settings
        self._threshold = settings.cache_similarity_threshold
        self._ttl = settings.cache_ttl_seconds
        self._config_hash = settings.retrieval_config_hash()
        self._ingest_version = settings.ingest_version
        self._client: redis.Redis = client or redis.Redis.from_url(
            settings.redis_url, decode_responses=True
        )

    def _namespace_prefix(self) -> str:
        return f"payguard:cache:{self._config_hash}:{self._ingest_version}:"

    def get_exact(self, question_hash: str) -> dict[str, Any] | None:
        """Fast path: exact question-hash lookup (O(1), always safe)."""
        prefix = self._namespace_prefix()
        exact_key = f"{prefix}{question_hash}"
        try:
            exact_val = self._client.get(exact_key)
            if exact_val is not None:
                data = json.loads(str(exact_val))
                if isinstance(data, dict):
                    logger.info(
                        "cache_hit_exact",
                        extra={"question_hash": question_hash, "similarity": 1.0},
                    )
                    return dict(data)
        except Exception as exc:  # noqa: BLE001
            logger.warning("cache_get_exact_error", extra={"error": str(exc)})
        return None

    def get_semantic(
        self,
        query_vector: list[float],
        *,
        question_hash: str = "",
    ) -> tuple[dict[str, Any] | None, float]:
        """Semantic path: compare query vector against cached vectors."""
        prefix = self._namespace_prefix()
        best_sim = -1.0
        best_payload: dict[str, Any] | None = None

        try:
            pattern = f"{prefix}*"
            for key in self._client.scan_iter(pattern, count=100):
                val = self._client.get(key)
                if not val:
                    continue
                entry = json.loads(str(val))
                if not isinstance(entry, dict):
                    continue
                cached_vec = entry.get("embedding")
                if not cached_vec or not isinstance(cached_vec, list):
                    continue
                sim = _cosine_similarity(query_vector, cached_vec)
                if sim > best_sim:
                    best_sim = sim
                    best_payload = dict(entry)

            if best_sim >= self._threshold and best_payload is not None:
                logger.info(
                    "cache_hit_semantic",
                    extra={
                        "question_hash": question_hash,
                        "similarity": round(best_sim, 4),
                        "cached_hash": best_payload.get("question_hash"),
                    },
                )
                return best_payload, best_sim
        except Exception as exc:  # noqa: BLE001
            logger.warning("cache_scan_semantic_error", extra={"error": str(exc)})

        return None, best_sim

    def get(
        self,
        question_hash: str,
        query_vector: list[float],
    ) -> dict[str, Any] | None:
        """Find cached response via exact match first, then semantic match."""
        # Fast path: exact hash match
        exact_hit = self.get_exact(question_hash)
        if exact_hit is not None:
            return exact_hit

        # Semantic path if enabled
        payload, best_sim = self.get_semantic(query_vector, question_hash=question_hash)
        if payload is not None:
            return payload

        logger.info(
            "cache_miss",
            extra={
                "question_hash": question_hash,
                "best_similarity": round(best_sim, 4) if best_sim > 0 else 0.0,
            },
        )
        return None

    def set(
        self,
        question_hash: str,
        query_vector: list[float],
        answer_payload: dict[str, Any],
    ) -> None:
        """Store question hash, vector, and answer payload (never raw question)."""
        prefix = self._namespace_prefix()
        key = f"{prefix}{question_hash}"
        record = {
            "question_hash": question_hash,
            "embedding": query_vector,
            "answer": answer_payload.get("answer"),
            "citations": answer_payload.get("citations", []),
            "retrieved_chunks": answer_payload.get("retrieved_chunks", []),
            "model": answer_payload.get("model", ""),
        }
        try:
            self._client.set(key, json.dumps(record), ex=self._ttl)
            logger.info(
                "cache_stored",
                extra={"question_hash": question_hash, "ttl": self._ttl},
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("cache_set_error", extra={"error": str(exc)})
