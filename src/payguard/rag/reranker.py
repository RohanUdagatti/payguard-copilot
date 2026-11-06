"""Cross-encoder reranker wrapping BAAI/bge-reranker-base (CPU)."""

from __future__ import annotations

import logging
import re
from typing import Any

from payguard.rag.interfaces import RetrievedChunk

logger = logging.getLogger(__name__)

_TABLE_TAG_RE = re.compile(r"\[/?TABLE\]", re.IGNORECASE)


def strip_chunk_markup(text: str) -> str:
    """Strip table tags and extraneous whitespace for reranker input ablation."""
    cleaned = _TABLE_TAG_RE.sub("", text)
    # Collapse multiple blank lines
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


class BGEReranker:
    """Cross-encoder reranker for reranking candidate chunks."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        *,
        max_length: int = 512,
        strip_markup: bool = False,
        torch_threads: int | None = None,
    ) -> None:
        import torch
        from sentence_transformers import CrossEncoder

        if torch_threads is not None and torch_threads > 0:
            torch.set_num_threads(torch_threads)

        self.model_name = model_name
        self.max_length = max_length
        self.strip_markup = strip_markup
        self._model: Any = CrossEncoder(
            model_name,
            max_length=max_length,
            device="cpu",
        )

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        *,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """Score candidate pairs and return top_k sorted by score."""
        if not candidates:
            return []
        if len(candidates) == 1:
            return candidates[:top_k]

        strip = getattr(self, "strip_markup", False)
        pairs = [
            (query, strip_chunk_markup(c.text) if strip else c.text) for c in candidates
        ]
        scores = self._model.predict(pairs)

        rescored: list[tuple[float, RetrievedChunk]] = []
        for c, score in zip(candidates, scores, strict=True):
            rescored.append((float(score), c))

        rescored.sort(key=lambda item: item[0], reverse=True)

        reranked: list[RetrievedChunk] = []
        for score, c in rescored[:top_k]:
            reranked.append(
                RetrievedChunk(
                    chunk_id=c.chunk_id,
                    text=c.text,
                    score=score,
                    metadata=c.metadata,
                )
            )
        return reranked
