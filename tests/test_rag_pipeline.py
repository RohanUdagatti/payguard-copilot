"""Pipeline and route tests with mocked embedder and LLM."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from payguard.config import Settings
from payguard.gateway.main import create_app
from payguard.rag.generator import NOT_FOUND_MESSAGE, GenerationResult
from payguard.rag.interfaces import ChunkRecord, Embedder, RetrievedChunk, VectorStore
from payguard.rag.pipeline import RAGPipeline
from payguard.rag.vectorstore import deterministic_point_id


class FakeEmbedder(Embedder):
    @property
    def dimension(self) -> int:
        return 3

    @property
    def model_name(self) -> str:
        return "fake-embedder"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t)), 0.0, 1.0] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 0.0, 1.0]


class FakeStore(VectorStore):
    def __init__(self, chunks: list[RetrievedChunk] | None = None) -> None:
        self._chunks = chunks or []

    def ensure_collection(self, dimension: int, *, recreate: bool = False) -> None:
        return None

    def upsert(self, chunks: list[ChunkRecord], vectors: list[list[float]]) -> int:
        return len(chunks)

    def search(
        self,
        query_vector: list[float],
        *,
        top_k: int,
        score_threshold: float | None = None,
    ) -> list[RetrievedChunk]:
        return self._chunks[:top_k]

    def delete_by_doc_id(self, doc_id: str) -> None:
        return None


class FakeGenerator:
    def __init__(self, result: GenerationResult) -> None:
        self._result = result

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> GenerationResult:
        if not chunks:
            return GenerationResult(
                answer=NOT_FOUND_MESSAGE,
                citations=[],
                model="fake",
                prompt_tokens=1,
                completion_tokens=1,
            )
        return self._result


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> Any:
    from payguard.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_pipeline_returns_citations_from_retrieved_chunks() -> None:
    retrieved = [
        RetrievedChunk(
            chunk_id="doc-a:0:abc",
            text="Settlement risk must be monitored daily.",
            score=0.91,
            metadata={
                "doc_id": "doc-a",
                "title": "Doc A",
                "section": "Risk",
                "page_start": 2,
                "effective_date": None,
            },
        )
    ]
    gen = FakeGenerator(
        GenerationResult(
            answer="Settlement risk must be monitored daily.",
            citations=[
                {
                    "doc_id": "doc-a",
                    "title": "Doc A",
                    "section": "Risk",
                    "page": 2,
                    "effective_date": None,
                }
            ],
            model="fake-llm",
            prompt_tokens=10,
            completion_tokens=5,
        )
    )
    pipeline = RAGPipeline(
        Settings(_env_file=None),
        embedder=FakeEmbedder(),
        store=FakeStore(retrieved),
        generator=gen,
    )
    result = pipeline.query("How is settlement risk handled?")
    assert result.answer.startswith("Settlement risk")
    assert result.citations[0]["doc_id"] == "doc-a"
    assert result.retrieved_chunks[0]["id"] == "doc-a:0:abc"


def test_insufficient_context_returns_exact_not_found() -> None:
    pipeline = RAGPipeline(
        Settings(_env_file=None),
        embedder=FakeEmbedder(),
        store=FakeStore([]),
        generator=FakeGenerator(GenerationResult(NOT_FOUND_MESSAGE, [], "fake", 1, 1)),
    )
    result = pipeline.query("What is the capital of Mars?")
    assert result.answer == NOT_FOUND_MESSAGE


def test_point_ids_stable_across_reingest() -> None:
    a = deterministic_point_id("doc:0:deadbeef")
    b = deterministic_point_id("doc:0:deadbeef")
    assert a == b


def test_rag_route_rejects_luhn_pan() -> None:
    settings = Settings(_env_file=None, log_level="WARNING")
    app = create_app(settings=settings)
    # Inject pipeline so route does not try real providers
    app.state.rag_pipeline = RAGPipeline(
        settings,
        embedder=FakeEmbedder(),
        store=FakeStore([]),
        generator=FakeGenerator(GenerationResult(NOT_FOUND_MESSAGE, [], "fake", 1, 1)),
    )
    with TestClient(app) as client:
        response = client.post(
            "/rag/query",
            json={"question": "Chargeback for 4111111111111111 please"},
        )
    assert response.status_code == 400


def test_rag_route_ok_with_fakes() -> None:
    settings = Settings(_env_file=None, log_level="WARNING")
    app = create_app(settings=settings)
    retrieved = [
        RetrievedChunk(
            chunk_id="fed:0:x",
            text="Payment system risk policy covers liquidity.",
            score=0.8,
            metadata={
                "doc_id": "fed_payment_risk",
                "title": "fed payment risk",
                "section": "Overview",
                "page_start": 1,
                "effective_date": None,
            },
        )
    ]
    app.state.rag_pipeline = RAGPipeline(
        settings,
        embedder=FakeEmbedder(),
        store=FakeStore(retrieved),
        generator=FakeGenerator(
            GenerationResult(
                answer="It covers liquidity risk.",
                citations=[
                    {
                        "doc_id": "fed_payment_risk",
                        "title": "fed payment risk",
                        "section": "Overview",
                        "page": 1,
                        "effective_date": None,
                    }
                ],
                model="fake",
                prompt_tokens=3,
                completion_tokens=2,
            )
        ),
    )
    with TestClient(app) as client:
        response = client.post(
            "/rag/query",
            json={"question": "What does the payment risk policy cover?"},
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["citations"][0]["doc_id"] == "fed_payment_risk"
    assert "retrieved_chunks" in payload
