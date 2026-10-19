"""Qdrant integration tests — skip cleanly when Qdrant is not reachable."""

from __future__ import annotations

import uuid

import pytest
from qdrant_client import QdrantClient

from payguard.rag.interfaces import ChunkRecord
from payguard.rag.vectorstore import QdrantVectorStore, deterministic_point_id


def _qdrant_available() -> bool:
    try:
        client = QdrantClient(url="http://localhost:6333", prefer_grpc=False, timeout=2)
        client.get_collections()
        return True
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(
    not _qdrant_available(),
    reason="Qdrant is not running on localhost:6333 — start with `make up`",
)


def test_reingest_does_not_duplicate_points() -> None:
    collection = f"payguard_test_{uuid.uuid4().hex[:8]}"
    store = QdrantVectorStore(url="http://localhost:6333", collection=collection)
    store.ensure_collection(3, recreate=True)

    chunk = ChunkRecord(
        chunk_id="testdoc:0:abc123",
        text="hello world",
        metadata={"doc_id": "testdoc", "region": None, "effective_date": None},
    )
    vector = [0.1, 0.2, 0.3]
    store.upsert([chunk], [vector])
    store.delete_by_doc_id("testdoc")
    store.upsert([chunk], [vector])
    store.upsert([chunk], [vector])  # second upsert same id

    client = QdrantClient(url="http://localhost:6333", prefer_grpc=False)
    count = client.count(collection_name=collection, exact=True).count
    assert count == 1
    point_id = deterministic_point_id(chunk.chunk_id)
    points = client.retrieve(collection_name=collection, ids=[point_id])
    assert len(points) == 1
    client.delete_collection(collection)
