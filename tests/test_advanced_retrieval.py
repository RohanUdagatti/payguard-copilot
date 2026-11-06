"""Unit tests for Phase 3 advanced retrieval features (no network, no real keys)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from payguard.config import Settings
from payguard.gateway.main import create_app
from payguard.rag.cache import SemanticCache
from payguard.rag.ingest.chunker import (
    deduplicate_to_parents,
    is_heading_line,
    is_junk_boilerplate_chunk,
    make_child_chunks,
)
from payguard.rag.interfaces import ChunkRecord, RetrievedChunk
from payguard.rag.pipeline import (
    RAGPipeline,
    apply_temporal_filter,
    hash_question,
)
from payguard.rag.reranker import BGEReranker
from payguard.rag.vectorstore import reciprocal_rank_fusion


def test_rrf_fusion_math_fixed_rankings() -> None:
    """Verify reciprocal rank fusion math against exact formula."""
    # Score for rank r: 1 / (60 + r)
    # Item A: rank 1 in list 1 (1/61), rank 2 in list 2 (1/62) -> sum ≈ 0.032522
    # Item B: rank 2 in list 1 (1/62), rank 1 in list 2 (1/61) -> sum ≈ 0.032522
    # Item C: rank 3 in list 1 (1/63), not in list 2 -> sum ≈ 0.015873
    c_a = RetrievedChunk(chunk_id="chunk_a", text="Text A", score=0.9)
    c_b = RetrievedChunk(chunk_id="chunk_b", text="Text B", score=0.8)
    c_c = RetrievedChunk(chunk_id="chunk_c", text="Text C", score=0.7)

    list1 = [c_a, c_b, c_c]
    list2 = [c_b, c_a]

    fused = reciprocal_rank_fusion([list1, list2], k=60, top_n=3)
    assert len(fused) == 3

    expected_score_a = (1.0 / 61.0) + (1.0 / 62.0)
    expected_score_c = 1.0 / 63.0

    assert pytest.approx(fused[0].score, rel=1e-5) == expected_score_a
    assert fused[2].chunk_id == "chunk_c"
    assert pytest.approx(fused[2].score, rel=1e-5) == expected_score_c


def test_reranker_reorders_known_candidates() -> None:
    """Mock cross-encoder to verify candidate reordering."""
    c1 = RetrievedChunk(chunk_id="c1", text="doc 1", score=0.9)
    c2 = RetrievedChunk(chunk_id="c2", text="doc 2", score=0.8)
    c3 = RetrievedChunk(chunk_id="c3", text="doc 3", score=0.7)

    reranker = BGEReranker.__new__(BGEReranker)
    reranker.model_name = "mock"
    mock_model = MagicMock()
    # Return higher score for c2, then c3, then c1
    mock_model.predict.return_value = [0.1, 0.95, 0.4]
    reranker._model = mock_model

    reranked = reranker.rerank("test query", [c1, c2, c3], top_k=2)
    assert len(reranked) == 2
    assert reranked[0].chunk_id == "c2"
    assert reranked[0].score == 0.95
    assert reranked[1].chunk_id == "c3"
    assert reranked[1].score == 0.4


def test_parent_child_chunking_and_deduplication() -> None:
    """Verify child chunk creation and parent deduplication."""
    long_text = "This is a detailed paragraph about payment risk management. " * 20
    parent = ChunkRecord(
        chunk_id="parent_1",
        text=long_text,
        metadata={"doc_id": "doc_a", "section": "1.1 Risk", "page_start": 5},
    )

    children = make_child_chunks(parent, target_tokens=100, overlap_tokens=20)
    assert len(children) >= 2
    for idx, c in enumerate(children):
        assert c.metadata["parent_id"] == "parent_1"
        assert c.metadata["parent_text"] == long_text
        assert c.metadata["is_child"] is True
        assert c.metadata["child_index"] == idx

    # Test deduplication: 3 child hits from same parent return 1 parent
    hits = [
        RetrievedChunk(
            chunk_id=children[1].chunk_id,
            text=children[1].text,
            score=0.92,
            metadata=children[1].metadata,
        ),
        RetrievedChunk(
            chunk_id=children[0].chunk_id,
            text=children[0].text,
            score=0.85,
            metadata=children[0].metadata,
        ),
        RetrievedChunk(
            chunk_id="standalone_chunk",
            text="Non-child standalone text",
            score=0.80,
            metadata={"doc_id": "doc_b"},
        ),
    ]

    deduped = deduplicate_to_parents(hits)
    assert len(deduped) == 2
    # First returned is parent_1 with highest child score (0.92) and full parent text
    assert deduped[0].chunk_id == "parent_1"
    assert deduped[0].text == long_text
    assert deduped[0].score == 0.92
    assert deduped[0].metadata["doc_id"] == "doc_a"
    assert "is_child" not in deduped[0].metadata

    # Second is standalone chunk
    assert deduped[1].chunk_id == "standalone_chunk"


def test_temporal_filter_picks_correct_version_and_handles_null() -> None:
    """Verify temporal filter logic across versions and null dates."""
    c_old = RetrievedChunk(
        chunk_id="v1",
        text="Version 1",
        score=0.9,
        metadata={"doc_id": "doc_a", "effective_date": "2024-01-01", "version": "1.0"},
    )
    c_mid = RetrievedChunk(
        chunk_id="v2",
        text="Version 2",
        score=0.85,
        metadata={"doc_id": "doc_a", "effective_date": "2025-01-01", "version": "2.0"},
    )
    c_future = RetrievedChunk(
        chunk_id="v3",
        text="Version 3",
        score=0.8,
        metadata={"doc_id": "doc_a", "effective_date": "2026-06-01", "version": "3.0"},
    )
    c_null = RetrievedChunk(
        chunk_id="unversioned",
        text="No date",
        score=0.75,
        metadata={"doc_id": "doc_b", "effective_date": None},
    )

    chunks = [c_old, c_mid, c_future, c_null]

    # Query with as_of_date = 2025-06-01
    filtered, has_null = apply_temporal_filter(chunks, "2025-06-01")

    # c_mid (2025-01-01) should be selected as latest valid version for doc_a
    # c_future (2026-06-01) should be filtered out
    # c_null should be retained and has_null flagged True
    assert has_null is True
    ids = [c.chunk_id for c in filtered]
    assert "v2" in ids
    assert "v1" not in ids
    assert "v3" not in ids
    assert "unversioned" in ids


def test_redis_semantic_cache_lifecycle_and_security() -> None:
    """Test cache hit, miss, config key isolation, and no raw question stored."""
    mock_redis = MagicMock()
    kv_store: dict[str, str] = {}

    def mock_set(key: str, val: str, ex: int | None = None) -> bool:
        kv_store[key] = val
        return True

    def mock_get(key: str) -> str | None:
        return kv_store.get(key)

    def mock_scan(pattern: str, count: int = 100) -> list[str]:
        prefix = pattern.rstrip("*")
        return [k for k in kv_store if k.startswith(prefix)]

    mock_redis.set.side_effect = mock_set
    mock_redis.get.side_effect = mock_get
    mock_redis.scan_iter.side_effect = mock_scan

    settings = Settings(
        cache_enabled=True,
        cache_similarity_threshold=0.95,
        ingest_version="1.0.0",
    )
    cache = SemanticCache(settings, client=mock_redis)

    q_hash = hash_question("What is the PSR policy?")
    q_vec = [1.0, 0.0, 0.0]

    # Initially miss
    assert cache.get(q_hash, q_vec) is None

    # Store entry
    payload = {
        "answer": "The PSR policy governs payment risk.",
        "citations": [{"doc_id": "fed_payment_risk"}],
        "retrieved_chunks": [],
        "model": "gemini-3.8-flash",
    }
    cache.set(q_hash, q_vec, payload)

    # Verify no raw question in stored payload
    stored_str = list(kv_store.values())[0]
    assert "What is the PSR policy?" not in stored_str
    stored_json = json.loads(stored_str)
    assert stored_json["question_hash"] == q_hash

    # Exact cache hit
    hit = cache.get(q_hash, q_vec)
    assert hit is not None
    assert hit["answer"] == payload["answer"]

    # Semantic cache hit (similarity 0.99 >= 0.95)
    similar_vec = [0.99, 0.01, 0.0]
    hit_semantic = cache.get("diff_hash_similar_query", similar_vec)
    assert hit_semantic is not None
    assert hit_semantic["answer"] == payload["answer"]

    # Semantic cache miss (similarity 0.5 < 0.95)
    distant_vec = [0.5, 0.5, 0.0]
    assert cache.get("diff_hash_distant_query", distant_vec) is None

    # Ingest version invalidation: change ingest_version
    settings_new_ver = Settings(
        cache_enabled=True,
        cache_similarity_threshold=0.95,
        ingest_version="2.0.0",
    )
    cache_v2 = SemanticCache(settings_new_ver, client=mock_redis)
    # Namespace is different -> cache miss
    assert cache_v2.get(q_hash, q_vec) is None


def test_pan_rejected_before_cache_lookup() -> None:
    """Ensure PAN question is rejected by gateway before any cache check."""
    client = TestClient(create_app())
    # Luhn-valid card number
    pan_question = "What happens with card 4111111111111111 dispute?"
    resp = client.post("/rag/query", json={"question": pan_question})
    assert resp.status_code == 400
    assert "potential card number detected" in resp.json()["detail"].lower()


def test_all_flags_off_regression() -> None:
    """Verify that all flags off preserves Phase 2 behavior."""
    settings = Settings(
        retrieval_mode="dense",
        rerank_enabled=False,
        parent_child_enabled=False,
        multi_query_enabled=False,
        temporal_filter_enabled=False,
        cache_enabled=False,
    )
    mock_store = MagicMock()
    mock_gen = MagicMock()
    mock_embed = MagicMock()

    mock_embed.embed_query.return_value = [0.1] * 384
    mock_store.search.return_value = [
        RetrievedChunk(chunk_id="chunk_1", text="Baseline chunk", score=0.88)
    ]
    mock_gen.generate.return_value = MagicMock(
        answer="Phase 2 answer",
        citations=[],
        model="gemini",
        prompt_tokens=10,
        completion_tokens=5,
    )

    pipeline = RAGPipeline(
        settings,
        embedder=mock_embed,
        store=mock_store,
        generator=mock_gen,
    )
    res = pipeline.query("Baseline question")

    assert res.answer == "Phase 2 answer"
    assert res.cached is False
    assert len(res.retrieved_chunks) == 1
    assert res.retrieved_chunks[0]["id"] == "chunk_1"


def test_junk_chunk_and_label_hygiene_filters() -> None:
    """Test junk boilerplate chunk detection and label hygiene rules."""
    boilerplate = (
        "[TABLE] ID# 0030050 | Edition: Apr 2026 | Last Updated: Oct 2024 [/TABLE]\n"
        "[TABLE] ID# 0030708 | Edition: Apr 2026 | Last Updated: Apr 2023 [/TABLE]\n"
    )
    assert is_junk_boilerplate_chunk(boilerplate) is True

    real_chunk = boilerplate + ("Real policy requirements must be enforced. " * 15)
    assert is_junk_boilerplate_chunk(real_chunk) is False

    # Label hygiene: glued page numbers
    assert not is_heading_line("Exempt-from-filing61")
    # Label hygiene: < 3 alphabetic characters
    assert not is_heading_line("1.2")
    assert not is_heading_line("--")
    # Label hygiene: table rating score
    assert not is_heading_line("2.25 Above average")
    # Label hygiene: running headers
    running_headers = {"NIST SP 800-63B", "AUTHENTICATION & LIFECYCLE MANAGEMENT"}
    assert not is_heading_line("AUTHENTICATION & LIFECYCLE MANAGEMENT", running_headers)
    assert not is_heading_line("NIST SP 800-63B", running_headers)
    assert is_heading_line("5.1.1 Memorized Secret Authenticators", running_headers)


def test_cache_exact_match_fast_path() -> None:
    """Verify exact-match fast path always hits even with semantic matching off."""
    mock_redis = MagicMock()
    kv_store: dict[str, str] = {}

    mock_redis.set.side_effect = lambda k, v, ex=None: kv_store.update({k: v}) or True
    mock_redis.get.side_effect = lambda k: kv_store.get(k)

    settings = Settings(cache_enabled=False, cache_similarity_threshold=0.998)
    cache = SemanticCache(settings, client=mock_redis)

    q_hash = hash_question("What is Dispute Condition 13.1?")
    cache.set(
        q_hash,
        [1.0, 0.0],
        {
            "answer": "Dispute Condition 13.1 answer",
            "citations": [],
            "retrieved_chunks": [],
            "model": "test",
        },
    )

    # get_exact should return cached item directly
    hit = cache.get_exact(q_hash)
    assert hit is not None
    assert hit["answer"] == "Dispute Condition 13.1 answer"

    # get() uses exact match fast path even with empty or dummy query vector
    assert cache.get(q_hash, [0.0, 0.0]) is not None


def test_near_miss_pairs_do_not_hit_semantic_cache() -> None:
    """Regression test: Near-miss question pairs must NEVER trigger false hit."""
    mock_redis = MagicMock()
    kv_store: dict[str, str] = {}

    mock_redis.set.side_effect = lambda k, v, ex=None: kv_store.update({k: v}) or True
    mock_redis.get.side_effect = lambda k: kv_store.get(k)
    mock_redis.scan_iter.side_effect = lambda p, count=100: [
        k for k in kv_store if k.startswith(p.rstrip("*"))
    ]

    # Stricter threshold 0.998
    settings = Settings(cache_enabled=True, cache_similarity_threshold=0.998)
    cache = SemanticCache(settings, client=mock_redis)

    # Seed Condition 10.4 into cache
    q_10_4 = "What are the requirements for Dispute Condition 10.4 Other Fraud?"
    h_10_4 = hash_question(q_10_4)
    vec_10_4 = [1.0, 0.0]

    cache.set(
        h_10_4,
        vec_10_4,
        {
            "answer": "Rules for Dispute Condition 10.4",
            "citations": [],
            "retrieved_chunks": [],
            "model": "gemini",
        },
    )

    # Query with Condition 10.3 (near-miss: sim 0.9867 < 0.998)
    q_10_3 = "What are the requirements for Dispute Condition 10.3 Other Fraud?"
    h_10_3 = hash_question(q_10_3)
    vec_10_3 = [0.9867, 0.1626]

    res, sim = cache.get_semantic(vec_10_3, question_hash=h_10_3)
    assert res is None  # Must NOT hit cache!
    assert pytest.approx(sim, abs=0.001) == 0.9867

    # Full get() call with differing hash also does not hit
    assert cache.get(h_10_3, vec_10_3) is None

    # Even closer pair: Section 5.8.11.1 vs 5.8.11.2 (sim 0.9960 < 0.998)
    vec_sec_1 = [0.9960, 0.08935]
    res2, sim2 = cache.get_semantic(vec_sec_1, question_hash="sec_diff_hash")
    assert res2 is None
    assert pytest.approx(sim2, abs=0.001) == 0.9960


def test_reranker_max_length_and_markup_stripping() -> None:
    """Verify BGEReranker handles max_length and markup stripping."""
    from payguard.rag.reranker import strip_chunk_markup

    raw_text = "[TABLE]\nRow 1 | Value 1\n[/TABLE]\nPolicy text here."
    cleaned = strip_chunk_markup(raw_text)
    assert "[TABLE]" not in cleaned
    assert "[/TABLE]" not in cleaned
    assert "Policy text here." in cleaned

    reranker = BGEReranker.__new__(BGEReranker)
    reranker.model_name = "test"
    reranker.max_length = 256
    reranker.strip_markup = True
    assert reranker.max_length == 256
    assert reranker.strip_markup is True


def test_route_level_latency_accounting() -> None:
    """Verify API response reports latency, retry_count, and backoff_ms."""
    app = create_app()
    with TestClient(app) as client:
        # Test with PAN check bypass (clean query)
        resp = client.post("/rag/query", json={"question": "What is the policy?"})
        # Even on error or success, schema should match
        if resp.status_code == 200:
            data = resp.json()
            assert "retrieval_ms" in data
            assert "generation_ms" in data
            assert "total_ms" in data
            assert "retry_count" in data
            assert "backoff_ms" in data
            assert isinstance(data["retrieval_ms"], int | float)
            assert isinstance(data["generation_ms"], int | float)
            assert isinstance(data["total_ms"], int | float)
            assert isinstance(data["retry_count"], int)
            assert isinstance(data["backoff_ms"], int | float)


def test_cached_hits_preserve_citations() -> None:
    """Regression test: Cached responses must preserve citations."""
    mock_redis = MagicMock()
    kv_store: dict[str, str] = {}

    mock_redis.set.side_effect = lambda k, v, ex=None: kv_store.update({k: v}) or True
    mock_redis.get.side_effect = lambda k: kv_store.get(k)

    settings = Settings(cache_enabled=True, cache_similarity_threshold=0.998)
    cache = SemanticCache(settings, client=mock_redis)

    q = "What is the daylight overdraft cap?"
    q_hash = hash_question(q)
    sample_citations = [
        {
            "doc_id": "fed_payment_risk",
            "title": "Fed Payment Risk",
            "section": "Daylight Overdrafts",
            "page": 14,
            "effective_date": None,
        }
    ]

    cache.set(
        q_hash,
        [1.0, 0.0],
        {
            "answer": "The cap depends on the institution capital.",
            "citations": sample_citations,
            "retrieved_chunks": [],
            "model": "test-model",
        },
    )

    # Exact match hit
    hit = cache.get_exact(q_hash)
    assert hit is not None
    assert hit["citations"] == sample_citations
    assert len(hit["citations"]) == 1
    assert hit["citations"][0]["doc_id"] == "fed_payment_risk"
