#!/usr/bin/env python3
"""Test live /rag/query with best config (hybrid + cache) measuring route-level latencies."""

from __future__ import annotations

import json
import time

from fastapi.testclient import TestClient

from payguard.config import get_settings
from payguard.gateway.main import create_app


def main() -> None:
    settings = get_settings()
    settings.retrieval_mode = "hybrid"
    settings.cache_enabled = True
    settings.rerank_enabled = False
    settings.parent_child_enabled = False

    app = create_app(settings)

    question = "What is the Federal Reserve payment system risk policy about?"
    payload = {"question": question}

    print("=== LIVE /rag/query WITH BEST CONFIG (hybrid + cache) ===")
    print(f"Question: {question}\n")

    # Use context manager so FastAPI lifespan (embedding pre-warming) executes BEFORE testing requests
    with TestClient(app) as client:
        # First call: cache miss (or pre-warmed miss)
        print("--- FIRST CALL (Cache Miss -> Retrieval + LLM Generation) ---")
        t0 = time.perf_counter()
        resp1 = client.post("/rag/query", json=payload)
        t1 = time.perf_counter()
        data1 = resp1.json()

        print(f"Status Code: {resp1.status_code}")
        print(f"Answer: {data1.get('answer')}")
        print(f"Citations: {json.dumps(data1.get('citations'), indent=2)}")
        print(f"Cached Flag: {data1.get('cached')}")
        print("Stage Timings Breakdown:")
        print(f"  Retrieval MS : {data1.get('retrieval_ms'):.2f} ms")
        print(f"  Generation MS: {data1.get('generation_ms'):.2f} ms")
        print(f"  Total MS     : {data1.get('total_ms'):.2f} ms")
        print(f"  Wall Time    : {(t1 - t0)*1000:.2f} ms\n")

        # Second call: cache hit (Redis)
        print("--- SECOND CALL (Cache Hit -> Redis Exact/Semantic Cache) ---")
        t2 = time.perf_counter()
        resp2 = client.post("/rag/query", json=payload)
        t3 = time.perf_counter()
        data2 = resp2.json()

        print(f"Status Code: {resp2.status_code}")
        print(f"Answer: {data2.get('answer')}")
        print(f"Cached Flag: {data2.get('cached')}")
        print("Stage Timings Breakdown:")
        print(f"  Retrieval MS : {data2.get('retrieval_ms'):.2f} ms")
        print(f"  Generation MS: {data2.get('generation_ms'):.2f} ms")
        print(f"  Total MS     : {data2.get('total_ms'):.2f} ms")
        print(f"  Wall Time    : {(t3 - t2)*1000:.2f} ms\n")

        print("=== LATENCY COMPARISON ===")
        print(f"First Call Total MS  (un-cached): {data1.get('total_ms'):.2f} ms (wall: {(t1 - t0)*1000:.2f} ms)")
        print(f"Second Call Total MS (cached)   : {data2.get('total_ms'):.2f} ms (wall: {(t3 - t2)*1000:.2f} ms)")
        speedup = data1.get("total_ms", 1.0) / max(data2.get("total_ms", 0.01), 0.01)
        print(f"Speedup                         : {speedup:.1f}x faster")


if __name__ == "__main__":
    main()

