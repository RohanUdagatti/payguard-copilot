#!/usr/bin/env python3
"""Benchmark retrieval configurations on eval/manual_questions.yaml."""

from __future__ import annotations

import argparse
import logging
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from payguard.config import Settings
from payguard.rag.embedders import build_embedder
from payguard.rag.ingest.chunker import deduplicate_to_parents
from payguard.rag.interfaces import RetrievedChunk
from payguard.rag.reranker import BGEReranker
from payguard.rag.vectorstore import QdrantVectorStore

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("payguard.benchmark")


def is_match(
    chunk: RetrievedChunk,
    expected_doc_id: str,
    section_hint: str | None,
) -> bool:
    """Return True if chunk matches expected_doc_id and section hint (when present)."""
    if chunk.metadata.get("doc_id") != expected_doc_id:
        return False
    if not section_hint:
        return True
    sec = str(chunk.metadata.get("section", "")).lower()
    hint = section_hint.lower()
    # Check substring containment in either direction or common token overlap
    if hint in sec or sec in hint:
        return True
    hint_words = set(hint.split())
    sec_words = set(sec.split())
    return len(hint_words.intersection(sec_words)) >= 2


def evaluate_ranking(
    results: list[RetrievedChunk],
    expected_doc_id: str,
    section_hint: str | None,
) -> tuple[float, float, float]:
    """Compute hit@1, hit@5, and reciprocal rank (RR)."""
    h1 = 1.0 if results and is_match(results[0], expected_doc_id, section_hint) else 0.0
    h5 = (
        1.0
        if any(is_match(c, expected_doc_id, section_hint) for c in results[:5])
        else 0.0
    )

    rr = 0.0
    for rank, c in enumerate(results[:5], start=1):
        if is_match(c, expected_doc_id, section_hint):
            rr = 1.0 / rank
            break
    return h1, h5, rr


def run_benchmark_for_config(
    name: str,
    questions: list[dict[str, Any]],
    settings: Settings,
    embedder: Any,
    store: QdrantVectorStore,
    reranker: BGEReranker | None,
    *,
    hybrid: bool,
    use_rerank: bool,
    parent_child: bool,
) -> dict[str, Any]:
    """Run retrieval-only benchmark for a specific configuration."""
    latencies: list[float] = []
    hits1: list[float] = []
    hits5: list[float] = []
    rrs: list[float] = []

    # Configure store retrieval mode
    store.retrieval_mode = "hybrid" if hybrid else "dense"
    search_k = 20 if use_rerank else 5

    for q in questions:
        query_text = q["question"]
        doc_id = q["expected_doc_id"]
        sec_hint = q.get("expected_section_hint")

        t0 = time.perf_counter()
        q_vec = embedder.embed_query(query_text)

        kwargs: dict[str, Any] = {
            "top_k": search_k,
            "filter_is_child": bool(parent_child),
        }
        if hybrid:
            kwargs["query_text"] = query_text

        retrieved = store.search(q_vec, **kwargs)

        if parent_child:
            retrieved = deduplicate_to_parents(retrieved)

        if use_rerank and reranker:
            retrieved = reranker.rerank(query_text, retrieved, top_k=5)
        else:
            retrieved = retrieved[:5]

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(elapsed_ms)

        h1, h5, rr = evaluate_ranking(retrieved, doc_id, sec_hint)
        hits1.append(h1)
        hits5.append(h5)
        rrs.append(rr)

    sorted_lats = sorted(latencies)
    p50 = statistics.median(sorted_lats)
    p95_idx = int(0.95 * len(sorted_lats))
    p95 = sorted_lats[min(p95_idx, len(sorted_lats) - 1)]

    return {
        "config": name,
        "hit@1": f"{sum(hits1) / len(hits1) * 100:.1f}%",
        "hit@5": f"{sum(hits5) / len(hits5) * 100:.1f}%",
        "mrr": f"{sum(rrs) / len(rrs):.3f}",
        "p50_ms": f"{p50:.2f} ms",
        "p95_ms": f"{p95:.2f} ms",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark PayGuard retrieval configs")
    parser.add_argument(
        "--questions-file",
        type=Path,
        default=Path("eval/manual_questions.yaml"),
        help="Path to questions YAML",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("eval/results/retrieval_ablation.md"),
        help="Path to write results markdown",
    )
    args = parser.parse_args()

    if not args.questions_file.is_file():
        print(
            f"ERROR: questions file not found: {args.questions_file}", file=sys.stderr
        )
        return 1

    with open(args.questions_file, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    questions = data.get("questions", [])
    valid_questions = [
        q
        for q in questions
        if q.get("question")
        and not q["question"].startswith("TODO")
        and q.get("expected_doc_id")
        and not q["expected_doc_id"].startswith("TODO")
    ]

    if len(valid_questions) < 10:
        print(
            f"ERROR: Only {len(valid_questions)} filled questions found in "
            f"{args.questions_file}. At least 10 filled questions are required. "
            "Never invent questions or numbers.",
            file=sys.stderr,
        )
        return 1

    settings = Settings()
    embedder = build_embedder(settings)
    store = QdrantVectorStore(
        url=settings.qdrant_url,
        collection=settings.qdrant_collection,
        api_key=settings.qdrant_api_key or None,
    )
    reranker = BGEReranker(settings.rerank_model)

    configs = [
        ("dense", False, False, False),
        ("hybrid", True, False, False),
        ("hybrid+rerank", True, True, False),
        ("hybrid+rerank+parent-child", True, True, True),
    ]

    rows: list[dict[str, Any]] = []
    for name, hybrid, use_rerank, parent_child in configs:
        row = run_benchmark_for_config(
            name,
            valid_questions,
            settings,
            embedder,
            store,
            reranker,
            hybrid=hybrid,
            use_rerank=use_rerank,
            parent_child=parent_child,
        )
        rows.append(row)

    # Format Markdown Table
    headers = ["Configuration", "Hit@1", "Hit@5", "MRR", "p50 Latency", "p95 Latency"]
    col_keys = ["config", "hit@1", "hit@5", "mrr", "p50_ms", "p95_ms"]
    col_widths = [
        max(len(h), max(len(str(r[k])) for r in rows))
        for h, k in zip(headers, col_keys, strict=True)
    ]

    header_line = (
        "| "
        + " | ".join(h.ljust(w) for h, w in zip(headers, col_widths, strict=True))
        + " |"
    )
    sep_line = "|-" + "-|-".join("-" * w for w in col_widths) + "-|"
    data_lines = [
        "| "
        + " | ".join(
            str(r[k]).ljust(w) for k, w in zip(col_keys, col_widths, strict=True)
        )
        + " |"
        for r in rows
    ]

    md_table = "\n".join([header_line, sep_line] + data_lines)
    print("\n" + md_table + "\n")

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        f.write("# Retrieval Ablation Benchmark Results\n\n")
        f.write(
            f"Evaluated on {len(valid_questions)} questions from "
            f"`{args.questions_file}`.\n\n"
        )
        f.write(md_table + "\n")

    print(f"Results written to {args.output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
