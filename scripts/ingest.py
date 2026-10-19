#!/usr/bin/env python3
"""CLI to ingest PDFs from data/raw_docs into Qdrant."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from payguard.config import get_settings
from payguard.rag.embedders import MissingProviderKeyError, build_embedder
from payguard.rag.ingest.chunker import chunk_document, count_tokens
from payguard.rag.ingest.metadata import load_metadata_index, resolve_doc_metadata
from payguard.rag.ingest.parser import parse_pdf
from payguard.rag.vectorstore import DimensionMismatchError, QdrantVectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("payguard.ingest")


def _parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Ingest PDFs into Qdrant")
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=Path(settings.rag_docs_dir),
        help="Directory containing PDF files",
    )
    parser.add_argument(
        "--collection",
        default=settings.qdrant_collection,
        help="Qdrant collection name",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate the collection",
    )
    return parser.parse_args()


def main() -> int:
    """Run ingestion and print a summary."""
    args = _parse_args()
    settings = get_settings()
    docs_dir: Path = args.docs_dir

    if not docs_dir.is_dir():
        print(f"ERROR: docs dir not found: {docs_dir}", file=sys.stderr)
        return 1

    pdfs = sorted(docs_dir.glob("*.pdf"))
    if not pdfs:
        print(f"ERROR: no PDF files in {docs_dir}", file=sys.stderr)
        return 1

    try:
        embedder = build_embedder(settings)
    except (MissingProviderKeyError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    store = QdrantVectorStore(
        url=settings.qdrant_url,
        collection=args.collection,
        api_key=settings.qdrant_api_key or None,
    )

    try:
        store.ensure_collection(embedder.dimension, recreate=args.recreate)
    except DimensionMismatchError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    overrides = load_metadata_index(docs_dir)
    started = time.perf_counter()
    total_chunks = 0
    total_tokens = 0

    for pdf in pdfs:
        logger.info("ingest_start", extra={"path": str(pdf)})
        meta = resolve_doc_metadata(pdf, overrides)
        parsed = parse_pdf(pdf)
        chunks = chunk_document(parsed, meta)
        store.delete_by_doc_id(meta.doc_id)
        vectors = embedder.embed_documents([c.text for c in chunks])
        written = store.upsert(chunks, vectors)
        tokens = sum(count_tokens(c.text) for c in chunks)
        total_chunks += written
        total_tokens += tokens
        print(
            f"  ingested {pdf.name}: pages={len(parsed.pages)} "
            f"chunks={written} tokens≈{tokens} doc_id={meta.doc_id}"
        )

    elapsed = time.perf_counter() - started
    print("\n=== Ingest summary ===")
    print(f"docs:        {len(pdfs)}")
    print(f"chunks:      {total_chunks}")
    print(f"tokens≈:     {total_tokens}")
    print(f"collection:  {args.collection}")
    print(f"embedder:    {embedder.model_name} (dim={embedder.dimension})")
    print(f"time_sec:    {elapsed:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
