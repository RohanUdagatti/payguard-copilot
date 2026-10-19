# RAG baseline (Phase 2)

Naive dense RAG over payment/compliance PDFs.

## Flow

1. Parse PDFs with PyMuPDF (page numbers + table blocks).
2. Chunk by heading/section, recursive splitter fallback (≈500–800 tokens, 100 overlap).
3. Embed with the configured `Embedder` (default: local `BAAI/bge-small-en-v1.5`).
4. Upsert into Qdrant with deterministic point IDs.
5. Query: embed → top‑k dense search → citation-enforced LLM answer.

## Configuration

| Variable | Default | Notes |
|----------|---------|-------|
| `LLM_PROVIDER` | `gemini` | `openai` optional |
| `LLM_MODEL` | `gemini-3.8-flash` | from env, never hardcoded in code paths |
| `GEMINI_API_KEY` | — | preferred; `GOOGLE_API_KEY` fallback |
| `EMBEDDING_PROVIDER` | `local` | also `gemini`, `openai` |
| `RAG_TOP_K` | `5` | override per request |
| `RAG_SCORE_THRESHOLD` | `0.2` | dense score floor |
| `QDRANT_COLLECTION` | `payguard_docs` | |

## Metadata

Without `configs/docs_sources.yaml` or `data/raw_docs/manifest.json`:

- `doc_id` = filename stem
- `title` = stem with `_` → spaces
- `version`, `effective_date`, `region` = `null` (never guessed)

A warning is logged per document when metadata files are missing.

## Limits (intentionally deferred)

- No hybrid search (BM25 / sparse)
- No reranker
- No Redis query cache
- No agents / SQL tools
- PAN check is Luhn-only; full Presidio tokenization is later

## Commands

```bash
make up
make ingest                 # or: make ingest INGEST_ARGS=--recreate
curl -s localhost:8000/rag/query -H 'content-type: application/json' \
  -d '{"question":"What is the Federal Reserve payment system risk policy about?"}'
```
