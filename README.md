# PayGuard Copilot

Enterprise GenAI platform for payment disputes, fraud and compliance.

## Requirements

- Python 3.11
- Docker and Docker Compose
- Make
- `GEMINI_API_KEY` in local `.env` (for LLM answers)

## Setup

```bash
cp .env.example .env
# add GEMINI_API_KEY=... to .env
conda activate payguard   # or any Python 3.11 env
pip install -e ".[core,dev,rag]"
```

## Run infrastructure + API

```bash
make up
curl http://localhost:8000/health
```

## RAG baseline (Phase 2)

Put PDFs in `data/raw_docs/` (gitignored), then:

```bash
make ingest
# rebuild collection after changing embedder:
# make ingest INGEST_ARGS=--recreate

curl -s http://localhost:8000/rag/query \
  -H 'content-type: application/json' \
  -d '{"question":"What does the Federal Reserve payment system risk policy cover?"}'
```

Default embedding provider is **local** `BAAI/bge-small-en-v1.5`.
Default LLM provider is **gemini** (`LLM_MODEL` from env).

Design notes: [docs/rag_baseline.md](docs/rag_baseline.md).

## Test and quality

```bash
make test
make lint
make typecheck
```

## Layout

```text
src/payguard/{gateway,rag,data,ml,agents,security,eval,platform}
tests/
scripts/
docs/
configs/
eval/
```

## Notes

- Secrets stay in `.env` (never commit).
- `/data/` is gitignored so PDFs are not committed.
- Langfuse is pinned to v2.
