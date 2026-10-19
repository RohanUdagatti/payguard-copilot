# PayGuard Copilot

Enterprise GenAI platform for payment disputes, fraud and compliance.

Phase 0 provides a production-style repository scaffold: FastAPI health endpoint,
configuration via environment variables, Docker Compose infrastructure, and CI.

## Requirements

- Python 3.11
- Docker and Docker Compose
- Make

## Setup

```bash
cp .env.example .env
python -m pip install -e ".[core,dev]"
```

## Run

```bash
make up
curl http://localhost:8000/health
```

`make up` waits until every service (including the API) reports healthy before returning.
This starts PostgreSQL, Redis, Qdrant, Langfuse (v2 + its own Postgres), and the API.

Stop services:

```bash
make down
```

## Test and quality

```bash
make test
make lint
make typecheck
make format
```

## Layout

```text
src/payguard/{gateway,rag,data,ml,agents,security,eval,platform}
tests/
scripts/
docs/
configs/
```

## Notes

- No business logic, LLM calls, or RAG code in Phase 0.
- Secrets come from `.env` (never commit real credentials).
- Langfuse is pinned to major version 2 to avoid the heavier v3 dependency stack.
