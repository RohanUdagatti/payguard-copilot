PYTHON ?= python
PIP ?= pip
COMPOSE ?= docker compose
PACKAGE_EXTRAS ?= .[core,dev,rag]

.PHONY: up down test lint format typecheck install env ingest rag-serve

ingest: env
	QDRANT_HOST=localhost $(PYTHON) scripts/ingest.py --docs-dir data/raw_docs $(INGEST_ARGS)

rag-serve: env
	QDRANT_HOST=localhost $(PYTHON) -m uvicorn payguard.gateway.main:app --host 0.0.0.0 --port 8000

env:
	@test -f .env || cp .env.example .env

install:
	$(PIP) install -e "$(PACKAGE_EXTRAS)"

up: env
	$(COMPOSE) up -d --build
	$(COMPOSE) up -d --wait
	$(COMPOSE) ps

down:
	$(COMPOSE) down

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m black --check src tests

format:
	$(PYTHON) -m ruff check --fix src tests
	$(PYTHON) -m black src tests

typecheck:
	$(PYTHON) -m mypy src/payguard
