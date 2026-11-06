PYTHON ?= python
PIP ?= pip
COMPOSE ?= docker compose
PACKAGE_EXTRAS ?= .[core,dev,rag,data,ml]

.PHONY: up down test lint format typecheck install env ingest rag-serve eval-gate eval-baseline migrate seed seed-large reset-db train ml-benchmark-kaggle

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

eval-gate:
	$(PYTHON) scripts/eval_gate.py $(EVAL_ARGS)

eval-baseline:
	$(PYTHON) scripts/eval_gate.py --create-baseline $(EVAL_ARGS)

migrate: env
	$(PYTHON) -m alembic upgrade head

seed: env
	$(PYTHON) scripts/seed_db.py --count $(or $(SEED_COUNT),500000)

seed-large: env
	$(PYTHON) scripts/seed_db.py --count 2000000

reset-db: env
	$(PYTHON) -m alembic downgrade base
	$(PYTHON) -m alembic upgrade head

train: env
	$(PYTHON) scripts/train_fraud.py

ml-benchmark-kaggle: env
	$(PYTHON) scripts/benchmark_kaggle.py


