PYTHON ?= python
PIP ?= pip
COMPOSE ?= docker compose
PACKAGE_EXTRAS ?= .[core,dev]

.PHONY: up down test lint format typecheck install env

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
