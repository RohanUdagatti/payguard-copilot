# syntax=docker/dockerfile:1.7

FROM python:3.11.11-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip==25.0.1 \
    && /opt/venv/bin/pip install ".[core]"


FROM python:3.11.11-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

RUN groupadd --gid 10001 payguard \
    && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin payguard

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --chown=payguard:payguard src ./src
COPY --chown=payguard:payguard pyproject.toml README.md ./

USER payguard

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["uvicorn", "payguard.gateway.main:app", "--host", "0.0.0.0", "--port", "8000"]
