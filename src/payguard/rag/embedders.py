"""Embedder implementations selected by EMBEDDING_PROVIDER."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

from pydantic import SecretStr
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from payguard.rag.interfaces import Embedder

if TYPE_CHECKING:
    from payguard.config import Settings

logger = logging.getLogger(__name__)

T = TypeVar("T")


class MissingProviderKeyError(RuntimeError):
    """Raised when the active embedding provider is missing its API key."""


def _with_retries(attempts: int, fn: Callable[[], T]) -> T:
    """Run ``fn`` with exponential backoff; log clear 429 warnings."""

    @retry(
        reraise=True,
        stop=stop_after_attempt(attempts),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception_type(Exception),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _inner() -> T:
        return fn()

    return _inner()


class LocalBGEEmbedder(Embedder):
    """CPU-friendly local embeddings via sentence-transformers."""

    def __init__(self, model_name: str, batch_size: int = 32) -> None:
        from sentence_transformers import SentenceTransformer

        self._model_name = model_name
        self._batch_size = batch_size
        self._model = SentenceTransformer(model_name)
        dim = self._model.get_sentence_embedding_dimension()
        if dim is None:
            raise RuntimeError(f"Could not resolve dimension for {model_name}")
        self._dimension = int(dim)

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class OpenAIEmbedder(Embedder):
    """OpenAI text-embedding models via langchain-openai."""

    def __init__(
        self,
        api_key: str,
        model_name: str,
        *,
        batch_size: int = 32,
        timeout_seconds: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        from langchain_openai import OpenAIEmbeddings

        self._model_name = model_name
        self._batch_size = batch_size
        self._max_retries = max_retries
        self._client = OpenAIEmbeddings(
            api_key=SecretStr(api_key),
            model=model_name,
            timeout=timeout_seconds,
            max_retries=0,
        )
        self._dimension = 1536 if "3-small" in model_name else 3072

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]

            def _call(batch: list[str] = batch) -> list[list[float]]:
                try:
                    return list(self._client.embed_documents(batch))
                except Exception as exc:  # noqa: BLE001
                    if "429" in str(exc):
                        logger.warning(
                            "embedding_rate_limited",
                            extra={
                                "provider": "openai",
                                "model": self._model_name,
                            },
                        )
                    raise

            out.extend(_with_retries(self._max_retries, _call))
        return out

    def embed_query(self, text: str) -> list[float]:
        def _call() -> list[float]:
            try:
                return list(self._client.embed_query(text))
            except Exception as exc:  # noqa: BLE001
                if "429" in str(exc):
                    logger.warning(
                        "embedding_rate_limited",
                        extra={"provider": "openai", "model": self._model_name},
                    )
                raise

        return _with_retries(self._max_retries, _call)


class GeminiEmbedder(Embedder):
    """Google Gemini embeddings via langchain-google-genai."""

    def __init__(
        self,
        api_key: str,
        model_name: str,
        *,
        batch_size: int = 32,
        timeout_seconds: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        self._model_name = model_name
        self._batch_size = batch_size
        self._max_retries = max_retries
        self._timeout_seconds = timeout_seconds
        self._client = GoogleGenerativeAIEmbeddings(
            google_api_key=SecretStr(api_key),
            model=model_name,
        )
        self._dimension = 768

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]

            def _call(batch: list[str] = batch) -> list[list[float]]:
                try:
                    return list(self._client.embed_documents(batch))
                except Exception as exc:  # noqa: BLE001
                    if "429" in str(exc):
                        logger.warning(
                            "embedding_rate_limited",
                            extra={
                                "provider": "gemini",
                                "model": self._model_name,
                            },
                        )
                    raise

            out.extend(_with_retries(self._max_retries, _call))
        return out

    def embed_query(self, text: str) -> list[float]:
        def _call() -> list[float]:
            try:
                return list(self._client.embed_query(text))
            except Exception as exc:  # noqa: BLE001
                if "429" in str(exc):
                    logger.warning(
                        "embedding_rate_limited",
                        extra={"provider": "gemini", "model": self._model_name},
                    )
                raise

        return _with_retries(self._max_retries, _call)


def build_embedder(settings: Settings) -> Embedder:
    """Construct the embedder selected by EMBEDDING_PROVIDER."""
    try:
        key = settings.require_active_embedding_key()
    except ValueError as exc:
        raise MissingProviderKeyError(str(exc)) from exc

    provider = settings.embedding_provider
    if provider == "local":
        return LocalBGEEmbedder(
            settings.embedding_model_local,
            batch_size=settings.embedding_batch_size,
        )
    if provider == "openai":
        assert key is not None
        return OpenAIEmbedder(
            api_key=key,
            model_name=settings.embedding_model_openai,
            batch_size=settings.embedding_batch_size,
            timeout_seconds=settings.embedding_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )
    if provider == "gemini":
        assert key is not None
        return GeminiEmbedder(
            api_key=key,
            model_name=settings.embedding_model_gemini,
            batch_size=settings.embedding_batch_size,
            timeout_seconds=settings.embedding_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )
    raise MissingProviderKeyError(f"Unsupported EMBEDDING_PROVIDER={provider!r}")
