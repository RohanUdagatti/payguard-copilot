"""Citation-enforced answer generation from retrieved context only."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from payguard.config import Settings
from payguard.rag.interfaces import RetrievedChunk

logger = logging.getLogger(__name__)

NOT_FOUND_MESSAGE = "Not found in the provided documents."

_SYSTEM_PROMPT = """You are PayGuard Copilot, a compliance assistant.
Answer ONLY using the provided context chunks.
If the context is insufficient to answer, return exactly this JSON:
{"answer":"Not found in the provided documents.","citations":[]}
Otherwise return JSON with this schema:
{"answer":"<string>","citations":[{"doc_id":"","title":"","section":"","page":0,"effective_date":null}]}
Rules:
- Never invent citations.
- Every citation must map to a provided chunk.
- page must be an integer from the chunk metadata (use page_start).
- Do not include any text outside the JSON object.
"""


@dataclass(frozen=True)
class GenerationResult:
    """Structured generation output."""

    answer: str
    citations: list[dict[str, Any]]
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None


class MissingLLMKeyError(RuntimeError):
    """Raised when the active LLM provider is missing its API key."""


def _format_context(chunks: list[RetrievedChunk]) -> str:
    blocks: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        meta = chunk.metadata
        header = (
            f"[chunk {idx}] chunk_id={chunk.chunk_id} "
            f"doc_id={meta.get('doc_id')} title={meta.get('title')} "
            f"section={meta.get('section')} page_start={meta.get('page_start')} "
            f"effective_date={meta.get('effective_date')}"
        )
        blocks.append(f"{header}\n{chunk.text}")
    return "\n\n".join(blocks)


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {"answer": NOT_FOUND_MESSAGE, "citations": []}
    try:
        data = json.loads(match.group(0))
        if isinstance(data, dict):
            return data
        return {"answer": NOT_FOUND_MESSAGE, "citations": []}
    except json.JSONDecodeError:
        return {"answer": NOT_FOUND_MESSAGE, "citations": []}


def _filter_citations(
    citations: list[Any],
    chunks: list[RetrievedChunk],
) -> list[dict[str, Any]]:
    allowed = {c.metadata.get("doc_id") for c in chunks}
    cleaned: list[dict[str, Any]] = []
    for item in citations:
        if not isinstance(item, dict):
            continue
        doc_id = item.get("doc_id")
        if doc_id not in allowed:
            continue
        cleaned.append(
            {
                "doc_id": doc_id,
                "title": item.get("title"),
                "section": item.get("section"),
                "page": item.get("page"),
                "effective_date": item.get("effective_date"),
            }
        )
    return cleaned


class AnswerGenerator:
    """LLM wrapper that enforces JSON answers with citations."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        try:
            self._api_key = settings.require_active_llm_key()
        except ValueError as exc:
            raise MissingLLMKeyError(str(exc)) from exc
        self._model = settings.llm_model
        self._provider = settings.llm_provider
        self._llm = self._build_llm()

    def _build_llm(self) -> Any:
        from pydantic import SecretStr

        if self._provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI

            # Pass key explicitly (GEMINI_API_KEY); avoid relying on GOOGLE_API_KEY env.
            kwargs: dict[str, Any] = {
                "model": self._model,
                "google_api_key": SecretStr(self._api_key),
                "temperature": 0,
                "timeout": self.settings.llm_timeout_seconds,
                "max_retries": 0,
            }
            return ChatGoogleGenerativeAI(**kwargs)
        if self._provider == "openai":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=self._model,
                api_key=SecretStr(self._api_key),
                temperature=0,
                timeout=self.settings.llm_timeout_seconds,
                max_retries=0,
            )
        raise MissingLLMKeyError(f"Unsupported LLM_PROVIDER={self._provider!r}")

    def generate(
        self,
        question: str,
        chunks: list[RetrievedChunk],
    ) -> GenerationResult:
        """Generate an answer from retrieved chunks only."""
        if not chunks:
            return GenerationResult(
                answer=NOT_FOUND_MESSAGE,
                citations=[],
                model=self._model,
                prompt_tokens=None,
                completion_tokens=None,
            )

        prompt = (
            f"{_SYSTEM_PROMPT}\n\n"
            f"CONTEXT:\n{_format_context(chunks)}\n\n"
            f"QUESTION:\n{question}\n"
        )

        @retry(
            reraise=True,
            stop=stop_after_attempt(self.settings.llm_max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=30),
            retry=retry_if_exception_type(Exception),
            before_sleep=before_sleep_log(logger, logging.WARNING),
        )
        def _call() -> Any:
            try:
                return self._llm.invoke(prompt)
            except Exception as exc:  # noqa: BLE001
                if "429" in str(exc):
                    logger.warning(
                        "llm_rate_limited",
                        extra={"provider": self._provider, "model": self._model},
                    )
                raise

        message = _call()
        content = getattr(message, "content", str(message))
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        parsed = _extract_json(str(content))
        answer = str(parsed.get("answer") or NOT_FOUND_MESSAGE)
        citations = _filter_citations(parsed.get("citations") or [], chunks)

        if answer.strip() != NOT_FOUND_MESSAGE and not citations:
            # Prefer honest not-found over uncited claims
            answer = NOT_FOUND_MESSAGE

        usage = getattr(message, "usage_metadata", None) or {}
        return GenerationResult(
            answer=answer,
            citations=citations,
            model=self._model,
            prompt_tokens=usage.get("input_tokens"),
            completion_tokens=usage.get("output_tokens"),
        )
