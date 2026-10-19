"""Document metadata loading. Missing fields are null — never guessed."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DocMetadata:
    """Known metadata for a source document."""

    doc_id: str
    title: str
    version: str | None
    effective_date: str | None
    region: str | None
    metadata_missing: bool


def _as_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data if isinstance(data, dict) else {}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def _index_from_sources(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Normalize docs_sources.yaml / manifest formats into doc_id -> fields."""
    index: dict[str, dict[str, Any]] = {}

    documents = data.get("documents")
    if isinstance(documents, list):
        for item in documents:
            if not isinstance(item, dict):
                continue
            doc_id = item.get("doc_id") or item.get("id") or item.get("filename")
            if not doc_id:
                continue
            index[str(doc_id)] = item

    # Allow top-level mapping of doc_id -> metadata
    for key, value in data.items():
        if key in {"documents", "version"}:
            continue
        if isinstance(value, dict) and (
            "title" in value or "effective_date" in value or "region" in value
        ):
            index[str(key)] = value

    return index


def load_metadata_index(
    docs_dir: Path,
    *,
    sources_path: Path | None = None,
    manifest_path: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Load optional docs_sources.yaml and manifest.json overrides."""
    index: dict[str, dict[str, Any]] = {}
    sources = sources_path or Path("configs/docs_sources.yaml")
    manifest = manifest_path or (docs_dir / "manifest.json")
    index.update(_index_from_sources(_load_yaml(sources)))
    index.update(_index_from_sources(_load_json(manifest)))
    return index


def resolve_doc_metadata(
    pdf_path: Path,
    overrides: dict[str, dict[str, Any]] | None = None,
) -> DocMetadata:
    """Resolve metadata for a PDF.

    Without Phase 1 artifacts:
    - doc_id = filename stem
    - title = stem with underscores replaced by spaces
    - version, effective_date, region = null
    """
    stem = pdf_path.stem
    defaults = {
        "doc_id": stem,
        "title": stem.replace("_", " "),
        "version": None,
        "effective_date": None,
        "region": None,
    }
    override = (overrides or {}).get(stem, {})
    # Also allow lookup by full filename
    if not override:
        override = (overrides or {}).get(pdf_path.name, {})

    merged = {**defaults, **{k: v for k, v in override.items() if v is not None}}
    metadata_missing = not bool(override)
    if metadata_missing:
        logger.warning(
            "document_metadata_missing",
            extra={"doc_id": stem, "path": str(pdf_path)},
        )

    return DocMetadata(
        doc_id=str(merged["doc_id"]),
        title=str(merged["title"]),
        version=_as_optional_str(merged.get("version")),
        effective_date=_as_optional_str(merged.get("effective_date")),
        region=_as_optional_str(merged.get("region")),
        metadata_missing=metadata_missing,
    )
