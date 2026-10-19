"""PDF parsing with PyMuPDF, preserving page numbers and table blocks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz


@dataclass(frozen=True)
class PageContent:
    """Structured content for a single PDF page."""

    page_number: int
    text: str
    tables: list[str]


@dataclass(frozen=True)
class ParsedDocument:
    """A parsed PDF with ordered pages."""

    path: Path
    pages: list[PageContent]


def _extract_tables(page: fitz.Page) -> list[str]:
    """Extract tables as multi-line text blocks (not flattened to one line)."""
    tables: list[str] = []
    try:
        finder = page.find_tables()
    except Exception:  # noqa: BLE001
        return tables

    for table in finder.tables:
        rows: list[str] = []
        try:
            extracted = table.extract()
        except Exception:  # noqa: BLE001
            continue
        for row in extracted:
            cells = [str(cell).strip() if cell is not None else "" for cell in row]
            rows.append(" | ".join(cells))
        if rows:
            tables.append("\n".join(rows))
    return tables


def parse_pdf(path: Path | str) -> ParsedDocument:
    """Parse a PDF into structured pages with table blocks preserved."""
    pdf_path = Path(path)
    doc = fitz.open(pdf_path)
    pages: list[PageContent] = []
    try:
        for index in range(doc.page_count):
            page = doc.load_page(index)
            text = page.get_text("text")
            tables = _extract_tables(page)
            pages.append(PageContent(page_number=index + 1, text=text, tables=tables))
    finally:
        doc.close()
    return ParsedDocument(path=pdf_path, pages=pages)


def page_to_text(page: PageContent) -> str:
    """Join page text and table blocks for chunking."""
    parts = [page.text.strip()]
    for table in page.tables:
        parts.append(f"[TABLE]\n{table}\n[/TABLE]")
    return "\n\n".join(p for p in parts if p)
