"""Parser and chunker tests using an in-memory PDF fixture."""

from __future__ import annotations

from pathlib import Path

import fitz
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from payguard.rag.ingest.chunker import (
    chunk_document,
    count_tokens,
    is_heading_line,
    make_chunk_id,
)
from payguard.rag.ingest.metadata import DocMetadata, resolve_doc_metadata
from payguard.rag.ingest.parser import PageContent, ParsedDocument, parse_pdf


def _write_fixture_pdf(path: Path) -> None:
    c = canvas.Canvas(str(path), pagesize=letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(72, 720, "SECTION 1 PAYMENT RISK")
    c.setFont("Helvetica", 11)
    c.drawString(72, 700, "Banks must monitor settlement risk daily.")
    c.drawString(72, 680, "Table of limits:")
    c.drawString(72, 660, "Tier | Limit")
    c.drawString(72, 645, "A | 100000")
    c.drawString(72, 630, "B | 250000")
    c.showPage()
    c.setFont("Helvetica-Bold", 14)
    c.drawString(72, 720, "SECTION 2 CONTROLS")
    c.setFont("Helvetica", 11)
    c.drawString(
        72,
        700,
        "Two-factor authentication is required for high-risk transfers.",
    )
    c.save()


def _meta(doc_id: str = "sample_policy") -> DocMetadata:
    return DocMetadata(
        doc_id=doc_id,
        title=doc_id.replace("_", " "),
        version=None,
        effective_date=None,
        region=None,
        metadata_missing=True,
    )


def test_heading_regex_rejects_false_positives() -> None:
    assert not is_heading_line("USD 3,000")
    assert not is_heading_line("1 Based on the technology supported by the Merchant")
    assert is_heading_line("SECTION 1 PAYMENT RISK")
    assert is_heading_line("1.3.3 Marks Use and Marketing Restrictions")


def test_parse_pdf_preserves_pages_and_text(tmp_path: Path) -> None:
    pdf = tmp_path / "sample_policy.pdf"
    _write_fixture_pdf(pdf)
    parsed = parse_pdf(pdf)
    assert len(parsed.pages) == 2
    assert parsed.pages[0].page_number == 1
    assert "PAYMENT RISK" in parsed.pages[0].text
    assert "CONTROLS" in parsed.pages[1].text
    page0 = parsed.pages[0].text
    assert "Tier | Limit" in page0 or "A | 100000" in page0


def test_metadata_fallback_without_manifest(tmp_path: Path) -> None:
    pdf = tmp_path / "fed_payment_risk.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    meta = resolve_doc_metadata(pdf, overrides={})
    assert meta.doc_id == "fed_payment_risk"
    assert meta.title == "fed payment risk"
    assert meta.version is None
    assert meta.effective_date is None
    assert meta.region is None
    assert meta.metadata_missing is True


def test_chunk_ids_deterministic(tmp_path: Path) -> None:
    pdf = tmp_path / "sample_policy.pdf"
    _write_fixture_pdf(pdf)
    parsed = parse_pdf(pdf)
    meta = _meta()
    first = chunk_document(parsed, meta)
    second = chunk_document(parsed, meta)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert first[0].metadata["page_start"] >= 1
    assert first[0].metadata["doc_id"] == "sample_policy"
    assert make_chunk_id("a", 0, "hello") == make_chunk_id("a", 0, "hello")


def test_packed_chunks_bounded_and_page_span() -> None:
    """Synthetic multi-page doc packs into <=800 token chunks with page spans."""
    pages: list[PageContent] = []
    # Parent 1.1 family — enough text to force packing / multi-window
    # ~200 tokens each so three 1.1* sections pack into one multi-page window
    body_a = "Settlement risk monitoring requires daily review. " * 35
    body_b = "Liquidity buffers protect payment systems under stress. " * 35
    body_c = "Intraday credit policies limit daylight overdraft exposure. " * 35
    pages.append(
        PageContent(
            1,
            f"1.1 Risk Overview\n{body_a}",
            [],
        )
    )
    pages.append(
        PageContent(
            2,
            f"1.1.1 Monitoring\n{body_b}",
            [],
        )
    )
    pages.append(
        PageContent(
            3,
            f"1.1.2 Controls\n{body_c}",
            [],
        )
    )
    # Different parent — should not pack into previous family
    pages.append(
        PageContent(
            4,
            (
                "2.1 Authentication\n"
                "Two-factor authentication is mandatory for high-risk transfers. "
            )
            * 20,
            [],
        )
    )
    parsed = ParsedDocument(path=Path("synthetic.pdf"), pages=pages)
    chunks = chunk_document(parsed, _meta("synthetic"))

    assert chunks
    for chunk in chunks:
        assert count_tokens(chunk.text) <= 800
        assert chunk.metadata["page_start"] <= chunk.metadata["page_end"]
        assert "section" in chunk.metadata
        assert "section_end" in chunk.metadata

    # No chunk under 100 tokens except when the doc has only one chunk
    if len(chunks) > 1:
        for chunk in chunks:
            assert count_tokens(chunk.text) >= 100

    # Page span for a window that absorbed 1.1* content should cover multiple pages
    multi = [
        c
        for c in chunks
        if str(c.metadata["section"]).startswith("1.1")
        and c.metadata["page_end"] > c.metadata["page_start"]
    ]
    assert multi, "expected at least one packed window spanning pages"


def test_no_tiny_chunks_except_singleton_doc() -> None:
    pages = [
        PageContent(1, "SECTION 1 SHORT\nTiny.", []),
    ]
    parsed = ParsedDocument(path=Path("short.pdf"), pages=pages)
    chunks = chunk_document(parsed, _meta("short"))
    assert len(chunks) == 1
    # Only chunk may be under 100 tokens
    assert count_tokens(chunks[0].text) < 100


def test_pymupdf_table_blocks_not_single_line(tmp_path: Path) -> None:
    """Tables extracted via find_tables stay multi-line when present."""
    pdf = tmp_path / "table_doc.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Col1 | Col2\nX | 1\nY | 2")
    doc.save(pdf)
    doc.close()
    parsed = parse_pdf(pdf)
    combined = parsed.pages[0].text
    assert "\n" in combined or parsed.pages[0].tables
