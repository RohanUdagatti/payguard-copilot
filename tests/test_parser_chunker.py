"""Parser and chunker tests using an in-memory PDF fixture."""

from __future__ import annotations

from pathlib import Path

import fitz
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from payguard.rag.ingest.chunker import (
    _split_line_heading,
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
    assert not is_heading_line("3.000")
    assert not is_heading_line("ID# 0025973")
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
    assert any(
        c.metadata["section"] == "1.1 Risk Overview"
        and c.metadata["section_end"] != c.metadata["section"]
        and c.metadata["page_start"] == 1
        and c.metadata["page_end"] >= 2
        for c in chunks
    ), "expected sibling sections to pack into a multi-page window"


def test_no_tiny_chunks_except_singleton_doc() -> None:
    pages = [
        PageContent(1, "SECTION 1 SHORT\nTiny.", []),
    ]
    parsed = ParsedDocument(path=Path("short.pdf"), pages=pages)
    chunks = chunk_document(parsed, _meta("short"))
    assert len(chunks) == 1
    # Only chunk may be under 100 tokens
    assert count_tokens(chunks[0].text) < 100


def test_toc_style_page_is_excluded_from_chunks() -> None:
    pages = [
        PageContent(
            1,
            "Payment Risk........................ 3\n"
            "Control Requirements................ 7\n"
            "Reporting........................... 12\n"
            "Settlement Monitoring............... 16",
            [],
        ),
        PageContent(2, "SECTION 1 PAYMENT RISK\nBanks must monitor risk daily.", []),
    ]
    parsed = ParsedDocument(path=Path("toc.pdf"), pages=pages)
    chunks = chunk_document(parsed, _meta("toc"))

    assert len(chunks) == 1
    assert "TABLE OF CONTENTS" not in chunks[0].text
    assert chunks[0].metadata["page_start"] == 2


def test_tiny_chunk_merges_with_neighbour_when_it_fits() -> None:
    body = "Settlement risk monitoring requires daily review. " * 105
    pages = [
        PageContent(1, f"SECTION 1 PAYMENT RISK\n{body}", []),
        PageContent(2, "SECTION 2 REPORTING\nReport exceptions.", []),
    ]
    parsed = ParsedDocument(path=Path("tiny.pdf"), pages=pages)
    chunks = chunk_document(parsed, _meta("tiny"))

    assert len(chunks) == 1
    assert count_tokens(chunks[0].text) <= 800
    assert "Report exceptions." in chunks[0].text
    assert chunks[0].metadata["page_start"] == 1
    assert chunks[0].metadata["page_end"] == 2


def test_section_carries_forward_to_page_without_heading() -> None:
    pages = [
        PageContent(1, "SECTION 1 PAYMENT RISK\nBanks must monitor risk daily.", []),
        PageContent(2, "Settlement exceptions require investigation.", []),
    ]
    parsed = ParsedDocument(path=Path("carried.pdf"), pages=pages)
    chunks = chunk_document(parsed, _meta("carried"))

    assert len(chunks) == 1
    assert chunks[0].metadata["section"] == "SECTION 1 PAYMENT RISK"
    assert chunks[0].metadata["section_end"] == "SECTION 1 PAYMENT RISK"


def test_split_line_heading_replaces_earlier_carried_label() -> None:
    earlier = "Settlement risk monitoring requires daily review. " * 105
    later = "Payroll card controls require verified identity. " * 30
    pages = [
        PageContent(1, f"SECTION 1 PAYMENT RISK\n{earlier}", []),
        PageContent(2, f"4.13.7\nPrepaid – Payroll/Salary\n{later}", []),
    ]
    parsed = ParsedDocument(path=Path("split_heading.pdf"), pages=pages)
    chunks = chunk_document(parsed, _meta("split_heading"))

    split_chunk = next(c for c in chunks if "Prepaid – Payroll/Salary" in c.text)
    assert split_chunk.metadata["section"] == "4.13.7 Prepaid – Payroll/Salary"
    assert split_chunk.metadata["section"] != "SECTION 1 PAYMENT RISK"


def test_chunk_uses_its_own_split_line_heading() -> None:
    text = (
        "4.22.2.2\n"
        "Visa Fleet Merchant Discount Program Requirements\n"
        + "Acquirers must apply program controls. " * 35
    )
    parsed = ParsedDocument(
        path=Path("own_heading.pdf"), pages=[PageContent(1, text, [])]
    )
    chunks = chunk_document(parsed, _meta("own_heading"))

    assert chunks[0].metadata["section"] == (
        "4.22.2.2 Visa Fleet Merchant Discount Program Requirements"
    )


def test_split_line_part_heading_replaces_earlier_section() -> None:
    pages = [
        PageContent(1, "SECTION 12 FEES\n" + "Fee requirements apply. " * 105, []),
        PageContent(2, "Part 4:\nGlossary\n" + "A glossary term definition. " * 30, []),
    ]
    parsed = ParsedDocument(path=Path("part_heading.pdf"), pages=pages)
    chunks = chunk_document(parsed, _meta("part_heading"))

    glossary_chunk = next(c for c in chunks if "glossary term" in c.text)
    assert glossary_chunk.metadata["section"] == "Part 4: Glossary"


def test_chunk_uses_later_heading_that_appears_in_its_own_text() -> None:
    earlier = "Fee requirements apply. " * 100
    text = f"SECTION 12 FEES\n{earlier}\nPart 1:\nVisa Core Rules\n"
    parsed = ParsedDocument(
        path=Path("part_boundary.pdf"), pages=[PageContent(1, text, [])]
    )
    chunks = chunk_document(parsed, _meta("part_boundary"))

    assert chunks[0].metadata["section"] == "SECTION 12 FEES"


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


def test_two_line_title_heading_joined() -> None:
    text = (
        "4.13.7\n"
        "Prepaid – Payroll/Salary\n"
        "Card Programs\n" + "Requirements for card programs apply. " * 35
    )
    parsed = ParsedDocument(path=Path("two_line.pdf"), pages=[PageContent(1, text, [])])
    chunks = chunk_document(parsed, _meta("two_line"))
    assert len(chunks) == 1
    assert chunks[0].metadata["section"] == (
        "4.13.7 Prepaid – Payroll/Salary Card Programs"
    )


def test_three_thousand_and_id_numbers_are_rejected() -> None:
    assert not is_heading_line("3.000")
    assert not is_heading_line("ID# 0025973")
    assert _split_line_heading(["3.000", "Some Title"], 0) is None
    assert _split_line_heading(["ID# 0025973", "Some Title"], 0) is None


def test_toc_page_excluded_and_headings_unused() -> None:
    toc_lines = [
        "12.5 Risk Assessments",
        "806",
        "12.6 Member Fees",
        "809",
        "12.6.4 Rewards for Recovered Cards",
        "813",
        "Glossary",
        "826",
    ]
    pages = [
        PageContent(1, "SECTION 1 OVERVIEW\n" + "Initial overview text. " * 35, []),
        PageContent(2, "\n".join(toc_lines), []),
        PageContent(3, "Subsequent text without heading. " * 35, []),
    ]
    parsed = ParsedDocument(path=Path("toc_leak.pdf"), pages=pages)
    chunks = chunk_document(parsed, _meta("toc_leak"))

    for c in chunks:
        assert "12.6.4 Rewards for Recovered Cards" not in str(c.metadata["section"])
        assert "813" not in c.text
    # Page 3 chunk should not carry across the TOC page
    p3_chunk = next(c for c in chunks if "Subsequent text" in c.text)
    assert p3_chunk.metadata["section"] != "SECTION 1 OVERVIEW"
    assert p3_chunk.metadata["section"] == "body"


def test_chunk_with_own_child_heading_uses_it() -> None:
    child_heading = (
        "11.10.10.1 Dispute Condition 13.9: Non-Receipt of Cash at an ATM "
        "– Dispute Reasons"
    )
    parent_text = (
        "11.10.10 Dispute Condition 13.9: Non-Receipt of Cash at an ATM\n"
        + "Parent dispute condition overview. " * 100
    )
    child_text = f"{child_heading}\n" + "Child dispute reason details. " * 35
    pages = [
        PageContent(1, parent_text, []),
        PageContent(2, child_text, []),
    ]
    parsed = ParsedDocument(path=Path("child.pdf"), pages=pages)
    chunks = chunk_document(parsed, _meta("child"))

    child_chunk = next(c for c in chunks if "Child dispute reason details" in c.text)
    assert child_chunk.metadata["section"] == child_heading


def test_later_heading_replaces_earlier_carried_label() -> None:
    pages = [
        PageContent(1, "SECTION 1 PAYMENT RISK\n" + "Risk text. " * 100, []),
        PageContent(2, "SECTION 2 SETTLEMENT CONTROLS\n" + "Controls text. " * 35, []),
    ]
    parsed = ParsedDocument(path=Path("replace.pdf"), pages=pages)
    chunks = chunk_document(parsed, _meta("replace"))

    s2_chunk = next(c for c in chunks if "Controls text" in c.text)
    assert s2_chunk.metadata["section"] == "SECTION 2 SETTLEMENT CONTROLS"
    assert s2_chunk.metadata["section"] != "SECTION 1 PAYMENT RISK"
