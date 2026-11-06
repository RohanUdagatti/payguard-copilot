"""Heading-aware chunking with recursive fallback (500–800 tokens, 100 overlap)."""

from __future__ import annotations

import hashlib
import logging
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from payguard.rag.ingest.metadata import DocMetadata
from payguard.rag.ingest.parser import ParsedDocument, page_to_text
from payguard.rag.interfaces import ChunkRecord, RetrievedChunk

# Tight heading detector: reject currency / "1 Based on..." false positives.
_HEADING_RE = re.compile(
    r"^(?:"
    r"(?:SECTION|ARTICLE|CHAPTER|ANNEX|APPENDIX)\s+[A-Z0-9].{2,}|"
    r"(?:\d+\.\d+(?:\.\d+)*)\s+[A-Z][\w/\-].{4,}|"
    r"[A-Z][A-Z0-9][A-Z0-9 /&,\-]{10,}"
    r")\s*$",
    re.MULTILINE,
)

_CURRENCY_LINE_RE = re.compile(
    r"^(?:USD|EUR|GBP|INR|\$)\s*[\d,]+(?:\.\d+)?\s*$|"
    r"^\$?\d[\d,]*(?:\.\d+)?(?:\s*(?:USD|EUR|GBP|INR))?\s*$",
    re.IGNORECASE,
)
_SINGLE_INT_PREFIX_RE = re.compile(r"^\d+\s+[A-Z]")

_TABLE_BLOCK_RE = re.compile(r"\[TABLE\].*?\[/TABLE\]", re.DOTALL)
_TOC_ENTRY_RE = re.compile(r"^.+?(?:\.{2,}|\s{2,})\s*\d{1,4}\s*$")
_PAGE_NUM_ALONE_RE = re.compile(r"^\d{1,4}$")
_SENTENCE_RE = re.compile(
    r"\b(?:must|shall|may|will|is|are|was|were|has|have|can)\b", re.IGNORECASE
)

_STANDALONE_NUMBER_RE = re.compile(r"^\d+(?:\.\d+)+\.?$")
_PART_LINE_RE = re.compile(r"^PART\s+\d+:?$", re.IGNORECASE)
_ID_LINE_RE = re.compile(r"^ID#\s*\d+", re.IGNORECASE)
_VERB_RE = re.compile(
    r"\b(?:must|shall|may|will|is|are|was|were|has|have|had|does|do|did|cannot|should|would|could)\b",
    re.IGNORECASE,
)
_GLUED_PAGE_NUM_RE = re.compile(r"[a-zA-Z]{3,}\d{1,4}$")
_RATING_LINE_RE = re.compile(
    r"^\d+\.\d{2}\s+(?:Above|Below|Average|Satisfactory|High|Medium|Low|Fair|Poor|Unsatisfactory)\b",
    re.IGNORECASE,
)

logger = logging.getLogger(__name__)

_ENC = tiktoken.get_encoding("cl100k_base")

_MIN_PACK = 500
_MAX_PACK = 800
_TAIL_MERGE = 100


def count_tokens(text: str) -> int:
    """Count tokens with tiktoken cl100k_base."""
    return len(_ENC.encode(text))


def make_chunk_id(doc_id: str, chunk_index: int, text: str) -> str:
    """Deterministic chunk id from doc, index, and content hash."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"{doc_id}:{chunk_index}:{digest}"


def is_heading_line(line: str, running_headers: set[str] | None = None) -> bool:
    """Return True if ``line`` is a real section heading (not a false positive)."""
    text = line.strip()
    if not text or len(text) > 180:
        return False
    if running_headers and text in running_headers:
        return False
    if text == "3.000" or re.match(r"^\d+\.000$", text):
        return False
    if _ID_LINE_RE.match(text):
        return False
    if _CURRENCY_LINE_RE.match(text):
        return False
    if _GLUED_PAGE_NUM_RE.search(text):
        return False
    if sum(1 for ch in text if ch.isalpha()) < 3:
        return False
    if _RATING_LINE_RE.match(text):
        return False
    # Reject "1 Based on..." (integer + text, no dotted numbering)
    single_int = _SINGLE_INT_PREFIX_RE.match(text) and not re.match(r"^\d+\.\d+", text)
    named = re.match(
        r"^(?:SECTION|ARTICLE|CHAPTER|ANNEX|APPENDIX)\b",
        text,
        re.IGNORECASE,
    )
    if single_int and not named:
        return False
    return bool(_HEADING_RE.match(text))


def _is_valid_first_title_line(line: str) -> bool:
    """Return whether line is eligible as first title line after a number."""
    text = line.strip()
    if not text or len(text) >= 120:
        return False
    if not re.match(r"^[A-Z]", text):
        return False
    return not (
        _STANDALONE_NUMBER_RE.match(text)
        or _ID_LINE_RE.match(text)
        or _CURRENCY_LINE_RE.match(text)
    )


def _is_strict_title_line(line: str) -> bool:
    """Check if line looks like a title (not a body sentence)."""
    text = line.strip()
    if not _is_valid_first_title_line(text):
        return False
    if text.endswith((".", ":", ";")):
        return False
    return not bool(_VERB_RE.search(text))


def _is_continuation_title_line(first_title: str, second_line: str) -> bool:
    """Return whether second_line can be the second title line of a heading."""
    text = second_line.strip()
    if not text or len(text) >= 120:
        return False
    if not re.match(r"^[A-Z]", text):
        return False
    if (
        _STANDALONE_NUMBER_RE.match(text)
        or _ID_LINE_RE.match(text)
        or _CURRENCY_LINE_RE.match(text)
    ):
        return False
    if text.endswith((".", ":", ";")):
        return False
    first_stripped = first_title.strip()
    if first_stripped.endswith(("-", "–", "—", "/", ",")):
        return True
    return not bool(_VERB_RE.search(text))


def _split_line_heading_with_span(
    lines: list[str], index: int
) -> tuple[str, int] | None:
    """Return (joined_heading, lines_consumed) for a numeric line and title line(s)."""
    number = lines[index].strip()
    if (
        not number
        or number == "3.000"
        or re.match(r"^\d+\.000$", number)
        or _ID_LINE_RE.match(number)
    ):
        return None

    is_number = bool(_STANDALONE_NUMBER_RE.match(number))
    is_part = bool(_PART_LINE_RE.match(number))
    if not is_number and not is_part:
        return None

    next_idx = None
    for j in range(index + 1, len(lines)):
        if lines[j].strip():
            next_idx = j
            break
    if next_idx is None:
        return None

    first_title = lines[next_idx].strip()
    if is_number and number.count(".") >= 2:
        if not _is_valid_first_title_line(first_title):
            return None
    else:
        if not _is_strict_title_line(first_title):
            return None

    consumed = next_idx - index
    title_parts = [first_title]

    second_idx = None
    for k in range(next_idx + 1, len(lines)):
        if lines[k].strip():
            second_idx = k
            break

    if second_idx is not None:
        second_line = lines[second_idx].strip()
        if _is_continuation_title_line(first_title, second_line):
            title_parts.append(second_line)
            consumed = second_idx - index

    full_title = " ".join(title_parts)
    return f"{number.rstrip('.')} {full_title}", consumed


def _split_line_heading(lines: list[str], index: int) -> str | None:
    """Return a heading made of a numeric line and up to 2 following title lines."""
    res = _split_line_heading_with_span(lines, index)
    return res[0] if res else None


def _first_heading_in_text(
    text: str, running_headers: set[str] | None = None
) -> str | None:
    """Find the first explicit heading in chunk text, including split-line headings."""
    lines = text.splitlines()
    for index, line in enumerate(lines):
        split_heading = _split_line_heading(lines, index)
        if split_heading and not (running_headers and split_heading in running_headers):
            return split_heading
        if is_heading_line(line, running_headers):
            return line.strip()
    return None


def parent_heading_key(title: str | None) -> str:
    """Parent key used to keep packing within the same heading family."""
    text = (title or "").strip()
    numbered = re.match(r"^(\d+)(?:\.(\d+))?", text)
    if numbered:
        if numbered.group(2) is not None:
            return f"{numbered.group(1)}.{numbered.group(2)}"
        return numbered.group(1)
    named = re.match(
        r"^(SECTION|ARTICLE|CHAPTER|ANNEX|APPENDIX)\s+(\S+)",
        text,
        re.IGNORECASE,
    )
    if named:
        return f"{named.group(1).upper()} {named.group(2)}"
    return text or "unsectioned"


@dataclass
class _Section:
    title: str | None
    page_start: int
    page_end: int
    text: str
    line_pos: int = 0


@dataclass
class _Atom:
    """Atomic text unit that should not be split casually (e.g. a table)."""

    text: str
    heading: str | None
    page_start: int
    page_end: int
    line_pos: int = 0
    is_table: bool = False


@dataclass
class _Window:
    text: str
    section: str | None
    section_end: str | None
    page_start: int
    page_end: int
    start_line_pos: int = 0


@dataclass(frozen=True)
class DocumentHeading:
    """Heading discovered in document on non-TOC pages."""

    page: int
    line_pos: int
    title: str


def _split_by_headings(
    text: str,
    page_number: int,
    running_headers: set[str] | None = None,
) -> list[_Section]:
    lines = text.splitlines(keepends=True)
    flat = ""
    line_spans: list[tuple[int, int, str]] = []
    for line in lines:
        start = len(flat)
        flat += line
        end = len(flat)
        line_spans.append((start, end, line))

    plain_lines = [line.rstrip("\n") for _start, _end, line in line_spans]
    heading_indexes: list[int] = []
    heading_titles: dict[int, str] = {}
    heading_line_positions: dict[int, int] = {}

    skip_until = -1
    for index, (start, _end, line) in enumerate(line_spans):
        if index <= skip_until:
            continue
        split_res = _split_line_heading_with_span(plain_lines, index)
        if split_res:
            heading_title, consumed = split_res
            if not (running_headers and heading_title in running_headers):
                heading_indexes.append(start)
                heading_titles[start] = heading_title
                heading_line_positions[start] = index
                skip_until = index + consumed
        elif is_heading_line(line.rstrip("\n"), running_headers):
            heading_indexes.append(start)
            heading_titles[start] = line.strip()
            heading_line_positions[start] = index

    if not heading_indexes:
        stripped = text.strip()
        if not stripped:
            return []
        return [_Section(None, page_number, page_number, stripped, line_pos=0)]

    sections: list[_Section] = []
    if heading_indexes[0] > 0:
        preface = flat[: heading_indexes[0]].strip()
        if preface:
            sections.append(
                _Section(None, page_number, page_number, preface, line_pos=0)
            )

    for idx, start in enumerate(heading_indexes):
        end = heading_indexes[idx + 1] if idx + 1 < len(heading_indexes) else len(flat)
        block = flat[start:end].strip()
        first_line = heading_titles.get(start) or (
            block.splitlines()[0].strip() if block else ""
        )
        l_pos = heading_line_positions.get(start, 0)
        sections.append(
            _Section(first_line, page_number, page_number, block, line_pos=l_pos)
        )
    return sections


def _is_toc_page(text: str) -> bool:
    """Identify a contents/index page without treating normal prose as a TOC."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 3:
        return False

    dot_or_num = sum(bool(_TOC_ENTRY_RE.match(line)) for line in lines)
    if dot_or_num >= 3 and (dot_or_num / len(lines)) >= 0.5:
        return True

    num_alone_lines = [line for line in lines if _PAGE_NUM_ALONE_RE.match(line)]
    distinct_nums = len(set(num_alone_lines))
    sentence_count = sum(
        bool(_SENTENCE_RE.search(line) and len(line) > 40) for line in lines
    )

    return bool(
        sentence_count == 0
        and len(num_alone_lines) >= 4
        and distinct_nums >= 4
        and (len(num_alone_lines) / len(lines)) >= 0.2
    )


def _resolve_chunk_section(
    chunk_text: str,
    page_start: int,
    line_pos: int,
    doc_headings: list[DocumentHeading],
    toc_pages: set[int],
    running_headers: set[str] | None = None,
) -> str:
    """Determine section label using the explicit label rule."""
    own = _first_heading_in_text(chunk_text, running_headers)
    if own:
        return own

    latest: str | None = None
    for h in doc_headings:
        if h.page > page_start:
            break
        if h.page == page_start and h.line_pos > line_pos:
            break
        if any(p in toc_pages for p in range(h.page, page_start + 1)):
            continue
        latest = h.title

    return latest or "body"


def _carry_forward_headings(
    sections: list[_Section], toc_pages: set[int]
) -> list[_Section]:
    """Label text after a heading with that heading until another heading appears."""
    current_heading: str | None = None
    last_page = 0
    carried: list[_Section] = []
    for section in sections:
        if (
            current_heading is not None
            and last_page > 0
            and any(p in toc_pages for p in range(last_page, section.page_start + 1))
        ):
            current_heading = None

        title = section.title
        if title is not None:
            current_heading = title
        elif current_heading is not None:
            title = current_heading
        carried.append(
            _Section(
                title,
                section.page_start,
                section.page_end,
                section.text,
                section.line_pos,
            )
        )
        last_page = section.page_end
    return carried


def _atoms_from_section(section: _Section) -> list[_Atom]:
    """Split a section into table / non-table atoms (tables stay whole)."""
    text = section.text
    atoms: list[_Atom] = []
    pos = 0
    for match in _TABLE_BLOCK_RE.finditer(text):
        if match.start() > pos:
            piece = text[pos : match.start()].strip()
            if piece:
                atoms.append(
                    _Atom(
                        piece,
                        section.title,
                        section.page_start,
                        section.page_end,
                        section.line_pos,
                        False,
                    )
                )
        table = match.group(0).strip()
        if table:
            atoms.append(
                _Atom(
                    table,
                    section.title,
                    section.page_start,
                    section.page_end,
                    section.line_pos,
                    True,
                )
            )
        pos = match.end()
    if pos < len(text):
        piece = text[pos:].strip()
        if piece:
            atoms.append(
                _Atom(
                    piece,
                    section.title,
                    section.page_start,
                    section.page_end,
                    section.line_pos,
                    False,
                )
            )
    if not atoms and text.strip():
        atoms.append(
            _Atom(
                text.strip(),
                section.title,
                section.page_start,
                section.page_end,
                section.line_pos,
                False,
            )
        )
    return atoms


def _token_suffix(text: str, n_tokens: int) -> str:
    tokens = _ENC.encode(text)
    if len(tokens) <= n_tokens:
        return text
    return _ENC.decode(tokens[-n_tokens:])


def _recursive_splitter(
    chunk_size: int = 700,
    chunk_overlap: int = 100,
) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=count_tokens,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def _split_oversized_table(atom: _Atom, overlap_tokens: int) -> list[str]:
    """Split a large table by whole rows; never mid-row."""
    inner = atom.text
    if inner.startswith("[TABLE]") and inner.endswith("[/TABLE]"):
        body = inner[len("[TABLE]") : -len("[/TABLE]")].strip("\n")
    else:
        body = inner
    rows = body.splitlines()
    windows: list[str] = []
    current: list[str] = []
    for row in rows:
        trial = "\n".join(current + [row])
        wrapped = f"[TABLE]\n{trial}\n[/TABLE]"
        if current and count_tokens(wrapped) > _MAX_PACK:
            windows.append("[TABLE]\n" + "\n".join(current) + "\n[/TABLE]")
            overlap_rows: list[str] = []
            for prev in reversed(current):
                overlap_rows.insert(0, prev)
                if count_tokens("\n".join(overlap_rows)) >= overlap_tokens:
                    break
            current = overlap_rows + [row]
        else:
            current.append(row)
    if current:
        windows.append("[TABLE]\n" + "\n".join(current) + "\n[/TABLE]")
    return windows


def _split_oversized_text(text: str, overlap_tokens: int) -> list[str]:
    splitter = _recursive_splitter(_MAX_PACK, overlap_tokens)
    return [p for p in splitter.split_text(text) if p.strip()]


def _expand_atom(atom: _Atom, overlap_tokens: int) -> list[_Atom]:
    """If an atom exceeds max size, split it carefully."""
    n = count_tokens(atom.text)
    if n <= _MAX_PACK:
        return [atom]
    if atom.is_table:
        pieces = _split_oversized_table(atom, overlap_tokens)
    else:
        pieces = _split_oversized_text(atom.text, overlap_tokens)
    return [
        _Atom(
            p,
            atom.heading,
            atom.page_start,
            atom.page_end,
            atom.line_pos,
            atom.is_table,
        )
        for p in pieces
        if p.strip()
    ]


def _pack_atoms(
    atoms: list[_Atom],
    *,
    overlap_tokens: int,
) -> list[_Window]:
    """Greedy pack atoms into 500–800 token windows within one parent group."""
    expanded: list[_Atom] = []
    for atom in atoms:
        expanded.extend(_expand_atom(atom, overlap_tokens))

    windows: list[_Window] = []
    buf: list[_Atom] = []
    buf_text = ""

    def flush() -> None:
        nonlocal buf, buf_text
        if not buf:
            return
        text = "\n\n".join(a.text for a in buf).strip()
        windows.append(
            _Window(
                text=text,
                section=_first_heading_in_text(text) or buf[0].heading,
                section_end=buf[-1].heading,
                page_start=min(a.page_start for a in buf),
                page_end=max(a.page_end for a in buf),
                start_line_pos=buf[0].line_pos,
            )
        )
        buf = []
        buf_text = ""

    for atom in expanded:
        candidate = atom.text if not buf else f"{buf_text}\n\n{atom.text}"
        cand_tokens = count_tokens(candidate)
        if buf and cand_tokens > _MAX_PACK:
            flush()
            if windows and overlap_tokens > 0:
                prefix = _token_suffix(windows[-1].text, overlap_tokens)
                buf = [
                    _Atom(
                        prefix,
                        windows[-1].section_end,
                        windows[-1].page_start,
                        windows[-1].page_end,
                        windows[-1].start_line_pos,
                        False,
                    )
                ]
                buf_text = prefix
                candidate = f"{buf_text}\n\n{atom.text}"
                if count_tokens(candidate) > _MAX_PACK:
                    flush()
                    buf = [atom]
                    buf_text = atom.text
                    if count_tokens(buf_text) >= _MIN_PACK:
                        flush()
                    continue
            else:
                buf = [atom]
                buf_text = atom.text
                if count_tokens(buf_text) >= _MIN_PACK:
                    flush()
                continue

        if buf:
            buf.append(atom)
            buf_text = candidate
        else:
            buf = [atom]
            buf_text = atom.text

        if count_tokens(buf_text) >= _MIN_PACK:
            flush()

    flush()
    return windows


def _merged_window(first: _Window, second: _Window) -> _Window:
    return _Window(
        text=f"{first.text}\n\n{second.text}".strip(),
        section=first.section,
        section_end=second.section_end,
        page_start=min(first.page_start, second.page_start),
        page_end=max(first.page_end, second.page_end),
        start_line_pos=first.start_line_pos,
    )


def _merge_tiny_windows(windows: list[_Window]) -> list[_Window]:
    """Merge every sub-100-token window with an adjacent window when it fits."""
    out = list(windows)
    index = 0
    while index < len(out):
        if count_tokens(out[index].text) >= _TAIL_MERGE or len(out) == 1:
            index += 1
            continue
        if index > 0:
            merged = _merged_window(out[index - 1], out[index])
            if count_tokens(merged.text) <= _MAX_PACK:
                out[index - 1] = merged
                out.pop(index)
                index = max(0, index - 1)
                continue
        if index + 1 < len(out):
            merged = _merged_window(out[index], out[index + 1])
            if count_tokens(merged.text) <= _MAX_PACK:
                out[index] = merged
                out.pop(index + 1)
                continue
        index += 1
    return out


def is_junk_boilerplate_chunk(text: str) -> bool:
    """Return True if chunk text is only boilerplate ID#/Edition/Last Updated tables."""
    has_boilerplate = bool(
        _TABLE_BLOCK_RE.search(text)
        or ("ID#" in text and ("Edition:" in text or "Last Updated:" in text))
    )
    if not has_boilerplate:
        return False
    stripped = _TABLE_BLOCK_RE.sub("", text)
    clean_lines = []
    for line in stripped.splitlines():
        line_s = line.strip()
        if not line_s:
            continue
        if ("ID#" in line_s or "Edition:" in line_s) and (
            "Last Updated:" in line_s or "Edition:" in line_s
        ):
            continue
        clean_lines.append(line_s)
    rem = "\n".join(clean_lines).strip()
    return count_tokens(rem) < 50


def make_child_chunks(
    parent: ChunkRecord,
    target_tokens: int = 200,
    overlap_tokens: int = 50,
) -> list[ChunkRecord]:
    """Split a parent chunk into ~200-token child chunks carrying parent metadata."""
    tokens = _ENC.encode(parent.text)
    if len(tokens) <= target_tokens:
        child_id = f"{parent.chunk_id}:c0"
        meta = {
            **parent.metadata,
            "parent_id": parent.chunk_id,
            "parent_text": parent.text,
            "is_child": True,
            "child_index": 0,
        }
        return [ChunkRecord(chunk_id=child_id, text=parent.text, metadata=meta)]

    children: list[ChunkRecord] = []
    step = max(1, target_tokens - overlap_tokens)
    c_idx = 0
    for start in range(0, len(tokens), step):
        sub_tokens = tokens[start : start + target_tokens]
        if not sub_tokens:
            break
        child_text = _ENC.decode(sub_tokens).strip()
        if not child_text:
            continue
        child_id = f"{parent.chunk_id}:c{c_idx}"
        meta = {
            **parent.metadata,
            "parent_id": parent.chunk_id,
            "parent_text": parent.text,
            "is_child": True,
            "child_index": c_idx,
        }
        children.append(ChunkRecord(chunk_id=child_id, text=child_text, metadata=meta))
        c_idx += 1
        if start + target_tokens >= len(tokens):
            break

    return children


def deduplicate_to_parents(retrieved: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Deduplicate child chunks to their parents, keeping the highest child score."""
    seen_parents: set[str] = set()
    parents: list[RetrievedChunk] = []

    for hit in retrieved:
        parent_id = hit.metadata.get("parent_id")
        parent_text = hit.metadata.get("parent_text")
        if not parent_id or not parent_text:
            if hit.chunk_id not in seen_parents:
                seen_parents.add(hit.chunk_id)
                parents.append(hit)
            continue

        if parent_id in seen_parents:
            continue
        seen_parents.add(parent_id)

        parent_meta = {
            k: v
            for k, v in hit.metadata.items()
            if k not in ("is_child", "child_index", "parent_id", "parent_text")
        }
        parents.append(
            RetrievedChunk(
                chunk_id=str(parent_id),
                text=str(parent_text),
                score=hit.score,
                metadata=parent_meta,
            )
        )

    return parents


def chunk_document(
    parsed: ParsedDocument,
    meta: DocMetadata,
    *,
    target_tokens: int = 700,
    overlap_tokens: int = 100,
) -> list[ChunkRecord]:
    """Chunk a parsed PDF into metadata-rich ChunkRecords (500–800 tokens)."""
    del target_tokens  # packing uses fixed _MIN_PACK/_MAX_PACK

    page_line_counts: Counter[str] = Counter()
    non_empty_pages = 0
    for page in parsed.pages:
        page_text = page_to_text(page).strip()
        if not page_text:
            continue
        non_empty_pages += 1
        page_lines = {line.strip() for line in page_text.splitlines() if line.strip()}
        for line in page_lines:
            page_line_counts[line] += 1

    running_headers: set[str] = {
        line
        for line, count in page_line_counts.items()
        if non_empty_pages >= 5 and count >= 3 and (count / non_empty_pages) > 0.30
    }

    toc_pages: set[int] = set()
    for page in parsed.pages:
        page_text = page_to_text(page)
        if _is_toc_page(page_text):
            toc_pages.add(page.page_number)

    logger.info(
        "toc_pages_excluded doc_id=%s count=%d",
        meta.doc_id,
        len(toc_pages),
        extra={"doc_id": meta.doc_id, "toc_pages_excluded": len(toc_pages)},
    )

    doc_headings: list[DocumentHeading] = []
    sections: list[_Section] = []
    for page in parsed.pages:
        if page.page_number in toc_pages:
            continue
        page_text = page_to_text(page)
        if not page_text.strip():
            continue
        page_sections = _split_by_headings(page_text, page.page_number, running_headers)
        sections.extend(page_sections)
        for s in page_sections:
            if s.title:
                doc_headings.append(
                    DocumentHeading(
                        page=s.page_start,
                        line_pos=s.line_pos,
                        title=s.title,
                    )
                )

    sections = _carry_forward_headings(sections, toc_pages)

    # Contiguous groups sharing the same parent heading
    groups: list[list[_Section]] = []
    for section in sections:
        key = parent_heading_key(section.title)
        if not groups:
            groups.append([section])
            continue
        prev_key = parent_heading_key(groups[-1][-1].title)
        if key == prev_key:
            groups[-1].append(section)
        else:
            groups.append([section])

    windows: list[_Window] = []
    for group in groups:
        atoms: list[_Atom] = []
        for section in group:
            atoms.extend(_atoms_from_section(section))
        windows.extend(_pack_atoms(atoms, overlap_tokens=overlap_tokens))

    windows = _merge_tiny_windows(windows)

    chunks: list[ChunkRecord] = []
    excluded_junk_count = 0
    for chunk_index, window in enumerate(windows):
        text = window.text.strip()
        if not text:
            continue
        if is_junk_boilerplate_chunk(text):
            excluded_junk_count += 1
            continue

        section_label = _resolve_chunk_section(
            text,
            window.page_start,
            window.start_line_pos,
            doc_headings,
            toc_pages,
            running_headers,
        )
        chunk_id = make_chunk_id(meta.doc_id, chunk_index, text)
        metadata: dict[str, Any] = {
            "doc_id": meta.doc_id,
            "title": meta.title,
            "section": section_label,
            "section_end": window.section_end or section_label,
            "page_start": window.page_start,
            "page_end": window.page_end,
            "version": meta.version,
            "effective_date": meta.effective_date,
            "region": meta.region,
            "chunk_id": chunk_id,
            "chunk_index": chunk_index,
            "is_child": False,
        }
        chunks.append(ChunkRecord(chunk_id=chunk_id, text=text, metadata=metadata))

    if excluded_junk_count > 0:
        logger.info(
            "junk_chunks_excluded doc_id=%s count=%d",
            meta.doc_id,
            excluded_junk_count,
            extra={"doc_id": meta.doc_id, "junk_chunks_excluded": excluded_junk_count},
        )

    return chunks
