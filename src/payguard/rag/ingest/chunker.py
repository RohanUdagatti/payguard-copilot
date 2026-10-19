"""Heading-aware chunking with recursive fallback (500–800 tokens, 100 overlap)."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from payguard.rag.ingest.metadata import DocMetadata
from payguard.rag.ingest.parser import ParsedDocument, page_to_text
from payguard.rag.interfaces import ChunkRecord

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


def is_heading_line(line: str) -> bool:
    """Return True if ``line`` is a real section heading (not a false positive)."""
    text = line.strip()
    if not text or len(text) > 180:
        return False
    if _CURRENCY_LINE_RE.match(text):
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


def parent_heading_key(title: str) -> str:
    """Parent key used to keep packing within the same heading family."""
    text = title.strip()
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
    return text or "body"


@dataclass
class _Section:
    title: str
    page_start: int
    page_end: int
    text: str


@dataclass
class _Atom:
    """Atomic text unit that should not be split casually (e.g. a table)."""

    text: str
    heading: str
    page_start: int
    page_end: int
    is_table: bool = False


@dataclass
class _Window:
    text: str
    section: str
    section_end: str
    page_start: int
    page_end: int


def _split_by_headings(text: str, page_number: int) -> list[_Section]:
    lines = text.splitlines(keepends=True)
    heading_indexes: list[int] = []
    flat = ""
    line_spans: list[tuple[int, int, str]] = []
    for line in lines:
        start = len(flat)
        flat += line
        end = len(flat)
        line_spans.append((start, end, line))

    for start, _end, line in line_spans:
        if is_heading_line(line.rstrip("\n")):
            heading_indexes.append(start)

    if not heading_indexes:
        stripped = text.strip()
        if not stripped:
            return []
        return [_Section("body", page_number, page_number, stripped)]

    sections: list[_Section] = []
    if heading_indexes[0] > 0:
        preface = flat[: heading_indexes[0]].strip()
        if preface:
            sections.append(_Section("body", page_number, page_number, preface))

    for idx, start in enumerate(heading_indexes):
        end = heading_indexes[idx + 1] if idx + 1 < len(heading_indexes) else len(flat)
        block = flat[start:end].strip()
        first_line = block.splitlines()[0].strip() if block else "body"
        sections.append(_Section(first_line, page_number, page_number, block))
    return sections


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
            # overlap: keep last few rows approximating overlap_tokens
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
        _Atom(p, atom.heading, atom.page_start, atom.page_end, atom.is_table)
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
                section=buf[0].heading,
                section_end=buf[-1].heading,
                page_start=min(a.page_start for a in buf),
                page_end=max(a.page_end for a in buf),
            )
        )
        buf = []
        buf_text = ""

    for atom in expanded:
        candidate = atom.text if not buf else f"{buf_text}\n\n{atom.text}"
        cand_tokens = count_tokens(candidate)
        if buf and cand_tokens > _MAX_PACK:
            flush()
            # start next window with overlap from previous
            if windows and overlap_tokens > 0:
                prefix = _token_suffix(windows[-1].text, overlap_tokens)
                buf = [
                    _Atom(
                        prefix,
                        windows[-1].section_end,
                        windows[-1].page_start,
                        windows[-1].page_end,
                        False,
                    )
                ]
                buf_text = prefix
                candidate = f"{buf_text}\n\n{atom.text}"
                if count_tokens(candidate) > _MAX_PACK:
                    # overlap + atom still too big: flush overlap alone then take atom
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


def _merge_tiny_tails(windows: list[_Window]) -> list[_Window]:
    """Merge a trailing window under 100 tokens into the previous one."""
    if len(windows) < 2:
        return windows
    out = list(windows)
    last = out[-1]
    if count_tokens(last.text) < _TAIL_MERGE:
        prev = out[-2]
        merged = _Window(
            text=f"{prev.text}\n\n{last.text}".strip(),
            section=prev.section,
            section_end=last.section_end,
            page_start=min(prev.page_start, last.page_start),
            page_end=max(prev.page_end, last.page_end),
        )
        # If merge exceeds max, keep as-is (only-chunk exception handled elsewhere)
        if count_tokens(merged.text) <= _MAX_PACK + 50:
            out[-2] = merged
            out.pop()
    return out


def chunk_document(
    parsed: ParsedDocument,
    meta: DocMetadata,
    *,
    target_tokens: int = 700,
    overlap_tokens: int = 100,
) -> list[ChunkRecord]:
    """Chunk a parsed PDF into metadata-rich ChunkRecords (500–800 tokens)."""
    del target_tokens  # packing uses fixed _MIN_PACK/_MAX_PACK

    sections: list[_Section] = []
    for page in parsed.pages:
        page_text = page_to_text(page)
        if not page_text.strip():
            continue
        sections.extend(_split_by_headings(page_text, page.page_number))

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

    windows = _merge_tiny_tails(windows)

    chunks: list[ChunkRecord] = []
    for chunk_index, window in enumerate(windows):
        text = window.text.strip()
        if not text:
            continue
        chunk_id = make_chunk_id(meta.doc_id, chunk_index, text)
        metadata: dict[str, Any] = {
            "doc_id": meta.doc_id,
            "title": meta.title,
            "section": window.section,
            "section_end": window.section_end,
            "page_start": window.page_start,
            "page_end": window.page_end,
            "version": meta.version,
            "effective_date": meta.effective_date,
            "region": meta.region,
            "chunk_id": chunk_id,
            "chunk_index": chunk_index,
        }
        chunks.append(ChunkRecord(chunk_id=chunk_id, text=text, metadata=metadata))
    return chunks
