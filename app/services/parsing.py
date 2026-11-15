import bisect
import io
import re
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from app.core.config import settings

# Section headings in statutes ("15. Budget, Finance and Audit.- (1) ...") and in
# rulings collections ("168. POINT OF ORDER: ...")
_SECTION_HEADING = re.compile(
    r"(?<![\w(,/])(?P<num>\d{1,3})(?P<suffix>[A-Z]?)\s?[.,]\s+"
    r"(?:(?P<title>[A-Z][^:]{2,120}?)\s?[.,]\s?[-—–―_]"
    r"|(?P<caps>[A-Z][A-Z ,'’/&()-]{3,60}):)"
)
SECTION_MAX_CHARS = 2500
SECTION_MIN_CHARS = 300
SECTION_OVERLAP = 200
SECTION_MAX_GAP = 15  # largest numbering jump allowed between consecutive detected headings


@dataclass
class SectionChunk:
    label: str  # e.g. "s.15 Budget, Finance and Audit"; "" when no section structure was found
    text: str
    page_number: int


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )


def split_text(text: str) -> list[str]:
    return _splitter().split_text(_clean(text))


def extract_text(content: bytes, mime_type: str) -> str:
    if mime_type == "application/pdf":
        reader = PdfReader(io.BytesIO(content))
        return _clean("\n\n".join(page.extract_text() or "" for page in reader.pages))
    return _clean(content.decode("utf-8", errors="replace"))


def extract_text_by_page(content: bytes) -> list[tuple[int, str]]:
    """Returns [(1-indexed page_num, page_text), ...] for PDFs."""
    reader = PdfReader(io.BytesIO(content))
    return [(i + 1, _clean(page.extract_text() or "")) for i, page in enumerate(reader.pages)]


def _find_sections(text: str) -> list[tuple[int, str]]:
    """(offset, label) of section headings, keeping the longest run that numbers in order.

    Taking the longest increasing run (rather than a strict n, n+1, ... chain) tolerates
    headings lost to OCR, while still dropping false matches such as sections quoted inside
    amendments, which restart or jump in numbering.
    """
    cands = []
    for m in _SECTION_HEADING.finditer(text):
        num, suffix = int(m["num"]), m["suffix"]
        prefix = "s." if m["title"] else "No."
        title = (m["title"] or m["caps"]).strip()
        key = num * 27 + (ord(suffix) - 64 if suffix else 0)  # 15 < 15A < 15B < 16
        cands.append((m.start(), num, key, f"{prefix}{num}{suffix} {title[:80]}"))

    best = [1] * len(cands)
    prev = [-1] * len(cands)
    for i, (_, num_i, key_i, _) in enumerate(cands):
        for j in range(i):
            _, num_j, key_j, _ = cands[j]
            if key_j < key_i and num_i - num_j <= SECTION_MAX_GAP and best[j] + 1 > best[i]:
                best[i], prev[i] = best[j] + 1, j

    if not cands:
        return []
    i = max(range(len(cands)), key=best.__getitem__)
    run = []
    while i != -1:
        run.append((cands[i][0], cands[i][3]))
        i = prev[i]
    return run[::-1]


def split_sections(pages: list[tuple[int, str]]) -> list[SectionChunk]:
    """Chunk a PDF along its numbered sections, across page breaks.

    Short sections are merged into the next one; long ones are split at subsection/clause
    boundaries, each piece keeping its section label. Documents without a usable section
    sequence fall back to fixed-size chunks per page.
    """
    text, page_starts = "", []
    for _, page_text in pages:
        page_starts.append(len(text))
        text += page_text + " "
    page_nums = [p for p, _ in pages]

    def page_at(offset: int) -> int:
        return page_nums[bisect.bisect_right(page_starts, offset) - 1]

    headings = _find_sections(text)
    if len(headings) < 2:
        return [SectionChunk("", c, p) for p, t in pages for c in split_text(t)]

    units = [(0, headings[0][0], "preamble")] if headings[0][0] > 0 else []
    for i, (start, label) in enumerate(headings):
        end = headings[i + 1][0] if i + 1 < len(headings) else len(text)
        units.append((start, end, label))

    merged: list[tuple[int, int, str]] = []
    pending: tuple[int, int, str] | None = None
    for start, end, label in units:
        if pending:
            start, label = pending[0], f"{pending[2]}; {label}"
            pending = None
        if end - start < SECTION_MIN_CHARS:
            pending = (start, end, label)
            continue
        merged.append((start, end, label))
    if pending:
        if merged:
            start, _, label = merged.pop()
            merged.append((start, pending[1], f"{label}; {pending[2]}"))
        else:
            merged.append(pending)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=SECTION_MAX_CHARS,
        chunk_overlap=SECTION_OVERLAP,
        separators=[" (", "; ", ". ", " ", ""],
    )
    chunks: list[SectionChunk] = []
    for start, end, label in merged:
        body = text[start:end].strip()
        if len(body) <= SECTION_MAX_CHARS:
            chunks.append(SectionChunk(label, body, page_at(start)))
            continue
        cursor = start
        for piece in splitter.split_text(body):
            piece = piece.strip()
            pos = text.find(piece[:50], cursor)
            cursor = pos if pos != -1 else cursor
            chunks.append(SectionChunk(label, piece, page_at(cursor)))
    return chunks
