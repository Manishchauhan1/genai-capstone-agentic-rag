"""Split parsed documents into chunks that carry their source (FR-4)."""

import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter

from docqa.models import Chunk, ParsedDocument

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
ROWS_PER_CHUNK = 20
MAX_TABLE_CHUNK_CHARS = 2000


def _render_row(columns: list[str], values) -> str:
    cells = ("" if pd.isna(value) else str(value) for value in values)
    return "; ".join(f"{column}: {cell}" for column, cell in zip(columns, cells))


def _table_chunks(kind: str, name: str, frame: pd.DataFrame) -> list[tuple[str, str]]:
    """Return (text, location) pairs for one table, repeating the headers in each."""
    columns = [str(column) for column in frame.columns]
    header = "Columns: " + " | ".join(columns)
    rows = [_render_row(columns, values) for values in frame.itertuples(index=False, name=None)]
    pieces: list[tuple[str, str]] = []
    start = 0
    while start < len(rows):
        size = ROWS_PER_CHUNK
        while True:
            text = header + "\n" + "\n".join(rows[start:start + size])
            if len(text) <= MAX_TABLE_CHUNK_CHARS or size == 1:
                break
            size = max(1, size // 2)
        if len(text) > MAX_TABLE_CHUNK_CHARS:
            text = text[: MAX_TABLE_CHUNK_CHARS - 1] + "…"
        taken = min(size, len(rows) - start)
        first, last = start + 1, start + taken
        prefix = "rows" if kind == "csv" else f"sheet {name} rows"
        pieces.append((text, f"{prefix} {first}-{last}"))
        start += taken
    return pieces


def chunk_document(doc: ParsedDocument) -> list[Chunk]:
    """Turn a ParsedDocument into chunks with file name and page/sheet metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    pairs: list[tuple[str, str]] = []
    for section in doc.sections:
        pairs.extend((piece, section.label) for piece in splitter.split_text(section.text))
    for name, frame in doc.tables.items():
        pairs.extend(_table_chunks(doc.kind, name, frame))
    return [
        Chunk(f"{doc.file_hash}:{n}", text, doc.filename, location, doc.file_hash)
        for n, (text, location) in enumerate(pairs)
    ]
