import pandas as pd

from docqa.chunking import MAX_TABLE_CHUNK_CHARS, chunk_document
from docqa.models import ParsedDocument, Section


def _text_doc(sections, kind="txt", file_hash="abc123"):
    return ParsedDocument("a.txt", file_hash, kind, sections, {})


def _table_doc(kind, tables, file_hash="abc123"):
    return ParsedDocument("t." + kind, file_hash, kind, [], tables)


def test_text_chunks_respect_size_and_cover_text():
    words = [f"alpha{i}" for i in range(500)]
    doc = _text_doc([Section("full text", " ".join(words))])
    chunks = chunk_document(doc)
    assert len(chunks) > 3
    assert all(len(c.text) <= 800 for c in chunks)
    seen = set(" ".join(c.text for c in chunks).split())
    assert set(words) <= seen


def test_pdf_chunks_keep_page_label():
    doc = _text_doc(
        [Section("page 1", "First page text."), Section("page 2", "Second page text.")],
        kind="pdf",
    )
    chunks = chunk_document(doc)
    assert [c.location for c in chunks] == ["page 1", "page 2"]
    assert chunks[1].filename == "a.txt"


def test_table_chunks_repeat_headers():
    frame = pd.DataFrame({"id": range(45), "name": [f"n{i}" for i in range(45)]})
    chunks = chunk_document(_table_doc("csv", {"csv": frame}))
    assert [c.location for c in chunks] == ["rows 1-20", "rows 21-40", "rows 41-45"]
    for chunk in chunks:
        assert chunk.text.startswith("Columns:")
        assert "id" in chunk.text.splitlines()[0] and "name" in chunk.text.splitlines()[0]
    assert "id: 0; name: n0" in chunks[0].text


def test_excel_location_names_the_sheet():
    frame = pd.DataFrame({"region": ["East", "West"], "amount": [1, 2]})
    chunks = chunk_document(_table_doc("xlsx", {"Sales": frame}))
    assert chunks[0].location == "sheet Sales rows 1-2"


def test_chunk_ids_are_unique_and_start_with_file_hash():
    frame = pd.DataFrame({"x": range(45)})
    doc = ParsedDocument(
        "t.csv", "deadbeef", "csv", [Section("full text", "hello " * 400)], {"csv": frame}
    )
    chunks = chunk_document(doc)
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))
    assert all(i.startswith("deadbeef:") for i in ids)
    assert all(c.file_hash == "deadbeef" for c in chunks)


def test_wide_table_chunks_stay_under_limit():
    columns = {f"c{i}": ["x" * 100] * 25 for i in range(30)}
    chunks = chunk_document(_table_doc("csv", {"csv": pd.DataFrame(columns)}))
    assert chunks
    assert all(len(c.text) <= MAX_TABLE_CHUNK_CHARS for c in chunks)


def test_missing_values_are_left_empty():
    frame = pd.DataFrame({"a": [1.0, None], "b": ["x", "y"]})
    chunks = chunk_document(_table_doc("csv", {"csv": frame}))
    assert "a: ; b: y" in chunks[0].text
    assert "nan" not in chunks[0].text.lower()
