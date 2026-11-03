import pytest

from docqa.errors import DocQAError
from docqa.retrieval import retrieve
from docqa.workspace import AddResult, Workspace


def _read(fixtures_dir, name):
    return (fixtures_dir / name).read_bytes()


def test_top_hit_contains_the_answer(fixtures_dir, offline_embedding):
    ws = Workspace(embedding_function=offline_embedding)
    ws.add_file("sample.pdf", _read(fixtures_dir, "sample.pdf"))
    result = retrieve(ws.store, "what is the refund window in days?")
    assert "30 days" in result.hits[0].chunk.text
    assert result.hits[0].chunk.location == "page 2"


def test_duplicate_upload_adds_nothing(fixtures_dir, offline_embedding):
    ws = Workspace(embedding_function=offline_embedding)
    data = _read(fixtures_dir, "sample.pdf")
    first = ws.add_file("sample.pdf", data)
    count = ws.store.count()
    second = ws.add_file("copy.pdf", data)
    assert first == AddResult("sample.pdf", count, False)
    assert second == AddResult("copy.pdf", 0, True)
    assert ws.store.count() == count


def test_tables_are_keyed_by_file_and_sheet(fixtures_dir, offline_embedding):
    ws = Workspace(embedding_function=offline_embedding)
    ws.add_file("multi.xlsx", _read(fixtures_dir, "multi.xlsx"))
    ws.add_file("cafe.csv", _read(fixtures_dir, "cafe_cp1252.csv"))
    assert set(ws.tables()) == {"multi.xlsx::Sales", "multi.xlsx::Staff", "cafe.csv::csv"}
    assert len(ws.tables()["multi.xlsx::Sales"]) == 3


def test_is_empty_until_first_file(fixtures_dir, offline_embedding):
    ws = Workspace(embedding_function=offline_embedding)
    assert ws.is_empty() is True
    ws.add_file("sample.pdf", _read(fixtures_dir, "sample.pdf"))
    assert ws.is_empty() is False
    assert ws.filenames() == ["sample.pdf"]


@pytest.mark.parametrize("name, data", [("a.docx", b"hello"), ("scanned.pdf", None)])
def test_bad_file_raises_and_leaves_workspace_empty(
    name, data, fixtures_dir, offline_embedding
):
    ws = Workspace(embedding_function=offline_embedding)
    payload = data if data is not None else _read(fixtures_dir, "scanned.pdf")
    with pytest.raises(DocQAError):
        ws.add_file(name, payload)
    assert ws.is_empty() is True
    assert ws.store.count() == 0
