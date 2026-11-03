import pytest

from docqa.errors import DocQAError
from docqa.ingestion import parse_file


def _read(fixtures_dir, name):
    return (fixtures_dir / name).read_bytes()


def test_pdf_has_one_section_per_page(fixtures_dir):
    doc = parse_file("sample.pdf", _read(fixtures_dir, "sample.pdf"))
    assert [s.label for s in doc.sections] == ["page 1", "page 2"]
    assert "30 days" in doc.sections[1].text
    assert doc.kind == "pdf"
    assert doc.tables == {}


def test_scanned_pdf_is_rejected(fixtures_dir):
    with pytest.raises(DocQAError) as exc:
        parse_file("scanned.pdf", _read(fixtures_dir, "scanned.pdf"))
    assert "No readable text found" in str(exc.value)


def test_txt_decodes_cp1252():
    doc = parse_file("a.txt", "café".encode("cp1252"))
    assert doc.sections[0].text == "café"
    assert doc.sections[0].label == "full text"


def test_csv_keeps_accents(fixtures_dir):
    doc = parse_file("cafe.csv", _read(fixtures_dir, "cafe_cp1252.csv"))
    assert doc.tables["csv"]["city"].iloc[0] == "Café"
    assert doc.sections == []


def test_header_only_csv_is_rejected(fixtures_dir):
    with pytest.raises(DocQAError):
        parse_file("header_only.csv", _read(fixtures_dir, "header_only.csv"))


def test_xlsx_reads_every_sheet(fixtures_dir):
    doc = parse_file("multi.xlsx", _read(fixtures_dir, "multi.xlsx"))
    assert set(doc.tables) == {"Sales", "Staff"}
    assert len(doc.tables["Sales"]) == 3


def test_corrupt_pdf_is_rejected():
    with pytest.raises(DocQAError) as exc:
        parse_file("a.pdf", b"%PDF-1.7 garbage")
    assert "Could not read" in str(exc.value)


def test_file_hash_is_stable():
    first = parse_file("a.txt", b"same text")
    again = parse_file("b.txt", b"same text")
    other = parse_file("a.txt", b"different text")
    assert first.file_hash == again.file_hash
    assert first.file_hash != other.file_hash
