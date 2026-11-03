import pytest

from docqa.errors import DocQAError
from docqa.guardrails import (
    MAX_QUESTION_CHARS,
    MAX_UPLOAD_BYTES,
    validate_question,
    validate_upload,
)


@pytest.mark.parametrize(
    "filename, data, kind",
    [
        ("a.pdf", b"%PDF-1.7 x", "pdf"),
        ("a.TXT", b"hello", "txt"),
        ("a.csv", b"a,b\n1,2", "csv"),
        ("a.xlsx", b"PK\x03\x04 x", "xlsx"),
    ],
)
def test_accepts_each_supported_type(filename, data, kind):
    assert validate_upload(filename, data) == kind


def test_rejects_unknown_extension():
    with pytest.raises(DocQAError) as exc:
        validate_upload("a.docx", b"hello")
    assert "PDF, TXT, CSV" in str(exc.value)


def test_rejects_empty_and_oversized():
    with pytest.raises(DocQAError) as empty:
        validate_upload("a.txt", b"")
    assert "empty" in str(empty.value)
    with pytest.raises(DocQAError) as big:
        validate_upload("a.txt", b"x" * (MAX_UPLOAD_BYTES + 1))
    assert "10 MB" in str(big.value)


def test_rejects_content_that_does_not_match_extension():
    with pytest.raises(DocQAError):
        validate_upload("fake.pdf", b"hello")
    with pytest.raises(DocQAError):
        validate_upload("x.csv", b"a\x00b")


def test_question_is_stripped():
    assert validate_question("  hi  ") == "hi"


def test_question_limits():
    with pytest.raises(DocQAError):
        validate_question("")
    with pytest.raises(DocQAError):
        validate_question("   ")
    with pytest.raises(DocQAError) as exc:
        validate_question("x" * (MAX_QUESTION_CHARS + 1))
    assert "1,000" in str(exc.value)
