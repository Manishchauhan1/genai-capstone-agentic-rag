"""Input validation for uploads and questions (FR-9, input half)."""

import os

from docqa.errors import DocQAError

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_QUESTION_CHARS = 1000

_EXTENSIONS = {".pdf": "pdf", ".txt": "txt", ".csv": "csv", ".xlsx": "xlsx"}


def validate_upload(filename: str, data: bytes) -> str:
    """Check an uploaded file and return its kind: "pdf", "txt", "csv" or "xlsx"."""
    extension = os.path.splitext(filename)[1].lower()
    kind = _EXTENSIONS.get(extension)
    if kind is None:
        raise DocQAError(
            f"Unsupported file type for {filename!r}. "
            "Please upload PDF, TXT, CSV or Excel (.xlsx) files."
        )
    if len(data) == 0:
        raise DocQAError(f"{filename!r} is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise DocQAError(f"{filename!r} is larger than 10 MB.")
    if kind == "pdf" and not data.startswith(b"%PDF-"):
        raise DocQAError(f"{filename!r} does not look like a real PDF file.")
    if kind == "xlsx" and not data.startswith(b"PK\x03\x04"):
        raise DocQAError(f"{filename!r} does not look like a real Excel (.xlsx) file.")
    if kind in ("txt", "csv") and b"\x00" in data:
        raise DocQAError(f"{filename!r} does not look like a text file.")
    return kind


def validate_question(text: str) -> str:
    """Return the stripped question, or raise if it is empty or too long."""
    question = text.strip()
    if not question:
        raise DocQAError("Please type a question.")
    if len(question) > MAX_QUESTION_CHARS:
        raise DocQAError("Your question is longer than 1,000 characters. Please shorten it.")
    return question
