"""Parse uploaded files (PDF, TXT, CSV, Excel) into ParsedDocument (FR-3)."""

import hashlib
import io

import pandas as pd
from pypdf import PdfReader

from docqa.errors import DocQAError
from docqa.guardrails import validate_upload
from docqa.models import ParsedDocument, Section


def _decode(data: bytes) -> str:
    """Decode text bytes: UTF-8 (BOM allowed), then Windows-1252, then Latin-1."""
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")


def _parse_pdf(filename: str, data: bytes) -> list[Section]:
    reader = PdfReader(io.BytesIO(data))
    sections = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            sections.append(Section(f"page {number}", text))
    if not sections:
        raise DocQAError(
            f"No readable text found in {filename!r}. Scanned PDFs are not supported."
        )
    return sections


def _parse_txt(filename: str, data: bytes) -> list[Section]:
    text = _decode(data)
    if not text.strip():
        raise DocQAError(f"{filename!r} contains no text.")
    return [Section("full text", text)]


def _parse_csv(filename: str, data: bytes) -> dict[str, pd.DataFrame]:
    frame = pd.read_csv(io.StringIO(_decode(data)))
    if len(frame) == 0 or len(frame.columns) == 0:
        raise DocQAError(f"{filename!r} has no data rows.")
    return {"csv": frame}


def _parse_xlsx(filename: str, data: bytes) -> dict[str, pd.DataFrame]:
    sheets = pd.read_excel(io.BytesIO(data), sheet_name=None, engine="openpyxl")
    tables = {str(name): frame for name, frame in sheets.items() if len(frame) > 0}
    if not tables:
        raise DocQAError(f"{filename!r} has no data rows in any sheet.")
    return tables


def parse_file(filename: str, data: bytes) -> ParsedDocument:
    """Validate and parse one uploaded file."""
    kind = validate_upload(filename, data)
    file_hash = hashlib.sha256(data).hexdigest()
    sections: list[Section] = []
    tables: dict[str, pd.DataFrame] = {}
    try:
        if kind == "pdf":
            sections = _parse_pdf(filename, data)
        elif kind == "txt":
            sections = _parse_txt(filename, data)
        elif kind == "csv":
            tables = _parse_csv(filename, data)
        else:
            tables = _parse_xlsx(filename, data)
    except DocQAError:
        raise
    except Exception as exc:
        raise DocQAError(f"Could not read {filename!r}. The file may be damaged.") from exc
    return ParsedDocument(filename, file_hash, kind, sections, tables)
