"""One user's session: their files, their tables and their vector store."""

from dataclasses import dataclass

import pandas as pd

from docqa.chunking import chunk_document
from docqa.ingestion import parse_file
from docqa.models import ParsedDocument
from docqa.vectorstore import VectorStore


@dataclass(frozen=True)
class AddResult:
    filename: str
    chunks_added: int
    duplicate: bool


class Workspace:
    def __init__(self, embedding_function=None):
        self.store = VectorStore(embedding_function=embedding_function)
        self._documents: dict[str, ParsedDocument] = {}  # keyed by file_hash

    def add_file(self, filename: str, data: bytes) -> AddResult:
        """Parse, chunk and store a file. A file already added changes nothing."""
        document = parse_file(filename, data)
        if document.file_hash in self._documents:
            return AddResult(filename, 0, True)
        added = self.store.add(chunk_document(document))
        self._documents[document.file_hash] = document
        return AddResult(filename, added, False)

    def is_empty(self) -> bool:
        return not self._documents

    def filenames(self) -> list[str]:
        names: list[str] = []
        for document in self._documents.values():
            if document.filename not in names:
                names.append(document.filename)
        return names

    def tables(self) -> dict[str, pd.DataFrame]:
        """All tables, keyed "<filename>::<table name>" (CSV tables are named "csv")."""
        return {
            f"{document.filename}::{name}": frame
            for document in self._documents.values()
            for name, frame in document.tables.items()
        }
