"""Shared data types. Every module imports these; none redefine them."""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Section:
    label: str
    text: str


@dataclass(frozen=True)
class ParsedDocument:
    filename: str
    file_hash: str
    kind: str
    sections: list[Section]
    tables: dict[str, pd.DataFrame]


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    filename: str
    location: str
    file_hash: str


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class RetrievalResult:
    hits: list[Hit]
    weak: bool


@dataclass(frozen=True)
class Source:
    filename: str
    location: str


@dataclass(frozen=True)
class AgentResult:
    answer: str
    sources: list[Source]
    trace: list[str]
    llm_calls: int
