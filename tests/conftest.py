"""Shared test fixtures. FakeLLM is added by Task 7."""

import hashlib
import re
from pathlib import Path

import numpy as np
import pytest
from chromadb import EmbeddingFunction

from make_fixtures import build_fixtures


class HashEmbedding(EmbeddingFunction):
    """Deterministic offline embedding: hashed bag of words, so texts that share
    words are close. Used so unit tests need no model download or network."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def __call__(self, input):
        vectors = []
        for text in input:
            vector = np.zeros(self.dim, dtype=np.float32)
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim
                vector[bucket] += 1.0
            norm = np.linalg.norm(vector)
            vectors.append(vector / norm if norm else vector)
        return vectors

    @staticmethod
    def name() -> str:
        return "hash-embedding-for-tests"

    def get_config(self):
        return {"dim": self.dim}

    @staticmethod
    def build_from_config(config):
        return HashEmbedding(**config)


class FakeLLM:
    """Scripted LLM for tests. Each reply is returned (or raised, if an Exception)
    in order; every call is recorded in `.calls` as (system, user)."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str, *, json_mode: bool = False) -> str:
        self.calls.append((system, user))
        if not self.replies:
            raise AssertionError("FakeLLM ran out of scripted replies")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


@pytest.fixture
def offline_embedding() -> HashEmbedding:
    return HashEmbedding()


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return build_fixtures(Path(__file__).parent / "fixtures")
