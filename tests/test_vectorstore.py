import pytest

from docqa.models import Chunk
from docqa.vectorstore import VectorStore


def _chunk(n: int, text: str, location: str = "full text") -> Chunk:
    return Chunk(id=f"h:{n}", text=text, filename="a.txt", location=location, file_hash="h")


def _three_chunks() -> list[Chunk]:
    return [
        _chunk(0, "Refund window is 30 days from the date of purchase.", "page 2"),
        _chunk(1, "Office parking rules: visitors must register at the front desk."),
        _chunk(2, "Quarterly sales totals grew by twelve percent in the east region."),
    ]


def test_add_counts_and_ignores_duplicates(offline_embedding):
    store = VectorStore(embedding_function=offline_embedding)
    chunks = _three_chunks()
    assert store.add(chunks) == 3
    assert store.count() == 3
    assert store.add(chunks) == 0
    assert store.count() == 3


def test_query_returns_most_similar_first(offline_embedding):
    store = VectorStore(embedding_function=offline_embedding)
    store.add(_three_chunks())
    hits = store.query("what is the refund window?", k=3)
    assert hits[0].chunk.id == "h:0"
    scores = [hit.score for hit in hits]
    assert scores == sorted(scores, reverse=True)


def test_hit_round_trips_chunk_fields(offline_embedding):
    store = VectorStore(embedding_function=offline_embedding)
    chunks = _three_chunks()
    store.add(chunks)
    hit = store.query("refund window", k=1)[0]
    assert hit.chunk == chunks[0]


def test_query_on_empty_store_returns_empty_list(offline_embedding):
    assert VectorStore(embedding_function=offline_embedding).query("anything") == []


def test_stores_are_isolated(offline_embedding):
    first = VectorStore(embedding_function=offline_embedding)
    second = VectorStore(embedding_function=offline_embedding)
    first.add(_three_chunks())
    assert second.count() == 0
    assert second.query("refund") == []


@pytest.mark.network
def test_default_embedding_matches_meaning_not_just_words():
    """Needs internet: Chroma downloads its small local model on first use."""
    store = VectorStore()
    store.add(_three_chunks())
    hits = store.query("how long do I have to return a purchase?", k=3)
    assert hits[0].chunk.id == "h:0"
