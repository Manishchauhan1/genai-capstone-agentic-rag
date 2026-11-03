"""Similarity retrieval with a weak-match flag (FR-6)."""

from docqa.models import RetrievalResult
from docqa.vectorstore import VectorStore

TOP_K = 5
MIN_SCORE = 0.25  # tuned in Task 11 against the evaluation set


def retrieve(
    store: VectorStore, query: str, k: int = TOP_K, min_score: float = MIN_SCORE
) -> RetrievalResult:
    """Return the top-k hits. `weak` is true when nothing matches well enough."""
    hits = store.query(query, k=k)
    weak = not hits or hits[0].score < min_score
    return RetrievalResult(hits=hits, weak=weak)
