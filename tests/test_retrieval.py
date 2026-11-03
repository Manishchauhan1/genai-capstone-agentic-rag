from docqa.models import Chunk, Hit
from docqa.retrieval import MIN_SCORE, TOP_K, retrieve


class FakeStore:
    """Returns hits with scores the test controls and records the k it was given."""

    def __init__(self, scores):
        self.scores = scores
        self.asked_k = None

    def query(self, text, k=5):
        self.asked_k = k
        chunk = Chunk("h:0", "text", "a.txt", "full text", "h")
        return [Hit(chunk, score) for score in self.scores][:k]


def test_weak_when_best_score_below_cutoff():
    assert retrieve(FakeStore([0.1]), "q").weak is True


def test_not_weak_when_best_score_above_cutoff():
    assert retrieve(FakeStore([0.6]), "q").weak is False


def test_no_hits_is_weak():
    result = retrieve(FakeStore([]), "q")
    assert result.weak is True
    assert result.hits == []


def test_retrieve_passes_k_to_store():
    store = FakeStore([0.9, 0.8, 0.7])
    result = retrieve(store, "q", k=2)
    assert store.asked_k == 2
    assert len(result.hits) == 2


def test_defaults_match_the_plan():
    assert (TOP_K, MIN_SCORE) == (5, 0.25)
