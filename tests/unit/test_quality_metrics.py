"""Unit tests for the retrieval-quality metric functions."""

import pytest

from rag_framework.metrics.quality import (
    document_ranking,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    summarize,
)
from rag_framework.models import SearchResult


def result(doc_id, score):
    return SearchResult(
        chunk_id=f"c-{doc_id}-{score}",
        document_id=doc_id,
        text="t",
        score=score,
        metadata={},
    )


class TestDocumentRanking:
    def test_collapses_chunks_keeping_best_rank(self):
        results = [
            result("a", 0.9),
            result("b", 0.8),
            result("a", 0.7),  # second chunk of a: ignored
            result("c", 0.6),
        ]
        assert document_ranking(results) == ["a", "b", "c"]

    def test_empty(self):
        assert document_ranking([]) == []


class TestRecall:
    def test_full_and_partial(self):
        assert recall_at_k(["a", "b", "c"], {"a", "c"}, 3) == 1.0
        assert recall_at_k(["a", "b", "c"], {"a", "z"}, 3) == 0.5
        assert recall_at_k(["a", "b"], {"z"}, 2) == 0.0

    def test_k_window_applies(self):
        assert recall_at_k(["x", "y", "a"], {"a"}, 2) == 0.0
        assert recall_at_k(["x", "y", "a"], {"a"}, 3) == 1.0

    def test_empty_relevance_set_refused(self):
        with pytest.raises(ValueError, match="undefined"):
            recall_at_k(["a"], set(), 5)

    def test_invalid_k_refused(self):
        with pytest.raises(ValueError, match="positive"):
            recall_at_k(["a"], {"a"}, 0)


class TestPrecision:
    def test_fixed_denominator(self):
        assert precision_at_k(["a", "b"], {"a"}, 2) == 0.5
        # only 2 retrieved but k=4: missing positions are misses
        assert precision_at_k(["a", "b"], {"a", "b"}, 4) == 0.5

    def test_zero_when_nothing_relevant(self):
        assert precision_at_k(["x", "y"], {"a"}, 2) == 0.0


class TestReciprocalRank:
    def test_first_hit_position(self):
        assert reciprocal_rank(["a", "b"], {"a"}) == 1.0
        assert reciprocal_rank(["x", "a"], {"a"}) == 0.5
        assert reciprocal_rank(["x", "y", "z", "a"], {"a"}) == 0.25

    def test_zero_when_absent(self):
        assert reciprocal_rank(["x", "y"], {"a"}) == 0.0

    def test_empty_ranking(self):
        assert reciprocal_rank([], {"a"}) == 0.0


class TestSummarize:
    def test_known_values(self):
        block = summarize([4.0, 1.0, 3.0, 2.0])
        assert block["n"] == 4
        assert block["min"] == 1.0
        assert block["median"] == 3.0  # upper median by index n//2
        assert block["max"] == 4.0
        assert block["mean"] == 2.5
        assert block["std"] == pytest.approx(1.118, abs=1e-3)

    def test_single_value(self):
        block = summarize([7.0])
        assert block["min"] == block["median"] == block["p95"] == 7.0
        assert block["std"] == 0.0

    def test_empty_refused(self):
        with pytest.raises(ValueError, match="empty"):
            summarize([])
