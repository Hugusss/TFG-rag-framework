"""Unit tests for the global top-k merge, independent of any store."""

import logging
import math
import random

import pytest

from rag_framework.models import SearchResult
from rag_framework.retrieval.base import RetrievalError
from rag_framework.retrieval.merge import merge_top_k


def result(chunk_id, score, partition=None, worker=None, document="d"):
    return SearchResult(
        chunk_id=chunk_id,
        document_id=document,
        text="t",
        score=score,
        metadata={},
        worker_id=worker,
        partition_id=partition,
    )


def ids(results):
    return [r.chunk_id for r in results]


class TestSelection:
    def test_global_top_k_across_partitions(self):
        partials = [
            [result("a", 0.9), result("b", 0.4)],
            [result("c", 0.8), result("d", 0.7)],
            [result("e", 0.95)],
        ]
        assert ids(merge_top_k(partials, 3)) == ["e", "a", "c"]

    def test_empty_partitions_are_skipped(self):
        partials = [[], [result("a", 0.5)], [], []]
        assert ids(merge_top_k(partials, 5)) == ["a"]

    def test_all_empty_gives_empty(self):
        assert merge_top_k([[], []], 3) == []
        assert merge_top_k([], 3) == []

    def test_fewer_candidates_than_k(self):
        partials = [[result("a", 0.5)], [result("b", 0.6)]]
        assert ids(merge_top_k(partials, 10)) == ["b", "a"]

    def test_results_pass_through_with_provenance(self):
        hit = result("a", 0.5, partition="p1", worker="w0")
        merged = merge_top_k([[hit]], 1)
        assert merged[0] is hit
        assert merged[0].partition_id == "p1" and merged[0].worker_id == "w0"

    def test_input_is_not_mutated(self):
        partition = [result("b", 0.1), result("a", 0.9)]
        merge_top_k([partition], 1)
        assert ids(partition) == ["b", "a"]


class TestDeterminism:
    def test_ties_break_on_chunk_id(self):
        partials = [[result("z", 0.5), result("m", 0.5)], [result("a", 0.5)]]
        assert ids(merge_top_k(partials, 3)) == ["a", "m", "z"]

    def test_partition_order_does_not_matter(self):
        partials = [
            [result("a", 0.9), result("b", 0.7)],
            [result("c", 0.7), result("d", 0.7)],
            [result("e", 0.95), result("f", 0.1)],
        ]
        reference = ids(merge_top_k(partials, 4))
        rng = random.Random(7)
        for _ in range(20):
            shuffled = [list(p) for p in partials]
            rng.shuffle(shuffled)
            for p in shuffled:
                rng.shuffle(p)
            assert ids(merge_top_k(shuffled, 4)) == reference


class TestDuplicates:
    def test_duplicate_keeps_best_score(self, caplog):
        partials = [[result("a", 0.3)], [result("a", 0.8)], [result("b", 0.5)]]
        with caplog.at_level(logging.WARNING):
            merged = merge_top_k(partials, 5)
        assert [(r.chunk_id, r.score) for r in merged] == [("a", 0.8), ("b", 0.5)]
        assert "1 duplicate" in caplog.text

    def test_equal_score_duplicate_resolved_by_provenance_not_order(self):
        first = result("a", 0.5, partition="p1")
        second = result("a", 0.5, partition="p0")
        assert merge_top_k([[first], [second]], 1)[0] is second
        assert merge_top_k([[second], [first]], 1)[0] is second

    def test_duplicates_do_not_eat_k_slots(self):
        partials = [[result("a", 0.9)], [result("a", 0.9)], [result("b", 0.1)]]
        assert ids(merge_top_k(partials, 2)) == ["a", "b"]


class TestInvalidInput:
    @pytest.mark.parametrize(
        "bad", [math.nan, math.inf, -math.inf, None, "0.9", True]
    )
    def test_invalid_score_raises(self, bad):
        partials = [[result("ok", 0.5)], [result("bad", bad)]]
        with pytest.raises(RetrievalError, match="bad"):
            merge_top_k(partials, 2)

    def test_invalid_score_beyond_k_still_raises(self):
        # validation covers every candidate, not just the winners
        partials = [[result("a", 0.9)], [result("nan", math.nan)]]
        with pytest.raises(RetrievalError):
            merge_top_k(partials, 1)

    @pytest.mark.parametrize("bad", [0, -3, 2.0, True, None])
    def test_invalid_k_raises(self, bad):
        with pytest.raises(RetrievalError, match="k must be"):
            merge_top_k([[result("a", 0.5)]], bad)
