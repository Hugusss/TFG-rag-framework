"""Unit tests for the ranking comparisons between two result lists."""

import pytest

from rag_framework.metrics.quality import (
    classify_differences,
    imbalance,
    discordant_pairs,
    max_score_difference,
    overlap_at_k,
    positions_preserved,
)
from rag_framework.models import SearchResult


def res(chunk_id, score):
    return SearchResult(chunk_id=chunk_id, document_id="d", text="", score=score, metadata={})


class TestPureComparisons:
    def test_overlap_and_positions(self):
        assert overlap_at_k(["a", "b", "c"], ["c", "b", "a"], 3) == 1.0
        assert positions_preserved(["a", "b", "c"], ["c", "b", "a"], 3) == pytest.approx(1 / 3)
        assert overlap_at_k(["a", "b"], ["a", "x", "y"], 3) == pytest.approx(1 / 3)
        assert overlap_at_k([], ["a"], 1) == 0.0
        with pytest.raises(ValueError):
            overlap_at_k(["a"], ["a"], 0)

    def test_discordant_pairs_only_over_common_ids(self):
        assert discordant_pairs(["a", "b", "c"], ["a", "b", "c"]) == []
        assert discordant_pairs(["b", "a", "c"], ["a", "b", "c"]) == [("b", "a")]
        assert discordant_pairs(["x", "b", "a"], ["a", "b", "y"]) == [("b", "a")]
        assert len(discordant_pairs(["c", "b", "a"], ["a", "b", "c"])) == 3

    def test_max_score_difference(self):
        assert max_score_difference({"a": 0.9, "b": 0.5}, {"a": 0.8, "b": 0.5}) == pytest.approx(0.1)
        assert max_score_difference({"a": 0.9}, {"b": 0.5}) is None


class TestClassification:
    def test_identical_lists_have_no_differences(self):
        exact = [res("a", 0.9), res("b", 0.8)]
        counts = classify_differences(exact, exact, exact)
        assert all(v == 0 for v in counts.values())

    def test_tie_reorder_is_not_an_error(self):
        candidate = [res("a", 0.5), res("b", 0.5)]
        baseline = [res("b", 0.5), res("a", 0.5)]
        counts = classify_differences(candidate, baseline, candidate)
        assert counts["tie_reorders"] == 1 and counts["score_reorders"] == 0

    def test_baseline_miss_attributed_with_exact_reference(self):
        exact = [res("a", 0.9), res("x", 0.7)]
        candidate = [res("a", 0.9), res("x", 0.7)]
        baseline = [res("a", 0.9), res("y", 0.6)]  # HNSW skipped x
        counts = classify_differences(candidate, baseline, exact)
        assert counts["baseline_misses"] == 1
        assert counts["candidate_misses"] == 0
        assert counts["outside_exact"] == 1  # y

    def test_candidate_miss_and_both_miss(self):
        exact = [res("a", 0.9), res("x", 0.7), res("z", 0.65)]
        candidate = [res("a", 0.9), res("q", 0.2)]
        baseline = [res("a", 0.9), res("x", 0.7)]
        counts = classify_differences(candidate, baseline, exact)
        assert counts["candidate_misses"] == 1  # x
        assert counts["both_miss"] == 1  # z
        assert counts["outside_exact"] == 1  # q

    def test_score_reorder_is_flagged(self):
        candidate = [res("a", 0.9), res("b", 0.8)]
        baseline = [res("b", 0.8), res("a", 0.9)]  # impossible for an exact backend
        counts = classify_differences(candidate, baseline, candidate)
        assert counts["score_reorders"] == 1 and counts["tie_reorders"] == 0


class TestImbalance:
    def test_balanced_and_skewed(self):
        assert imbalance([10, 10, 10]) == {"max_over_mean": 1.0, "min_over_mean": 1.0, "mean": 10.0}
        skewed = imbalance([5, 15])
        assert skewed["max_over_mean"] == pytest.approx(1.5)
        assert skewed["min_over_mean"] == pytest.approx(0.5)

    def test_degenerate_inputs(self):
        assert imbalance([0, 0])["max_over_mean"] is None
        with pytest.raises(ValueError):
            imbalance([])
