"""Unit tests for the exact brute-force reference."""

import math

import pytest

from rag_framework.metrics.exact import cosine_similarity, exact_top_k


def rows(*vectors):
    return [(f"c{i}", f"d{i}", list(v)) for i, v in enumerate(vectors)]


class TestCosine:
    def test_values(self):
        assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
        assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
        assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)
        assert cosine_similarity([2, 0], [0.5, 0.5]) == pytest.approx(math.sqrt(0.5))

    def test_scale_invariant(self):
        assert cosine_similarity([3, 4], [30, 40]) == pytest.approx(1.0)

    def test_dimension_mismatch_and_zero_vector(self):
        with pytest.raises(ValueError, match="dimension"):
            cosine_similarity([1, 0], [1, 0, 0])
        with pytest.raises(ValueError, match="zero vector"):
            cosine_similarity([0, 0], [1, 0])


class TestExactTopK:
    def test_ranks_by_similarity(self):
        data = rows([1, 0], [0, 1], [0.6, 0.8], [0.8, 0.6])
        top = exact_top_k([1.0, 0.0], data, 3)
        assert [r.chunk_id for r in top] == ["c0", "c3", "c2"]
        assert [round(r.score, 3) for r in top] == [1.0, 0.8, 0.6]
        assert top[0].document_id == "d0" and top[0].text == ""

    def test_ties_break_on_chunk_id_and_k_caps(self):
        data = [("z", "d", [1.0, 0.0]), ("a", "d", [1.0, 0.0]), ("m", "d", [0.0, 1.0])]
        top = exact_top_k([1.0, 0.0], data, 2)
        assert [r.chunk_id for r in top] == ["a", "z"]
        assert len(exact_top_k([1.0, 0.0], data, 10)) == 3

    def test_empty_rows_and_invalid_k(self):
        assert exact_top_k([1.0], [], 5) == []
        with pytest.raises(ValueError, match="k must be"):
            exact_top_k([1.0], rows([1.0]), 0)
