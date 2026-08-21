"""Exact nearest-neighbour reference (spec section 13.5).

Brute-force cosine similarity over every stored vector, in pure
Python (``math.sumprod`` is C-implemented: the whole 6k × 1024 corpus
scores in ~40 ms per query). It exists because both retrieval modes
search HNSW graphs, which are *approximate*: comparing two approximate
answers cannot tell which one is wrong. This function can.

Ordering is the same total order as ``merge_top_k`` — score
descending, then ``chunk_id`` — so exact ties are never attributed to
either index.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

from rag_framework.models import SearchResult


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} vs {len(b)}")
    norm = math.sqrt(math.sumprod(a, a)) * math.sqrt(math.sumprod(b, b))
    if norm == 0.0:
        raise ValueError("cosine similarity is undefined for a zero vector")
    return math.sumprod(a, b) / norm


def exact_top_k(
    query_vector: list[float],
    rows: Iterable[tuple[str, str, list[float]]],
    k: int,
) -> list[SearchResult]:
    """Top-``k`` of ``rows`` (``chunk_id, document_id, vector``) by exact
    cosine similarity. ``text`` is empty: the reference ranks ids, it
    does not carry content."""
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError(f"k must be a positive int, got {k!r}")
    scored = [
        SearchResult(
            chunk_id=chunk_id,
            document_id=document_id,
            text="",
            score=cosine_similarity(query_vector, vector),
            metadata={},
        )
        for chunk_id, document_id, vector in rows
    ]
    scored.sort(key=lambda r: (-r.score, r.chunk_id))
    return scored[:k]
