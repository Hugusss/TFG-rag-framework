"""Exact nearest-neighbour reference by brute force.

Cosine similarity against every stored vector, in pure Python
(``math.sumprod`` is C-implemented). It exists because both retrieval
modes search approximate HNSW graphs, and comparing two approximate
answers to each other cannot say which one is wrong. Ordering is the
same total order as ``merge_top_k``, so a tie is never scored as a
difference between indexes.
"""

from __future__ import annotations

import heapq
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


def exact_top_k_many(
    query_vectors: dict[str, list[float]],
    rows: Iterable[tuple[str, str, list[float]]],
    k: int,
) -> dict[str, list[SearchResult]]:
    """Exact top-``k`` for many queries in ONE streaming pass.

    Never materialises the corpus: each row is scored against every
    query as it streams by (one norm per row, one dot product per
    query) and only per-query top candidates are kept. Equivalent to
    calling :func:`exact_top_k` per query — pinned by test — but usable
    on corpora whose vectors do not fit in memory as Python lists. A
    small buffer above ``k`` absorbs score ties (duplicate texts have
    identical vectors) so the final ``(score desc, chunk_id)`` order
    matches the single-query function.

    Cost: O(N * Q * d) time for ``N`` stored vectors, ``Q`` queries and
    ``d`` dimensions — the price of an exact answer, and the reason this
    is a reference rather than a retrieval mode — but only O(Q * k)
    memory, which is what makes it usable at corpus scale.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError(f"k must be a positive int, got {k!r}")
    keep = 3 * k
    norms = {}
    for qid, vector in query_vectors.items():
        norm = math.sqrt(math.sumprod(vector, vector))
        if norm == 0.0:
            raise ValueError("cosine similarity is undefined for a zero vector")
        norms[qid] = norm
    heaps: dict[str, list] = {qid: [] for qid in query_vectors}
    for chunk_id, document_id, vector in rows:
        row_norm = math.sqrt(math.sumprod(vector, vector))
        if row_norm == 0.0:
            raise ValueError(
                f"cosine similarity is undefined for a zero vector ({chunk_id})"
            )
        for qid, qvec in query_vectors.items():
            score = math.sumprod(qvec, vector) / (norms[qid] * row_norm)
            heap = heaps[qid]
            # max-heap by (score, reversed chunk_id) via negation on pop
            # is awkward; keep a min-heap of the top `keep` and repair
            # tie order in the final exact sort
            if len(heap) < keep:
                heapq.heappush(heap, (score, chunk_id, document_id))
            elif score > heap[0][0]:
                heapq.heapreplace(heap, (score, chunk_id, document_id))
    results = {}
    for qid, heap in heaps.items():
        scored = [
            SearchResult(
                chunk_id=chunk_id, document_id=document_id, text="",
                score=score, metadata={},
            )
            for score, chunk_id, document_id in heap
        ]
        scored.sort(key=lambda r: (-r.score, r.chunk_id))
        results[qid] = scored[:k]
    return results
