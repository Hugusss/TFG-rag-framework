"""Retrieval-quality metrics (spec section 17.3), document level.

Pure functions over ranked document ids and a relevance set — no
retrieval, no storage, no I/O — so the same code scores the chunk-size
experiment, the collective-correctness comparison, and anything later,
and every function is exhaustively unit-testable.

Relevance in this project is judged at document level
(evaluation/queries.jsonl), so rankings are *document* rankings: use
:func:`document_ranking` to collapse a chunk-level result list first.
"""

from __future__ import annotations

from rag_framework.models import SearchResult


def document_ranking(results: list[SearchResult]) -> list[str]:
    """Unique document ids in best-first order.

    A document retrieved through several chunks ranks at its best
    (first) chunk; later occurrences are ignored.
    """
    seen: set[str] = set()
    ranking: list[str] = []
    for result in results:
        if result.document_id not in seen:
            seen.add(result.document_id)
            ranking.append(result.document_id)
    return ranking


def recall_at_k(ranking: list[str], relevant: set[str], k: int) -> float:
    """Fraction of the relevant documents found in the top ``k``.

    Undefined for an empty relevance set — callers must not score
    unanswerable queries with recall (raise instead of guessing).
    """
    if not relevant:
        raise ValueError("recall is undefined for an empty relevance set")
    if k < 1:
        raise ValueError(f"k must be positive, got {k}")
    return len(set(ranking[:k]) & relevant) / len(relevant)


def precision_at_k(ranking: list[str], relevant: set[str], k: int) -> float:
    """Fraction of the top ``k`` positions holding a relevant document.

    Conventional fixed-denominator form: fewer than ``k`` retrieved
    documents count as misses, so shallow result lists are penalized,
    not hidden.
    """
    if k < 1:
        raise ValueError(f"k must be positive, got {k}")
    return len(set(ranking[:k]) & relevant) / k


def reciprocal_rank(ranking: list[str], relevant: set[str]) -> float:
    """1/rank of the first relevant document, 0.0 if none appears."""
    for rank, document_id in enumerate(ranking, start=1):
        if document_id in relevant:
            return 1.0 / rank
    return 0.0


def summarize(values: list[float]) -> dict:
    """The spec-section-17.2 summary block: min/median/p95/mean/std."""
    if not values:
        raise ValueError("cannot summarize an empty list")
    ordered = sorted(values)
    n = len(ordered)
    mean = sum(ordered) / n
    variance = sum((x - mean) ** 2 for x in ordered) / n
    return {
        "n": n,
        "min": ordered[0],
        "median": ordered[n // 2],
        "p95": ordered[min(n - 1, round(0.95 * (n - 1)))],
        "max": ordered[-1],
        "mean": mean,
        "std": variance**0.5,
    }
