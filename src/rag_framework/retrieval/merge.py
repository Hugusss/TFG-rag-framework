"""Global top-k over per-partition candidates.

A pure function — no backend, no threads — so the same code reduces
results whether the partials came from threads, processes or remote
workers, and can be tested exhaustively on its own. Ordering is total
and deterministic: score descending, then ``chunk_id`` ascending, so
equal scores never depend on which partition answered first (ADR-006).
"""

from __future__ import annotations

import logging
import math

from rag_framework.models import SearchResult
from rag_framework.retrieval.base import RetrievalError

_logger = logging.getLogger(__name__)


def merge_top_k(partial_results: list[list[SearchResult]], k: int) -> list[SearchResult]:
    """Select the global top-``k`` from per-partition result lists.

    - Empty partitions and short lists are fine: the result has at most
      ``k`` items, fewer when the candidates run out.
    - A ``chunk_id`` seen more than once keeps its best-scoring
      candidate (ties resolved by provenance, then input order) and is
      logged: with document-level partitioning a duplicate cannot
      happen, so one signals overlapping partitions upstream.
    - A non-finite or non-numeric score raises :class:`RetrievalError`
      — NaN would silently corrupt the order.

    Cost: one pass over the ``C`` candidates plus a sort of the unique
    ids, so O(C log C) time and O(C) memory. With document-level
    partitioning ``C`` is at most ``P * k``, which is why the reduction
    stays negligible next to the searches that feed it.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise RetrievalError(f"k must be a positive int, got {k!r}")

    best: dict[str, SearchResult] = {}
    duplicates = 0
    for partition in partial_results:
        for result in partition:
            score = result.score
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(score)
            ):
                raise RetrievalError(
                    f"chunk {result.chunk_id}: invalid score {score!r}"
                )
            current = best.get(result.chunk_id)
            if current is None:
                best[result.chunk_id] = result
                continue
            duplicates += 1
            if _prefer(result, current):
                best[result.chunk_id] = result
    if duplicates:
        _logger.warning(
            "merge_top_k: %d duplicate chunk id(s) across partitions —"
            " partitions overlap upstream?",
            duplicates,
        )

    ranked = sorted(best.values(), key=lambda r: (-r.score, r.chunk_id))
    return ranked[:k]


def _prefer(candidate: SearchResult, current: SearchResult) -> bool:
    """Higher score wins; equal scores fall back to provenance order so
    the choice does not depend on partition arrival order."""
    if candidate.score != current.score:
        return candidate.score > current.score
    return (candidate.partition_id or "", candidate.worker_id or "") < (
        current.partition_id or "",
        current.worker_id or "",
    )
