"""The Retriever interface and its timing contract."""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag_framework.models import SearchResult


class RetrievalError(Exception):
    """A condition that invalidates retrieval work.

    Invalid merge input (non-finite scores, bad ``k``), impossible
    partition arguments, or a worker failure in collective mode — all
    surface as this type, never as a partial or reordered result list.
    """


class Retriever(ABC):
    """Turns a query string into ranked SearchResults.

    Contract for every implementation:

    - :meth:`retrieve` returns at most ``k`` results ordered by
      ``score`` descending (similarity, ADR-003).
    - Per-stage wall times of the last call are exposed in ``timings``
      (seconds): at least ``embed_seconds``, ``search_seconds``, and
      ``total_seconds``. Result conversion — the backend-row to
      SearchResult construction, including score conversion — happens
      inside the vector store adapter (ADR-003), so it is included in
      ``search_seconds``; a separate conversion stage would be
      accounting fiction. ``timings`` is
      reset at the start of each call — read it before the next one.
    - Failures from the embedding or store seams propagate as their
      own typed errors; retrievers add no error translation and never
      degrade to partial results silently.
    """

    def __init__(self) -> None:
        self.timings: dict[str, float] = {}

    @abstractmethod
    def retrieve(self, query: str, k: int) -> list[SearchResult]:
        """Return the top-``k`` results for ``query``, best first."""
