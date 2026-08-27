"""The VectorStore interface and its contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from rag_framework.models import Chunk, SearchResult


class VectorStoreError(Exception):
    """A condition that invalidates store work.

    Incompatible index configuration, duplicate ids inside one batch,
    mismatched chunk/vector counts, backend failures — all surface as
    this type; backend exceptions never escape raw.
    """


class VectorStore(ABC):
    """Persists chunks with their vectors and searches them.

    Contract for every implementation:

    - :meth:`create_or_open` prepares a named collection and is
      idempotent; opening an existing collection whose stored embedding
      configuration conflicts with the expected one raises
      :class:`VectorStoreError`, because an incompatible index must never
      be silently reused.
    - :meth:`add` stores one vector per chunk, rejects mismatched
      lengths and duplicate chunk ids within a batch, and is
      **idempotent across re-ingestion**: chunk ids are deterministic,
      so re-adding the same chunks must not create duplicates.
      Stored per-row metadata is exactly the
      chunk's metadata plus the reserved provenance keys
      ``document_id`` and ``position``; a non-scalar metadata value or
      a reserved-key collision is an error, never a silent drop.
    - :meth:`search` returns at most ``k`` results ordered by
      ``SearchResult.score`` — **similarity, higher is better** — with
      each adapter converting its native metric at this boundary
      (ADR-003). ``filters`` is a flat ``{key: value}`` equality
      mapping over chunk metadata; adapters translate it to their
      native syntax so backend query languages never leak through the
      seam.
    - Collections persist across processes; :meth:`count` reports the
      stored vector count; :meth:`reset` empties the collection but
      leaves it usable, so a controlled experiment can rebuild in place.
    - :meth:`iter_vectors` streams every stored ``(chunk_id,
      document_id, vector)`` in a stable order, so an exact brute-force
      reference can be computed over precisely what the index holds —
      differences between two approximate indexes are only attributable
      against an exact answer.
    """

    @abstractmethod
    def create_or_open(self, collection_name: str) -> None:
        """Create the collection or open it if it already exists."""

    @abstractmethod
    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Store one vector per chunk; idempotent for identical ids."""

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        k: int,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        """Return the top-``k`` most similar chunks, best first."""

    @abstractmethod
    def count(self) -> int:
        """Number of vectors currently stored."""

    @abstractmethod
    def reset(self) -> None:
        """Empty the collection, keeping it usable."""

    @abstractmethod
    def iter_vectors(self) -> Iterator[tuple[str, str, list[float]]]:
        """Stream ``(chunk_id, document_id, vector)`` for every row."""
