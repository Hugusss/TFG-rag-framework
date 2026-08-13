"""The EmbeddingProvider interface and its contract."""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag_framework.models import Chunk


class EmbeddingError(Exception):
    """A condition that invalidates embedding.

    A chunk without its official vector, inconsistent dimensions, or
    corrupt vector data mean the experiment cannot be what it claims.
    Embedding problems are never repaired silently (Rule 6): no
    skipping, no padding, no quiet re-encoding.
    """


class EmbeddingProvider(ABC):
    """Turns chunks and queries into vectors.

    ``embed_documents`` deliberately takes Chunks, not bare strings —
    a documented deviation from the spec's sketch (ADR-005): adapters
    that look vectors up by *identity* (the precomputed provider is
    keyed by ``(document_id, position)``) cannot express that lookup
    through a text-only interface without side channels, which would
    force exactly the orchestrator branch ADR-002 forbids. Providers
    that only need text simply read ``chunk.text``.

    Contract for every implementation:

    - ``embed_documents`` returns exactly one vector per chunk, in
      input order, each of dimension ``dimension``. On any
      inconsistency it raises :class:`EmbeddingError` instead of
      skipping, padding, or re-encoding.
    - ``embed_query`` embeds free text (queries have no identity) with
      the same model, dimension, and normalization as document vectors
      — mixing embedding spaces in one collection is forbidden
      (spec section 10). An adapter whose query path is not yet
      available must fail loudly (``NotImplementedError`` naming the
      reason), never return a degraded vector.
    - Deterministic: the same inputs produce the same vectors.
    - The embedding identity is machine-readable: ``model_id``,
      ``dimension``, and ``normalized`` describe the embedding space,
      so the vector-store layer can stamp collection names/metadata and
      result files can carry the spec-section-10 record without any
      component hardcoding model facts outside this seam.
    """

    model_id: str
    dimension: int
    normalized: bool

    @abstractmethod
    def embed_documents(self, chunks: list[Chunk]) -> list[list[float]]:
        """Return one vector per chunk, in input order."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Return the vector for one query text."""
