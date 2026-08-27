"""The EmbeddingProvider interface and its contract."""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag_framework.models import Chunk


class EmbeddingError(Exception):
    """A condition that invalidates embedding.

    A chunk without its official vector, inconsistent dimensions, or
    corrupt vector data mean the experiment cannot be what it claims.
    Embedding problems are never repaired silently: no skipping, no
    padding, no quiet re-encoding.
    """


class EmbeddingProvider(ABC):
    """Turns chunks and queries into vectors.

    ``embed_documents`` takes Chunks rather than bare strings (ADR-005):
    a provider that looks vectors up by *identity* — the precomputed one
    is keyed by ``(document_id, position)`` — cannot express that lookup
    through a text-only interface without a side channel, and a side
    channel would put corpus knowledge back into the orchestrator.
    Providers that only need text simply read ``chunk.text``.

    Contract for every implementation:

    - ``embed_documents`` returns exactly one vector per chunk, in
      input order, each of dimension ``dimension``. On any
      inconsistency it raises :class:`EmbeddingError` instead of
      skipping, padding, or re-encoding.
    - ``embed_query`` embeds free text (queries have no identity) with
      the same model, dimension, and normalization as document vectors
      — one collection holds one embedding space, never a mixture.
      An adapter whose query path is not yet
      available must fail loudly (``NotImplementedError`` naming the
      reason), never return a degraded vector.
    - Deterministic: the same inputs produce the same vectors.
    - The embedding identity is machine-readable: ``model_id``,
      ``dimension``, and ``normalized`` describe the embedding space,
      and ``device`` records where encoding executes: it does not change
      the space, but a latency without it is uninterpretable. The
      vector-store layer stamps collection
      metadata and result files carry the record without any component
      hardcoding model facts outside this seam.
    """

    model_id: str
    dimension: int
    normalized: bool
    device: str

    @abstractmethod
    def embed_documents(self, chunks: list[Chunk]) -> list[list[float]]:
        """Return one vector per chunk, in input order."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Return the vector for one query text."""
