"""The Generator interface and its contract."""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag_framework.models import SearchResult


class GenerationError(Exception):
    """A condition that invalidates generation (model failure,
    timeout). Never silently degraded into an empty answer."""


class Generator(ABC):
    """Produces a grounded answer from a query and retrieved context.

    Contract for every implementation:

    - :meth:`generate` returns the answer text; it must ground itself
      in the given context and degrade honestly when the context is
      empty (say so — never invent). Prompt construction happens entirely
      inside the implementation; no other component knows what a
      prompt is, so swapping models never touches the pipeline.
    - Deterministic implementations state so; sampling ones record
      their parameters. The mock is fully deterministic.
    - Failures raise :class:`GenerationError`; credentials, if any,
      come from the environment, never from source code or from a
      configuration file kept in the repository.
    """

    @abstractmethod
    def generate(self, query: str, context: list[SearchResult]) -> str:
        """Return an answer for ``query`` grounded in ``context``."""
