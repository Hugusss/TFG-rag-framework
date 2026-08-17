"""The Generator interface and its contract."""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag_framework.models import SearchResult


class GenerationError(Exception):
    """A condition that invalidates generation (model failure,
    timeout). Never silently degraded into an empty answer (Rule 6)."""


class Generator(ABC):
    """Produces a grounded answer from a query and retrieved context.

    Contract for every implementation:

    - :meth:`generate` returns the answer text; it must ground itself
      in the given context and degrade honestly when the context is
      empty (say so — never invent). Prompt construction — the spec's
      section-14 stage — happens entirely inside the implementation;
      no other component knows what a prompt is (Rule 2).
    - Deterministic implementations state so; sampling ones record
      their parameters. The mock is fully deterministic.
    - Failures raise :class:`GenerationError`; credentials, if any,
      come from the environment, never from source code (spec
      section 14 — this project extends the rule to configuration
      files).
    """

    @abstractmethod
    def generate(self, query: str, context: list[SearchResult]) -> str:
        """Return an answer for ``query`` grounded in ``context``."""
