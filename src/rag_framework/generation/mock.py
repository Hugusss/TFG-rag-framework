"""Deterministic mock generator: the seam without a model.

Exists so the full RAG path — retrieval, context assembly, generation,
answer-with-citations — can run and be tested without any language
model. It is *extractive*: the answer is built verbatim from the
retrieved chunks with explicit source attribution, and it labels
itself, so no report, figure or transcript can mistake a placeholder
for model-generated prose.
"""

from __future__ import annotations

import re

from rag_framework.generation.base import Generator
from rag_framework.models import SearchResult

_SNIPPET_CHARS = 200


def _snippet(text: str) -> str:
    """First words of a chunk, HTML stripped, whitespace collapsed.

    Only tag-shaped spans (``<`` followed by a letter, ``/`` or ``!``)
    are stripped, so prose like ``5 < 6 y 7 > 2`` survives verbatim.
    Truncation is word-safe except for a leading token longer than the
    snippet budget (then a hard cut is the only option).
    """
    clean = re.sub(
        r"\s+", " ", re.sub(r"<[/!]?[a-zA-Z][^>]*>", " ", text)
    ).strip()
    if len(clean) <= _SNIPPET_CHARS:
        return clean
    cut = clean[:_SNIPPET_CHARS]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut + "…"


class MockGenerator(Generator):
    """Extractive, deterministic, dependency-free."""

    def generate(self, query: str, context: list[SearchResult]) -> str:
        if not context:
            return (
                "[mock answer] No relevant passages were retrieved for"
                " this question; the corpus may not contain an answer."
            )
        lines = [
            f"[mock answer] Extractive summary of the top"
            f" {len(context)} retrieved passages:"
        ]
        for rank, source in enumerate(context, start=1):
            origin = source.metadata.get("url") or source.document_id
            lines.append(f"[{rank}] ({origin}) {_snippet(source.text)}")
        return "\n".join(lines)
