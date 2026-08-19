"""Recursive chunking strategy: paragraph-aware word windows (spec §9).

The strategy the pipeline uses when the corpus does not bring its own
chunk boundaries (or when an experiment wants boundaries it controls):

1. Text is split into paragraphs (blank-line boundaries), each with
   whitespace normalized.
2. Consecutive paragraphs are packed into one chunk while the running
   word count stays within ``target_tokens`` — paragraph boundaries
   are meaningful and are retained (§9).
3. A single paragraph larger than the target is split into word
   windows of ``target_tokens`` with ``overlap_tokens`` of overlap
   between consecutive windows. Overlap exists to guard *arbitrary*
   cuts; paragraph boundaries are not arbitrary, so packed-paragraph
   seams get no overlap (recorded in ADR-008).
4. Chunks with fewer than ``minimum_tokens`` words are dropped and
   counted (``chunks_dropped_below_minimum``) — never silently.

Token policy (ADR-008): a "token" is a whitespace-delimited word.
Stdlib-only and deterministic; the imprecision relative to model
subword tokens is accepted and documented — the §9 experiment
documents a trade-off, it does not tune a model budget.

Unlike the publisher path, this strategy owns its boundaries and its
text: whitespace normalization is allowed here because the vectors for
these chunks are computed live from exactly this text (the
publisher-path byte-exactness rule protects *precomputed* alignment,
which this strategy never uses).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Iterator

from rag_framework.chunking.base import Chunker
from rag_framework.models import Chunk, Document, make_chunk_id

_logger = logging.getLogger(__name__)

# two newlines with any whitespace between; covers \n\n, \r\n\r\n and
# unicode-space blank lines (a CR-only "\r\r" break is not recognized —
# extinct in practice, and content still flows into windows unharmed)
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")

# publisher-path metadata keys; meaningless for computed boundaries
_DISCARDED_KEYS = ("chunk_offsets", "start", "end", "clipped")


class RecursiveChunker(Chunker):
    """Paragraph-packing word-window chunker."""

    def __init__(
        self, *, target_tokens: int, overlap_tokens: int, minimum_tokens: int
    ) -> None:
        super().__init__()
        # the config seam validates these for pipeline use; this guard
        # covers direct construction and must survive python -O (an
        # invalid stride would silently produce zero windows — the
        # Rule-6 failure mode), so it is a real raise, not an assert
        if target_tokens < 1 or not 0 <= overlap_tokens < target_tokens:
            raise ValueError(
                f"invalid chunking geometry: target={target_tokens},"
                f" overlap={overlap_tokens}"
            )
        self._target = target_tokens
        self._overlap = overlap_tokens
        self._minimum = minimum_tokens
        self.chunks_dropped_below_minimum = 0

    def split(self, documents: Iterable[Document]) -> Iterator[Chunk]:
        self.documents_processed = 0
        self.documents_without_chunks = 0
        self.documents_rejected = 0
        self.chunks_dropped_below_minimum = 0
        self.rejections = []
        return self._iter_chunks(documents)

    def _iter_chunks(self, documents: Iterable[Document]) -> Iterator[Chunk]:
        for document in documents:
            self.documents_processed += 1
            inherited = {
                key: value
                for key, value in document.metadata.items()
                if key not in _DISCARDED_KEYS
            }
            produced = 0
            for position, words in enumerate(self._segments(document.text)):
                if len(words) < self._minimum:
                    self.chunks_dropped_below_minimum += 1
                    continue
                text = " ".join(words)
                produced += 1
                yield Chunk(
                    chunk_id=make_chunk_id(
                        document.document_id, position, text
                    ),
                    document_id=document.document_id,
                    text=text,
                    position=position,
                    metadata=dict(inherited),
                )
            if produced == 0:
                self.documents_without_chunks += 1
                _logger.warning(
                    "document %s produced no chunks (below"
                    " minimum_tokens=%d)",
                    document.document_id,
                    self._minimum,
                )

    def _segments(self, text: str) -> Iterator[list[str]]:
        """Word lists for each chunk, in order.

        Positions are assigned to every segment produced here — also
        the below-minimum ones that are later dropped — so a chunk's
        position never depends on the minimum filter.
        """
        paragraphs = [
            words
            for block in _PARAGRAPH_BREAK.split(text)
            if (words := block.split())
        ]
        packed: list[str] = []
        for paragraph in paragraphs:
            if len(packed) + len(paragraph) <= self._target:
                packed.extend(paragraph)
                continue
            if packed:
                yield packed
                packed = []
            if len(paragraph) <= self._target:
                packed = list(paragraph)
                continue
            # oversized paragraph: overlapping word windows
            stride = self._target - self._overlap
            for start in range(0, len(paragraph), stride):
                window = paragraph[start : start + self._target]
                yield window
                if start + self._target >= len(paragraph):
                    break
        if packed:
            yield packed
