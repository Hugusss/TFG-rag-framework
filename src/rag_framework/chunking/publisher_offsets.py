"""Chunker strategy that reuses the publisher's official chunk windows.

The OWI dataset ships its own chunk boundaries alongside its official
embeddings, and they enter through the Chunker seam like any other
strategy instead of as a branch in the orchestrator (ADR-002), so
nothing downstream can tell publisher chunks from computed ones. Windows
come from ``metadata["chunk_offsets"]``; malformed or out-of-range ones
are rejected with a reason, and slight overhangs are clipped and counted.
"""

from __future__ import annotations

import logging
import operator
from collections.abc import Iterable, Iterator

from rag_framework.chunking.base import Chunker
from rag_framework.models import Chunk, Document, make_chunk_id

_logger = logging.getLogger(__name__)

# About a third of real windows end a few characters past the text end,
# always by the same small amount — a publisher-side artifact, not
# corruption. This is a validity threshold measured from the corpus, not
# a knob an experiment varies, so it lives in code rather than config.
MAX_OVERHANG = 10  # characters

# chunker-owned metadata keys; document-level values are discarded
_RESERVED_KEYS = ("chunk_offsets", "start", "end", "clipped")


class PublisherOffsetsChunker(Chunker):
    """Cuts chunks along the ``metadata["chunk_offsets"]`` windows."""

    def __init__(self) -> None:
        super().__init__()
        self.windows_clipped = 0

    def split(self, documents: Iterable[Document]) -> Iterator[Chunk]:
        self.documents_processed = 0
        self.documents_without_chunks = 0
        self.documents_rejected = 0
        self.windows_clipped = 0
        self.rejections = []
        return self._iter_chunks(documents)

    def _iter_chunks(self, documents: Iterable[Document]) -> Iterator[Chunk]:
        for document in documents:
            self.documents_processed += 1
            offsets = document.metadata.get("chunk_offsets")
            if offsets is None:
                self.documents_rejected += 1
                self._reject(
                    document.document_id,
                    "document has no chunk_offsets metadata; this corpus"
                    " cannot use the publisher_offsets strategy",
                )
                continue

            length = len(document.text)
            inherited = {
                key: value
                for key, value in document.metadata.items()
                if key not in _RESERVED_KEYS
            }
            produced = 0
            # position is the publisher's own window index, not a running
            # count of accepted windows: the precomputed provider looks
            # vectors up by (document_id, position), so a rejected window
            # must not renumber the ones after it
            for position, window in enumerate(offsets):
                source = f"{document.document_id}#window{position}"
                try:
                    # operator.index accepts only true integers — a
                    # float or string bound is malformed, never truncated
                    start = operator.index(window[0])
                    end = operator.index(window[1])
                except (TypeError, ValueError, KeyError, IndexError):
                    self._reject(source, f"malformed window {window!r}")
                    continue
                if start < 0 or start >= length:
                    self._reject(
                        source,
                        f"window start {start} outside text of length {length}",
                    )
                    continue
                if end <= start:
                    self._reject(source, f"degenerate window ({start}, {end})")
                    continue

                clipped = False
                if end > length:
                    overhang = end - length
                    if overhang > MAX_OVERHANG:
                        self._reject(
                            source,
                            f"window end {end} overhangs text of length"
                            f" {length} by {overhang} chars"
                            f" (limit {MAX_OVERHANG})",
                        )
                        continue
                    end = length
                    clipped = True
                    self.windows_clipped += 1

                # sliced verbatim: the official embeddings were computed
                # over exactly these substrings, so cleaning or
                # normalizing here would break vector alignment. Nor is
                # the content filtered — a whitespace-only window still
                # has an official vector, so dropping it would leave a
                # hole in the (document_id, position) key space.
                text = document.text[start:end]
                metadata = dict(inherited)
                metadata["start"] = start
                metadata["end"] = end
                if clipped:
                    metadata["clipped"] = True

                produced += 1
                yield Chunk(
                    chunk_id=make_chunk_id(document.document_id, position, text),
                    document_id=document.document_id,
                    text=text,
                    position=position,
                    metadata=metadata,
                )

            if produced == 0:
                self.documents_without_chunks += 1
                _logger.warning(
                    "document %s produced no chunks", document.document_id
                )
