"""Chunker strategy that reuses the publisher's official chunk windows.

This is half of the project's central design move (ADR-002): the OWI
dataset ships its own chunk boundaries (and embeddings), and they enter
the pipeline as an ordinary strategy of the Chunker seam — never as a
special branch in the orchestrator. Downstream components cannot tell
publisher chunks from computed ones.

Two properties are load-bearing:

- ``Chunk.position`` equals the publisher's window index (``chunk_idx``
  in the official embeddings parquet). The precomputed embedding
  provider will look vectors up by ``(document_id, position)``, so a
  rejected window never renumbers its successors.
- Chunk text is byte-exact ``text[start:end]`` — no cleaning, no
  normalization. The official embeddings were computed over exactly
  these strings, so any change to the text breaks vector alignment;
  non-cosmetic changes also change the chunk id (the id scheme itself
  normalizes whitespace and unicode composition before hashing).

Overhang policy (measured on the real corpus, recorded in ADR-002):
about a third of real windows end up to 10 characters past the text end
— a known artifact of the publisher pipeline. Window ends overhanging by
at most :data:`MAX_OVERHANG` are clipped to the text end and counted in
``windows_clipped`` (chunk metadata gains ``clipped: True``); anything
larger is rejected as corruption, not quirk. ``MAX_OVERHANG`` is
deliberately code, not configuration: the spec's rule that chunking
parameters live in config covers values an experimenter may vary, and
this is a validity threshold measured from the corpus — if it needs
changing, the corpus itself changed.

Publisher windows are never content-filtered: even a whitespace-only
window is kept, because the publisher delimited it and its official
embedding exists. The spec's "avoid empty chunks" guidance applies to
strategies that choose their own boundaries. Document-level metadata
keys ``start``, ``end``, and ``clipped`` are reserved for the chunker
and discarded if present on a document.
"""

from __future__ import annotations

import logging
import operator
from collections.abc import Iterable, Iterator

from rag_framework.chunking.base import Chunker
from rag_framework.models import Chunk, Document, make_chunk_id

_logger = logging.getLogger(__name__)

MAX_OVERHANG = 10  # characters; the measured corpus bound

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
