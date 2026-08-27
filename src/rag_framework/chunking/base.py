"""The Chunker interface and its accounting contract."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator

from rag_framework.models import Chunk, Document, Rejection

_logger = logging.getLogger(__name__)


class Chunker(ABC):
    """Splits Documents into Chunks under one configured strategy.

    Contract for every implementation:

    - :meth:`split` is deterministic and streaming: the same documents
      produce the same chunks, ids, and order on every run, without
      materializing the corpus in memory.
    - Chunks never silently disappear: every input document
      counts in ``documents_processed``; a document whose windows
      produce no chunks counts in ``documents_without_chunks`` and is
      logged; an individual window that cannot be produced is appended
      to ``rejections`` with a reason. A document that cannot
      participate in the strategy at all (e.g. missing the metadata the
      strategy requires) is rejected as a whole — one ``Rejection``
      whose source is the document id, counted in
      ``documents_rejected`` — and is *not* counted in
      ``documents_without_chunks``. Counters reset per :meth:`split`
      call and are complete once the returned iterator is exhausted —
      one active split per instance at a time.
    - Strategies never mix: whether publisher boundaries or computed
      ones are used is decided by configuration, once, for the whole
      corpus — never per document.
    """

    def __init__(self) -> None:
        self.documents_processed = 0
        self.documents_without_chunks = 0
        self.documents_rejected = 0
        self.rejections: list[Rejection] = []

    @abstractmethod
    def split(self, documents: Iterable[Document]) -> Iterator[Chunk]:
        """Yield Chunks for ``documents``."""

    def _reject(self, source: str, reason: str) -> None:
        # source may be a window ref or a whole document id: keep the
        # noun neutral so the audit trail never mislabels
        self.rejections.append(Rejection(source=source, reason=reason))
        _logger.warning("rejected %s: %s", source, reason)
