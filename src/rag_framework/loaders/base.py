"""The DocumentLoader interface and its failure-handling contract."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path

from rag_framework.models import Document, Rejection

_logger = logging.getLogger(__name__)


class LoaderError(Exception):
    """A condition that invalidates the load as a whole.

    Missing or unpaired input files mean the corpus cannot be what the
    experiment claims, so loading must stop loudly instead of yielding a
    silently smaller corpus. Per-record problems are rejections, not
    errors.
    """


class DocumentLoader(ABC):
    """Yields canonical Documents from a source location.

    Contract for every implementation:

    - :meth:`load` is deterministic: the same source produces the same
      documents, ids, and order on every run.
    - Structural problems (missing files, unpaired shards, unreadable or
      wrong-schema input) raise :class:`LoaderError` when :meth:`load`
      is called; corruption discoverable only while decoding a shard
      raises :class:`LoaderError` from the iterator. Backend exceptions
      never escape raw.
    - Per-record problems never pass silently: each skipped
      record is logged and appended to ``rejections``; re-encountered
      document ids are counted in ``duplicates_skipped`` (first
      occurrence wins). Both attributes reset per :meth:`load` call and
      are complete only once the returned iterator is exhausted — one
      load at a time per instance; do not interleave two active
      :meth:`load` iterators of the same loader.
    """

    def __init__(self) -> None:
        self.rejections: list[Rejection] = []
        self.duplicates_skipped: int = 0

    @abstractmethod
    def load(self, source: str | Path) -> Iterator[Document]:
        """Yield Documents from ``source``."""

    def _reject(self, source: str, reason: str) -> None:
        self.rejections.append(Rejection(source=source, reason=reason))
        _logger.warning("rejected record %s: %s", source, reason)
