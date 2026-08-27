"""Deterministic corpus sampling, as a decorator of any loader (ADR-011).

Keeps a document iff ``stable_bucket(document_id, 100) < percent``. The
subsets are nested — every document of the 10 % subset is also in the
25 % subset — and independent of file order and of the machine, so
"50 % of the corpus" names the same documents on every run — dataset
size is only a variable if each size is reproducible. Dropped documents
are counted in ``documents_sampled_out`` and never reach the chunker.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from rag_framework.loaders.base import DocumentLoader
from rag_framework.models import Document, stable_bucket


class SampledLoader(DocumentLoader):
    """Keeps a deterministic ``percent`` of the inner loader's documents."""

    def __init__(self, inner: DocumentLoader, percent: int) -> None:
        if isinstance(percent, bool) or not isinstance(percent, int):
            raise TypeError(f"percent must be an int, got {type(percent).__name__}")
        if not 1 <= percent <= 100:
            raise ValueError(f"percent must be between 1 and 100, got {percent}")
        self._inner = inner
        self.percent = percent
        self.documents_sampled_out = 0

    @property
    def rejections(self):  # type: ignore[override]
        return self._inner.rejections

    @property
    def duplicates_skipped(self):  # type: ignore[override]
        return self._inner.duplicates_skipped

    def load(self, source: str | Path) -> Iterator[Document]:
        self.documents_sampled_out = 0
        for document in self._inner.load(source):
            if stable_bucket(document.document_id, 100) < self.percent:
                yield document
            else:
                self.documents_sampled_out += 1
