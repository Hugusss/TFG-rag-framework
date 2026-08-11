"""Canonical data model shared by every pipeline component.

The rest of the pipeline depends on these types, never on a raw dataset
format (assignment spec, section 7). All types are frozen dataclasses:
components exchange values, they do not mutate shared state. Freezing is
shallow — the ``metadata`` dictionaries themselves remain mutable, so
components must treat received metadata as read-only.

Identifier scheme (spec section 7, "Stable identifiers"):

- ``document_id``: the publisher's stable identifier when the source
  provides one (loaders decide; the OWI ``record_id`` is adopted as-is),
  otherwise ``make_document_id(source_identifier)``.
- ``chunk_id``: always ``make_chunk_id(document_id, position, text)``.

Both functions are pure and deterministic, so re-ingesting unchanged input
produces identical ids — the property deduplication relies on.

Score convention: ``SearchResult.score`` is a similarity, HIGHER IS BETTER.
Vector-store adapters convert their native distance metrics before
returning results.
"""

from __future__ import annotations

import hashlib
import unicodedata
from dataclasses import dataclass


def _normalize(text: str) -> str:
    """Unicode NFC, collapse whitespace runs to single spaces, strip ends."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def make_document_id(source_identifier: str) -> str:
    """Deterministic document id for sources without a stable publisher id.

    Exact scheme: ``sha256(normalize(source_identifier))`` hex digest,
    where ``normalize`` is Unicode NFC plus whitespace collapsing. Changing
    this scheme invalidates every stored collection; it is pinned by a
    known-value unit test.

    Raises ``ValueError`` for identifiers that are empty after
    normalization: they carry no identity and would all collide.
    """
    normalized = _normalize(source_identifier)
    if not normalized:
        raise ValueError("source_identifier is empty after normalization")
    return hashlib.sha256(normalized.encode()).hexdigest()


def make_chunk_id(document_id: str, position: int, text: str) -> str:
    """Deterministic chunk id, stable across re-ingestion.

    Exact scheme: ``sha256(payload)`` hex digest with the injective
    encoding ``payload = str(len(document_id)) + ":" + document_id + ":"
    + str(position) + ":" + normalize(text)``, where ``normalize`` is the
    same as in :func:`make_document_id`. The length prefix makes the
    encoding unambiguous even when a document id itself contains ``":"``
    — without it, two different (document_id, position, text) triples
    could produce the same payload and therefore the same id, and the
    vector store's duplicate-id protection would silently drop a real
    chunk. Including the position keeps ids distinct when a document
    repeats the same text in two places; including the text makes an id
    change when the underlying content changes.
    """
    payload = f"{len(document_id)}:{document_id}:{position}:{_normalize(text)}"
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class Document:
    """A normalized source document, independent of any dataset format."""

    document_id: str
    text: str
    metadata: dict


@dataclass(frozen=True, slots=True)
class Chunk:
    """A retrievable piece of one document; keeps its provenance.

    ``position`` is the 0-based index of the chunk within its document.
    """

    chunk_id: str
    document_id: str
    text: str
    position: int
    metadata: dict


@dataclass(frozen=True, slots=True)
class SearchResult:
    """One retrieval hit.

    ``score`` is a similarity — higher is better, whatever the backend's
    native metric. ``worker_id`` and ``partition_id`` carry provenance in
    collective mode and stay ``None`` in sequential mode.
    """

    chunk_id: str
    document_id: str
    text: str
    score: float
    metadata: dict
    worker_id: str | None = None
    partition_id: str | None = None


@dataclass(frozen=True, slots=True)
class RAGResult:
    """Answer plus the evidence it rests on and per-stage metrics.

    ``answer`` is ``None`` in retrieval-only mode.
    """

    query: str
    answer: str | None
    sources: list[SearchResult]
    metrics: dict


@dataclass(frozen=True, slots=True)
class IngestionReport:
    """Counters and timings every ingestion run must produce.

    All fields are required so an ingestion cannot silently omit one
    (spec sections 8 and 17.1).
    """

    documents_read: int
    documents_rejected: int
    duplicates_skipped: int
    chunks_created: int
    bytes_processed: int
    embedding_time_seconds: float
    indexing_time_seconds: float
    total_time_seconds: float
    final_vector_count: int
    index_size_bytes: int
