"""Canonical data model shared by every pipeline component.

Components depend on these types, never on a raw dataset format, and
exchange them as values: all are frozen dataclasses. Freezing is shallow,
so received ``metadata`` dictionaries must be treated as read-only.
Defined here too: the deterministic id functions every seam relies on for
idempotent re-ingestion, and the convention that ``SearchResult.score``
is always a similarity — higher is better, whatever the backend returns.
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
class Rejection:
    """One skipped input record: where it came from and why.

    Loaders and chunkers append these instead of dropping data quietly,
    and ingestion reports count them, so "handled" never turns into
    "hidden".
    """

    source: str
    reason: str


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

    All fields are required, so an ingestion cannot omit a counter by
    forgetting to set it.
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


def stable_bucket(key: str, buckets: int) -> int:
    """Deterministic bucket of ``key`` in ``[0, buckets)``: the first
    8 bytes of SHA-256 over the UTF-8 key, big-endian, modulo
    ``buckets``. Stable across processes, machines and Python versions
    (unlike ``hash()``, which is salted per process for strings); used
    for partition assignment and for corpus sampling, so both agree on
    what a "stable" assignment means (ADR-006, ADR-011).

    Cost: one hash of the key, independent of ``buckets`` and of how
    many keys have been bucketed before."""
    if isinstance(buckets, bool) or not isinstance(buckets, int):
        raise TypeError(f"buckets must be an int, got {type(buckets).__name__}")
    if buckets < 1:
        raise ValueError(f"buckets must be at least 1, got {buckets}")
    if not key:
        raise ValueError("key must be a non-empty string")
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % buckets
