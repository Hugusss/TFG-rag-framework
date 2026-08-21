"""Deterministic partition assignment (spec section 13.3).

    partition_id = sha256(document_id) mod P

All chunks of a document land in the same partition, so a document is
never split across workers and the document-level ranking that the
evaluation judges (metrics/quality.py) is unaffected by partitioning.

The hash is SHA-256 over the UTF-8 document id — **not** Python's
``hash()``, which is salted per process for strings and would assign
documents differently on every run. A rebuild with the same input and
the same ``P`` reproduces the same assignment on any machine (ADR-006).
"""

from __future__ import annotations

import hashlib

from rag_framework.models import Chunk
from rag_framework.retrieval.base import RetrievalError


def partition_for(document_id: str, partitions: int) -> int:
    """The partition index in ``[0, partitions)`` owning ``document_id``."""
    if isinstance(partitions, bool) or not isinstance(partitions, int):
        raise RetrievalError(
            f"partitions must be an int, got {type(partitions).__name__}"
        )
    if partitions < 1:
        raise RetrievalError(f"partitions must be at least 1, got {partitions}")
    if not document_id:
        raise RetrievalError("document_id must be a non-empty string")
    digest = hashlib.sha256(document_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % partitions


def group_by_partition(chunks: list[Chunk], partitions: int) -> list[list[Chunk]]:
    """Split ``chunks`` into ``partitions`` lists, input order preserved
    within each; every chunk appears in exactly one list."""
    groups: list[list[Chunk]] = [[] for _ in range(partitions)]
    for chunk in chunks:
        groups[partition_for(chunk.document_id, partitions)].append(chunk)
    return groups
