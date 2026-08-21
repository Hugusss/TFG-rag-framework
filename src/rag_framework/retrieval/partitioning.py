"""Deterministic partition assignment (spec section 13.3).

    partition_id = sha256(document_id) mod P

All chunks of a document land in the same partition, so a document is
never split across workers and the document-level ranking that the
evaluation judges (metrics/quality.py) is unaffected by partitioning.

The hash is ``models.stable_bucket`` — SHA-256 over the UTF-8 document
id, **not** Python's ``hash()``, which is salted per process for strings
and would assign documents differently on every run. A rebuild with the
same input and the same ``P`` reproduces the same assignment on any
machine (ADR-006); corpus sampling (ADR-011) uses the same function.
"""

from __future__ import annotations

from rag_framework.models import Chunk, stable_bucket
from rag_framework.retrieval.base import RetrievalError


def partition_for(document_id: str, partitions: int) -> int:
    """The partition index in ``[0, partitions)`` owning ``document_id``."""
    try:
        return stable_bucket(document_id, partitions)
    except TypeError as error:
        raise RetrievalError(
            f"partitions must be an int, got {type(partitions).__name__}"
        ) from error
    except ValueError as error:
        if "buckets" in str(error):
            raise RetrievalError(
                f"partitions must be at least 1, got {partitions}"
            ) from error
        raise RetrievalError("document_id must be a non-empty string") from error


def group_by_partition(chunks: list[Chunk], partitions: int) -> list[list[Chunk]]:
    """Split ``chunks`` into ``partitions`` lists, input order preserved
    within each; every chunk appears in exactly one list."""
    groups: list[list[Chunk]] = [[] for _ in range(partitions)]
    for chunk in chunks:
        groups[partition_for(chunk.document_id, partitions)].append(chunk)
    return groups
