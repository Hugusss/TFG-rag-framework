"""Which partition owns a document, and how to group chunks by it.

    partition_id = sha256(document_id) mod P

All chunks of a document land in the same partition, so a document's
evidence is never split across workers. The hash is
``models.stable_bucket``, not Python's ``hash()``: rebuilding an index
with the same input and the same P must reproduce it exactly (ADR-006).
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
