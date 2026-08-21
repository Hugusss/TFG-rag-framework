"""Partitioned store: P collections behind one VectorStore (spec 13.2).

The orchestrator ingests into "a vector store"; this adapter makes that
store P inner stores — one collection per logical partition — and
routes every chunk to the partition that owns its document
(``partition_for``, ADR-006). Ingest code does not change: partitioning
enters as an adapter of an existing seam, the same move that brought
the official OWI artifacts in (ADR-002).

Layout is part of the index identity: partition ``i`` of ``P`` lives in
collection ``{name}-p{i:02d}of{P:02d}`` and is stamped with
``partitions``/``partition_index`` metadata, so an index built for one
P can never be opened as another (spec section 22).

No concurrency here. :meth:`search_partition` is the unit of work a
collective retriever fans out through the executor seam;
:meth:`search` runs the partitions one after another and merges — the
"P partitions, one worker" path, correct by construction and useful
as a reference.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

from rag_framework.models import Chunk, SearchResult
from rag_framework.retrieval.merge import merge_top_k
from rag_framework.retrieval.partitioning import group_by_partition
from rag_framework.vectorstores.base import VectorStore, VectorStoreError


def partition_collection_name(name: str, index: int, partitions: int) -> str:
    return f"{name}-p{index:02d}of{partitions:02d}"


class PartitionedVectorStore(VectorStore):
    """One VectorStore made of ``partitions`` inner stores."""

    def __init__(
        self,
        store_factory: Callable[[dict], VectorStore],
        partitions: int,
    ) -> None:
        """``store_factory(extra_metadata)`` builds one inner store; the
        adapter passes the partition layout to stamp on its collection.
        Backend construction stays in the factory (Rule 2)."""
        if isinstance(partitions, bool) or not isinstance(partitions, int):
            raise VectorStoreError(
                f"partitions must be an int, got {type(partitions).__name__}"
            )
        if partitions < 1:
            raise VectorStoreError(
                f"partitions must be at least 1, got {partitions}"
            )
        self._partitions = partitions
        self._stores = [
            store_factory({"partitions": partitions, "partition_index": index})
            for index in range(partitions)
        ]
        self._name: str | None = None

    @property
    def partitions(self) -> int:
        return self._partitions

    def create_or_open(self, collection_name: str) -> None:
        for index, store in enumerate(self._stores):
            store.create_or_open(
                partition_collection_name(
                    collection_name, index, self._partitions
                )
            )
        self._name = collection_name

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise VectorStoreError(
                f"{len(chunks)} chunks but {len(embeddings)} embeddings"
            )
        vector_of = dict(zip((c.chunk_id for c in chunks), embeddings))
        if len(vector_of) != len(chunks):
            raise VectorStoreError(
                "duplicate chunk ids within one add() batch"
            )
        groups = group_by_partition(chunks, self._partitions)
        for store, group in zip(self._stores, groups):
            if group:
                store.add(group, [vector_of[c.chunk_id] for c in group])

    def search_partition(
        self,
        index: int,
        query_embedding: list[float],
        k: int,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        """Top-``k`` of one partition, each hit stamped with its
        ``partition_id`` (provenance, spec section 13.1 step 8)."""
        if not 0 <= index < self._partitions:
            raise VectorStoreError(
                f"partition index {index} out of range"
                f" [0, {self._partitions})"
            )
        return [
            dataclasses.replace(result, partition_id=str(index))
            for result in self._stores[index].search(query_embedding, k, filters)
        ]

    def search(
        self,
        query_embedding: list[float],
        k: int,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        partials = [
            self.search_partition(index, query_embedding, k, filters)
            for index in range(self._partitions)
        ]
        return merge_top_k(partials, k)

    def count(self) -> int:
        return sum(self.partition_counts())

    def partition_counts(self) -> list[int]:
        """Vectors per partition — the imbalance measure of spec 18 B."""
        return [store.count() for store in self._stores]

    def reset(self) -> None:
        for store in self._stores:
            store.reset()
