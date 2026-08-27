"""Partitioned store: P collections behind one VectorStore.

Holds P inner stores, one collection per logical partition, and routes
every chunk to the partition that owns its document (``partition_for``,
ADR-006). Ingest code does not change: partitioning arrives as an
adapter of an existing seam. :meth:`search_partition` exposes one
partition's top-k as the unit of work a collective retriever fans out;
:meth:`search` walks the partitions in turn and merges.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterator

from rag_framework.models import Chunk, SearchResult
from rag_framework.retrieval.merge import merge_top_k
from rag_framework.retrieval.partitioning import group_by_partition
from rag_framework.vectorstores.base import VectorStore, VectorStoreError


def partition_collection_name(name: str, index: int, partitions: int) -> str:
    # P is in the name, and the pair (partitions, partition_index) is
    # stamped on the collection metadata: together they make an index
    # built for one P impossible to open as another, whether by a
    # renamed config or by a partition opened in the wrong slot
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
        Backend construction stays in the factory, so this class holds no
        knowledge of which store it is composing."""
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
        """Top-``k`` of one partition, each hit stamped with the
        ``partition_id`` it came from."""
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
        # one partition after another, never in parallel: concurrency is
        # the executor seam's business, so this stays the "P partitions,
        # one worker" reference every threaded run is compared against
        partials = [
            self.search_partition(index, query_embedding, k, filters)
            for index in range(self._partitions)
        ]
        return merge_top_k(partials, k)

    def count(self) -> int:
        return sum(self.partition_counts())

    def partition_counts(self) -> list[int]:
        """Vectors per partition — how evenly the hash spread the corpus."""
        return [store.count() for store in self._stores]

    def reset(self) -> None:
        for store in self._stores:
            store.reset()

    def iter_vectors(self) -> Iterator[tuple[str, str, list[float]]]:
        for store in self._stores:
            yield from store.iter_vectors()
