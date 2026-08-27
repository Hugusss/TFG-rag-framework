"""Collective retrieval: partitioned local search plus a global merge.

    query -> query embedding -> broadcast to P partition searches
          -> (executor runs them on W workers) -> merge_top_k -> top-k

This retriever owns the query flow and its timing only: layout belongs
to the PartitionedVectorStore, concurrency to the Executor, reduction
to merge_top_k. Workers are local, so this simulates a distributed
search rather than being one.
"""

from __future__ import annotations

import dataclasses
import time

from rag_framework.embeddings.base import EmbeddingProvider
from rag_framework.executors.base import Executor
from rag_framework.models import SearchResult
from rag_framework.retrieval.base import RetrievalError, Retriever
from rag_framework.retrieval.merge import merge_top_k
from rag_framework.vectorstores.partitioned import PartitionedVectorStore


class CollectiveRetriever(Retriever):
    """Fan-out over the partitions of a PartitionedVectorStore."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        store: PartitionedVectorStore,
        executor: Executor,
    ) -> None:
        super().__init__()
        self._provider = embedding_provider
        self._store = store
        self._executor = executor
        # per-partition detail of the last call — worker times, candidate
        # counts, imbalance — emitted on the normal query path so scaling
        # runs need no separate instrumentation; reset with timings
        self.partition_searches: list[dict] = []

    def retrieve(self, query: str, k: int) -> list[SearchResult]:
        self.timings = {}
        self.partition_searches = []
        total_start = time.perf_counter()

        start = time.perf_counter()
        query_vector = self._provider.embed_query(query)
        self.timings["embed_seconds"] = time.perf_counter() - start

        store = self._store
        start = time.perf_counter()
        outcomes = self._executor.map(
            lambda index: store.search_partition(index, query_vector, k),
            range(store.partitions),
        )
        self.timings["search_seconds"] = time.perf_counter() - start

        # the executor reports failures as data; turning them into a failed
        # query is this layer's decision. Answering from the partitions that
        # did succeed would look like a complete result over a silently
        # smaller corpus, so one lost partition loses the query.
        failed = [outcome for outcome in outcomes if not outcome.ok]
        if failed:
            first = failed[0]
            raise RetrievalError(
                f"{len(failed)} of {store.partitions} partition searches"
                f" failed; first: partition {first.index} on worker"
                f" {first.worker_id}: {first.error!r}"
            ) from first.error

        partials = [
            [
                dataclasses.replace(result, worker_id=outcome.worker_id)
                for result in outcome.result
            ]
            for outcome in outcomes
        ]
        start = time.perf_counter()
        merged = merge_top_k(partials, k)
        self.timings["merge_seconds"] = time.perf_counter() - start

        self.partition_searches = [
            {
                "partition_id": str(outcome.index),
                "worker_id": outcome.worker_id,
                "seconds": outcome.seconds,
                "candidates": len(outcome.result),
            }
            for outcome in outcomes
        ]
        worker_seconds = [outcome.seconds for outcome in outcomes]
        self.timings["worker_max_seconds"] = max(worker_seconds)
        self.timings["worker_mean_seconds"] = sum(worker_seconds) / len(
            worker_seconds
        )
        self.timings["candidates_returned"] = sum(
            len(outcome.result) for outcome in outcomes
        )
        self.timings["total_seconds"] = time.perf_counter() - total_start
        return merged
