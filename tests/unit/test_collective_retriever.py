"""Unit tests for the collective retriever (stub store and provider)."""

import math

import pytest

from rag_framework.executors.local import SerialExecutor, ThreadExecutor
from rag_framework.models import SearchResult
from rag_framework.retrieval.base import RetrievalError
from rag_framework.retrieval.collective import CollectiveRetriever
from rag_framework.vectorstores.base import VectorStoreError
from rag_framework.vectorstores.partitioned import PartitionedVectorStore


def result(chunk_id, score, partition=None):
    return SearchResult(
        chunk_id=chunk_id,
        document_id="d",
        text="t",
        score=score,
        metadata={},
        partition_id=partition,
    )


class StubProvider:
    def __init__(self):
        self.queries = []

    def embed_query(self, text):
        self.queries.append(text)
        return [1.0, 0.0]


class StubPartitionedStore(PartitionedVectorStore):
    """Preset per-partition results; records every search_partition."""

    def __init__(self, per_partition, failing=()):
        self._partitions = len(per_partition)
        self._per_partition = per_partition
        self._failing = set(failing)
        self.calls = []

    def search_partition(self, index, query_embedding, k, filters=None):
        self.calls.append((index, list(query_embedding), k))
        if index in self._failing:
            raise VectorStoreError(f"partition {index} is broken")
        return [
            result(cid, score, partition=str(index))
            for cid, score in self._per_partition[index][:k]
        ]


PARTIALS = [
    [("a", 0.9), ("b", 0.4)],
    [],
    [("c", 0.8), ("d", 0.7), ("e", 0.1)],
    [("f", 0.95)],
]


@pytest.mark.parametrize(
    "executor", [SerialExecutor(), ThreadExecutor(2), ThreadExecutor(8)],
    ids=lambda e: f"{e.name}{e.workers}",
)
class TestRetrieve:
    def test_fans_out_once_per_partition_and_merges(self, executor):
        store = StubPartitionedStore(PARTIALS)
        retriever = CollectiveRetriever(StubProvider(), store, executor)

        results = retriever.retrieve("pregunta", k=3)

        assert sorted(c[0] for c in store.calls) == [0, 1, 2, 3]
        assert all(c[1] == [1.0, 0.0] and c[2] == 3 for c in store.calls)
        assert [r.chunk_id for r in results] == ["f", "a", "c"]
        executor.close()

    def test_provenance_carries_partition_and_worker(self, executor):
        store = StubPartitionedStore(PARTIALS)
        retriever = CollectiveRetriever(StubProvider(), store, executor)
        results = retriever.retrieve("q", k=10)
        assert [r.partition_id for r in results] == ["3", "0", "2", "2", "0", "2"]
        assert all(isinstance(r.worker_id, str) and r.worker_id for r in results)
        executor.close()

    def test_failed_partition_fails_the_query_loudly(self, executor):
        store = StubPartitionedStore(PARTIALS, failing={2})
        retriever = CollectiveRetriever(StubProvider(), store, executor)
        with pytest.raises(RetrievalError, match="partition 2 on worker") as info:
            retriever.retrieve("q", k=3)
        assert isinstance(info.value.__cause__, VectorStoreError)
        assert retriever.partition_searches == []  # no partial bookkeeping
        executor.close()

    def test_timings_and_per_partition_metrics(self, executor):
        store = StubPartitionedStore(PARTIALS)
        retriever = CollectiveRetriever(StubProvider(), store, executor)
        retriever.retrieve("q", k=10)
        t = retriever.timings
        assert set(t) >= {
            "embed_seconds", "search_seconds", "merge_seconds",
            "worker_max_seconds", "worker_mean_seconds",
            "candidates_returned", "total_seconds",
        }
        assert t["candidates_returned"] == 6
        assert t["worker_max_seconds"] >= t["worker_mean_seconds"] >= 0
        assert t["total_seconds"] >= t["embed_seconds"] + t["search_seconds"]
        assert [p["partition_id"] for p in retriever.partition_searches] == ["0", "1", "2", "3"]
        assert [p["candidates"] for p in retriever.partition_searches] == [2, 0, 3, 1]
        assert all(math.isfinite(p["seconds"]) for p in retriever.partition_searches)
        executor.close()

    def test_state_is_reset_per_call(self, executor):
        store = StubPartitionedStore(PARTIALS)
        retriever = CollectiveRetriever(StubProvider(), store, executor)
        retriever.retrieve("q", k=1)
        retriever.retrieve("q", k=1)
        assert len(retriever.partition_searches) == 4
        executor.close()
