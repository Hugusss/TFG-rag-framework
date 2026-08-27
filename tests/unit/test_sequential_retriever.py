"""Unit tests for the sequential retriever (stub provider and store)."""

import pytest

from rag_framework.embeddings.base import EmbeddingError
from rag_framework.models import SearchResult
from rag_framework.retrieval.sequential import SequentialRetriever


def result(chunk_id, score):
    return SearchResult(
        chunk_id=chunk_id,
        document_id="d",
        text="t",
        score=score,
        metadata={},
    )


class StubProvider:
    def __init__(self, vector=(1.0, 0.0)):
        self.vector = list(vector)
        self.queries = []

    def embed_query(self, text):
        self.queries.append(text)
        return self.vector


class StubStore:
    def __init__(self, results=()):
        self.results = list(results)
        self.calls = []

    def search(self, query_embedding, k, filters=None):
        self.calls.append((list(query_embedding), k, filters))
        return self.results[:k]


class TestRetrieve:
    def test_embeds_then_searches_and_passes_results_through(self):
        provider = StubProvider(vector=(0.1, 0.9))
        store = StubStore([result("a", 0.9), result("b", 0.5)])
        retriever = SequentialRetriever(provider, store)

        results = retriever.retrieve("una consulta", k=2)

        assert provider.queries == ["una consulta"]
        assert store.calls == [([0.1, 0.9], 2, None)]
        assert [r.chunk_id for r in results] == ["a", "b"]

    def test_k_caps_results(self):
        store = StubStore([result("a", 0.9), result("b", 0.5)])
        retriever = SequentialRetriever(StubProvider(), store)
        assert len(retriever.retrieve("q", k=1)) == 1

    def test_empty_store_gives_empty_results(self):
        retriever = SequentialRetriever(StubProvider(), StubStore())
        assert retriever.retrieve("q", k=5) == []


class TestTimings:
    def test_all_stage_timings_recorded(self):
        retriever = SequentialRetriever(StubProvider(), StubStore())
        retriever.retrieve("q", k=5)
        timings = retriever.timings
        assert set(timings) == {"embed_seconds", "search_seconds", "total_seconds"}
        assert all(value >= 0 for value in timings.values())
        assert timings["total_seconds"] >= timings["embed_seconds"]

    def test_timings_are_recomputed_not_stale(self):
        import time as time_module

        class SlowOnceProvider(StubProvider):
            def __init__(self):
                super().__init__()
                self.first = True

            def embed_query(self, text):
                if self.first:
                    self.first = False
                    time_module.sleep(0.05)
                return super().embed_query(text)

        retriever = SequentialRetriever(SlowOnceProvider(), StubStore())
        retriever.retrieve("q1", k=5)
        slow = retriever.timings["embed_seconds"]
        retriever.retrieve("q2", k=5)
        fast = retriever.timings["embed_seconds"]
        assert slow >= 0.05
        assert fast < slow  # values recomputed, not carried over

    def test_provider_failure_propagates_untranslated(self):
        class BrokenProvider:
            def embed_query(self, text):
                raise EmbeddingError("no encoder")

        retriever = SequentialRetriever(BrokenProvider(), StubStore())
        with pytest.raises(EmbeddingError, match="no encoder"):
            retriever.retrieve("q", k=5)

    def test_store_failure_propagates_untranslated(self):
        from rag_framework.vectorstores.base import VectorStoreError

        class BrokenStore:
            def search(self, query_embedding, k, filters=None):
                raise VectorStoreError("index gone")

        retriever = SequentialRetriever(StubProvider(), BrokenStore())
        with pytest.raises(VectorStoreError, match="index gone"):
            retriever.retrieve("q", k=5)
