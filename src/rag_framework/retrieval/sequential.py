"""Sequential retrieval: search one collection, return the top-k.

    query -> query embedding -> one search over the complete local
    collection -> top-k

Deliberately minimal: this is the reference every collective
configuration is compared against, so it must stay simple enough to be
obviously correct by reading it.
"""

from __future__ import annotations

import time

from rag_framework.embeddings.base import EmbeddingProvider
from rag_framework.models import SearchResult
from rag_framework.retrieval.base import Retriever
from rag_framework.vectorstores.base import VectorStore


class SequentialRetriever(Retriever):
    """Full-collection top-k search through the seam interfaces."""

    def __init__(
        self, embedding_provider: EmbeddingProvider, vector_store: VectorStore) -> None:
        super().__init__()
        self._provider = embedding_provider
        self._store = vector_store

    def retrieve(self, query: str, k: int) -> list[SearchResult]:
        self.timings = {}
        total_start = time.perf_counter()

        start = time.perf_counter()
        query_vector = self._provider.embed_query(query)
        self.timings["embed_seconds"] = time.perf_counter() - start

        start = time.perf_counter()
        results = self._store.search(query_vector, k)
        self.timings["search_seconds"] = time.perf_counter() - start

        self.timings["total_seconds"] = time.perf_counter() - total_start
        return results
