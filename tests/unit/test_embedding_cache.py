"""Unit tests for the disk embedding cache."""

import pytest

from rag_framework.embeddings.base import EmbeddingError
from rag_framework.embeddings.cache import CachedEmbeddingProvider
from rag_framework.models import Chunk


class StubProvider:
    model_id = "stub-model"
    normalized = True
    device = "stub"
    dimension = 2

    def __init__(self):
        self.encoded: list[str] = []

    def embed_documents(self, chunks):
        self.encoded.extend(c.chunk_id for c in chunks)
        return [[float(len(c.text)), 1.0] for c in chunks]

    def embed_query(self, text):
        return [9.0, 9.0]


def chunk(cid, text="texto"):
    return Chunk(chunk_id=cid, document_id="d", text=text, position=0, metadata={})


class TestCache:
    def test_miss_encodes_then_hit_skips_inner(self, tmp_path):
        inner = StubProvider()
        cache = CachedEmbeddingProvider(inner, tmp_path / "cache")
        first = cache.embed_documents([chunk("a"), chunk("b", "xx")])
        assert inner.encoded == ["a", "b"]
        second = cache.embed_documents([chunk("a"), chunk("b", "xx")])
        assert inner.encoded == ["a", "b"]  # no re-encode
        assert first == second
        assert cache.hits == 2 and cache.misses == 2

    def test_persists_across_instances(self, tmp_path):
        inner1 = StubProvider()
        CachedEmbeddingProvider(inner1, tmp_path / "cache").embed_documents(
            [chunk("a")]
        )
        inner2 = StubProvider()
        cache2 = CachedEmbeddingProvider(inner2, tmp_path / "cache")
        cache2.embed_documents([chunk("a")])
        assert inner2.encoded == []  # resumed from disk

    def test_mixed_hits_and_misses_preserve_order(self, tmp_path):
        inner = StubProvider()
        cache = CachedEmbeddingProvider(inner, tmp_path / "cache")
        cache.embed_documents([chunk("a", "x")])
        vectors = cache.embed_documents(
            [chunk("b", "yy"), chunk("a", "x"), chunk("c", "zzz")]
        )
        assert vectors == [[2.0, 1.0], [1.0, 1.0], [3.0, 1.0]]
        assert inner.encoded == ["a", "b", "c"]

    def test_wrong_model_refused(self, tmp_path):
        CachedEmbeddingProvider(StubProvider(), tmp_path / "cache")

        class OtherModel(StubProvider):
            model_id = "another-model"

        with pytest.raises(EmbeddingError, match="refusing to mix"):
            CachedEmbeddingProvider(OtherModel(), tmp_path / "cache")

    def test_identity_delegates(self, tmp_path):
        cache = CachedEmbeddingProvider(StubProvider(), tmp_path / "cache")
        assert cache.model_id == "stub-model"
        assert cache.normalized is True
        assert cache.dimension == 2
        assert cache.embed_query("q") == [9.0, 9.0]
