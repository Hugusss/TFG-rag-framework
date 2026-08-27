"""Unit tests for the local embedding provider (stub encoder, no model)."""

import pytest

from rag_framework.embeddings.base import EmbeddingError
from rag_framework.embeddings.local import LocalEmbeddingProvider
from rag_framework.models import Chunk


class StubEncoder:
    """Records every encode call; returns index-tagged 3-d vectors."""

    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        return [[float(i), 1.0, 0.0] for i, _ in enumerate(texts)]


def chunk(text, position=0):
    return Chunk(
        chunk_id=f"c{position}",
        document_id="d",
        text=text,
        position=position,
        metadata={},
    )


def make_provider():
    stub = StubEncoder()
    loads = []

    def factory(model_id):
        loads.append(model_id)
        return stub

    provider = LocalEmbeddingProvider("test-model", encoder_factory=factory)
    return provider, stub, loads


class TestLaziness:
    def test_construction_never_loads_the_model(self):
        _, _, loads = make_provider()
        assert loads == []

    def test_model_loads_once_across_calls(self):
        provider, _, loads = make_provider()
        provider.embed_query("a")
        provider.embed_query("b")
        provider.embed_documents([chunk("c")])
        assert loads == ["test-model"]


class TestRecipe:
    def test_exact_reference_recipe_is_pinned(self):
        # drift here silently changes the embedding space (ADR-009)
        provider, stub, _ = make_provider()
        provider.embed_query("una consulta")
        (texts, kwargs), = stub.calls
        assert texts == ["una consulta"]
        assert kwargs["task"] == "retrieval"
        assert kwargs["normalize_embeddings"] is True
        assert kwargs["batch_size"] == 32  # the documented default

    def test_custom_batch_size_forwarded(self):
        stub = StubEncoder()
        provider = LocalEmbeddingProvider(
            "m", batch_size=64, encoder_factory=lambda model_id: stub
        )
        provider.embed_query("q")
        assert stub.calls[0][1]["batch_size"] == 64
        assert provider.batch_size == 64  # recorded in the run manifest

    def test_loader_half_of_the_recipe_is_pinned(self, monkeypatch):
        # the constructor half — trust_remote_code and device — must be
        # as unforgeable as the encode half
        import sys
        import types

        captured = {}

        class FakeSentenceTransformer:
            def __init__(self, model_id, **kwargs):
                captured["model_id"] = model_id
                captured.update(kwargs)

        fake = types.ModuleType("sentence_transformers")
        fake.SentenceTransformer = FakeSentenceTransformer
        monkeypatch.setitem(sys.modules, "sentence_transformers", fake)

        from rag_framework.embeddings.local import _load_sentence_transformer

        _load_sentence_transformer("some-model")
        assert captured == {
            "model_id": "some-model",
            "trust_remote_code": True,
            "device": "cpu",
        }

    def test_documents_encode_chunk_texts_in_order(self):
        provider, stub, _ = make_provider()
        vectors = provider.embed_documents([chunk("uno"), chunk("dos", 1)])
        assert stub.calls[0][0] == ["uno", "dos"]
        assert vectors == [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]

    def test_empty_documents_is_a_no_op(self):
        provider, _, loads = make_provider()
        assert provider.embed_documents([]) == []
        assert loads == []


class TestContract:
    def test_dimension_probes_and_caches(self):
        provider, stub, _ = make_provider()
        assert provider.dimension == 3
        assert provider.dimension == 3
        assert len(stub.calls) == 1  # one probe, then cached

    def test_identity_is_machine_readable(self):
        provider, _, _ = make_provider()
        assert provider.model_id == "test-model"
        assert provider.normalized is True
        assert provider.device == "cpu"
        assert provider.batch_size == 32
        assert provider.revision is None  # resolved only after a load

    def test_model_load_failure_is_an_embedding_error(self):
        def broken_factory(model_id):
            raise RuntimeError("no network")

        provider = LocalEmbeddingProvider("m", encoder_factory=broken_factory)
        with pytest.raises(EmbeddingError, match="could not load"):
            provider.embed_query("q")

    def test_encode_failure_is_an_embedding_error(self):
        class Broken:
            def encode(self, texts, **kwargs):
                raise RuntimeError("boom")

        provider = LocalEmbeddingProvider("m", encoder_factory=lambda m: Broken())
        with pytest.raises(EmbeddingError, match="encoding failed"):
            provider.embed_query("q")

    def test_inconsistent_dimensions_rejected(self):
        class Ragged:
            def encode(self, texts, **kwargs):
                return [[1.0, 0.0], [1.0, 0.0, 0.0]][: len(texts)]

        provider = LocalEmbeddingProvider("m", encoder_factory=lambda m: Ragged())
        with pytest.raises(EmbeddingError, match="inconsistent"):
            provider.embed_documents([chunk("a"), chunk("b", 1)])

    def test_wrong_vector_count_rejected(self):
        class Short:
            def encode(self, texts, **kwargs):
                return [[1.0, 0.0]]  # always one vector

        provider = LocalEmbeddingProvider("m", encoder_factory=lambda m: Short())
        with pytest.raises(EmbeddingError, match="1 vectors for 2 texts"):
            provider.embed_documents([chunk("a"), chunk("b", 1)])
