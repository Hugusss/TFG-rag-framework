"""Unit tests for the orchestration factories (name resolution)."""

from types import SimpleNamespace

import pytest

from rag_framework.chunking.publisher_offsets import PublisherOffsetsChunker
from rag_framework.config import (
    ChunkingConfig,
    ConfigError,
    DatasetConfig,
    EmbeddingConfig,
    GenerationConfig,
    MetricsConfig,
    PipelineConfig,
    RetrievalConfig,
    VectorStoreConfig,
)
from rag_framework.loaders.owi import OwiLoader
from rag_framework.orchestration.pipeline import (
    build_chunker,
    build_embedding_provider,
    build_loader,
    build_vector_store,
)
from rag_framework.vectorstores.chroma import ChromaVectorStore


def config(**overrides) -> PipelineConfig:
    values = {
        "dataset": DatasetConfig(loader="owi", path="./data/owi"),
        "chunking": ChunkingConfig(strategy="publisher_offsets"),
        "embedding": EmbeddingConfig(provider="precomputed", model="m"),
        "vector_store": VectorStoreConfig(
            type="chroma", path="./state/chroma", collection="col"
        ),
        "retrieval": RetrievalConfig(mode="sequential"),
        "generation": GenerationConfig(),
        "metrics": MetricsConfig(output="./results"),
    }
    values.update(overrides)
    return PipelineConfig(**values)


FAKE_PROVIDER = SimpleNamespace(model_id="m", dimension=4, normalized=True)


class TestFactories:
    def test_known_names_build_the_right_types(self, tmp_path):
        cfg = config(
            vector_store=VectorStoreConfig(
                type="chroma", path=str(tmp_path), collection="col"
            )
        )
        assert isinstance(build_loader(cfg), OwiLoader)
        assert isinstance(build_chunker(cfg), PublisherOffsetsChunker)
        assert isinstance(
            build_vector_store(cfg, FAKE_PROVIDER), ChromaVectorStore
        )

    def test_unknown_loader_lists_known(self):
        cfg = config(dataset=DatasetConfig(loader="warc", path="p"))
        with pytest.raises(ConfigError, match="known: owi"):
            build_loader(cfg)

    def test_recursive_strategy_builds_with_config_parameters(self):
        from rag_framework.chunking.recursive import RecursiveChunker

        cfg = config(
            chunking=ChunkingConfig(
                strategy="recursive",
                target_tokens=500,
                overlap_tokens=50,
                minimum_tokens=50,
            )
        )
        chunker = build_chunker(cfg)
        assert isinstance(chunker, RecursiveChunker)
        assert chunker._target == 500 and chunker._overlap == 50

    def test_recursive_with_precomputed_provider_refused_at_build(self, tmp_path):
        from rag_framework.orchestration.pipeline import RAGPipeline

        cfg = config(
            chunking=ChunkingConfig(
                strategy="recursive",
                target_tokens=500,
                overlap_tokens=50,
                minimum_tokens=50,
            )
        )
        with pytest.raises(ConfigError, match="cannot be combined"):
            RAGPipeline(cfg)

    def test_unknown_embedding_provider(self):
        cfg = config(embedding=EmbeddingConfig(provider="openai", model="m"))
        with pytest.raises(ConfigError, match="known: precomputed, local"):
            build_embedding_provider(cfg)

    def test_local_provider_builds_without_loading_a_model(self):
        from rag_framework.embeddings.local import LocalEmbeddingProvider

        cfg = config(embedding=EmbeddingConfig(provider="local", model="m"))
        provider = build_embedding_provider(cfg)
        assert isinstance(provider, LocalEmbeddingProvider)
        assert provider.model_id == "m"

    def test_normalize_false_is_rejected_loudly(self):
        cfg = config(
            embedding=EmbeddingConfig(provider="local", model="m", normalize=False)
        )
        with pytest.raises(ConfigError, match="normalize"):
            build_embedding_provider(cfg)

    def test_precomputed_branch_wires_a_matching_query_delegate(self, monkeypatch):
        # the one guarantee that config.model reaches BOTH halves of the
        # split provider — captured without parquet or a real model
        from rag_framework.embeddings.local import LocalEmbeddingProvider
        from rag_framework.orchestration import pipeline as pipeline_module

        captured = {}

        class Recorder:
            def __init__(self, source, *, model_id, query_encoder_factory):
                captured["model_id"] = model_id
                captured["factory"] = query_encoder_factory

        monkeypatch.setattr(
            pipeline_module, "PrecomputedEmbeddingProvider", Recorder
        )
        cfg = config(
            embedding=EmbeddingConfig(
                provider="precomputed", model="m", batch_size=64
            )
        )
        build_embedding_provider(cfg)
        assert captured["model_id"] == "m"
        delegate = captured["factory"]()
        assert isinstance(delegate, LocalEmbeddingProvider)
        assert delegate.model_id == "m"
        assert delegate.batch_size == 64

    def test_unknown_store_type(self):
        cfg = config(
            vector_store=VectorStoreConfig(
                type="blocksdb", path="p", collection="c"
            )
        )
        with pytest.raises(ConfigError, match="known: chroma"):
            build_vector_store(cfg, FAKE_PROVIDER)

    def test_sequential_mode_builds_sequential_retriever(self, tmp_path):
        from types import SimpleNamespace

        from rag_framework.orchestration.pipeline import build_retriever
        from rag_framework.retrieval.sequential import SequentialRetriever

        cfg = config()
        retriever = build_retriever(cfg, FAKE_PROVIDER, SimpleNamespace())
        assert isinstance(retriever, SequentialRetriever)

    def test_collective_mode_not_implemented_yet(self):
        from types import SimpleNamespace

        from rag_framework.orchestration.pipeline import build_retriever

        cfg = config(
            retrieval=RetrievalConfig(mode="collective", partitions=4, workers=4)
        )
        with pytest.raises(ConfigError, match="not implemented"):
            build_retriever(cfg, FAKE_PROVIDER, SimpleNamespace())

    def test_mock_generator_builds(self):
        from rag_framework.generation.mock import MockGenerator
        from rag_framework.orchestration.pipeline import build_generator

        assert isinstance(build_generator(config()), MockGenerator)

    def test_unknown_generation_provider(self):
        from rag_framework.orchestration.pipeline import build_generator

        cfg = config(generation=GenerationConfig(enabled=True, provider="openai"))
        with pytest.raises(ConfigError, match="known: mock"):
            build_generator(cfg)

    def test_store_is_stamped_with_provider_identity(self, tmp_path):
        cfg = config(
            vector_store=VectorStoreConfig(
                type="chroma", path=str(tmp_path), collection="col"
            )
        )
        store = build_vector_store(cfg, FAKE_PROVIDER)
        assert store._metadata["model_id"] == "m"
        assert store._metadata["dimension"] == 4
