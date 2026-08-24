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

    def test_sampled_dataset_wraps_the_loader(self):
        from rag_framework.loaders.sample import SampledLoader

        cfg = config(dataset=DatasetConfig(loader="owi", path="./data/owi", sample_percent=25))
        loader = build_loader(cfg)
        assert isinstance(loader, SampledLoader) and loader.percent == 25
        assert isinstance(loader._inner, OwiLoader)
        assert isinstance(build_loader(config()), OwiLoader)  # 100 % = plain

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
        cfg = config(embedding=EmbeddingConfig(provider="remote-api", model="m"))
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

    def test_collective_mode_builds_collective_retriever(self, tmp_path):
        from rag_framework.executors.local import SerialExecutor, ThreadExecutor
        from rag_framework.orchestration.pipeline import build_retriever
        from rag_framework.retrieval.collective import CollectiveRetriever

        cfg = config(
            vector_store=VectorStoreConfig(
                type="chroma", path=str(tmp_path), collection="col"
            ),
            retrieval=RetrievalConfig(mode="collective", partitions=4, workers=3),
        )
        store = build_vector_store(cfg, FAKE_PROVIDER)
        retriever = build_retriever(cfg, FAKE_PROVIDER, store)
        assert isinstance(retriever, CollectiveRetriever)
        assert isinstance(retriever._executor, ThreadExecutor)
        assert retriever._executor.workers == 3

        serial = config(
            vector_store=cfg.vector_store,
            retrieval=RetrievalConfig(
                mode="collective", partitions=2, workers=1, executor="serial"
            ),
        )
        retriever = build_retriever(serial, FAKE_PROVIDER, build_vector_store(serial, FAKE_PROVIDER))
        assert isinstance(retriever._executor, SerialExecutor)

    def test_collective_mode_refuses_a_plain_store(self):
        from types import SimpleNamespace

        from rag_framework.orchestration.pipeline import build_retriever

        cfg = config(
            retrieval=RetrievalConfig(mode="collective", partitions=4, workers=4)
        )
        with pytest.raises(ConfigError, match="partitioned store"):
            build_retriever(cfg, FAKE_PROVIDER, SimpleNamespace())

    def test_mock_generator_builds(self):
        from rag_framework.generation.mock import MockGenerator
        from rag_framework.orchestration.pipeline import build_generator

        assert isinstance(build_generator(config()), MockGenerator)

    def test_ollama_generator_builds_with_config_values(self):
        from rag_framework.generation.ollama import OllamaGenerator
        from rag_framework.orchestration.pipeline import build_generator

        cfg = config(generation=GenerationConfig(
            enabled=True, provider="ollama", model="ministral-3",
            endpoint="http://localhost:11434", timeout_seconds=30,
        ))
        generator = build_generator(cfg)
        assert isinstance(generator, OllamaGenerator)
        assert generator.model == "ministral-3"
        assert generator.timeout_seconds == 30

    def test_unknown_generation_provider(self):
        from rag_framework.orchestration.pipeline import build_generator

        cfg = config(generation=GenerationConfig(enabled=True, provider="remote-api"))
        with pytest.raises(ConfigError, match="known: mock"):
            build_generator(cfg)

    def test_collective_mode_builds_partitioned_store(self, tmp_path):
        from rag_framework.vectorstores.partitioned import PartitionedVectorStore

        cfg = config(
            vector_store=VectorStoreConfig(
                type="chroma", path=str(tmp_path), collection="col"
            ),
            retrieval=RetrievalConfig(mode="collective", partitions=4, workers=2),
        )
        store = build_vector_store(cfg, FAKE_PROVIDER)
        assert isinstance(store, PartitionedVectorStore)
        assert store.partitions == 4
        inner = store._stores[3]
        assert isinstance(inner, ChromaVectorStore)
        assert inner._metadata["model_id"] == "m"
        assert inner._metadata["partitions"] == 4
        assert inner._metadata["partition_index"] == 3

    def test_store_is_stamped_with_provider_identity(self, tmp_path):
        cfg = config(
            vector_store=VectorStoreConfig(
                type="chroma", path=str(tmp_path), collection="col"
            )
        )
        store = build_vector_store(cfg, FAKE_PROVIDER)
        assert store._metadata["model_id"] == "m"
        assert store._metadata["dimension"] == 4
