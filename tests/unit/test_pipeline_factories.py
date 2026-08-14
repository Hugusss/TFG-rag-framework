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

    def test_recursive_strategy_not_implemented_yet(self):
        cfg = config(
            chunking=ChunkingConfig(
                strategy="recursive",
                target_tokens=500,
                overlap_tokens=50,
                minimum_tokens=50,
            )
        )
        with pytest.raises(ConfigError, match="not.*implemented"):
            build_chunker(cfg)

    def test_unknown_embedding_provider(self):
        cfg = config(embedding=EmbeddingConfig(provider="local", model="m"))
        with pytest.raises(ConfigError, match="known: precomputed"):
            build_embedding_provider(cfg)

    def test_unknown_store_type(self):
        cfg = config(
            vector_store=VectorStoreConfig(
                type="blocksdb", path="p", collection="c"
            )
        )
        with pytest.raises(ConfigError, match="known: chroma"):
            build_vector_store(cfg, FAKE_PROVIDER)

    def test_store_is_stamped_with_provider_identity(self, tmp_path):
        cfg = config(
            vector_store=VectorStoreConfig(
                type="chroma", path=str(tmp_path), collection="col"
            )
        )
        store = build_vector_store(cfg, FAKE_PROVIDER)
        assert store._metadata["model_id"] == "m"
        assert store._metadata["dimension"] == 4
