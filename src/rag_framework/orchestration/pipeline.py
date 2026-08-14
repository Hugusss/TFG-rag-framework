"""RAGPipeline: configuration-driven wiring of the component seams.

Factories resolve component *names* to implementations — the validation
boundary defined with the config seam: `config.py` checks structure,
factories check that names are registered, components check runtime
facts. An unknown name raises :class:`ConfigError` listing what exists,
so a typo'd config fails at build time, before any work runs.

`ingest()` streams documents through loader → chunker → embedding
provider → vector store in bounded batches, accumulates per-stage
timings, and produces the spec-section-8 :class:`IngestionReport` from
the counters the seams themselves expose. The orchestrator adds no
data logic of its own — wiring and timing only.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from rag_framework import __version__
from rag_framework.chunking.base import Chunker
from rag_framework.chunking.publisher_offsets import PublisherOffsetsChunker
from rag_framework.config import ConfigError, PipelineConfig, load_config
from rag_framework.embeddings.base import EmbeddingProvider
from rag_framework.embeddings.precomputed import PrecomputedEmbeddingProvider
from rag_framework.loaders.base import DocumentLoader
from rag_framework.loaders.owi import OwiLoader
from rag_framework.models import Chunk, IngestionReport
from rag_framework.vectorstores.base import VectorStore
from rag_framework.vectorstores.chroma import ChromaVectorStore

_logger = logging.getLogger(__name__)

_BATCH_SIZE = 512  # chunks per embed/store round trip; bounds memory


def build_loader(config: PipelineConfig) -> DocumentLoader:
    if config.dataset.loader == "owi":
        return OwiLoader()
    raise ConfigError(
        f"dataset.loader: unknown loader '{config.dataset.loader}'"
        " (known: owi)"
    )


def build_chunker(config: PipelineConfig) -> Chunker:
    if config.chunking.strategy == "publisher_offsets":
        return PublisherOffsetsChunker()
    # config already guarantees the strategy is a known name; recursive
    # is scheduled for the Week-2 chunk-size experiments
    raise ConfigError(
        f"chunking.strategy: '{config.chunking.strategy}' is not"
        " implemented yet (implemented: publisher_offsets)"
    )


def build_embedding_provider(config: PipelineConfig) -> EmbeddingProvider:
    if config.embedding.provider == "precomputed":
        return PrecomputedEmbeddingProvider(
            config.dataset.path, model_id=config.embedding.model
        )
    raise ConfigError(
        f"embedding.provider: unknown provider"
        f" '{config.embedding.provider}' (known: precomputed)"
    )


def build_vector_store(
    config: PipelineConfig, provider: EmbeddingProvider
) -> VectorStore:
    if config.vector_store.type == "chroma":
        return ChromaVectorStore(
            config.vector_store.path,
            collection_metadata={
                "model_id": provider.model_id,
                "dimension": provider.dimension,
                "normalized": provider.normalized,
                "ingestion_version": __version__,
            },
        )
    raise ConfigError(
        f"vector_store.type: unknown type '{config.vector_store.type}'"
        " (known: chroma)"
    )


class RAGPipeline:
    """The application-level pipeline, built entirely from config."""

    def __init__(self, config: PipelineConfig) -> None:
        # component construction is real ingestion work (the precomputed
        # provider reads and indexes the embeddings parquet here), so it
        # is timed and included in the report's total
        setup_start = time.perf_counter()
        self.config = config
        self.loader = build_loader(config)
        self.chunker = build_chunker(config)
        self.embedding_provider = build_embedding_provider(config)
        self.vector_store = build_vector_store(config, self.embedding_provider)
        self.setup_seconds = round(time.perf_counter() - setup_start, 3)
        self.vectors_before: int | None = None

    @classmethod
    def from_config(cls, path: str | Path) -> "RAGPipeline":
        return cls(load_config(path))

    def ingest(self) -> IngestionReport:
        """Ingest the configured corpus; safe to run twice.

        The corpus location is always ``dataset.path`` from the config:
        the embedding provider was built from that same path, and
        loading a different corpus against it could silently attach
        wrong official vectors (overlapping ids, different text).
        ``total_time_seconds`` includes component setup.
        """
        source = Path(self.config.dataset.path)
        total_start = time.perf_counter()

        self.vector_store.create_or_open(self.config.vector_store.collection)
        # makes idempotence auditable from the report payload alone
        self.vectors_before = self.vector_store.count()

        bytes_processed = 0
        chunks_created = 0
        embedding_seconds = 0.0
        indexing_seconds = 0.0

        def counted(documents):
            nonlocal bytes_processed
            for document in documents:
                bytes_processed += len(document.text.encode("utf-8"))
                yield document

        batch: list[Chunk] = []

        def flush() -> None:
            nonlocal chunks_created, embedding_seconds, indexing_seconds
            if not batch:
                return
            start = time.perf_counter()
            vectors = self.embedding_provider.embed_documents(batch)
            embedding_seconds += time.perf_counter() - start
            start = time.perf_counter()
            self.vector_store.add(batch, vectors)
            indexing_seconds += time.perf_counter() - start
            chunks_created += len(batch)
            batch.clear()

        for chunk in self.chunker.split(counted(self.loader.load(source))):
            batch.append(chunk)
            if len(batch) >= _BATCH_SIZE:
                flush()
        flush()

        final_vector_count = self.vector_store.count()
        report = IngestionReport(
            documents_read=(
                self.chunker.documents_processed
                + len(self.loader.rejections)
                + self.loader.duplicates_skipped
            ),
            documents_rejected=(
                len(self.loader.rejections) + self.chunker.documents_rejected
            ),
            duplicates_skipped=self.loader.duplicates_skipped,
            chunks_created=chunks_created,
            bytes_processed=bytes_processed,
            embedding_time_seconds=round(embedding_seconds, 3),
            indexing_time_seconds=round(indexing_seconds, 3),
            total_time_seconds=round(
                self.setup_seconds + time.perf_counter() - total_start, 3
            ),
            final_vector_count=final_vector_count,
            index_size_bytes=_directory_size(self.config.vector_store.path),
        )
        _logger.info(
            "ingest complete: %d documents read, %d rejected, %d duplicates,"
            " %d chunks, %d vectors stored",
            report.documents_read,
            report.documents_rejected,
            report.duplicates_skipped,
            report.chunks_created,
            report.final_vector_count,
        )
        if final_vector_count != chunks_created:
            # legitimate on re-ingest into an existing collection, but
            # always worth a visible note (Rule 6)
            _logger.warning(
                "stored vector count (%d) differs from chunks processed"
                " this run (%d): pre-existing collection content?",
                final_vector_count,
                chunks_created,
            )
        return report

    def query(self, question: str, k: int = 10):
        raise NotImplementedError(
            "retrieval lands with the Week-2 increments; the pipeline is"
            " ingest-only for now"
        )


def _directory_size(path: str | Path) -> int:
    """Total size in bytes of the persistent index directory.

    Deliberate simplification: assumes a local-path store. When a
    remote store lands (BlocksDB, ADR-007), index size must become a
    VectorStore method — flagged so this deferral does not silently
    become permanent.
    """
    root = Path(path)
    if not root.exists():
        return 0
    return sum(f.stat().st_size for f in root.rglob("*") if f.is_file())
