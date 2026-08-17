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
from rag_framework.embeddings.local import LocalEmbeddingProvider
from rag_framework.embeddings.precomputed import PrecomputedEmbeddingProvider
from rag_framework.generation.base import Generator
from rag_framework.generation.mock import MockGenerator
from rag_framework.loaders.base import DocumentLoader
from rag_framework.loaders.owi import OwiLoader
from rag_framework.models import Chunk, IngestionReport, RAGResult
from rag_framework.retrieval.base import Retriever
from rag_framework.retrieval.sequential import SequentialRetriever
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
    if config.embedding.normalize is False:
        # a config knob that silently does nothing would lie in every
        # result file: both implemented providers normalize, because
        # the corpus vectors and the reference recipe do
        raise ConfigError(
            "embedding.normalize: only true is supported — the official"
            " corpus vectors and the reference encoding recipe are"
            " normalized"
        )
    model_id = config.embedding.model
    batch_size = config.embedding.batch_size
    if config.embedding.provider == "precomputed":
        # queries are encoded live with the same model as the corpus
        # vectors; the delegate is built on first query, never at ingest
        return PrecomputedEmbeddingProvider(
            config.dataset.path,
            model_id=model_id,
            query_encoder_factory=lambda: LocalEmbeddingProvider(
                model_id, batch_size=batch_size
            ),
        )
    if config.embedding.provider == "local":
        return LocalEmbeddingProvider(model_id, batch_size=batch_size)
    raise ConfigError(
        f"embedding.provider: unknown provider"
        f" '{config.embedding.provider}' (known: precomputed, local)"
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


def build_generator(config: PipelineConfig) -> Generator:
    if config.generation.provider == "mock":
        return MockGenerator()
    # the real backend is deliberately deferred until the expected
    # model is agreed (an embedding model cannot generate; a small
    # CPU-runnable instruct model is the planned candidate)
    raise ConfigError(
        f"generation.provider: unknown provider"
        f" '{config.generation.provider}' (known: mock)"
    )


def build_retriever(
    config: PipelineConfig,
    provider: EmbeddingProvider,
    store: VectorStore,
) -> Retriever:
    if config.retrieval.mode == "sequential":
        return SequentialRetriever(provider, store)
    # config already guarantees the mode is a known name; collective
    # arrives with the Week-3 increments
    raise ConfigError(
        f"retrieval.mode: '{config.retrieval.mode}' is not implemented"
        " yet (implemented: sequential)"
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
        self.retriever = build_retriever(
            config, self.embedding_provider, self.vector_store
        )
        # built even when generation is disabled: an invalid provider
        # name must fail at construction, not on the first generating
        # query (fail-fast, ADR-004 boundary)
        self.generator = build_generator(config)
        self.setup_seconds = round(time.perf_counter() - setup_start, 3)
        self.vectors_before: int | None = None
        self._collection_open = False

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
        self._collection_open = True
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

    def query(
        self,
        question: str,
        k: int | None = None,
        *,
        retrieval_only: bool = False,
    ) -> RAGResult:
        """Retrieve — and, when enabled, generate — for ``question``.

        ``retrieval_only=True`` skips generation regardless of the
        configuration (spec section 14). ``k`` defaults to
        ``retrieval.k``; caller-supplied values are validated like the
        config would (a typo must not silently change an experiment).
        """
        generate = self.config.generation.enabled and not retrieval_only
        if k is None:
            k = self.config.retrieval.k
        elif k < 1:
            raise ConfigError(f"k: must be positive, got {k}")
        if not self._collection_open:
            self.vector_store.create_or_open(
                self.config.vector_store.collection
            )
            self._collection_open = True
        if self.vector_store.count() == 0:
            # querying an empty collection is never a valid experiment;
            # the classic cause is a typo'd collection name (which
            # create_or_open would otherwise silently satisfy with a
            # fresh empty collection and a success exit code)
            raise ConfigError(
                f"vector_store.collection: collection"
                f" '{self.config.vector_store.collection}' is empty —"
                " has ingest run, and is the name spelled exactly as"
                " ingested?"
            )

        total_start = time.perf_counter()
        sources = self.retriever.retrieve(question, k)
        answer = None
        generation_seconds = None
        if generate:
            start = time.perf_counter()
            # context assembly (spec section 14) is the identity step
            # here: the ranked SearchResults ARE the context; prompt
            # construction happens inside the generator (Rule 2)
            answer = self.generator.generate(question, sources)
            generation_seconds = round(time.perf_counter() - start, 6)
        # stamp the request total before any bookkeeping calls: count()
        # is cheap locally but becomes a network round trip on a remote
        # store, and it must never pollute the section-12 request total
        total_seconds = round(time.perf_counter() - total_start, 6)
        timings = self.retriever.timings

        def rounded(key):  # one rounding policy for every stage
            value = timings.get(key)
            return None if value is None else round(value, 6)

        metrics = {
            "k": k,
            "results_returned": len(sources),
            "collection_count": self.vector_store.count(),
            "embed_seconds": rounded("embed_seconds"),
            "search_seconds": rounded("search_seconds"),
            "retrieval_seconds": rounded("total_seconds"),
            # generation reported separately so it can never hide
            # retrieval behavior (spec section 12)
            "generation_seconds": generation_seconds,
            "total_seconds": total_seconds,
        }
        return RAGResult(
            query=question, answer=answer, sources=sources, metrics=metrics
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
