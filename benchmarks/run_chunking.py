"""Chunk-size experiment: how chunk boundaries change cost and quality.

Runs three self-chunked configurations (small/medium/large, recursive
strategy with live encoding) plus the publisher baseline (official
chunks and vectors) over the same corpus and evaluation queries, each
into its own store directory so index sizes stay comparable. Records
chunk count, index size, ingestion time, per-stage latency and
document-level quality; the medium run also measures the corpus
word-to-subword ratio behind the "token = whitespace word" policy.

Writes: results/chunkexp-<configuration>-<timestamp>.json.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_framework.config import (
    ChunkingConfig,
    DatasetConfig,
    EmbeddingConfig,
    GenerationConfig,
    MetricsConfig,
    PipelineConfig,
    RetrievalConfig,
    VectorStoreConfig,
)
from rag_framework.metrics.manifest import run_manifest, write_report
from rag_framework.metrics.quality import (
    document_ranking,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    summarize,
)
from rag_framework.orchestration.pipeline import RAGPipeline

_logger = logging.getLogger("benchmarks.run_chunking")

MODEL_ID = "jinaai/jina-embeddings-v5-text-small"
DATASET_VERSION = "owi-v2.0.0-gpu-spa-2026-07-28"
WARM_UP_QUERIES = 3

# name -> (target_tokens, overlap_tokens, minimum_tokens); minimum is
# overlap is a tenth of the target, the conventional 500 -> 50 ratio
RECURSIVE_CONFIGS = {
    "small": (200, 20, 20),
    "medium": (500, 50, 50),
    "large": (1000, 100, 100),
}


def pipeline_config(name: str, args) -> PipelineConfig:
    if name == "publisher":
        chunking = ChunkingConfig(strategy="publisher_offsets")
        embedding = EmbeddingConfig(provider="precomputed", model=MODEL_ID)
        collection = "owi-spa0728-publisherchunks-jinav5small-v1"
    else:
        target, overlap, minimum = RECURSIVE_CONFIGS[name]
        chunking = ChunkingConfig(
            strategy="recursive",
            target_tokens=target,
            overlap_tokens=overlap,
            minimum_tokens=minimum,
        )
        # batch size 8 measured ~2x faster than 32 on the experiment
        # machine (memory-bound CPU encode); recorded in the manifest
        embedding = EmbeddingConfig(
            provider="local", model=MODEL_ID, batch_size=8
        )
        collection = f"owi-spa0728-rec{target}o{overlap}-jinav5small-v1"
    return PipelineConfig(
        dataset=DatasetConfig(
            loader="owi", path=args.data, version=DATASET_VERSION
        ),
        chunking=chunking,
        embedding=embedding,
        vector_store=VectorStoreConfig(
            type="chroma",
            path=str(Path(args.state) / name),
            collection=collection,
        ),
        retrieval=RetrievalConfig(mode="sequential", k=args.k),
        generation=GenerationConfig(),
        metrics=MetricsConfig(output=args.output),
    )


def load_queries(path: str) -> list[dict]:
    lines = Path(path).read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(line) for line in lines]


def measure_queries(pipeline: RAGPipeline, queries: list[dict], args) -> dict:
    for query in queries[:WARM_UP_QUERIES]:
        pipeline.query(query["query"], retrieval_only=True)

    latencies: dict[str, list[float]] = {
        "embed_seconds": [],
        "search_seconds": [],
        "retrieval_seconds": [],
    }
    per_query: list[dict] = []
    unanswerable_top_scores: list[float] = []

    for repetition in range(args.repetitions):
        for query in queries:
            result = pipeline.query(query["query"], retrieval_only=True)
            for key in latencies:
                latencies[key].append(result.metrics[key])
            if repetition > 0:
                continue  # quality is deterministic; score once
            ranking = document_ranking(result.sources)
            relevant = set(query["relevant_document_ids"])
            entry = {
                "query_id": query["query_id"],
                "answerable": query["answerable"],
                "top_score": result.sources[0].score if result.sources else None,
            }
            if query["answerable"]:
                entry["recall_at_k"] = recall_at_k(ranking, relevant, args.k)
                entry["precision_at_k"] = precision_at_k(
                    ranking, relevant, args.k
                )
                entry["reciprocal_rank"] = reciprocal_rank(ranking, relevant)
            else:
                unanswerable_top_scores.append(entry["top_score"])
            per_query.append(entry)

    answerable = [e for e in per_query if e["answerable"]]
    return {
        "latency": {key: summarize(values) for key, values in latencies.items()},
        "quality": {
            "answerable_queries": len(answerable),
            "mean_recall_at_k": sum(e["recall_at_k"] for e in answerable)
            / len(answerable),
            "mean_precision_at_k": sum(e["precision_at_k"] for e in answerable)
            / len(answerable),
            "mean_reciprocal_rank": sum(e["reciprocal_rank"] for e in answerable)
            / len(answerable),
        },
        "unanswerable_top_scores": summarize(unanswerable_top_scores),
        "per_query": per_query,
    }


def word_subword_ratio(pipeline: RAGPipeline, sample_size: int = 300) -> dict:
    """Measured constant behind the token=word policy (ADR-008)."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    chunks = []
    for chunk in pipeline.chunker.split(
        pipeline.loader.load(pipeline.config.dataset.path)
    ):
        chunks.append(chunk.text)
        if len(chunks) >= sample_size:
            break
    words = sum(len(text.split()) for text in chunks)
    subwords = sum(
        len(tokenizer.encode(text, add_special_tokens=False))
        for text in chunks
    )
    return {
        "sample_chunks": len(chunks),
        "words": words,
        "subword_tokens": subwords,
        "subwords_per_word": round(subwords / words, 3),
    }


def run_configuration(name: str, queries: list[dict], args) -> None:
    _logger.info("=== configuration %s ===", name)
    config = pipeline_config(name, args)
    pipeline = RAGPipeline(config)
    if config.embedding.provider == "local":
        # resumable encodes: an interrupted run (the first medium sweep
        # was OOM-killed 3.2 h in) loses at most one 512-chunk call;
        # cache stats are reported so embedding_time_seconds is always
        # interpretable (hits make it a partial cost, not a raw cost)
        from rag_framework.embeddings.cache import CachedEmbeddingProvider

        pipeline.embedding_provider = CachedEmbeddingProvider(
            pipeline.embedding_provider,
            Path(args.state) / f"{name}-embedding-cache",
        )
    report = pipeline.ingest()
    _logger.info(
        "%s: %d chunks ingested in %.1f s",
        name,
        report.chunks_created,
        report.total_time_seconds,
    )

    measurements = measure_queries(pipeline, queries, args)
    payload = {
        "experiment": "chunk-size",
        "configuration": name,
        "k": args.k,
        "ingest_report": dataclasses.asdict(report),
        "ingest_details": {
            "pipeline_setup_seconds": pipeline.setup_seconds,
            "documents_without_chunks": pipeline.chunker.documents_without_chunks,
            "chunks_dropped_below_minimum": getattr(
                pipeline.chunker, "chunks_dropped_below_minimum", None
            ),
            "windows_clipped": getattr(
                pipeline.chunker, "windows_clipped", None
            ),
            "embedding_cache_hits": getattr(
                pipeline.embedding_provider, "hits", None
            ),
            "embedding_cache_misses": getattr(
                pipeline.embedding_provider, "misses", None
            ),
        },
        **measurements,
        "warm_up_policy": (
            f"{WARM_UP_QUERIES} unmeasured warm-up queries per"
            " configuration (the first also loads the query encoder)"
        ),
        "repetitions": args.repetitions,
        "manifest": run_manifest(config),
    }
    if name == "medium":
        payload["word_subword_ratio"] = word_subword_ratio(pipeline)
    path = write_report(payload, args.output, f"chunkexp-{name}")
    _logger.info("%s: report written to %s", name, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="./data/owi")
    parser.add_argument("--queries", default="./evaluation/queries.jsonl")
    parser.add_argument("--output", default="./results")
    parser.add_argument("--state", default="./state/chunkexp")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument(
        "--configurations",
        default="publisher,small,medium,large",
        help="comma-separated subset to run",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    names = args.configurations.split(",")
    for name in names:
        if name not in ("publisher", *RECURSIVE_CONFIGS):
            parser.error(f"unknown configuration {name!r}")

    if len(names) > 1:
        # one child process per configuration: hours of CPU encoding
        # fragment the torch arena without bound (a full multi-config
        # run was OOM-killed at 19 GB RSS), so memory must reset
        # between configurations by construction
        import subprocess

        for name in names:
            child = subprocess.run(
                [
                    sys.executable,
                    __file__,
                    "--data", args.data,
                    "--queries", args.queries,
                    "--output", args.output,
                    "--state", args.state,
                    "--k", str(args.k),
                    "--repetitions", str(args.repetitions),
                    "--configurations", name,
                ]
            )
            if child.returncode != 0:
                # an incomplete sweep must never look complete
                _logger.error(
                    "configuration %s failed with exit code %d; aborting"
                    " the remaining configurations",
                    name,
                    child.returncode,
                )
                return child.returncode
        _logger.info("all configurations completed")
        return 0

    queries = load_queries(args.queries)
    run_configuration(names[0], queries, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
