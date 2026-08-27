"""Dataset-size experiment: behaviour as the corpus grows.

Holds the retrieval configuration constant and grows the corpus through
deterministic, nested subsets (``dataset.sample_percent``, ADR-011).
Each subset gets a freshly built index in its own directory — index
sizes are only comparable between fresh builds, and the script never
deletes anything. Records construction time and size, median and p95
latency, throughput, and quality reported next to the fraction of
relevant documents the subset still contains.

Writes: results/scaling-dataset-<timestamp>.json.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_framework.config import load_config
from rag_framework.metrics.manifest import run_manifest, write_report
from rag_framework.metrics.quality import (
    document_ranking,
    recall_at_k,
    reciprocal_rank,
    summarize,
)
from rag_framework.models import stable_bucket
from rag_framework.orchestration.pipeline import RAGPipeline, build_retriever

_logger = logging.getLogger("benchmarks.run_dataset_scaling")


def fresh_directory(root: Path, run_id: str, name: str) -> Path:
    """A never-used store directory: ``root/<run_id>/<name>``. The
    script creates, it never deletes — old runs are the user's to
    remove (the backend caches clients per path, so deleting a
    directory under a live client corrupts the run)."""
    path = root / run_id / name
    if path.exists():
        raise SystemExit(f"{path} already exists; refusing to reuse an index")
    path.mkdir(parents=True)
    return path


def measure(retriever, queries: list[dict], k: int, warm_up: int, repetitions: int) -> dict:
    for query in queries[:warm_up]:
        retriever.retrieve(query["query"], k)
    totals: list[float] = []
    searches: list[float] = []
    per_query: list[dict] = []
    wall_start = time.perf_counter()
    for repetition in range(repetitions):
        for query in queries:
            results = retriever.retrieve(query["query"], k)
            totals.append(retriever.timings["total_seconds"])
            searches.append(retriever.timings["search_seconds"])
            if repetition == 0 and query["answerable"]:
                ranking = document_ranking(results)
                relevant = set(query["relevant_document_ids"])
                per_query.append(
                    {
                        "query_id": query["query_id"],
                        "recall_at_k": recall_at_k(ranking, relevant, k),
                        "reciprocal_rank": reciprocal_rank(ranking, relevant),
                    }
                )
    wall = time.perf_counter() - wall_start
    n = repetitions * len(queries)
    return {
        "latency": {
            "retrieval_seconds": summarize(totals),
            "search_seconds": summarize(searches),
        },
        "throughput": {
            "queries_per_second": n / wall,
            "search_only_queries_per_second": n / sum(searches),
            "measured_queries": n,
        },
        "quality": {
            "answerable_queries": len(per_query),
            "mean_recall_at_k": sum(q["recall_at_k"] for q in per_query) / len(per_query),
            "mean_reciprocal_rank": sum(q["reciprocal_rank"] for q in per_query) / len(per_query),
        },
        "per_query": per_query,
    }


def relevant_coverage(queries: list[dict], percent: int) -> dict:
    relevant = {d for q in queries if q["answerable"] for d in q["relevant_document_ids"]}
    kept = {d for d in relevant if stable_bucket(d, 100) < percent}
    return {"relevant_documents": len(relevant), "in_subset": len(kept), "fraction": len(kept) / len(relevant)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/local_chroma.yaml")
    parser.add_argument("--percents", nargs="+", type=int, default=[10, 25, 50, 100])
    parser.add_argument("--state", default="./state/dataset-scaling")
    parser.add_argument("--queries", default="./evaluation/queries.jsonl")
    parser.add_argument("--output", default="./results")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warm-up", type=int, default=3)
    args = parser.parse_args(argv)
    if any(not 1 <= p <= 100 for p in args.percents):
        parser.error("percents must be between 1 and 100")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    queries = [
        json.loads(line)
        for line in Path(args.queries).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    base_config = load_config(args.config)
    root = Path(args.state)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%f")
    encoder = None
    rows = []
    for percent in args.percents:
        store_path = fresh_directory(root, run_id, f"p{percent:03d}")
        config = dataclasses.replace(
            base_config,
            dataset=dataclasses.replace(base_config.dataset, sample_percent=percent),
            vector_store=dataclasses.replace(base_config.vector_store, path=str(store_path)),
        )
        pipeline = RAGPipeline(config)
        report = pipeline.ingest()
        if encoder is None:
            encoder = pipeline.embedding_provider  # one model load for all subsets
        retriever = build_retriever(config, encoder, pipeline.vector_store)
        row = {
            "percent": percent,
            "documents": report.documents_read,
            "documents_sampled_out": getattr(pipeline.loader, "documents_sampled_out", 0),
            "chunks": report.chunks_created,
            "vectors": report.final_vector_count,
            "index_construction_seconds": report.total_time_seconds,
            "indexing_seconds": report.indexing_time_seconds,
            "embedding_seconds": report.embedding_time_seconds,
            "index_size_bytes": report.index_size_bytes,
            "relevant_coverage": relevant_coverage(queries, percent),
            **measure(retriever, queries, args.k, args.warm_up, args.repetitions),
        }
        if hasattr(retriever, "_executor"):
            retriever._executor.close()
        _logger.info(
            "%3d%%: %d docs, %d vectors, built in %.1f s (%.0f MB); search median %.2f ms;"
            " %.1f q/s; recall@%d %.3f (relevant docs in subset %.0f%%)",
            percent, row["documents"], row["vectors"], row["index_construction_seconds"],
            row["index_size_bytes"] / 1e6, 1000 * row["latency"]["search_seconds"]["median"],
            row["throughput"]["queries_per_second"], args.k, row["quality"]["mean_recall_at_k"],
            100 * row["relevant_coverage"]["fraction"],
        )
        rows.append(row)

    payload = {
        "experiment": "dataset-size-scaling",
        "k": args.k,
        "queries": len(queries),
        "repetitions": args.repetitions,
        "warm_up_policy": (
            f"{args.warm_up} unmeasured warm-up queries per subset; the query"
            " encoder is loaded once and shared by every subset"
        ),
        "index_policy": f"every subset is built fresh in its own directory under {root / run_id}",
        "rows": rows,
        "manifest": run_manifest(base_config),
    }
    path = write_report(payload, args.output, "scaling-dataset")
    _logger.info("report written to %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
