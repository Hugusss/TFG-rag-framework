"""Partition- and worker-scaling experiments.

Partition scaling: the collective layouts P = 1, 2, 4, 8 (workers = P)
against the sequential baseline, same corpus. Worker scaling: one layout
searched by the serial executor and by 1, 2, 4, 8 threads. Both record
total latency, query-embedding time, fan-out wall time, merge time,
maximum and mean worker time, candidates returned, partition-size and
worker-time imbalance, and index size — after a stated warm-up and for a
stated number of repetitions. All configurations share the baseline's
query encoder, so every one sees the identical query vector.

Writes: results/scaling-partitions-<timestamp>.json and
results/scaling-workers-<timestamp>.json.
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

from rag_framework.config import PipelineConfig, load_config
from rag_framework.metrics.manifest import run_manifest, write_report
from rag_framework.metrics.quality import imbalance, summarize
from rag_framework.orchestration.pipeline import RAGPipeline, build_retriever

_logger = logging.getLogger("benchmarks.run_scaling")

TIMING_KEYS = (
    "total_seconds",
    "embed_seconds",
    "search_seconds",
    "merge_seconds",
    "worker_max_seconds",
    "worker_mean_seconds",
    "candidates_returned",
)


def directory_size(path: str) -> int:
    root = Path(path)
    if not root.exists():
        return 0
    return sum(f.stat().st_size for f in root.rglob("*") if f.is_file())


def measure(retriever, queries: list[str], k: int, warm_up: int, repetitions: int) -> dict:
    for query in queries[:warm_up]:
        retriever.retrieve(query, k)
    samples: dict[str, list[float]] = {key: [] for key in TIMING_KEYS}
    per_partition: dict[str, list[float]] = {}
    for _ in range(repetitions):
        for query in queries:
            retriever.retrieve(query, k)
            timings = retriever.timings
            for key in TIMING_KEYS:
                if key in timings:
                    samples[key].append(timings[key])
            for entry in getattr(retriever, "partition_searches", []):
                per_partition.setdefault(entry["partition_id"], []).append(entry["seconds"])
    result = {
        key: summarize(values) for key, values in samples.items() if values
    }
    if per_partition:
        mean_by_partition = {
            pid: sum(v) / len(v) for pid, v in sorted(per_partition.items(), key=lambda kv: int(kv[0]))
        }
        result["partition_mean_seconds"] = mean_by_partition
        result["worker_time_imbalance"] = imbalance(list(mean_by_partition.values()))
    return result


def describe(config: PipelineConfig, store) -> dict:
    info = {
        "mode": config.retrieval.mode,
        "partitions": config.retrieval.partitions,
        "workers": config.retrieval.workers,
        "executor": config.retrieval.executor if config.retrieval.mode == "collective" else None,
        "vectors": store.count(),
        "index_size_bytes": directory_size(config.vector_store.path),
    }
    if hasattr(store, "partition_counts"):
        counts = store.partition_counts()
        info["partition_counts"] = counts
        info["partition_size_imbalance"] = imbalance(counts)
    return info


def open_layout(path: str, encoder):
    config = load_config(path)
    pipeline = RAGPipeline(config)
    pipeline.vector_store.create_or_open(config.vector_store.collection)
    if pipeline.vector_store.count() == 0:
        raise SystemExit(f"{path}: collection is empty — run ingest first")
    return config, pipeline


def run_baseline(base: RAGPipeline, queries, args) -> dict:
    base.vector_store.create_or_open(base.config.vector_store.collection)
    base._collection_open = True
    row = {"label": "sequential", **describe(base.config, base.vector_store)}
    row["timings"] = measure(base.retriever, queries, args.k, args.warm_up, args.repetitions)
    _logger.info("sequential: search median %.2f ms", 1000 * row["timings"]["search_seconds"]["median"])
    return row


def run_partition_scaling(base: RAGPipeline, queries, args) -> list[dict]:
    rows = [run_baseline(base, queries, args)]
    for path in args.layouts:
        config, pipeline = open_layout(path, base.embedding_provider)
        retriever = build_retriever(config, base.embedding_provider, pipeline.vector_store)
        row = {"label": f"P={config.retrieval.partitions}", "config": path, **describe(config, pipeline.vector_store)}
        row["timings"] = measure(retriever, queries, args.k, args.warm_up, args.repetitions)
        retriever._executor.close()
        t = row["timings"]
        _logger.info(
            "%s (%s x%d): search median %.2f ms, worker max %.2f / mean %.2f ms, merge %.3f ms",
            row["label"], config.retrieval.executor, config.retrieval.workers,
            1000 * t["search_seconds"]["median"], 1000 * t["worker_max_seconds"]["median"],
            1000 * t["worker_mean_seconds"]["median"], 1000 * t["merge_seconds"]["median"],
        )
        rows.append(row)
    return rows


def run_worker_scaling(base: RAGPipeline, queries, args) -> list[dict]:
    config, pipeline = open_layout(args.worker_layout, base.embedding_provider)
    settings = [("serial", 1)] + [("threads", w) for w in args.workers]
    rows = []
    for executor, workers in settings:
        variant = dataclasses.replace(
            config,
            retrieval=dataclasses.replace(config.retrieval, executor=executor, workers=workers),
        )
        retriever = build_retriever(variant, base.embedding_provider, pipeline.vector_store)
        row = {"label": f"{executor} x{workers}", "config": args.worker_layout, **describe(variant, pipeline.vector_store)}
        row["timings"] = measure(retriever, queries, args.k, args.warm_up, args.repetitions)
        retriever._executor.close()
        t = row["timings"]
        _logger.info(
            "%s: search median %.2f ms, worker max %.2f / mean %.2f ms",
            row["label"], 1000 * t["search_seconds"]["median"],
            1000 * t["worker_max_seconds"]["median"], 1000 * t["worker_mean_seconds"]["median"],
        )
        rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("partitions", "workers", "both"), default="both")
    parser.add_argument("--baseline", default="configs/local_chroma.yaml")
    parser.add_argument(
        "--layouts", nargs="+",
        default=[f"configs/collective_{p}.yaml" for p in (1, 2, 4, 8)],
    )
    parser.add_argument("--worker-layout", default="configs/collective_8.yaml")
    parser.add_argument("--workers", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--queries", default="./evaluation/queries.jsonl")
    parser.add_argument("--output", default="./results")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warm-up", type=int, default=3)
    args = parser.parse_args(argv)
    if args.repetitions < 1 or args.warm_up < 0:
        parser.error("repetitions must be >= 1 and warm-up >= 0")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    queries = [
        json.loads(line)["query"]
        for line in Path(args.queries).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    base = RAGPipeline(load_config(args.baseline))
    common = {
        "k": args.k,
        "queries": len(queries),
        "repetitions": args.repetitions,
        "warm_up_policy": (
            f"{args.warm_up} unmeasured warm-up queries per configuration;"
            " the query encoder is shared and loaded before any measurement"
        ),
        "manifest": run_manifest(base.config),
    }
    if args.experiment in ("partitions", "both"):
        rows = run_partition_scaling(base, queries, args)
        path = write_report(
            {"experiment": "partition-scaling", "rows": rows, **common}, args.output, "scaling-partitions"
        )
        _logger.info("partition-scaling report written to %s", path)
    if args.experiment in ("workers", "both"):
        rows = run_worker_scaling(base, queries, args)
        path = write_report(
            {"experiment": "worker-scaling", "rows": rows, **common}, args.output, "scaling-workers"
        )
        _logger.info("worker-scaling report written to %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
