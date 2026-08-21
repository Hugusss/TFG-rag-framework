"""Collective-correctness experiment (spec sections 13.5 and 18, Experiment E).

For every evaluation query, compares three answers over the same
vectors, embedding model and distance metric:

- the **exact** top-k (brute-force cosine over every stored vector,
  metrics/exact.py) — the reference that makes differences attributable;
- the **baseline**: sequential search over the full collection;
- the **collective** top-k for each partition count given.

Per query and layout it records overlap, positions preserved,
discordant pairs, score differences, recall relative to the baseline
and to the exact reference, and a classification of every difference
(tie reorder / HNSW miss in the baseline / miss in a partition /
outside the exact top-k). One raw result file through the metrics
layer; plots come later from that file, never from this script.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_framework.config import load_config
from rag_framework.metrics.exact import exact_top_k
from rag_framework.metrics.manifest import run_manifest, write_report
from rag_framework.metrics.quality import (
    classify_differences,
    discordant_pairs,
    max_score_difference,
    overlap_at_k,
    positions_preserved,
    recall_at_k,
)
from rag_framework.orchestration.pipeline import RAGPipeline, build_retriever

_logger = logging.getLogger("benchmarks.run_correctness")


def ids(results):
    return [r.chunk_id for r in results]


def scores(results):
    return {r.chunk_id: r.score for r in results}


def compare(candidate, reference, k: int) -> dict:
    cand, ref = ids(candidate), ids(reference)
    return {
        "overlap_at_k": overlap_at_k(cand, ref, k),
        "positions_preserved": positions_preserved(cand, ref, k),
        "identical_set": set(cand) == set(ref),
        "identical_order": cand == ref,
        "discordant_pairs": len(discordant_pairs(cand, ref)),
        "max_score_difference": max_score_difference(
            scores(candidate), scores(reference)
        ),
        "recall": recall_at_k(cand, set(ref), k) if ref else None,
    }


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    numeric = ("overlap_at_k", "positions_preserved", "recall", "discordant_pairs")
    summary = {
        key: sum(r[key] for r in rows) / n for key in numeric if rows and all(r[key] is not None for r in rows)
    }
    summary["identical_set"] = sum(r["identical_set"] for r in rows)
    summary["identical_order"] = sum(r["identical_order"] for r in rows)
    diffs = [r["max_score_difference"] for r in rows if r["max_score_difference"] is not None]
    summary["max_score_difference"] = max(diffs) if diffs else None
    summary["queries"] = n
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="configs/local_chroma.yaml")
    parser.add_argument(
        "--collective",
        nargs="+",
        default=["configs/collective_2.yaml", "configs/collective_4.yaml", "configs/collective_8.yaml"],
    )
    parser.add_argument("--queries", default="./evaluation/queries.jsonl")
    parser.add_argument("--output", default="./results")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    k = args.k
    queries = [
        json.loads(line)
        for line in Path(args.queries).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    base_config = load_config(args.baseline)
    base = RAGPipeline(base_config)
    base.vector_store.create_or_open(base_config.vector_store.collection)
    base._collection_open = True
    start = time.perf_counter()
    rows = list(base.vector_store.iter_vectors())
    export_seconds = time.perf_counter() - start
    _logger.info("exported %d vectors in %.1f s", len(rows), export_seconds)

    layouts = []
    for path in args.collective:
        config = load_config(path)
        pipeline = RAGPipeline(config)
        pipeline.vector_store.create_or_open(config.vector_store.collection)
        if pipeline.vector_store.count() != len(rows):
            raise SystemExit(
                f"{path}: {pipeline.vector_store.count()} vectors but the"
                f" baseline holds {len(rows)} — not the same index content"
            )
        # share the baseline's query encoder: same vector for all three
        # answers by construction, one model load
        retriever = build_retriever(config, base.embedding_provider, pipeline.vector_store)
        layouts.append((path, config, pipeline, retriever))

    baseline_vs_exact = []
    per_layout = {path: [] for path, *_ in layouts}
    classifications = {path: [] for path, *_ in layouts}
    exact_seconds = []
    for query in queries:
        vector = base.embedding_provider.embed_query(query["query"])
        start = time.perf_counter()
        exact = exact_top_k(vector, rows, k)
        exact_seconds.append(time.perf_counter() - start)
        baseline = base.query(query["query"], k=k, retrieval_only=True).sources
        baseline_vs_exact.append(
            {"query_id": query["query_id"], **compare(baseline, exact, k)}
        )
        for path, config, _, retriever in layouts:
            collective = retriever.retrieve(query["query"], k)
            per_layout[path].append(
                {
                    "query_id": query["query_id"],
                    "vs_baseline": compare(collective, baseline, k),
                    "vs_exact": compare(collective, exact, k),
                    "differences": classify_differences(collective, baseline, exact),
                }
            )

    def totals(entries: list[dict]) -> dict:
        keys = entries[0]["differences"].keys() if entries else ()
        return {key: sum(e["differences"][key] for e in entries) for key in keys}

    payload = {
        "experiment": "collective-correctness",
        "k": k,
        "vectors": len(rows),
        "queries": len(queries),
        "exact_reference": {
            "method": "brute-force cosine over every stored vector (pure Python)",
            "export_seconds": round(export_seconds, 3),
            "mean_seconds_per_query": round(sum(exact_seconds) / len(exact_seconds), 4),
        },
        "baseline_vs_exact": {
            "summary": summarize(baseline_vs_exact),
            "per_query": baseline_vs_exact,
        },
        "layouts": [
            {
                "config": path,
                "partitions": config.retrieval.partitions,
                "workers": config.retrieval.workers,
                "executor": config.retrieval.executor,
                "partition_counts": pipeline.vector_store.partition_counts(),
                "vs_baseline": summarize([e["vs_baseline"] for e in per_layout[path]]),
                "vs_exact": summarize([e["vs_exact"] for e in per_layout[path]]),
                "differences": totals(per_layout[path]),
                "per_query": per_layout[path],
                "config_values": run_manifest(config)["config"],
            }
            for path, config, pipeline, _ in layouts
        ],
        "manifest": run_manifest(base_config),
    }
    for layout in payload["layouts"]:
        _logger.info(
            "P=%d: vs baseline overlap %.3f, identical order %d/%d; vs exact"
            " recall %.3f; differences %s",
            layout["partitions"],
            layout["vs_baseline"]["overlap_at_k"],
            layout["vs_baseline"]["identical_order"],
            len(queries),
            layout["vs_exact"]["recall"],
            layout["differences"],
        )
    _logger.info(
        "baseline vs exact: recall %.3f, identical order %d/%d",
        payload["baseline_vs_exact"]["summary"]["recall"],
        payload["baseline_vs_exact"]["summary"]["identical_order"],
        len(queries),
    )
    path = write_report(payload, args.output, "correctness")
    _logger.info("report written to %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
