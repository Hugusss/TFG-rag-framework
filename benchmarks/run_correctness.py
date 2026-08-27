"""Collective-correctness experiment: is the merged answer the right one?

For every evaluation query it compares three answers over the same
vectors, model and metric: the exact top-k (brute-force cosine over
every stored vector), the sequential baseline, and the collective result
for each partition count given. The exact answer is what makes a
difference attributable, since baseline and collective are both
approximate. Records overlap, positions preserved, discordant pairs,
score differences and a classification of every difference.

Writes: results/correctness-<timestamp>.json.
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
from rag_framework.metrics.exact import exact_top_k_many
from rag_framework.metrics.manifest import run_manifest, write_report
from rag_framework.metrics.quality import (
    classify_differences,
    discordant_pairs,
    max_score_difference,
    overlap_at_k,
    positions_preserved,
    recall_at_k,
)
from rag_framework.orchestration.pipeline import (
    RAGPipeline,
    build_retriever,
    build_vector_store,
)

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

    layouts = []
    for path in args.collective:
        config = load_config(path)
        # only the layout's STORE is built — never a second pipeline: a
        # second precomputed provider would duplicate the multi-GB
        # vector table for nothing (the baseline's encoder is shared)
        store = build_vector_store(config, base.embedding_provider)
        store.create_or_open(config.vector_store.collection)
        if store.count() != base.vector_store.count():
            raise SystemExit(
                f"{path}: {store.count()} vectors but the baseline"
                f" holds {base.vector_store.count()} — not the same"
                " index content"
            )
        # share the baseline's query encoder: same vector for all three
        # answers by construction, one model load
        retriever = build_retriever(config, base.embedding_provider, store)
        layouts.append((path, config, store, retriever))

    # exact reference: one streaming pass for every query at once — the
    # corpus is never materialised (a 1.36M-vector corpus as Python
    # lists is tens of GB and was OOM-killed)
    vectors = {
        query["query_id"]: base.embedding_provider.embed_query(query["query"])
        for query in queries
    }
    vector_count = 0

    def counted():
        nonlocal vector_count
        for row in base.vector_store.iter_vectors():
            vector_count += 1
            yield row

    start = time.perf_counter()
    exacts = exact_top_k_many(vectors, counted(), k)
    exact_pass_seconds = time.perf_counter() - start
    _logger.info(
        "exact reference over %d vectors in %.1f s (streaming)",
        vector_count, exact_pass_seconds,
    )

    baseline_vs_exact = []
    per_layout = {path: [] for path, *_ in layouts}
    for query in queries:
        exact = exacts[query["query_id"]]
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
        "vectors": vector_count,
        "queries": len(queries),
        "exact_reference": {
            "method": (
                "brute-force cosine over every stored vector, one"
                " streaming pass for all queries (pure Python)"
            ),
            "pass_seconds": round(exact_pass_seconds, 3),
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
                "partition_counts": store.partition_counts(),
                "vs_baseline": summarize([e["vs_baseline"] for e in per_layout[path]]),
                "vs_exact": summarize([e["vs_exact"] for e in per_layout[path]]),
                "differences": totals(per_layout[path]),
                "per_query": per_layout[path],
                "config_values": run_manifest(config)["config"],
            }
            for path, config, store, _ in layouts
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
