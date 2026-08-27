"""Retrieval quality and exactness for one corpus configuration.

For every evaluation query it retrieves through the configured pipeline
and also computes the exact brute-force top-k over the stored vectors,
then reports both side by side: the gap between them is the cost of
approximate indexing. Also records overlap, per-query detail, relevant
document coverage and latency after a stated warm-up.

Made for corpus-growth studies: run once per corpus configuration and
compare the rows. Writes: results/quality-<config>-<timestamp>.json.
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
    document_ranking,
    overlap_at_k,
    recall_at_k,
    reciprocal_rank,
    summarize,
)
from rag_framework.orchestration.pipeline import RAGPipeline

_logger = logging.getLogger("benchmarks.run_quality")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--label", default=None, help="report suffix (default: config stem)")
    parser.add_argument("--queries", default="./evaluation/queries.jsonl")
    parser.add_argument("--output", default="./results")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warm-up", type=int, default=3)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    label = args.label or Path(args.config).stem
    queries = [
        json.loads(line)
        for line in Path(args.queries).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    pipeline = RAGPipeline.from_config(args.config)
    pipeline.vector_store.create_or_open(pipeline.config.vector_store.collection)
    pipeline._collection_open = True
    vectors = {q["query_id"]: pipeline.embedding_provider.embed_query(q["query"]) for q in queries}

    # exact reference: ONE streaming pass over the stored vectors for
    #all queries at once, the corpus is never materialised; so this
    # scales to millions of vectors (a list of python floats would not)
    vector_count = 0
    indexed_docs: set[str] = set()

    def counted():
        nonlocal vector_count
        for chunk_id, document_id, vector in pipeline.vector_store.iter_vectors():
            vector_count += 1
            indexed_docs.add(document_id)
            yield chunk_id, document_id, vector

    start = time.perf_counter()
    exacts = exact_top_k_many(vectors, counted(), args.k)
    exact_pass_seconds = time.perf_counter() - start
    _logger.info(
        "%s: exact reference over %d vectors in %.1f s (streaming)",
        label, vector_count, exact_pass_seconds,
    )

    for q in queries[: args.warm_up]:
        pipeline.vector_store.search(vectors[q["query_id"]], args.k)

    per_query = []
    search_lat: list[float] = []
    for q in queries:
        vector = vectors[q["query_id"]]
        for _ in range(args.repetitions):
            t = time.perf_counter()
            hnsw = pipeline.vector_store.search(vector, args.k)
            search_lat.append(time.perf_counter() - t)
        exact = exacts[q["query_id"]]
        entry = {
            "query_id": q["query_id"],
            "answerable": q["answerable"],
            "overlap_at_k": overlap_at_k(
                [r.chunk_id for r in hnsw], [r.chunk_id for r in exact], args.k
            ),
        }
        if q["answerable"]:
            relevant = set(q["relevant_document_ids"])
            entry["recall_at_k"] = recall_at_k(document_ranking(hnsw), relevant, args.k)
            entry["reciprocal_rank"] = reciprocal_rank(document_ranking(hnsw), relevant)
            entry["exact_recall_at_k"] = recall_at_k(document_ranking(exact), relevant, args.k)
            entry["exact_reciprocal_rank"] = reciprocal_rank(document_ranking(exact), relevant)
        per_query.append(entry)

    answerable = [e for e in per_query if e["answerable"]]
    relevant_ids = {d for q in queries if q["answerable"] for d in q["relevant_document_ids"]}
    ef = pipeline.config.vector_store.ef_search
    payload = {
        "experiment": "corpus-quality",
        "label": label,
        "k": args.k,
        "vectors": vector_count,
        "documents": len(indexed_docs),
        "ef_search": ef if ef is not None else "backend default",
        "relevant_coverage": {
            "relevant_documents": len(relevant_ids),
            "in_corpus": len(relevant_ids & indexed_docs),
        },
        "quality": {
            "answerable_queries": len(answerable),
            "mean_recall_at_k": sum(e["recall_at_k"] for e in answerable) / len(answerable),
            "mean_reciprocal_rank": sum(e["reciprocal_rank"] for e in answerable) / len(answerable),
            "exact_mean_recall_at_k": sum(e["exact_recall_at_k"] for e in answerable) / len(answerable),
            "exact_mean_reciprocal_rank": sum(e["exact_reciprocal_rank"] for e in answerable) / len(answerable),
        },
        "exactness": {
            "mean_overlap_at_k": sum(e["overlap_at_k"] for e in per_query) / len(per_query),
            "queries_below_full_overlap": sum(1 for e in per_query if e["overlap_at_k"] < 0.999),
        },
        "latency": {
            "search_seconds": summarize(search_lat),
            "exact_reference_pass_seconds": round(exact_pass_seconds, 3),
        },
        "per_query": per_query,
        "warm_up_policy": f"{args.warm_up} unmeasured warm-up searches; query vectors precomputed once",
        "repetitions": args.repetitions,
        "manifest": run_manifest(pipeline.config),
    }
    _logger.info(
        "%s: R@%d %.3f (exact %.3f) MRR %.3f overlap %.3f (<1 on %d/%d) search median %.2f ms",
        label, args.k, payload["quality"]["mean_recall_at_k"],
        payload["quality"]["exact_mean_recall_at_k"],
        payload["quality"]["mean_reciprocal_rank"],
        payload["exactness"]["mean_overlap_at_k"],
        payload["exactness"]["queries_below_full_overlap"], len(per_query),
        1000 * payload["latency"]["search_seconds"]["median"],
    )
    path = write_report(payload, args.output, f"quality-{label}")
    _logger.info("report written to %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
