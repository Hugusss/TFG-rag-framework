"""Reproducibility manifest and result-file writing.

Every result file carries the git commit, configuration, dataset
version, machine, timestamp and software versions that produced it.
:func:`run_manifest` builds that block once from live sources;
:func:`write_report` stamps it into a JSON file. Nothing else in the
project writes result files.
"""

from __future__ import annotations

import dataclasses
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path

from rag_framework import __version__
from rag_framework.config import PipelineConfig

_DEPENDENCIES = (
    "chromadb",
    "pyarrow",
    "pyyaml",
    "sentence-transformers",
    "transformers",
    "torch",
    "peft",
)


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent), *args],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout if result.returncode == 0 else None


def _git_commit() -> str:
    """The current commit, "-dirty"-suffixed when the tree differs.

    A result stamped with a commit that does not contain the code that
    produced it would be worse than one with no commit at all; the dirty
    marker makes that state visible instead of silently plausible.
    """
    commit = _git("rev-parse", "HEAD")
    if commit is None:
        return "unknown"
    status = _git("status", "--porcelain")
    dirty = "-dirty" if status is None or status.strip() else ""
    return commit.strip() + dirty


def run_manifest(config: PipelineConfig) -> dict:
    """The reproducibility block stamped into every result file."""
    versions = {"rag-framework": __version__}
    for package in _DEPENDENCIES:
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "dataset_version": config.dataset.version or "unspecified",
        "machine": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
        },
        "versions": versions,
        "config": dataclasses.asdict(config),
    }


def write_report(payload: dict, output_dir: str | Path, name_prefix: str) -> Path:
    """Write one result payload as pretty JSON; returns the file path.

    File names carry a microsecond UTC timestamp and are opened
    exclusively, so runs never overwrite each other — a colliding name
    gets a counter suffix instead of clobbering the earlier result: a
    measurement is evidence, and evidence is added to, never replaced.
    """
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%S") + f".{now.microsecond:06d}Z"
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    for attempt in range(1000):
        suffix = "" if attempt == 0 else f"-{attempt}"
        path = directory / f"{name_prefix}-{stamp}{suffix}.json"
        try:
            with path.open("x", encoding="utf-8") as handle:
                handle.write(text)
        except FileExistsError:
            continue
        return path
    raise OSError(f"could not find a free report name under {directory}")


def build_ingest_payload(pipeline, report) -> dict:
    """The complete ingest result payload.

    Result-file shape is owned by the metrics layer: the ingestion
    report, the per-seam accounting details the fixed report cannot
    carry, the warm-up and repetition policy, and the manifest.
    ``pipeline`` is a RAGPipeline (untyped to keep metrics free of an
    orchestration import).
    """
    loader, chunker = pipeline.loader, pipeline.chunker
    provider = pipeline.embedding_provider
    sample = _REJECTION_SAMPLE
    return {
        "report": dataclasses.asdict(report),
        "details": {
            "pipeline_setup_seconds": pipeline.setup_seconds,
            "embedding_identity": {
                "model_id": provider.model_id,
                "dimension": provider.dimension,
                "normalized": provider.normalized,
                "device": provider.device,
                "revision": getattr(provider, "revision", None),
            },
            "vectors_before_run": pipeline.vectors_before,
            "documents_sampled_out": getattr(loader, "documents_sampled_out", None),
            "duplicate_vectors_skipped": getattr(
                pipeline.embedding_provider, "duplicate_vectors_skipped", None
            ),
            "loader_rejections": len(loader.rejections),
            "loader_rejection_sample": [
                dataclasses.asdict(r) for r in loader.rejections[:sample]
            ],
            "chunker_rejections": len(chunker.rejections),
            "chunker_rejection_sample": [
                dataclasses.asdict(r) for r in chunker.rejections[:sample]
            ],
            "documents_without_chunks": chunker.documents_without_chunks,
            # strategy-specific counters surface here and only here
            "windows_clipped": getattr(chunker, "windows_clipped", None),
        },
        "warm_up_policy": "none",
        "repetitions": 1,
        "manifest": run_manifest(pipeline.config),
    }


_REJECTION_SAMPLE = 20  # full detail is in the logs; reports carry a sample


def build_query_payload(pipeline, result) -> dict:
    """The complete query result payload.

    ``result`` is a RAGResult; sources are recorded as compact rows
    (ids, score, provenance) — full texts live in the store, not in
    every report. Deferred the remaining reproducibility fields
    until the benchmark
    runner exists: experiment id and query id (single ad-hoc queries
    have neither); failed queries write no report — failures live in
    logs and exit codes.

    ``query_encoder`` records the provider that actually encoded the
    query vector (for the precomputed provider that is its live
    delegate, whose device and resolved model revision differ from the
    corpus vectors') — the attribution ADR-009's unpinned revision
    depends on.
    """
    provider = pipeline.embedding_provider
    encoder = getattr(provider, "query_encoder", None) or provider
    return {
        "query": result.query,
        "retrieval_mode": pipeline.config.retrieval.mode,
        "partitions": pipeline.config.retrieval.partitions,
        "workers": pipeline.config.retrieval.workers,
        "executor": pipeline.config.retrieval.executor,
        "collection": pipeline.config.vector_store.collection,
        "metrics": result.metrics,
        "answer": result.answer,
        "generator": {
            "provider": pipeline.config.generation.provider,
            "model": pipeline.config.generation.model,
            "enabled": pipeline.config.generation.enabled,
        },
        "sources": [
            {
                "rank": rank,
                "score": source.score,
                "chunk_id": source.chunk_id,
                "document_id": source.document_id,
                "partition_id": source.partition_id,
                "worker_id": source.worker_id,
                "url": source.metadata.get("url"),
                "position": source.metadata.get("position"),
            }
            for rank, source in enumerate(result.sources, start=1)
        ],
        "embedding_identity": {
            "model_id": provider.model_id,
            "dimension": provider.dimension,
            "normalized": provider.normalized,
            "device": provider.device,
            "revision": getattr(provider, "revision", None),
        },
        "query_encoder": {
            "model_id": encoder.model_id,
            "device": encoder.device,
            "revision": getattr(encoder, "revision", None),
        },
        "warm_up_policy": "none",
        "repetitions": 1,
        "manifest": run_manifest(pipeline.config),
    }
