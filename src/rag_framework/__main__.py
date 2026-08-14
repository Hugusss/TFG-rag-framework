"""Command-line entry point: ``python -m rag_framework <command>``."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys

from rag_framework.config import ConfigError
from rag_framework.embeddings.base import EmbeddingError
from rag_framework.loaders.base import LoaderError
from rag_framework.metrics.manifest import build_ingest_payload, write_report
from rag_framework.orchestration.pipeline import RAGPipeline
from rag_framework.vectorstores.base import VectorStoreError

_logger = logging.getLogger("rag_framework")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rag_framework",
        description="Modular Collective RAG Pipeline over the Open Web Index",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    ingest = subcommands.add_parser(
        "ingest", help="ingest the configured dataset into the vector store"
    )
    ingest.add_argument("--config", required=True, help="pipeline YAML file")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.command == "ingest":
        return _ingest(args.config)
    parser.error(f"unknown command {args.command!r}")
    return 2


def _ingest(config_path: str) -> int:
    try:
        pipeline = RAGPipeline.from_config(config_path)
        report = pipeline.ingest()
    except (ConfigError, LoaderError, EmbeddingError, VectorStoreError) as error:
        # a user-fixable failure gets a message, never a traceback
        _logger.error("%s", error)
        return 1

    # print before writing: the metrics of a successful ingest must
    # survive even if the report file cannot be written
    print(json.dumps(dataclasses.asdict(report), indent=2))

    payload = build_ingest_payload(pipeline, report)
    try:
        path = write_report(payload, pipeline.config.metrics.output, "ingest")
    except OSError as error:
        _logger.error(
            "ingest succeeded but the report file could not be written: %s",
            error,
        )
        return 3
    print(f"report written to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
