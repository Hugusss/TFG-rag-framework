"""Loader for the OWI v2.0.0 "gpu" parquet layout.

On disk (as pulled by the official OWI CLI), a corpus root contains two
dataset directories sharing one hive partition scheme:

    <root>/**/<embeddings dataset>/year=Y/month=M/day=D/language=L/
        metadata_N_records.parquet     id, chunk_offsets, quality signals
        metadata_N_embeddings.parquet  record_id, chunk_idx, embedding
    <root>/**/<text dataset>/year=Y/month=M/day=D/language=L/
        metadata_N.parquet             id, main_content, url, title, ...

The records files define the corpus: exactly the documents the publisher
chunked and embedded. Text arrives by joining the sibling text file on
``id``. Emitted documents carry the publisher chunk offsets in
``metadata["chunk_offsets"]`` and the crawl date rebuilt from the hive
partition; validating offsets against the text is the chunker's job.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from rag_framework import __version__
from rag_framework.loaders.base import DocumentLoader, LoaderError
from rag_framework.models import Document

_logger = logging.getLogger(__name__)

_RECORDS_NAME = re.compile(r"^metadata_(\d+)_records\.parquet$")
_TEXT_NAME = re.compile(r"^metadata_(\d+)\.parquet$")

# the sibling embeddings parquet is deliberately not read here: vectors
# belong to the embedding-provider seam, so this loader behaves the same
# whether vectors come from the corpus or from a live encoder
_RECORDS_COLUMNS = ["id", "chunk_offsets"]
_TEXT_COLUMNS = ["id", "url", "title", "main_content", "language"]


@dataclass(frozen=True, slots=True)
class _Shard:
    records_file: Path
    text_file: Path
    partition: tuple[str, ...]  #e.g ("year=2026", ..., "language=spa")
    index: int


def _partition_of(path: Path) -> tuple[str, ...]:
    """The hive ``key=value`` segments of a file's directory path."""
    return tuple(part for part in path.parent.parts if "=" in part)


def _crawl_date(partition: tuple[str, ...]) -> str | None:
    values = dict(part.split("=", 1) for part in partition)
    try:
        year, month, day = (int(values[k]) for k in ("year", "month", "day"))
    except (KeyError, ValueError):
        return None
    return f"{year:04d}-{month:02d}-{day:02d}"


def _check_schema(path: Path, required: list[str]) -> None:
    """Raise LoaderError if a file is unreadable or lacks required columns.

    Reads parquet metadata only (cheap), so schema problems surface at
    load() call time instead of as backend exceptions mid-iteration.
    """
    try:
        names = set(pq.read_schema(path).names)
    except (OSError, ValueError) as error:
        raise LoaderError(f"unreadable parquet file {path}: {error}") from error
    missing = set(required) - names
    if missing:
        raise LoaderError(
            f"{path}: missing required column(s): {', '.join(sorted(missing))}"
        )


class OwiLoader(DocumentLoader):
    """DocumentLoader for the paired OWI v2.0.0 parquet corpus."""

    def load(self, source: str | Path) -> Iterator[Document]:
        self.rejections = []
        self.duplicates_skipped = 0
        root = Path(source)
        #discovery is eager so structural problems raise here, not on
        # the first next() somewhere inside the ingestion loop
        shards = self._discover(root)
        return self._iter_documents(root, shards)

    def _discover(self, root: Path) -> list[_Shard]:
        if not root.is_dir():
            raise LoaderError(f"source is not a directory: {root}")

        records: dict[tuple[tuple[str, ...], int], Path] = {}
        texts: dict[tuple[tuple[str, ...], int], Path] = {}
        for path in sorted(root.rglob("metadata_*.parquet")):
            if match := _RECORDS_NAME.match(path.name):
                key = (_partition_of(path), int(match.group(1)))
                target = records
            elif match := _TEXT_NAME.match(path.name):
                key = (_partition_of(path), int(match.group(1)))
                target = texts
            else:
                continue
            # the pairing key spans dataset directories on purpose, so a
            #second file with the same key would silently shadow the
            # real one, and a mispaired corpus must fail loudly
            if key in target:
                raise LoaderError(
                    f"conflicting files for shard {key[1]} in partition "
                    f"{'/'.join(key[0])}: {target[key]} and {path}"
                )
            target[key] = path

        if not records:
            raise LoaderError(f"no records parquet files found under {root}")

        shards: list[_Shard] = []
        unpaired: list[str] = []
        for (partition, index), path in records.items():
            text_file = texts.get((partition, index))
            if text_file is None:
                unpaired.append(str(path))
            else:
                shards.append(_Shard(path, text_file, partition, index))
        if unpaired:
            # a partly joinable corpus is worse than none: the run would
            # succeed over a silently smaller document set
            raise LoaderError(
                "records shard(s) without a text sibling: " + ", ".join(unpaired)
            )

        for shard in shards:
            _check_schema(shard.records_file, _RECORDS_COLUMNS)
            _check_schema(shard.text_file, _TEXT_COLUMNS)

        shards.sort(key=lambda shard: (shard.partition, shard.index))
        return shards

    def _iter_documents(
        self, root: Path, shards: list[_Shard]
    ) -> Iterator[Document]:
        seen: set[str] = set()
        for shard in shards:
            yield from self._load_shard(root, shard, seen)

    def _load_shard(
        self, root: Path, shard: _Shard, seen: set[str]
    ) -> Iterator[Document]:
        try:
            records = pq.read_table(shard.records_file, columns=_RECORDS_COLUMNS)
            text = pq.read_table(shard.text_file, columns=_TEXT_COLUMNS)
        except (OSError, ValueError) as error:
            # schema was pre-validated; this catches data corruption
            # discovered only while decoding pages,still LoaderError,
            #never a bare backend exception
            raise LoaderError(
                f"failed reading shard {shard.records_file} / "
                f"{shard.text_file}: {error}"
            ) from error

        text_rows: dict[str, tuple] = {}
        for row in zip(*(text.column(name).to_pylist() for name in _TEXT_COLUMNS)):
            text_id = row[0]
            if text_id in text_rows:
                _logger.warning(
                    "duplicate id in text file %s: %s (keeping first)",
                    shard.text_file.name,
                    text_id,
                )
                continue
            text_rows[text_id] = row

        date = _crawl_date(shard.partition)
        hive = dict(part.split("=", 1) for part in shard.partition)
        records_ref = str(shard.records_file.relative_to(root))
        text_ref = str(shard.text_file.relative_to(root))

        ids = records.column("id").to_pylist()
        all_offsets = records.column("chunk_offsets").to_pylist()
        for row_index, (record_id, raw_offsets) in enumerate(zip(ids, all_offsets)):
            ref = f"{records_ref}#{row_index}"
            if record_id is None or not record_id.strip():
                self._reject(ref, "missing id")
                continue
            if record_id in seen:
                self.duplicates_skipped += 1
                _logger.warning("duplicate document id skipped: %s (%s)", record_id, ref)
                continue
            row = text_rows.get(record_id)
            if row is None:
                self._reject(ref, f"no text row for id {record_id}")
                continue
            _, url, title, main_content, language = row
            if main_content is None or not main_content.strip():
                self._reject(ref, f"empty main_content for id {record_id}")
                continue
            try:
                offsets = [
                    (int(offset["start"]), int(offset["end"]))
                    for offset in (raw_offsets or [])
                ]
            except (TypeError, KeyError):
                self._reject(ref, f"malformed chunk_offsets for id {record_id}")
                continue

            metadata = {
                "chunk_offsets": offsets,
                "source_records_file": records_ref,
                "source_text_file": text_ref,
                "ingestion_version": __version__,
            }
            #missing values are absent keys, never invented placeholders
            if url:
                metadata["url"] = url
            if title:
                metadata["title"] = title
            if language or hive.get("language"):
                metadata["language"] = language or hive["language"]
            if date:
                metadata["date"] = date

            seen.add(record_id)
            # the publisher id is adopted verbatim rather than re-hashed
            # (ADR-001): it is already a content hash, and keeping it makes
            #every official artifact directly joinable
            yield Document(document_id=record_id, text=main_content, metadata=metadata)
