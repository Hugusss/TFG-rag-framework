"""Embedding adapter that serves the corpus's official vectors.

The OWI dataset ships one official vector per publisher chunk, and they
enter through the EmbeddingProvider seam like any other provider rather
than as a path in the orchestrator (ADR-002). The official embeddings
parquet (``{record_id, chunk_idx, embedding}``) is read here and nowhere
else; document vectors are looked up by identity, and queries — which no
corpus contains — are delegated to a live encoder built on first use.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from rag_framework.embeddings.base import EmbeddingError, EmbeddingProvider
from rag_framework.models import Chunk

_logger = logging.getLogger(__name__)

_EMBEDDINGS_NAME = re.compile(r"^metadata_\d+_embeddings\.parquet$")
_COLUMNS = ["record_id", "chunk_idx", "embedding"]


class PrecomputedEmbeddingProvider(EmbeddingProvider):
    """Serves official corpus vectors by (document_id, position)."""

    def __init__(
        self,
        source: str | Path,
        *,
        model_id: str,
        query_encoder_factory=None,
    ) -> None:
        """``query_encoder_factory`` is a zero-argument callable
        returning an EmbeddingProvider for query encoding (built on
        first query, so ingest never pays for it)."""
        self.model_id = model_id
        self.normalized = True  # v2.0.0 corpus contract: unit vectors
        self.device = "none"  # vectors are read, never computed here
        self._query_encoder_factory = query_encoder_factory
        # public once built: the provider that actually encodes queries,
        # so reports can attribute query vectors to the encoder that
        # produced them (its device and resolved model revision differ
        # from this provider's)
        self.query_encoder: EmbeddingProvider | None = None
        root = Path(source)
        files = sorted(
            path
            for path in root.rglob("metadata_*_embeddings.parquet")
            if _EMBEDDINGS_NAME.match(path.name)
        )
        if not files:
            raise EmbeddingError(
                f"no embeddings parquet files found under {root}"
            )

        tables = []
        file_of_row: list[Path] = []
        reference_schema = None
        reference_file = None
        for file in files:
            try:
                names = set(pq.read_schema(file).names)
            except (OSError, ValueError) as error:
                raise EmbeddingError(
                    f"unreadable embeddings file {file}: {error}"
                ) from error
            missing = set(_COLUMNS) - names
            if missing:
                raise EmbeddingError(
                    f"{file}: missing required column(s):"
                    f" {', '.join(sorted(missing))}"
                )
            try:
                shard = pq.read_table(file, columns=_COLUMNS)
            except (OSError, ValueError) as error:
                raise EmbeddingError(
                    f"failed reading embeddings file {file}: {error}"
                ) from error
            if reference_schema is None:
                reference_schema, reference_file = shard.schema, file
            elif shard.schema != reference_schema:
                raise EmbeddingError(
                    f"{file}: schema differs from {reference_file}"
                    " (mixed corpus versions?)"
                )
            tables.append(shard)
            file_of_row.extend([file] * shard.num_rows)

        try:
            table = pa.concat_tables(tables)
        except (OSError, ValueError) as error:
            raise EmbeddingError(
                f"could not combine embeddings shards under {root}: {error}"
            ) from error
        if table.num_rows == 0:
            raise EmbeddingError(f"embeddings files under {root} are empty")

        embedding_column = table.column("embedding")
        if embedding_column.null_count:
            raise EmbeddingError(
                f"{embedding_column.null_count} null embedding value(s)"
                f" under {root}"
            )
        if pc.list_flatten(embedding_column).null_count:
            raise EmbeddingError(
                f"embedding vectors under {root} contain null elements"
            )
        lengths = pc.list_value_length(embedding_column)
        low, high = pc.min(lengths).as_py(), pc.max(lengths).as_py()
        if low is None or low != high:
            raise EmbeddingError(
                f"inconsistent embedding dimensions under {root}:"
                f" {low}..{high}"
            )
        if low < 1:
            raise EmbeddingError(
                f"zero-dimensional embedding vectors under {root}"
            )
        self.dimension = high

        index: dict[tuple[str, int], int] = {}
        record_ids = table.column("record_id").to_pylist()
        chunk_idxs = table.column("chunk_idx").to_pylist()
        # a key seen twice in ONE file is corruption and raises; the
        # same key in different files is a legitimate recrawl (multi-day
        # corpora republish unchanged documents — record ids are content
        # hashes, so the text behind both vectors is identical): the
        # first occurrence wins, mirroring the loader, and the count is
        # reported so no report can absorb it unnoticed
        self.duplicate_vectors_skipped = 0
        for row, key in enumerate(zip(record_ids, chunk_idxs)):
            if key[0] is None or key[1] is None:
                raise EmbeddingError(
                    f"embeddings row {row} has a null record_id/chunk_idx"
                )
            if key in index:
                if file_of_row[index[key]] == file_of_row[row]:
                    raise EmbeddingError(
                        f"duplicate vector for record {key[0]} chunk"
                        f" {key[1]} within {file_of_row[row]}"
                    )
                self.duplicate_vectors_skipped += 1
                continue
            index[key] = row
        if self.duplicate_vectors_skipped:
            _logger.warning(
                "embeddings under %s: %d duplicate (record, chunk) row(s)"
                " across files — recrawled documents; first occurrence kept",
                root,
                self.duplicate_vectors_skipped,
            )
        self._index = index
        # vectors stay in the fp16 arrow column (~13 MB for the full
        # slice); they are converted to python floats per lookup
        self._vectors = table.column("embedding")

    def embed_documents(self, chunks: list[Chunk]) -> list[list[float]]:
        vectors = []
        for chunk in chunks:
            # keyed by identity rather than by text (ADR-005), so clipping
            # a window's text cannot change which vector it receives; this
            # is why Chunk.position must stay the publisher's chunk_idx
            row = self._index.get((chunk.document_id, chunk.position))
            if row is None:
                raise EmbeddingError(
                    f"no official vector for document {chunk.document_id}"
                    f" position {chunk.position}: the chunker and the"
                    " embeddings table disagree about the corpus"
                )
            vectors.append(self._vectors[row].as_py())
        return vectors

    def embed_query(self, text: str) -> list[float]:
        if self._query_encoder_factory is None:
            raise NotImplementedError(
                "no query encoder configured: this provider serves"
                " precomputed document vectors only"
            )
        if self.query_encoder is None:
            # built on first query, never at pipeline construction, so an
            # ingest that only serves corpus vectors pays no model load
            encoder = self._query_encoder_factory()
            if encoder.model_id != self.model_id:
                raise EmbeddingError(
                    f"query encoder model '{encoder.model_id}' does not"
                    f" match the corpus vectors' model '{self.model_id}':"
                    " refusing to mix embedding spaces"
                )
            if encoder.normalized != self.normalized:
                raise EmbeddingError(
                    f"query encoder normalization ({encoder.normalized})"
                    f" does not match the corpus vectors"
                    f" ({self.normalized}): scores would be incomparable"
                )
            self.query_encoder = encoder
        vector = self.query_encoder.embed_query(text)
        if len(vector) != self.dimension:
            raise EmbeddingError(
                f"query vector dimension {len(vector)} does not match"
                f" the corpus dimension {self.dimension}"
            )
        return vector
