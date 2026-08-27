"""Disk cache decorator for embedding providers.

Wraps any :class:`EmbeddingProvider` and persists document vectors keyed
by ``chunk_id``, which is deterministic over (document, position,
normalized text), so a hit is exact by construction. Vectors are
appended as parquet shards, one per call that produced misses, which
makes a long encode resumable. The cache belongs to a single embedding
model and refuses to open under another; queries are never cached.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from rag_framework.embeddings.base import EmbeddingError, EmbeddingProvider
from rag_framework.models import Chunk

_logger = logging.getLogger(__name__)


class CachedEmbeddingProvider(EmbeddingProvider):
    """Serves vectors from disk; delegates misses to the inner provider."""

    def __init__(self, inner: EmbeddingProvider, cache_dir: str | Path) -> None:
        self._inner = inner
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

        meta_path = self._dir / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("model_id") != inner.model_id:
                raise EmbeddingError(
                    f"embedding cache at {self._dir} belongs to model"
                    f" '{meta.get('model_id')}', not '{inner.model_id}':"
                    " refusing to mix embedding spaces"
                )
        else:
            meta_path.write_text(
                json.dumps({"model_id": inner.model_id}), encoding="utf-8"
            )

        self._vectors: dict[str, list[float]] = {}
        for shard in sorted(self._dir.glob("shard-*.parquet")):
            try:
                table = pq.read_table(shard)
            except (OSError, ValueError) as error:
                raise EmbeddingError(
                    f"unreadable cache shard {shard}: {error}"
                ) from error
            for chunk_id, vector in zip(
                table.column("chunk_id").to_pylist(),
                table.column("vector").to_pylist(),
            ):
                self._vectors[chunk_id] = vector
        if self._vectors:
            _logger.info(
                "embedding cache at %s: %d vectors loaded",
                self._dir,
                len(self._vectors),
            )

    # identity delegates to the inner provider (device reflects where
    # misses would actually be computed)
    @property
    def model_id(self):  # type: ignore[override]
        return self._inner.model_id

    @property
    def normalized(self):  # type: ignore[override]
        return self._inner.normalized

    @property
    def device(self):  # type: ignore[override]
        return self._inner.device

    @property
    def revision(self):
        return getattr(self._inner, "revision", None)

    @property
    def dimension(self):  # type: ignore[override]
        if self._vectors:
            return len(next(iter(self._vectors.values())))
        return self._inner.dimension

    def embed_documents(self, chunks: list[Chunk]) -> list[list[float]]:
        missing = [c for c in chunks if c.chunk_id not in self._vectors]
        if missing:
            vectors = self._inner.embed_documents(missing)
            shard_path = (
                self._dir / f"shard-{len(list(self._dir.glob('shard-*.parquet'))):06d}.parquet"
            )
            pq.write_table(
                pa.table(
                    {
                        "chunk_id": pa.array(
                            [c.chunk_id for c in missing], pa.string()
                        ),
                        "vector": pa.array(
                            vectors, pa.list_(pa.float32())
                        ),
                    }
                ),
                shard_path,
            )
            for chunk, vector in zip(missing, vectors):
                self._vectors[chunk.chunk_id] = [float(x) for x in vector]
        self.misses += len(missing)
        self.hits += len(chunks) - len(missing)
        return [self._vectors[c.chunk_id] for c in chunks]

    def embed_query(self, text: str) -> list[float]:
        return self._inner.embed_query(text)
