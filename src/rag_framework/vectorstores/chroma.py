"""ChromaDB adapter for the VectorStore seam.

The only module in the project that imports chromadb; everything
backend-specific stays behind this boundary — the cosine space and its
distance-to-similarity conversion (ADR-003), upsert semantics, batch
limits and metadata encoding. Collections are stamped at creation with
the embedding space they hold, and reopening one under a conflicting
identity raises instead of quietly searching the wrong vectors.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import chromadb

from rag_framework.models import Chunk, SearchResult
from rag_framework.vectorstores.base import VectorStore, VectorStoreError

# collection-metadata keys that define the embedding space and, for a
# partitioned index, its layout; a mismatch on any of them makes an
# existing collection incompatible
_IDENTITY_KEYS = (
    "model_id", "dimension", "normalized", "partitions", "partition_index"
)

_SCALAR_TYPES = (str, int, float, bool)


class ChromaVectorStore(VectorStore):
    """VectorStore backed by a persistent local ChromaDB collection."""

    def __init__(
        self,
        path: str | Path,
        *,
        collection_metadata: dict | None = None,
        ef_search: int | None = None,
    ) -> None:
        """``collection_metadata`` carries the embedding identity to
        stamp at creation (model_id, dimension, normalized, ...); the
        orchestrator fills it from the EmbeddingProvider. ``ef_search``
        overrides the HNSW search width — the approximation knob:
        the backend default (100) measurably loses true neighbours
        beyond ~10^4 vectors."""
        self._path = Path(path)
        self._metadata = dict(collection_metadata or {})
        self._ef_search = ef_search
        self._client = None
        self._collection = None
        self._name: str | None = None
        self._max_batch: int | None = None

    def create_or_open(self, collection_name: str) -> None:
        try:
            self._client = chromadb.PersistentClient(
                path=str(self._path),
                settings=chromadb.config.Settings(anonymized_telemetry=False),
            )
            collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine", **self._metadata},
            )
        except Exception as error:
            raise VectorStoreError(
                f"could not create or open collection '{collection_name}'"
                f" at {self._path}: {error}"
            ) from error

        if self._ef_search is not None:
            current = ((getattr(collection, "configuration_json", None) or {})
                       .get("hnsw") or {}).get("ef_search")
            if current != self._ef_search:
                try:
                    collection.modify(
                        configuration={"hnsw": {"ef_search": self._ef_search}}
                    )
                    # the modified width only takes effect on a fresh
                    # handle (measured: a stale handle keeps searching
                    # with the old ef), so reopen before any search
                    collection = self._client.get_collection(collection_name)
                except Exception as error:
                    raise VectorStoreError(
                        f"could not set ef_search={self._ef_search} on"
                        f" collection '{collection_name}': {error}"
                    ) from error

        stored = collection.metadata or {}
        space = self._collection_space(collection, stored)
        if space != "cosine":
            raise VectorStoreError(
                f"collection '{collection_name}' uses distance space"
                f" '{space}', expected 'cosine': scores would be"
                " converted wrongly (ADR-003)"
            )
        # only the keys the caller actually declares are compared: a store
        # built without collection_metadata states no expectations and can
        # open anything, which inspection tooling needs. The orchestrator
        # always passes the provider's identity, so real runs are guarded.
        for key in _IDENTITY_KEYS:
            expected = self._metadata.get(key)
            if expected is not None and stored.get(key) != expected:
                detail = (
                    f"was built with {key}={stored[key]!r}"
                    if key in stored
                    else f"has no recorded {key}"
                )
                raise VectorStoreError(
                    f"collection '{collection_name}' {detail}, expected"
                    f" {expected!r}: refusing to mix embedding spaces"
                )
        self._collection = collection
        self._name = collection_name

    @staticmethod
    def _collection_space(collection, stored_metadata: dict) -> str:
        """The collection's real distance space.

        chromadb 1.x is migrating this setting from collection metadata
        to the configuration API; read the authoritative configuration
        first and fall back to legacy metadata.
        """
        config = getattr(collection, "configuration_json", None) or {}
        hnsw = config.get("hnsw") or {}
        return (
            hnsw.get("space")
            or stored_metadata.get("hnsw:space")
            or "unknown"
        )

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        collection = self._require_collection()
        if len(chunks) != len(embeddings):
            raise VectorStoreError(
                f"{len(chunks)} chunks but {len(embeddings)} embeddings"
            )
        if not chunks:
            return
        ids = [chunk.chunk_id for chunk in chunks]
        if len(set(ids)) != len(ids):
            # upsert makes re-ingestion idempotent across batches, but
            # within one batch it would collapse two distinct chunks into
            # a single row and report both as stored
            raise VectorStoreError(
                "duplicate chunk ids within one add() batch"
            )

        documents = [chunk.text for chunk in chunks]
        metadatas = [self._chunk_metadata(chunk) for chunk in chunks]

        if self._max_batch is None:
            self._max_batch = self._backend_max_batch()
        try:
            for lo in range(0, len(ids), self._max_batch):
                hi = lo + self._max_batch
                collection.upsert(
                    ids=ids[lo:hi],
                    embeddings=embeddings[lo:hi],
                    documents=documents[lo:hi],
                    metadatas=metadatas[lo:hi],
                )
        except Exception as error:
            raise VectorStoreError(f"add failed: {error}") from error

    def search(
        self,
        query_embedding: list[float],
        k: int,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        collection = self._require_collection()
        n_results = min(k, self.count())
        if n_results == 0:
            return []
        try:
            response = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=self._translate_filters(filters),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as error:
            raise VectorStoreError(f"search failed: {error}") from error

        results = []
        for chunk_id, document, metadata, distance in zip(
            response["ids"][0],
            response["documents"][0],
            response["metadatas"][0],
            response["distances"][0],
        ):
            metadata = dict(metadata or {})
            document_id = metadata.pop("document_id", None)
            if not document_id:
                raise VectorStoreError(
                    f"stored row {chunk_id} has no document_id —"
                    " provenance is broken (written by another tool?)"
                )
            results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    text=document,
                    score=1.0 - distance,  # cosine distance -> similarity
                    metadata=metadata,
                )
            )
        return results

    def count(self) -> int:
        collection = self._require_collection()
        try:
            return collection.count()
        except Exception as error:
            raise VectorStoreError(f"count failed: {error}") from error

    def reset(self) -> None:
        collection = self._require_collection()
        assert self._client is not None and self._name is not None
        try:
            self._client.delete_collection(self._name)
        except Exception as error:
            raise VectorStoreError(f"reset failed: {error}") from error
        self._collection = None
        self.create_or_open(self._name)

    def iter_vectors(self) -> Iterator[tuple[str, str, list[float]]]:
        collection = self._require_collection()
        page = self._backend_max_batch() if self._max_batch is None else self._max_batch
        offset = 0
        while True:
            try:
                response = collection.get(
                    include=["embeddings", "metadatas"],
                    limit=page,
                    offset=offset,
                )
            except Exception as error:
                raise VectorStoreError(f"vector export failed: {error}") from error
            ids = response["ids"]
            if not ids:
                return
            for chunk_id, metadata, vector in zip(
                ids, response["metadatas"], response["embeddings"]
            ):
                document_id = (metadata or {}).get("document_id")
                if not document_id:
                    raise VectorStoreError(
                        f"stored row {chunk_id} has no document_id —"
                        " provenance is broken (written by another tool?)"
                    )
                # the backend hands back array types; the seam speaks lists
                yield chunk_id, document_id, [float(x) for x in vector]
            offset += len(ids)

    def _require_collection(self):
        if self._collection is None:
            raise VectorStoreError("create_or_open must be called first")
        return self._collection

    def _backend_max_batch(self) -> int:
        try:
            return int(self._client.get_max_batch_size())
        except Exception:
            return 1000  # conservative fallback; correctness unaffected

    @staticmethod
    def _chunk_metadata(chunk: Chunk) -> dict:
        """Per-row metadata as stored: the chunk's metadata plus the
        reserved provenance keys ``document_id`` and ``position``.

        A non-scalar metadata value or a reserved-key collision raises:
        dropping a value quietly would make later filters on it return
        nothing, with no trace of why.
        """
        metadata: dict = {}
        for key, value in chunk.metadata.items():
            if key in ("document_id", "position"):
                raise VectorStoreError(
                    f"chunk {chunk.chunk_id}: metadata key '{key}' is"
                    " reserved for the store"
                )
            if not isinstance(value, _SCALAR_TYPES):
                raise VectorStoreError(
                    f"chunk {chunk.chunk_id}: metadata '{key}' has"
                    f" non-scalar type {type(value).__name__}; store"
                    " metadata must be scalar"
                )
            metadata[key] = value
        metadata["document_id"] = chunk.document_id
        metadata["position"] = chunk.position
        return metadata

    @staticmethod
    def _translate_filters(filters: dict | None):
        """Translate the seam's flat equality mapping to Chroma syntax."""
        if not filters:
            return None
        clauses = [{key: {"$eq": value}} for key, value in filters.items()]
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}
