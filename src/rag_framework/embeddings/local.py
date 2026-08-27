"""Live embedding adapter over sentence-transformers.

Encodes text with the same recipe that produced the corpus's official
vectors (ADR-009), which is the whole contract of this module:

    SentenceTransformer(model_id, trust_remote_code=True, device="cpu")
    model.encode(texts, task="retrieval", normalize_embeddings=True)

The model loads on first use; tests inject a stub ``encoder_factory``.
"""

from __future__ import annotations

import logging
import time

from rag_framework.embeddings.base import EmbeddingError, EmbeddingProvider
from rag_framework.models import Chunk

_logger = logging.getLogger(__name__)


def _load_sentence_transformer(model_id: str):
    # imported here so the heavy dependency chain (torch) loads only
    # when a live encoder is actually needed
    from sentence_transformers import SentenceTransformer

    # trust_remote_code executes model-repository code; the risk is
    # accepted knowingly because the reference encoder that validated the
    # corpus vectors is built the same way, and any deviation here would
    # produce vectors from a subtly different embedding space
    return SentenceTransformer(model_id, trust_remote_code=True, device="cpu")


def _cached_revision(model_id: str) -> str | None:
    """The resolved HF commit of the cached model, without network.

    The revision is deliberately not pinned (ADR-009), so recording
    which one actually ran is what keeps results attributable if
    upstream drifts.
    """
    try:
        from huggingface_hub import try_to_load_from_cache

        path = try_to_load_from_cache(model_id, "config.json")
        if isinstance(path, str) and "/snapshots/" in path:
            return path.split("/snapshots/")[1].split("/")[0]
    except Exception:  # never let bookkeeping break encoding
        return None
    return None


class LocalEmbeddingProvider(EmbeddingProvider):
    """Encodes chunks and queries with a local sentence-transformers
    model, using the pinned retrieval recipe."""

    def __init__(
        self,
        model_id: str,
        *,
        batch_size: int | None = None,
        encoder_factory=None,
    ) -> None:
        self.model_id = model_id
        self.normalized = True  # the reference recipe normalizes
        self.device = "cpu"  # the reference recipe's execution device
        self.batch_size = batch_size or 32
        self.revision: str | None = None  # resolved HF revision after load
        self._encoder_factory = encoder_factory or _load_sentence_transformer
        self._encoder = None
        self._dimension: int | None = None

    @property
    def dimension(self) -> int:
        """Vector dimension — note: first access loads the model and
        runs one probe encode (cached afterwards)."""
        if self._dimension is None:
            self._encode(["dimension probe"])
        assert self._dimension is not None
        return self._dimension

    def embed_documents(self, chunks: list[Chunk]) -> list[list[float]]:
        if not chunks:
            return []
        return self._encode([chunk.text for chunk in chunks])

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text])[0]

    def _require_encoder(self):
        if self._encoder is None:
            _logger.info("loading embedding model %s (first use)", self.model_id)
            start = time.perf_counter()
            try:
                self._encoder = self._encoder_factory(self.model_id)
            except Exception as error:
                raise EmbeddingError(
                    f"could not load embedding model {self.model_id}: {error}"
                ) from error
            self.revision = _cached_revision(self.model_id)
            _logger.info(
                "embedding model loaded in %.1f s (revision %s)",
                time.perf_counter() - start,
                self.revision or "unknown",
            )
        return self._encoder

    def _encode(self, texts: list[str]) -> list[list[float]]:
        encoder = self._require_encoder()
        try:
            raw = encoder.encode(
                texts,
                task="retrieval",
                normalize_embeddings=True,
                batch_size=self.batch_size,
            )
            vectors = [[float(x) for x in vector] for vector in raw]
        except Exception as error:
            raise EmbeddingError(
                f"encoding failed with model {self.model_id}: {error}"
            ) from error

        if len(vectors) != len(texts):
            raise EmbeddingError(
                f"model returned {len(vectors)} vectors for {len(texts)} texts"
            )
        for vector in vectors:
            if self._dimension is None:
                self._dimension = len(vector)
            elif len(vector) != self._dimension:
                raise EmbeddingError(
                    f"inconsistent vector dimensions from model"
                    f" {self.model_id}: {self._dimension} and {len(vector)}"
                )
        return vectors
