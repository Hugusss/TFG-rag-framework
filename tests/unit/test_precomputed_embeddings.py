"""Unit tests for the precomputed embedding provider."""

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from rag_framework.embeddings.base import EmbeddingError
from rag_framework.embeddings.precomputed import PrecomputedEmbeddingProvider
from rag_framework.models import Chunk

PARTITION = "year=2026/month=7/day=28/language=spa"


def write_embeddings(root, rows, *, index=0, partition=PARTITION, dim=4):
    """Write one embeddings shard; rows = [(record_id, chunk_idx, vector)].

    Vectors are stored as fp16 like the real corpus; use values exactly
    representable in fp16 (0.5, 0.25, ...) when asserting equality.
    """
    directory = root / "owie-dataset" / partition
    directory.mkdir(parents=True, exist_ok=True)

    def as_fp16(vector):
        if vector is None:
            return None
        return [None if x is None else np.float16(x) for x in vector]

    table = pa.table(
        {
            "record_id": pa.array([r[0] for r in rows], pa.string()),
            "chunk_idx": pa.array([r[1] for r in rows], pa.int32()),
            "embedding": pa.array(
                [as_fp16(r[2]) for r in rows], pa.list_(pa.float16())
            ),
        }
    )
    pq.write_table(table, directory / f"metadata_{index}_embeddings.parquet")


def chunk(doc_id, position, text="some text"):
    return Chunk(
        chunk_id=f"c-{doc_id}-{position}",
        document_id=doc_id,
        text=text,
        position=position,
        metadata={},
    )


VEC_A0 = [0.5, 0.25, -0.5, 1.0]
VEC_A1 = [0.125, -0.25, 0.75, 0.0]
VEC_B0 = [1.0, 0.0, -1.0, 0.5]


def provider_for(root):
    return PrecomputedEmbeddingProvider(root, model_id="test-model")


def standard_corpus(tmp_path):
    write_embeddings(
        tmp_path, [("aaa", 0, VEC_A0), ("aaa", 1, VEC_A1), ("bbb", 0, VEC_B0)]
    )
    return provider_for(tmp_path)


class TestLookup:
    def test_vectors_served_by_identity_in_input_order(self, tmp_path):
        provider = standard_corpus(tmp_path)
        vectors = provider.embed_documents(
            [chunk("bbb", 0), chunk("aaa", 1), chunk("aaa", 0)]
        )
        assert vectors == [VEC_B0, VEC_A1, VEC_A0]

    def test_dimension_exposed(self, tmp_path):
        assert standard_corpus(tmp_path).dimension == 4

    def test_lookup_ignores_text_entirely(self, tmp_path):
        # a clipped window changes text but never position: the vector
        # must be found by identity alone (ADR-002/ADR-005)
        provider = standard_corpus(tmp_path)
        original = provider.embed_documents([chunk("aaa", 0, text="full text")])
        clipped = provider.embed_documents([chunk("aaa", 0, text="full te")])
        assert original == clipped

    def test_deterministic(self, tmp_path):
        provider = standard_corpus(tmp_path)
        first = provider.embed_documents([chunk("aaa", 0)])
        second = provider.embed_documents([chunk("aaa", 0)])
        assert first == second

    def test_fp16_values_round_trip(self, tmp_path):
        write_embeddings(tmp_path, [("aaa", 0, [0.1, 0.2, 0.3, 0.4])])
        provider = provider_for(tmp_path)
        (vector,) = provider.embed_documents([chunk("aaa", 0)])
        assert vector == pytest.approx([0.1, 0.2, 0.3, 0.4], abs=1e-3)

    def test_multiple_shards_merged(self, tmp_path):
        write_embeddings(tmp_path, [("aaa", 0, VEC_A0)], index=0)
        write_embeddings(tmp_path, [("bbb", 0, VEC_B0)], index=1)
        provider = provider_for(tmp_path)
        assert provider.embed_documents([chunk("bbb", 0)]) == [VEC_B0]

    def test_embedding_identity_is_machine_readable(self, tmp_path):
        provider = standard_corpus(tmp_path)
        assert provider.model_id == "test-model"
        assert provider.normalized is True
        assert provider.dimension == 4


class TestFailures:
    def test_missing_vector_is_a_loud_error(self, tmp_path):
        provider = standard_corpus(tmp_path)
        with pytest.raises(EmbeddingError, match="aaa.*position 7"):
            provider.embed_documents([chunk("aaa", 7)])

    def test_no_embeddings_files(self, tmp_path):
        (tmp_path / "empty").mkdir()
        with pytest.raises(EmbeddingError, match="no embeddings parquet"):
            provider_for(tmp_path / "empty")

    def test_inconsistent_dimensions_rejected_at_init(self, tmp_path):
        write_embeddings(
            tmp_path, [("aaa", 0, [0.5, 0.5, 0.5, 0.5]), ("bbb", 0, [0.5, 0.5])]
        )
        with pytest.raises(EmbeddingError, match="inconsistent embedding dimensions"):
            provider_for(tmp_path)

    def test_cross_file_duplicate_keeps_first_and_counts(self, tmp_path):
        write_embeddings(tmp_path, [("aaa", 0, VEC_A0)], index=0)
        write_embeddings(tmp_path, [("aaa", 0, VEC_A1), ("bbb", 0, VEC_B0)], index=1)
        provider = provider_for(tmp_path)
        assert provider.duplicate_vectors_skipped == 1
        # the first file's vector is the one served
        assert provider.embed_documents([chunk("aaa", 0)])[0] == VEC_A0
        assert provider.embed_documents([chunk("bbb", 0)])[0] == VEC_B0

    def test_duplicate_key_rejected_at_init(self, tmp_path):
        write_embeddings(tmp_path, [("aaa", 0, VEC_A0), ("aaa", 0, VEC_A1)])
        with pytest.raises(EmbeddingError, match="duplicate vector"):
            provider_for(tmp_path)

    def test_null_key_rejected_at_init(self, tmp_path):
        write_embeddings(tmp_path, [("aaa", None, VEC_A0)])
        with pytest.raises(EmbeddingError, match="null record_id/chunk_idx"):
            provider_for(tmp_path)

    def test_missing_column_named_in_error(self, tmp_path):
        directory = tmp_path / "owie-dataset" / PARTITION
        directory.mkdir(parents=True)
        pq.write_table(
            pa.table({"record_id": pa.array(["aaa"], pa.string())}),
            directory / "metadata_0_embeddings.parquet",
        )
        with pytest.raises(EmbeddingError, match="missing required column"):
            provider_for(tmp_path)

    def test_embed_query_unavailable_without_a_delegate(self, tmp_path):
        provider = standard_corpus(tmp_path)
        with pytest.raises(NotImplementedError, match="no query encoder"):
            provider.embed_query("una consulta")

    def test_null_vector_rejected_at_init(self, tmp_path):
        # a null embedding cell must never silently become None downstream
        write_embeddings(
            tmp_path, [("aaa", 0, VEC_A0), ("bbb", 0, None), ("ccc", 0, VEC_B0)]
        )
        with pytest.raises(EmbeddingError, match="null embedding value"):
            provider_for(tmp_path)

    def test_null_element_inside_vector_rejected_at_init(self, tmp_path):
        write_embeddings(tmp_path, [("aaa", 0, [0.5, None, 0.5, 0.5])])
        with pytest.raises(EmbeddingError, match="null elements"):
            provider_for(tmp_path)

    def test_all_zero_dimensional_vectors_rejected(self, tmp_path):
        write_embeddings(tmp_path, [("aaa", 0, []), ("bbb", 0, [])])
        with pytest.raises(EmbeddingError, match="zero-dimensional"):
            provider_for(tmp_path)

    def test_mixed_schema_shards_rejected_with_named_files(self, tmp_path):
        write_embeddings(tmp_path, [("aaa", 0, VEC_A0)], index=0)
        # second shard drifts to fp32 vectors
        directory = tmp_path / "owie-dataset" / PARTITION
        pq.write_table(
            pa.table(
                {
                    "record_id": pa.array(["bbb"], pa.string()),
                    "chunk_idx": pa.array([0], pa.int32()),
                    "embedding": pa.array([VEC_B0], pa.list_(pa.float32())),
                }
            ),
            directory / "metadata_1_embeddings.parquet",
        )
        with pytest.raises(EmbeddingError, match="schema differs"):
            provider_for(tmp_path)

    def test_duplicate_key_across_partitions_keeps_first(self, tmp_path):
        # recrawled documents republish identical content under the same
        # content-hash id on later days; the provider keeps the first
        # vector and counts the rest (same-file duplicates still raise)
        write_embeddings(tmp_path, [("aaa", 0, VEC_A0)], index=0)
        write_embeddings(
            tmp_path, [("aaa", 0, VEC_A1)], index=0,
            partition="year=2026/month=8/day=13/language=spa",
        )
        provider = provider_for(tmp_path)
        assert provider.duplicate_vectors_skipped == 1
        assert provider.embed_documents([chunk("aaa", 0)])[0] == VEC_A0


    def test_zero_row_corpus_rejected(self, tmp_path):
        write_embeddings(tmp_path, [])
        with pytest.raises(EmbeddingError, match="empty"):
            provider_for(tmp_path)

    def test_corrupt_file_rejected_with_named_file(self, tmp_path):
        directory = tmp_path / "owie-dataset" / PARTITION
        directory.mkdir(parents=True)
        (directory / "metadata_0_embeddings.parquet").write_bytes(b"not parquet")
        with pytest.raises(EmbeddingError, match="unreadable"):
            provider_for(tmp_path)


class StubQueryEncoder:
    def __init__(self, model_id="test-model", dimension=4, normalized=True):
        self.model_id = model_id
        self.normalized = normalized
        self.device = "cpu"
        self._dimension = dimension
        self.queries = []

    def embed_query(self, text):
        self.queries.append(text)
        return [0.5] * self._dimension


class TestQueryDelegation:
    def test_delegate_built_lazily_and_reused(self, tmp_path):
        built = []

        def factory():
            built.append(True)
            return StubQueryEncoder()

        write_embeddings(tmp_path, [("aaa", 0, VEC_A0)])
        provider = PrecomputedEmbeddingProvider(
            tmp_path, model_id="test-model", query_encoder_factory=factory
        )
        assert built == []  # ingest-time cost: zero
        assert provider.embed_query("q1") == [0.5] * 4
        provider.embed_query("q2")
        assert built == [True]

    def test_mismatched_delegate_model_refused(self, tmp_path):
        write_embeddings(tmp_path, [("aaa", 0, VEC_A0)])
        provider = PrecomputedEmbeddingProvider(
            tmp_path,
            model_id="corpus-model",
            query_encoder_factory=lambda: StubQueryEncoder("other-model"),
        )
        with pytest.raises(EmbeddingError, match="refusing to mix"):
            provider.embed_query("q")

    def test_mismatched_delegate_normalization_refused(self, tmp_path):
        write_embeddings(tmp_path, [("aaa", 0, VEC_A0)])
        provider = PrecomputedEmbeddingProvider(
            tmp_path,
            model_id="test-model",
            query_encoder_factory=lambda: StubQueryEncoder(normalized=False),
        )
        with pytest.raises(EmbeddingError, match="normalization"):
            provider.embed_query("q")

    def test_mismatched_query_dimension_refused(self, tmp_path):
        write_embeddings(tmp_path, [("aaa", 0, VEC_A0)])
        provider = PrecomputedEmbeddingProvider(
            tmp_path,
            model_id="test-model",
            query_encoder_factory=lambda: StubQueryEncoder(dimension=3),
        )
        with pytest.raises(EmbeddingError, match="dimension 3"):
            provider.embed_query("q")
