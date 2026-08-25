"""Tests for the ChromaDB vector-store adapter (temporary directories)."""

import math

import pytest

from rag_framework.models import Chunk
from rag_framework.vectorstores.base import VectorStoreError
from rag_framework.vectorstores.chroma import ChromaVectorStore

IDENTITY = {"model_id": "test-model", "dimension": 2, "normalized": True}


def chunk(doc_id, position, text=None, **extra):
    metadata = {"language": "spa", "start": 0, "end": 5}
    metadata.update(extra)
    return Chunk(
        chunk_id=f"c-{doc_id}-{position}",
        document_id=doc_id,
        text=text or f"text of {doc_id}/{position}",
        position=position,
        metadata=metadata,
    )


def norm(vector):
    length = math.sqrt(sum(x * x for x in vector))
    return [x / length for x in vector]


def open_store(tmp_path, name="col", metadata=IDENTITY):
    store = ChromaVectorStore(tmp_path / "chroma", collection_metadata=metadata)
    store.create_or_open(name)
    return store


def seeded_store(tmp_path):
    store = open_store(tmp_path)
    store.add(
        [chunk("a", 0), chunk("b", 0), chunk("c", 0)],
        [norm([1.0, 0.0]), norm([0.0, 1.0]), norm([0.7, 0.7])],
    )
    return store


class TestLifecycle:
    def test_create_then_count_zero(self, tmp_path):
        assert open_store(tmp_path).count() == 0

    def test_add_and_count(self, tmp_path):
        assert seeded_store(tmp_path).count() == 3

    def test_re_add_same_ids_is_idempotent(self, tmp_path):
        store = seeded_store(tmp_path)
        store.add([chunk("a", 0)], [norm([1.0, 0.0])])
        assert store.count() == 3

    def test_persistence_across_reopen(self, tmp_path):
        seeded_store(tmp_path)
        again = open_store(tmp_path)
        assert again.count() == 3

    def test_reset_empties_but_stays_usable(self, tmp_path):
        store = seeded_store(tmp_path)
        store.reset()
        assert store.count() == 0
        store.add([chunk("z", 0)], [norm([1.0, 1.0])])
        assert store.count() == 1

    def test_methods_require_create_or_open(self, tmp_path):
        store = ChromaVectorStore(tmp_path / "chroma")
        with pytest.raises(VectorStoreError, match="create_or_open"):
            store.count()

    def test_incompatible_embedding_identity_refused(self, tmp_path):
        seeded_store(tmp_path)
        other = ChromaVectorStore(
            tmp_path / "chroma",
            collection_metadata={**IDENTITY, "model_id": "another-model"},
        )
        with pytest.raises(VectorStoreError, match="refusing to mix"):
            other.create_or_open("col")

    def test_non_cosine_collection_refused(self, tmp_path):
        # fixture setup only: an externally created index; settings must
        # match the adapter's because chroma caches one client per path
        import chromadb

        client = chromadb.PersistentClient(
            path=str(tmp_path / "chroma"),
            settings=chromadb.config.Settings(anonymized_telemetry=False),
        )
        client.create_collection("foreign")  # chroma default space: l2
        store = ChromaVectorStore(tmp_path / "chroma")
        with pytest.raises(VectorStoreError, match="distance space"):
            store.create_or_open("foreign")

    def test_collection_without_identity_refused_when_expected(self, tmp_path):
        import chromadb

        client = chromadb.PersistentClient(
            path=str(tmp_path / "chroma"),
            settings=chromadb.config.Settings(anonymized_telemetry=False),
        )
        client.create_collection("legacy", metadata={"hnsw:space": "cosine"})
        store = ChromaVectorStore(tmp_path / "chroma", collection_metadata=IDENTITY)
        with pytest.raises(VectorStoreError, match="no recorded model_id"):
            store.create_or_open("legacy")

    def test_store_without_expectations_opens_anything(self, tmp_path):
        seeded_store(tmp_path)
        inspector = ChromaVectorStore(tmp_path / "chroma")  # no expectations
        inspector.create_or_open("col")
        assert inspector.count() == 3


class TestSearch:
    def test_scores_are_similarity_higher_is_better(self, tmp_path):
        store = seeded_store(tmp_path)
        results = store.search(norm([1.0, 0.0]), k=3)
        assert [r.chunk_id for r in results] == ["c-a-0", "c-c-0", "c-b-0"]
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
        assert scores[0] == pytest.approx(1.0, abs=1e-5)
        assert scores[1] == pytest.approx(math.cos(math.pi / 4), abs=1e-3)
        assert scores[2] == pytest.approx(0.0, abs=1e-5)

    def test_results_are_traceable(self, tmp_path):
        store = seeded_store(tmp_path)
        top = store.search(norm([1.0, 0.0]), k=1)[0]
        assert top.document_id == "a"
        assert top.text == "text of a/0"
        assert top.metadata["position"] == 0
        assert top.metadata["language"] == "spa"
        assert top.metadata["start"] == 0

    def test_k_larger_than_collection(self, tmp_path):
        store = seeded_store(tmp_path)
        assert len(store.search(norm([1.0, 0.0]), k=50)) == 3

    def test_search_on_empty_collection(self, tmp_path):
        assert open_store(tmp_path).search(norm([1.0, 0.0]), k=5) == []

    def test_equality_filter(self, tmp_path):
        store = open_store(tmp_path)
        store.add(
            [chunk("a", 0, language="spa"), chunk("b", 0, language="cat")],
            [norm([1.0, 0.0]), norm([1.0, 0.1])],
        )
        results = store.search(norm([1.0, 0.0]), k=5, filters={"language": "cat"})
        assert [r.document_id for r in results] == ["b"]

    def test_multi_key_filter(self, tmp_path):
        store = open_store(tmp_path)
        store.add(
            [chunk("a", 0, language="spa"), chunk("a", 1, language="spa")],
            [norm([1.0, 0.0]), norm([0.0, 1.0])],
        )
        results = store.search(
            norm([1.0, 0.0]), k=5, filters={"language": "spa", "position": 1}
        )
        assert [r.chunk_id for r in results] == ["c-a-1"]


class TestAddValidation:
    def test_length_mismatch_rejected(self, tmp_path):
        store = open_store(tmp_path)
        with pytest.raises(VectorStoreError, match="1 chunks but 2"):
            store.add([chunk("a", 0)], [[1.0, 0.0], [0.0, 1.0]])

    def test_duplicate_ids_in_batch_rejected(self, tmp_path):
        store = open_store(tmp_path)
        with pytest.raises(VectorStoreError, match="duplicate chunk ids"):
            store.add(
                [chunk("a", 0), chunk("a", 0)],
                [norm([1.0, 0.0]), norm([0.0, 1.0])],
            )

    def test_internal_batching_preserves_all_rows(self, tmp_path):
        store = open_store(tmp_path)
        store._max_batch = 2  # force several backend batches
        chunks = [chunk("d", i) for i in range(5)]
        vectors = [norm([1.0, float(i)]) for i in range(5)]
        store.add(chunks, vectors)
        assert store.count() == 5

    def test_empty_add_is_a_no_op(self, tmp_path):
        store = open_store(tmp_path)
        store.add([], [])
        assert store.count() == 0

    def test_non_scalar_metadata_rejected(self, tmp_path):
        store = open_store(tmp_path)
        bad = chunk("a", 0, offsets_list=[1, 2])
        with pytest.raises(VectorStoreError, match="non-scalar"):
            store.add([bad], [norm([1.0, 0.0])])

    def test_reserved_metadata_key_rejected(self, tmp_path):
        store = open_store(tmp_path)
        bad = chunk("a", 0, document_id="impostor")
        with pytest.raises(VectorStoreError, match="reserved"):
            store.add([bad], [norm([1.0, 0.0])])

    def test_backend_exception_wrapped_as_store_error(self, tmp_path):
        store = seeded_store(tmp_path)  # 2-d collection
        with pytest.raises(VectorStoreError, match="add failed"):
            store.add([chunk("z", 0)], [[1.0, 0.0, 0.0]])  # 3-d vector


class TestIterVectors:
    def test_exports_every_row_with_provenance(self, tmp_path):
        store = seeded_store(tmp_path)
        rows = list(store.iter_vectors())
        assert sorted(cid for cid, _, _ in rows) == ["c-a-0", "c-b-0", "c-c-0"]
        by_id = {cid: (doc, vec) for cid, doc, vec in rows}
        assert by_id["c-a-0"][0] == "a"
        assert by_id["c-a-0"][1] == pytest.approx([1.0, 0.0])
        assert all(isinstance(x, float) for _, _, vec in rows for x in vec)

    def test_pages_through_large_collections(self, tmp_path, monkeypatch):
        store = seeded_store(tmp_path)
        store._max_batch = 2  # force two pages for three rows
        assert len(list(store.iter_vectors())) == 3

    def test_requires_open_collection(self, tmp_path):
        store = ChromaVectorStore(tmp_path / "chroma")
        with pytest.raises(VectorStoreError, match="create_or_open"):
            list(store.iter_vectors())


class TestEfSearch:
    def read_ef(self, store):
        return ((getattr(store._collection, "configuration_json", None) or {})
                .get("hnsw") or {}).get("ef_search")

    def test_default_backend_ef_untouched(self, tmp_path):
        assert self.read_ef(seeded_store(tmp_path)) == 100

    def test_ef_applied_on_create_and_on_reopen(self, tmp_path):
        store = ChromaVectorStore(
            tmp_path / "chroma", collection_metadata=IDENTITY, ef_search=384
        )
        store.create_or_open("col")
        assert self.read_ef(store) == 384
        again = ChromaVectorStore(
            tmp_path / "chroma", collection_metadata=IDENTITY, ef_search=512
        )
        again.create_or_open("col")  # existing collection: modify + reopen
        assert self.read_ef(again) == 512
        assert again.count() == 0

    def test_search_still_correct_with_custom_ef(self, tmp_path):
        store = ChromaVectorStore(
            tmp_path / "chroma", collection_metadata=IDENTITY, ef_search=256
        )
        store.create_or_open("col")
        store.add(
            [chunk("a", 0), chunk("b", 0)], [norm([1.0, 0.0]), norm([0.0, 1.0])]
        )
        results = store.search(norm([1.0, 0.1]), 2)
        assert [r.document_id for r in results] == ["a", "b"]
