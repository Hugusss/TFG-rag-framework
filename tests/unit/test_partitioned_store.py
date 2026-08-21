"""Tests for the partitioned store: routing and merge with stub inner
stores, plus layout guards on real Chroma collections."""

import math

import pytest

from rag_framework.models import Chunk, SearchResult
from rag_framework.retrieval.partitioning import partition_for
from rag_framework.vectorstores.base import VectorStoreError
from rag_framework.vectorstores.chroma import ChromaVectorStore
from rag_framework.vectorstores.partitioned import (
    PartitionedVectorStore,
    partition_collection_name,
)


def chunk(doc_id, position):
    return Chunk(
        chunk_id=f"c-{doc_id}-{position}",
        document_id=doc_id,
        text=f"text {doc_id}/{position}",
        position=position,
        metadata={},
    )


class StubStore:
    """Records calls; searches by dot product over what was added."""

    def __init__(self, extra):
        self.extra = extra
        self.opened = None
        self.rows = {}
        self.adds = []
        self.resets = 0

    def create_or_open(self, name):
        self.opened = name

    def add(self, chunks, embeddings):
        self.adds.append((list(chunks), list(embeddings)))
        for c, v in zip(chunks, embeddings):
            self.rows[c.chunk_id] = (c, v)

    def search(self, query_embedding, k, filters=None):
        scored = [
            SearchResult(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                text=c.text,
                score=sum(a * b for a, b in zip(query_embedding, v)),
                metadata={},
            )
            for c, v in self.rows.values()
        ]
        return sorted(scored, key=lambda r: -r.score)[:k]

    def count(self):
        return len(self.rows)

    def reset(self):
        self.resets += 1
        self.rows = {}

    def iter_vectors(self):
        for c, v in self.rows.values():
            yield c.chunk_id, c.document_id, list(v)


def stub_store(partitions):
    made = []

    def factory(extra):
        store = StubStore(extra)
        made.append(store)
        return store

    store = PartitionedVectorStore(factory, partitions)
    return store, made


class TestLayout:
    @pytest.mark.parametrize("bad", [0, -2, 2.0, True])
    def test_invalid_partition_count(self, bad):
        with pytest.raises(VectorStoreError, match="partitions"):
            PartitionedVectorStore(lambda extra: StubStore(extra), bad)

    def test_factory_receives_layout_and_names_encode_it(self):
        store, inner = stub_store(4)
        store.create_or_open("col")
        assert [s.extra for s in inner] == [
            {"partitions": 4, "partition_index": i} for i in range(4)
        ]
        assert [s.opened for s in inner] == [
            "col-p00of04", "col-p01of04", "col-p02of04", "col-p03of04"
        ]
        assert partition_collection_name("x", 3, 16) == "x-p03of16"
        assert store.partitions == 4


class TestAdd:
    def test_routes_whole_documents_with_their_own_vectors(self):
        store, inner = stub_store(3)
        store.create_or_open("col")
        chunks = [chunk(f"d{i % 4}", i) for i in range(12)]
        vectors = [[float(i), 0.0] for i in range(12)]

        store.add(chunks, vectors)

        assert store.count() == 12
        for index, s in enumerate(inner):
            for c, v in s.rows.values():
                assert partition_for(c.document_id, 3) == index
                assert v == [float(c.position), 0.0]  # pairing kept
        assert sum(len(s.adds) for s in inner) <= 3  # one add per partition

    def test_empty_partitions_get_no_add_call(self):
        store, inner = stub_store(8)
        store.create_or_open("col")
        store.add([chunk("only", 0)], [[1.0]])
        assert sum(len(s.adds) for s in inner) == 1

    def test_length_mismatch_and_duplicates_rejected(self):
        store, _ = stub_store(2)
        store.create_or_open("col")
        with pytest.raises(VectorStoreError, match="2 chunks but 1"):
            store.add([chunk("a", 0), chunk("a", 1)], [[1.0]])
        with pytest.raises(VectorStoreError, match="duplicate chunk ids"):
            store.add([chunk("a", 0), chunk("a", 0)], [[1.0], [2.0]])


class TestSearch:
    def seeded(self, partitions=4):
        store, inner = stub_store(partitions)
        store.create_or_open("col")
        chunks = [chunk(f"d{i}", 0) for i in range(10)]
        angles = [i / 10 * math.pi / 2 for i in range(10)]
        vectors = [[math.cos(a), math.sin(a)] for a in angles]
        store.add(chunks, vectors)
        return store, inner

    def test_search_partition_stamps_provenance(self):
        store, inner = self.seeded()
        for index in range(4):
            hits = store.search_partition(index, [1.0, 0.0], 3)
            assert len(hits) <= 3 and len(hits) <= inner[index].count()
            assert all(h.partition_id == str(index) for h in hits)
        with pytest.raises(VectorStoreError, match="out of range"):
            store.search_partition(4, [1.0, 0.0], 3)

    def test_search_is_global_top_k_over_all_partitions(self):
        store, _ = self.seeded()
        hits = store.search([1.0, 0.0], 4)
        # d0..d3 are the closest to the x axis, in that order
        assert [h.chunk_id for h in hits] == ["c-d0-0", "c-d1-0", "c-d2-0", "c-d3-0"]
        assert {h.partition_id for h in hits} == {
            str(partition_for(f"d{i}", 4)) for i in range(4)
        }

    def test_iter_vectors_chains_every_partition(self):
        store, _ = self.seeded()
        exported = list(store.iter_vectors())
        assert len(exported) == 10
        assert {cid for cid, _, _ in exported} == {f"c-d{i}-0" for i in range(10)}
        assert all(len(vector) == 2 for _, _, vector in exported)

    def test_counts_and_reset_cover_every_partition(self):
        store, inner = self.seeded()
        assert sum(store.partition_counts()) == store.count() == 10
        store.reset()
        assert store.count() == 0
        assert all(s.resets == 1 for s in inner)


IDENTITY = {"model_id": "test-model", "dimension": 2, "normalized": True}


def norm(vector):
    length = math.sqrt(sum(x * x for x in vector))
    return [x / length for x in vector]


class TestOnChroma:
    def chroma_store(self, tmp_path, partitions):
        return PartitionedVectorStore(
            lambda extra: ChromaVectorStore(
                tmp_path / "chroma", collection_metadata={**IDENTITY, **extra}
            ),
            partitions,
        )

    def test_ingest_search_and_reopen(self, tmp_path):
        store = self.chroma_store(tmp_path, 2)
        store.create_or_open("col")
        chunks = [chunk(f"d{i}", 0) for i in range(6)]
        vectors = [norm([1.0, i / 5]) for i in range(6)]
        store.add(chunks, vectors)
        assert store.count() == 6
        assert sum(store.partition_counts()) == 6

        hits = store.search(norm([1.0, 0.0]), 3)
        assert [h.chunk_id for h in hits] == ["c-d0-0", "c-d1-0", "c-d2-0"]
        assert all(h.partition_id in ("0", "1") for h in hits)
        merged_by_hand = sorted(
            store.search_partition(0, norm([1.0, 0.0]), 3)
            + store.search_partition(1, norm([1.0, 0.0]), 3),
            key=lambda h: (-h.score, h.chunk_id),
        )[:3]
        assert [h.chunk_id for h in merged_by_hand] == [h.chunk_id for h in hits]

        again = self.chroma_store(tmp_path, 2)
        again.create_or_open("col")  # idempotent reopen, same layout
        assert again.count() == 6

    def test_wrong_layout_is_refused(self, tmp_path):
        store = self.chroma_store(tmp_path, 2)
        store.create_or_open("col")
        # someone points a differently laid-out store at an existing
        # partition collection: index 1 of 2 opened as index 0 of 2
        impostor = ChromaVectorStore(
            tmp_path / "chroma",
            collection_metadata={**IDENTITY, "partitions": 2, "partition_index": 0},
        )
        with pytest.raises(VectorStoreError, match="partition_index"):
            impostor.create_or_open("col-p01of02")
