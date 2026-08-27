"""Unit tests for deterministic partition assignment and grouping."""

from collections import Counter

import pytest

from rag_framework.models import Chunk
from rag_framework.retrieval.base import RetrievalError
from rag_framework.retrieval.partitioning import group_by_partition, partition_for


class TestPartitionFor:
    def test_in_range_and_stable_across_calls(self):
        for p in (1, 2, 3, 4, 8, 16):
            values = {partition_for("some-document", p) for _ in range(5)}
            assert len(values) == 1
            assert 0 <= values.pop() < p

    def test_single_partition_takes_everything(self):
        assert all(partition_for(f"doc-{i}", 1) == 0 for i in range(50))

    def test_pinned_values_guard_the_hash_definition(self):
        # sha256(utf-8 id), first 8 bytes big-endian, mod P — changing
        # any of that reassigns every existing partitioned index, so
        # the constants are pinned here on purpose
        assert partition_for("abc", 4) == 2
        assert partition_for("abc", 8) == 2
        assert partition_for("abc", 16) == 10
        assert partition_for("é", 2) == 1  # non-ASCII ids hash as UTF-8
        owi_id = "271b3c2fb511e598ce3ee824a8ca99cad573bd6e4bf9ef213535487eb2937755"
        assert partition_for(owi_id, 8) == 6

    def test_not_python_hash(self, monkeypatch):
        # Python's str hash is salted per process; ours must not be
        before = partition_for("document", 8)
        monkeypatch.setattr("builtins.hash", lambda value: 12345)
        assert partition_for("document", 8) == before

    def test_roughly_balanced_on_many_ids(self):
        counts = Counter(partition_for(f"doc-{i}", 8) for i in range(10000))
        assert len(counts) == 8
        mean = 10000 / 8
        assert max(counts.values()) - min(counts.values()) < 0.15 * mean

    @pytest.mark.parametrize("bad", [0, -1, 2.0, True, "4"])
    def test_invalid_partition_count(self, bad):
        with pytest.raises(RetrievalError, match="partitions"):
            partition_for("doc", bad)

    def test_empty_document_id_refused(self):
        with pytest.raises(RetrievalError, match="document_id"):
            partition_for("", 4)


def chunk(doc_id, position):
    return Chunk(
        chunk_id=f"{doc_id}:{position}",
        document_id=doc_id,
        text="t",
        position=position,
        metadata={},
    )


class TestGroupByPartition:
    def test_documents_stay_whole_and_order_is_preserved(self):
        chunks = [chunk(f"doc-{i % 5}", i) for i in range(25)]
        groups = group_by_partition(chunks, 4)

        assert len(groups) == 4
        assert sum(len(g) for g in groups) == len(chunks)
        assert {c.chunk_id for g in groups for c in g} == {
            c.chunk_id for c in chunks
        }
        for group in groups:
            positions = [c.position for c in group]
            assert positions == sorted(positions)  # input order kept
        # a document's chunks are all in exactly one group
        for doc in {c.document_id for c in chunks}:
            owners = [i for i, g in enumerate(groups) if any(c.document_id == doc for c in g)]
            assert len(owners) == 1
            assert owners[0] == partition_for(doc, 4)

    def test_empty_input_gives_empty_groups(self):
        assert group_by_partition([], 3) == [[], [], []]
