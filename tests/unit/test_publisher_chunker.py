"""Unit tests for the publisher_offsets chunking strategy."""

import logging

from rag_framework.chunking.publisher_offsets import (
    MAX_OVERHANG,
    PublisherOffsetsChunker,
)
from rag_framework.models import Document, make_chunk_id


def make_doc(doc_id="doc1", text="0123456789ABCDEF", offsets=((0, 5), (5, 10)), **extra):
    metadata = {"url": f"https://example.org/{doc_id}", "language": "spa"}
    if offsets is not None:
        metadata["chunk_offsets"] = [tuple(w) for w in offsets]
    metadata.update(extra)
    return Document(document_id=doc_id, text=text, metadata=metadata)


def split(*documents):
    chunker = PublisherOffsetsChunker()
    chunks = list(chunker.split(list(documents)))
    return chunker, chunks


class TestCutting:
    def test_windows_become_byte_exact_slices(self):
        _, chunks = split(make_doc())
        assert [c.text for c in chunks] == ["01234", "56789"]
        assert [c.position for c in chunks] == [0, 1]
        assert all(c.document_id == "doc1" for c in chunks)

    def test_html_passes_through_untouched(self):
        html = "<h2><a href='x'>t&iacute;tulo</a></h2> body"
        _, chunks = split(make_doc(text=html, offsets=[(0, len(html))]))
        assert chunks[0].text == html

    def test_chunk_ids_use_the_canonical_scheme(self):
        _, chunks = split(make_doc())
        assert chunks[0].chunk_id == make_chunk_id("doc1", 0, "01234")
        assert chunks[1].chunk_id == make_chunk_id("doc1", 1, "56789")

    def test_deterministic_across_runs(self):
        first = [(c.chunk_id, c.text) for c in split(make_doc())[1]]
        second = [(c.chunk_id, c.text) for c in split(make_doc())[1]]
        assert first == second

    def test_identical_text_at_two_positions_gets_distinct_ids(self):
        _, chunks = split(make_doc(offsets=[(0, 5), (0, 5)]))
        assert chunks[0].text == chunks[1].text
        assert chunks[0].chunk_id != chunks[1].chunk_id

    def test_metadata_inherited_without_offsets_list(self):
        _, chunks = split(make_doc(title="a title"))
        metadata = chunks[0].metadata
        assert metadata["url"] == "https://example.org/doc1"
        assert metadata["title"] == "a title"
        assert metadata["start"] == 0 and metadata["end"] == 5
        assert "chunk_offsets" not in metadata
        assert "clipped" not in metadata


class TestOverhangPolicy:
    def test_overhang_within_limit_is_clipped_and_counted(self):
        text = "0123456789"  # length 10
        chunker, chunks = split(
            make_doc(text=text, offsets=[(0, 5), (5, 10 + MAX_OVERHANG)])
        )
        assert chunks[1].text == "56789"
        assert chunks[1].metadata["end"] == 10
        assert chunks[1].metadata["clipped"] is True
        assert chunker.windows_clipped == 1
        assert chunker.rejections == []

    def test_overhang_beyond_limit_is_rejected(self):
        text = "0123456789"
        chunker, chunks = split(
            make_doc(text=text, offsets=[(0, 5), (5, 10 + MAX_OVERHANG + 1)])
        )
        assert [c.position for c in chunks] == [0]
        (rejection,) = chunker.rejections
        assert "overhangs" in rejection.reason
        assert chunker.windows_clipped == 0

    def test_clipped_chunk_id_hashes_the_clipped_text(self):
        text = "0123456789"
        _, chunks = split(make_doc(text=text, offsets=[(5, 12)]))
        assert chunks[0].chunk_id == make_chunk_id("doc1", 0, "56789")


class TestRejectionsAndPositions:
    def test_rejected_window_does_not_renumber_successors(self):
        chunker, chunks = split(
            make_doc(offsets=[(0, 5), (99, 120), (5, 10)])
        )
        # the middle window dies; the third keeps its publisher index
        assert [c.position for c in chunks] == [0, 2]
        (rejection,) = chunker.rejections
        assert "#window1" in rejection.source

    def test_degenerate_window_rejected(self):
        chunker, chunks = split(make_doc(offsets=[(5, 5), (0, 5)]))
        assert [c.position for c in chunks] == [1]
        (rejection,) = chunker.rejections
        assert "degenerate" in rejection.reason

    def test_start_outside_text_rejected(self):
        chunker, _ = split(make_doc(text="0123", offsets=[(4, 8)]))
        (rejection,) = chunker.rejections
        assert "outside text" in rejection.reason

    def test_malformed_window_rejected(self):
        chunker, chunks = split(make_doc(offsets=[(0, 5), ("a", "b")]))
        assert [c.position for c in chunks] == [0]
        (rejection,) = chunker.rejections
        assert "malformed" in rejection.reason

    def test_float_window_bounds_rejected_not_truncated(self):
        chunker, chunks = split(make_doc(offsets=[(0, 5.7)]))
        assert chunks == []
        (rejection,) = chunker.rejections
        assert "malformed" in rejection.reason

    def test_negative_start_rejected(self):
        chunker, chunks = split(make_doc(offsets=[(-1, 5)]))
        assert chunks == []
        (rejection,) = chunker.rejections
        assert "outside text" in rejection.reason

    def test_whitespace_only_window_is_kept(self):
        # the publisher delimited it and an official embedding exists
        # for it; content judgment is not this strategy's job
        chunker, chunks = split(make_doc(text="ab   cd", offsets=[(2, 5)]))
        assert chunks[0].text == "   "
        assert chunker.rejections == []


class TestAccounting:
    def test_empty_offsets_count_document_without_chunks(self, caplog):
        chunker = PublisherOffsetsChunker()
        with caplog.at_level(logging.WARNING):
            chunks = list(chunker.split([make_doc(offsets=[]), make_doc(doc_id="doc2")]))
        assert len(chunks) == 2
        assert chunker.documents_processed == 2
        assert chunker.documents_without_chunks == 1
        assert "produced no chunks" in caplog.text

    def test_missing_offsets_key_is_a_rejection(self):
        chunker, chunks = split(make_doc(offsets=None))
        assert chunks == []
        (rejection,) = chunker.rejections
        assert rejection.source == "doc1"
        assert "no chunk_offsets metadata" in rejection.reason
        assert chunker.documents_rejected == 1
        assert chunker.documents_without_chunks == 0  # rejected, not zero-window

    def test_all_windows_rejected_counts_document_without_chunks(self):
        chunker, chunks = split(make_doc(text="0123", offsets=[(9, 12)]))
        assert chunks == []
        assert chunker.documents_without_chunks == 1
        assert len(chunker.rejections) == 1

    def test_counters_reset_between_split_calls(self):
        chunker = PublisherOffsetsChunker()
        list(
            chunker.split(
                [
                    make_doc(text="0123456789", offsets=[(5, 12)]),
                    make_doc(doc_id="doc2", offsets=[]),
                    make_doc(doc_id="doc3", text="0123", offsets=[(9, 12)]),
                ]
            )
        )
        assert chunker.windows_clipped == 1
        assert chunker.documents_without_chunks == 2
        assert len(chunker.rejections) == 1
        list(chunker.split([make_doc()]))
        assert chunker.windows_clipped == 0
        assert chunker.documents_processed == 1
        assert chunker.documents_without_chunks == 0
        assert chunker.rejections == []

    def test_reserved_document_metadata_keys_are_discarded(self):
        chunker, chunks = split(
            make_doc(clipped="doc-level", start=99, end=99)
        )
        metadata = chunks[0].metadata
        assert metadata["start"] == 0  # the chunker's value, not 99
        assert metadata["end"] == 5
        assert "clipped" not in metadata  # unclipped chunk stays unmarked

    def test_split_is_streaming(self):
        chunker = PublisherOffsetsChunker()
        iterator = chunker.split(make_doc() for _ in range(1))
        assert chunker.documents_processed == 0  # nothing consumed yet
        list(iterator)
        assert chunker.documents_processed == 1
