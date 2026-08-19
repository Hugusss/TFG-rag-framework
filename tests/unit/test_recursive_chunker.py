"""Unit tests for the recursive chunking strategy."""

from rag_framework.chunking.recursive import RecursiveChunker
from rag_framework.models import Document, make_chunk_id


def doc(text, doc_id="doc1", **extra):
    metadata = {"url": "https://example.org", "language": "spa"}
    metadata.update(extra)
    return Document(document_id=doc_id, text=text, metadata=metadata)


def words(n, prefix="w"):
    return " ".join(f"{prefix}{i}" for i in range(n))


def split(chunker, *documents):
    return list(chunker.split(list(documents)))


def make(target=10, overlap=3, minimum=2):
    return RecursiveChunker(
        target_tokens=target, overlap_tokens=overlap, minimum_tokens=minimum
    )


class TestPacking:
    def test_short_document_is_one_chunk(self):
        chunks = split(make(), doc(words(6)))
        assert len(chunks) == 1
        assert chunks[0].text == words(6)
        assert chunks[0].position == 0

    def test_paragraphs_pack_up_to_target(self):
        text = f"{words(4, 'a')}\n\n{words(4, 'b')}\n\n{words(4, 'c')}"
        chunks = split(make(target=10, overlap=3, minimum=2), doc(text))
        # a+b fit in 10; c starts a new chunk — the paragraph boundary
        # is retained, not split mid-paragraph
        assert len(chunks) == 2
        assert chunks[0].text == f"{words(4, 'a')} {words(4, 'b')}"
        assert chunks[1].text == words(4, "c")

    def test_whitespace_normalized_within_paragraphs(self):
        chunks = split(make(), doc("uno   dos\t tres\ncuatro"))
        assert chunks[0].text == "uno dos tres cuatro"

    def test_oversized_paragraph_splits_with_exact_overlap(self):
        chunks = split(make(target=10, overlap=3, minimum=2), doc(words(24)))
        texts = [c.text.split() for c in chunks]
        assert [len(t) for t in texts] == [10, 10, 10]  # strides 0,7,14
        assert texts[0][-3:] == texts[1][:3]  # overlap verbatim
        assert texts[1][-3:] == texts[2][:3]
        assert texts[2][-1] == "w23"  # tail covered

    def test_window_coverage_is_complete(self):
        chunks = split(make(target=10, overlap=3, minimum=1), doc(words(25)))
        seen = {w for c in chunks for w in c.text.split()}
        assert seen == set(words(25).split())


class TestMinimumFilter:
    def test_below_minimum_tail_window_dropped_and_counted(self):
        chunker = make(target=10, overlap=3, minimum=5)
        # 25 words, stride 7: windows 10,10,10 then a 4-word tail
        chunks = split(chunker, doc(words(25)))
        assert [len(c.text.split()) for c in chunks] == [10, 10, 10]
        assert chunker.chunks_dropped_below_minimum == 1

    def test_no_spurious_tail_when_a_window_reaches_the_end(self):
        chunker = make(target=10, overlap=3, minimum=5)
        # 17 words: windows w0-9 and w7-16 — the second reaches the
        # end, so no fragment tail exists and nothing is dropped
        chunks = split(chunker, doc(words(17)))
        assert [len(c.text.split()) for c in chunks] == [10, 10]
        assert chunker.chunks_dropped_below_minimum == 0

    def test_small_paragraphs_pack_across_boundaries(self):
        # two small paragraphs fitting one target become ONE chunk
        chunker = make(target=10, overlap=3, minimum=2)
        text = f"{words(3, 'b')}\n\n{words(6, 'c')}"
        (chunk,) = split(chunker, doc(text))
        assert chunk.text == f"{words(3, 'b')} {words(6, 'c')}"

    def test_positions_do_not_depend_on_the_filter(self):
        # same document, two minimum settings: surviving chunks keep
        # identical (position, chunk_id) — the filter never renumbers
        text = words(25)
        strict = split(make(target=10, overlap=3, minimum=5), doc(text))
        loose = split(make(target=10, overlap=3, minimum=1), doc(text))
        loose_by_position = {c.position: c.chunk_id for c in loose}
        assert len(loose) == len(strict) + 1
        for chunk in strict:
            assert loose_by_position[chunk.position] == chunk.chunk_id

    def test_document_below_minimum_counts_without_chunks(self, caplog):
        import logging

        chunker = make(minimum=5)
        with caplog.at_level(logging.WARNING):
            assert split(chunker, doc("una sola palabra")) == []
        assert chunker.documents_without_chunks == 1
        assert "produced no chunks" in caplog.text  # contract: counted AND logged

    def test_empty_document_counts_without_chunks(self):
        chunker = make()
        assert split(chunker, doc("   \n\n  ")) == []
        assert chunker.documents_without_chunks == 1


class TestContract:
    def test_ids_use_canonical_scheme_and_are_distinct(self):
        chunks = split(make(target=5, overlap=1, minimum=1), doc(words(12)))
        assert chunks[0].chunk_id == make_chunk_id("doc1", 0, chunks[0].text)
        assert len({c.chunk_id for c in chunks}) == len(chunks)

    def test_deterministic_across_runs(self):
        first = [(c.chunk_id, c.text) for c in split(make(), doc(words(30)))]
        second = [(c.chunk_id, c.text) for c in split(make(), doc(words(30)))]
        assert first == second

    def test_metadata_inherited_and_publisher_keys_discarded(self):
        document = doc(
            words(6),
            chunk_offsets=[(0, 5)],
            start=0,
            end=5,
            clipped=True,
            title="a title",
        )
        (chunk,) = split(make(), document)
        assert chunk.metadata["url"] == "https://example.org"
        assert chunk.metadata["title"] == "a title"
        for key in ("chunk_offsets", "start", "end", "clipped"):
            assert key not in chunk.metadata

    def test_counters_reset_between_calls(self):
        chunker = make(minimum=5)
        split(chunker, doc("corto"))
        assert chunker.documents_without_chunks == 1
        split(chunker, doc(words(8)))
        assert chunker.documents_without_chunks == 0
        assert chunker.documents_processed == 1

    def test_streaming(self):
        chunker = make()
        iterator = chunker.split(doc(words(5)) for _ in range(1))
        assert chunker.documents_processed == 0
        list(iterator)
        assert chunker.documents_processed == 1

    def test_zero_overlap_tiles_without_duplication(self):
        chunker = make(target=5, overlap=0, minimum=1)
        chunks = split(chunker, doc(words(12)))
        assert [len(c.text.split()) for c in chunks] == [5, 5, 2]
        all_words = [w for c in chunks for w in c.text.split()]
        assert len(all_words) == len(set(all_words)) == 12

    def test_crlf_paragraph_breaks_recognized(self):
        text = f"{words(4, 'a')}\r\n\r\n{words(4, 'b')}"
        chunks = split(make(target=5, overlap=1, minimum=1), doc(text))
        # two paragraphs that cannot pack into one 5-word target
        assert [c.text for c in chunks] == [words(4, "a"), words(4, "b")]

    def test_invalid_geometry_raises_at_construction(self):
        import pytest

        with pytest.raises(ValueError, match="geometry"):
            RecursiveChunker(
                target_tokens=10, overlap_tokens=10, minimum_tokens=1
            )
