"""Unit tests for the canonical data model and deterministic identifiers."""

import dataclasses

import pytest

from rag_framework.models import (
    Chunk,
    Document,
    RAGResult,
    SearchResult,
    make_chunk_id,
    make_document_id,
)


class TestMakeDocumentId:
    def test_deterministic(self):
        a = make_document_id("https://example.org/page")
        b = make_document_id("https://example.org/page")
        assert a == b

    def test_distinct_sources_get_distinct_ids(self):
        assert make_document_id("source-a") != make_document_id("source-b")

    def test_whitespace_is_normalized(self):
        assert make_document_id("  a \t b\n") == make_document_id("a b")

    def test_unicode_nfc_nfd_equivalence(self):
        composed = "caf\u00e9"    # "e with acute" as one code point (NFC)
        decomposed = "cafe\u0301"  # "e" + combining acute accent (NFD)
        assert composed != decomposed  # genuinely different code points
        assert make_document_id(composed) == make_document_id(decomposed)

    def test_is_sha256_hex(self):
        digest = make_document_id("x")
        assert len(digest) == 64
        int(digest, 16)  # raises ValueError if not hexadecimal

    def test_known_value_pins_the_scheme(self):
        # sha256("a"). If this fails, the id scheme changed and every
        # stored collection is invalidated — that must never be silent.
        assert make_document_id("a") == (
            "ca978112ca1bbdcafac231b39a23dc4da786eff8147c4e72b9807785afee48bb"
        )

    def test_blank_identifier_is_rejected(self):
        for blank in ("", "   ", "\t\n"):
            with pytest.raises(ValueError):
                make_document_id(blank)


class TestMakeChunkId:
    def test_deterministic(self):
        assert make_chunk_id("doc", 3, "text") == make_chunk_id("doc", 3, "text")

    def test_sensitive_to_document(self):
        assert make_chunk_id("doc-a", 0, "t") != make_chunk_id("doc-b", 0, "t")

    def test_sensitive_to_position(self):
        assert make_chunk_id("doc", 0, "t") != make_chunk_id("doc", 1, "t")

    def test_sensitive_to_text(self):
        assert make_chunk_id("doc", 0, "ta") != make_chunk_id("doc", 0, "tb")

    def test_text_whitespace_is_normalized(self):
        assert make_chunk_id("doc", 0, "a  b\n") == make_chunk_id("doc", 0, "a b")

    def test_encoding_is_injective_across_field_boundaries(self):
        # Without the length prefix both calls would encode to the payload
        # "doc:1:2:x" and collide — two different chunks, one id, and the
        # store's duplicate-id protection would silently drop one.
        assert make_chunk_id("doc", 1, "2:x") != make_chunk_id("doc:1", 2, "x")

    def test_known_value_pins_the_scheme(self):
        # sha256("3:doc:0:hello world") — length-prefixed payload
        assert make_chunk_id("doc", 0, "hello world") == (
            "35e3e6af5aa0036689b22687bd558df54bcfa0ef7995d0e62a31343dccb9342f"
        )


class TestDataclasses:
    def test_search_result_provenance_defaults_to_none(self):
        result = SearchResult(
            chunk_id="c", document_id="d", text="t", score=0.5, metadata={}
        )
        assert result.worker_id is None
        assert result.partition_id is None

    def test_instances_are_frozen(self):
        doc = Document(document_id="d", text="t", metadata={})
        with pytest.raises(dataclasses.FrozenInstanceError):
            doc.text = "changed"

    def test_chunk_keeps_document_provenance(self):
        doc = Document(document_id="d1", text="hello world", metadata={})
        chunk = Chunk(
            chunk_id=make_chunk_id(doc.document_id, 0, "hello"),
            document_id=doc.document_id,
            text="hello",
            position=0,
            metadata={},
        )
        assert chunk.document_id == doc.document_id

    def test_rag_result_answer_may_be_none(self):
        result = RAGResult(query="q", answer=None, sources=[], metrics={})
        assert result.answer is None
        assert result.sources == []
