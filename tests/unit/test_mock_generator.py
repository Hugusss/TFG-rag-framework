"""Unit tests for the mock generator."""

from rag_framework.generation.mock import MockGenerator
from rag_framework.models import SearchResult


def source(doc_id, text, url=None):
    metadata = {"url": url} if url else {}
    return SearchResult(
        chunk_id=f"c-{doc_id}",
        document_id=doc_id,
        text=text,
        score=0.9,
        metadata=metadata,
    )


class TestMockGenerator:
    def test_labels_itself_as_mock(self):
        answer = MockGenerator().generate("q", [source("d1", "some text")])
        assert answer.startswith("[mock answer]")

    def test_cites_every_source_in_rank_order(self):
        answer = MockGenerator().generate(
            "q",
            [
                source("d1", "primero", url="https://a.example"),
                source("d2", "segundo", url="https://b.example"),
            ],
        )
        assert answer.index("[1] (https://a.example)") < answer.index(
            "[2] (https://b.example)"
        )
        assert "primero" in answer and "segundo" in answer

    def test_falls_back_to_document_id_without_url(self):
        answer = MockGenerator().generate("q", [source("doc-xyz", "texto")])
        assert "(doc-xyz)" in answer

    def test_empty_context_degrades_honestly(self):
        answer = MockGenerator().generate("q", [])
        assert "[mock answer]" in answer
        assert "No relevant passages" in answer

    def test_deterministic(self):
        context = [source("d1", "texto de prueba", url="https://a.example")]
        assert MockGenerator().generate("q", context) == MockGenerator().generate(
            "q", context
        )

    def test_html_stripped_and_long_text_truncated_wordsafe(self):
        text = "<h1>Título</h1> " + "palabra " * 60
        answer = MockGenerator().generate("q", [source("d1", text)])
        assert "<h1>" not in answer
        assert "Título" in answer
        assert "…" in answer
        # never cuts a word in half before the ellipsis
        cited = answer.split(") ", 1)[1]
        assert cited.rsplit(" ", 1)[-1].rstrip("…") in ("", "palabra")

    def test_non_markup_angle_brackets_survive(self):
        text = "valores 5 < 6 y también 7 > 2 final"
        answer = MockGenerator().generate("q", [source("d1", text)])
        assert "5 < 6" in answer and "7 > 2" in answer

    def test_unbroken_long_token_gets_hard_cut(self):
        text = "x" * 300  # no spaces: word-safe truncation impossible
        answer = MockGenerator().generate("q", [source("d1", text)])
        cited = answer.split(") ", 1)[1]
        assert cited == "x" * 200 + "…"
