"""Unit tests for the Ollama generator (stubbed transport, no network)."""

import pytest

from rag_framework.generation.base import GenerationError
from rag_framework.generation.ollama import REFUSAL, OllamaGenerator
from rag_framework.models import SearchResult


def source(rank, text, url=None):
    return SearchResult(
        chunk_id=f"c{rank}",
        document_id=f"doc-{rank}",
        text=text,
        score=1.0 - rank / 10,
        metadata={"url": url} if url else {},
    )


class StubTransport:
    def __init__(self, answer="La Alhambra está en Granada [1]."):
        self.calls = []
        self.answer = answer

    def __call__(self, url, payload, timeout):
        self.calls.append((url, payload, timeout))
        return {"message": {"content": self.answer}}


def generator(transport, **kwargs):
    return OllamaGenerator(
        model="test-model", endpoint="http://localhost:11434",
        timeout_seconds=60, transport=transport, **kwargs,
    )


class TestPromptAndCall:
    def test_grounded_prompt_carries_context_query_and_settings(self):
        transport = StubTransport()
        answer = generator(transport).generate(
            "¿dónde está la Alhambra?",
            [source(1, "La Alhambra se encuentra en Granada.", "https://e.org/a"),
             source(2, "Otra cosa distinta.")],
        )
        assert answer == "La Alhambra está en Granada [1]."
        (url, payload, timeout), = transport.calls
        assert url == "http://localhost:11434/api/chat"
        assert timeout == 60.0
        assert payload["model"] == "test-model"
        assert payload["stream"] is False
        assert payload["options"]["temperature"] == 0
        system, user = payload["messages"]
        assert system["role"] == "system"
        assert REFUSAL in system["content"]
        assert "ÚNICAMENTE" in system["content"]
        assert "[1] (fuente: https://e.org/a)" in user["content"]
        assert "La Alhambra se encuentra en Granada." in user["content"]
        assert "[2] (fuente: doc-2)" in user["content"]  # id when no url
        assert "Pregunta: ¿dónde está la Alhambra?" in user["content"]

    def test_long_fragments_are_capped(self):
        transport = StubTransport()
        generator(transport).generate("q", [source(1, "x" * 10_000)])
        user = transport.calls[0][1]["messages"][1]["content"]
        assert "x" * 4000 + "…" in user
        assert "x" * 4001 not in user

    def test_empty_context_refuses_without_calling_the_model(self):
        transport = StubTransport()
        assert generator(transport).generate("q", []) == REFUSAL
        assert transport.calls == []

    def test_answer_is_stripped(self):
        assert generator(StubTransport("  hola \n")).generate("q", [source(1, "t")]) == "hola"

    def test_trailing_slash_endpoint_normalised(self):
        transport = StubTransport()
        OllamaGenerator("m", endpoint="http://localhost:11434/", transport=transport).generate("q", [source(1, "t")])
        assert transport.calls[0][0] == "http://localhost:11434/api/chat"


class TestFailures:
    def test_transport_errors_propagate_as_generation_errors(self):
        def broken(url, payload, timeout):
            raise GenerationError("cannot reach Ollama at http://localhost:11434")

        with pytest.raises(GenerationError, match="cannot reach"):
            generator(broken).generate("q", [source(1, "t")])

    @pytest.mark.parametrize("response", [None, {}, {"message": {}}, {"message": None}])
    def test_unexpected_shape_raises(self, response):
        with pytest.raises(GenerationError, match="unexpected Ollama response"):
            generator(lambda *a: response).generate("q", [source(1, "t")])

    def test_empty_answer_raises(self):
        with pytest.raises(GenerationError, match="empty answer"):
            generator(StubTransport("   ")).generate("q", [source(1, "t")])
