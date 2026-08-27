"""Generator backed by a language model served by Ollama.

Talks to an Ollama server over its HTTP API (``/api/chat``,
non-streaming) using only the standard library. The endpoint is
configuration and always local — typically an SSH tunnel to a GPU host —
so no address or credential ever lives in the repository. Prompt
construction and the grounding contract (answer only from the retrieved
fragments, cite them, otherwise refuse) belong to this module alone.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable

from rag_framework.generation.base import GenerationError, Generator
from rag_framework.models import SearchResult

# Refusal is instructed rather than gated on a score threshold: the top
# similarity of an unanswerable query overlaps the answerable range, so
# no cut-off can tell them apart. The generator is the only stage that
# sees whether the retrieved text actually answers the question.
REFUSAL = "No puedo responder con el contexto disponible."

_SYSTEM = (
    "Eres un asistente de búsqueda documental. Respondes ÚNICAMENTE con"
    " la información contenida en los fragmentos de contexto que se te"
    " proporcionan. Citas cada afirmación con el número de su fragmento,"
    " por ejemplo [2]. Respondes en el idioma de la pregunta, de forma"
    " breve y concreta. Si el contexto no contiene la respuesta,"
    f' respondes exactamente: "{REFUSAL}"'
)

_CHUNK_CHARS = 4000  # per-fragment cap keeps the prompt bounded
_NUM_CTX = 16384  # fragments (k×~500 words) plus answer fit comfortably


def _post_json(url: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as error:
        detail = error.read()[:200].decode("utf-8", "replace")
        raise GenerationError(
            f"Ollama at {url} answered HTTP {error.code}: {detail}"
        ) from error
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise GenerationError(
            f"cannot reach Ollama at {url}: {error} — is the service"
            " running and the SSH tunnel open?"
        ) from error
    try:
        return json.loads(body)
    except ValueError as error:
        raise GenerationError(
            f"Ollama at {url} returned malformed JSON: {error}"
        ) from error


class OllamaGenerator(Generator):
    """Grounded question answering through a configured Ollama model."""

    def __init__(
        self,
        model: str,
        endpoint: str = "http://localhost:11434",
        timeout_seconds: int = 120,
        transport: Callable[[str, dict, float], dict] | None = None,
    ) -> None:
        self.model = model
        self.endpoint = endpoint.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._transport = transport or _post_json

    def generate(self, query: str, context: list[SearchResult]) -> str:
        if not context:
            # no evidence: refusing is the only grounded answer, and it
            # needs no model call
            return REFUSAL
        response = self._transport(
            f"{self.endpoint}/api/chat",
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": self._prompt(query, context)},
                ],
                "stream": False,
                # greedy decoding: the most reproducible setting a sampling
                # model offers, though the server's own version still
                # decides the weights, so reports record model and endpoint
                "options": {"temperature": 0, "num_ctx": _NUM_CTX},
            },
            float(self.timeout_seconds),
        )
        try:
            answer = response["message"]["content"]
        except (TypeError, KeyError) as error:
            raise GenerationError(
                f"unexpected Ollama response shape: {str(response)[:200]}"
            ) from error
        answer = (answer or "").strip()
        if not answer:
            raise GenerationError(
                f"Ollama model '{self.model}' returned an empty answer"
            )
        return answer

    @staticmethod
    def _prompt(query: str, context: list[SearchResult]) -> str:
        lines = ["Fragmentos de contexto:"]
        for rank, source in enumerate(context, start=1):
            origin = source.metadata.get("url") or source.document_id
            text = source.text[:_CHUNK_CHARS]
            if len(source.text) > _CHUNK_CHARS:
                text += "…"
            lines.append(f"[{rank}] (fuente: {origin})\n{text}")
        lines.append(f"\nPregunta: {query}")
        return "\n\n".join(lines)
