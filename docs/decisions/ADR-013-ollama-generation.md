# ADR-013 — Real generation through Ollama; model choice

Status: extends ADR-010, which shipped the deterministic mock first. The
mock remains the default; this record adds the real model beside it.

Increment: Ollama generator

## Problem

ADR-010 deferred real generation behind a mock until a model and venue
were agreed. A GPU host running Ollama is now reachable through an SSH
tunnel as `localhost:11434`. The Generator seam needs its first real
adapter, and a model.

## Decision 1 — adapter against the Ollama HTTP API, stdlib only

`generation/ollama.py` posts to `/api/chat` (non-streaming) with
`urllib.request` — no new dependency. `temperature 0` (greedy: the most
reproducible setting a sampler offers; not bit-identical across
hardware — recorded). The endpoint is configuration and is always a
*local* port (the tunnel), so no remote address or credential can ever
reach the repository. Prompt construction lives entirely in the
adapter: numbered fragments with provenance, instruction to
answer only from context, cite `[n]`, and refuse with a fixed sentence
otherwise. An empty context short-circuits to the refusal without a
model call. Failures are `GenerationError`s naming the endpoint and
the likely cause ("is the tunnel open?").

The refusal-by-instruction design is load-bearing: the measured
overlap of answerable and unanswerable top scores means no similarity
threshold can gate answers — the generator is the
only honest place for "I don't know", and the live smoke confirmed it
(exact refusal sentence on an out-of-corpus question, 0.3 s).

## Decision 2 — model: `ministral-3` as shipped default

Options on the host: ministral-3 (8.9B, strong multilingual, 262k
ctx), hermes3:8b, gemma4:31b / qwen 27B (better but 3–8× slower),
gpt-oss:20b; pullable: `hf.co/BSC-LT/salamandra-7b-instruct-gguf`
(ADR-010's original family, BSC's Spanish public model); rejected:
hosted APIs (credentials, cost, not local).

Chosen: **ministral-3** in the shipped config — already present,
strong Spanish, fast enough for interactive use (0.3–2.1 s warm;
~30 s first call while the model loads into VRAM). The adapter is
model-agnostic (`generation.model`), so Salamandra-7B-instruct remains
the thesis-coherent comparison to pull when generation quality is
evaluated: a Spanish public-sector model answering over a Spanish
public web corpus is the better story, and the ADR-010 deviation
(2B → 7B, Ollama GGUF) would be recorded then.

## Consequences

- The fourth seam swap: mock ↔ real model is a config change; the
  pipeline, retrievers and reports did not change (reports now record
  provider and model).
- Generation is only as good as its fragments: with publisher chunks
  (HTML navigation noise) the model *correctly refused* a question the
  corpus can answer; with the 500-word recursive chunks it produced a
  complete cited answer. Chunking quality propagates to generation —
  Experiment D's conclusion, now visible end to end.
- Reproducibility is weaker than everywhere else in the project
  (greedy decoding, GPU nondeterminism): answers are attributable
  (model, provider recorded) but not bit-stable, so generation quality
  must be judged by rubric, not by diff.
