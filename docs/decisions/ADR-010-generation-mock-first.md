# ADR-010 — Generation: mock-first, real backend deferred

Status: extended by ADR-013, which added the real Ollama generator beside
the mock. The reasoning below still explains why the mock came first and
why it remains the default.

Increment: generation

## Problem

The generation seam needs a mock and, eventually, a real backend. Which
real backend to build was still open at this point — a local model and a
hosted API lead to different adapters, different dependencies and
different reproducibility claims — and guessing would have meant writing
an adapter likely to be thrown away.

## Decision

Implement the Generator seam with the mock only, and defer the real
backend until the target is known. The deferral is loud, not implicit:
requesting any other provider raises a `ConfigError` naming `mock` as
the only one registered, so an unimplemented backend can never degrade
into a silent fallback.

## Facts settled now, for the future decision

- jina-embeddings-v5 cannot generate (embedding-only); the generator
  must be a separate language model.
- Available hardware (laptop CPU + compute6's 128 CPU cores, no GPU)
  comfortably runs a 1–2B-parameter instruct model; no new hardware
  or paid API is required for a demo-grade backend.
- Leading candidate: **Salamandra-2B-instruct** (BSC — a Spanish,
  European public model over the EU's open web index is thematically
  coherent); alternatives: Llama-3.2-1B-instruct, Qwen-class small
  models. transformers/torch are already project dependencies, so no
  new dependency is expected.
- The mock is extractive, deterministic, and self-labeling
  (`[mock answer]` prefix) so no artifact can pass it off as model
  output.

## Consequences

- Retrieval experiments and the end-to-end pipeline behaviour are
  unblocked; evaluating generation quality stays blocked until a real
  backend lands.
- The deferral was later resolved by ADR-013, which supersedes this
  record's open question with a concrete backend.
