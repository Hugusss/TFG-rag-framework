# ADR-005 — `embed_documents` takes Chunks, not bare texts

Increment: embeddings

## Problem

The obvious signature for an embedding provider is
`embed_documents(texts: list[str])`. But the corpus's official vectors
are looked up by *identity* — `(record_id, chunk_idx)` — not computed
from text. A text-only interface cannot express that adapter.

## Options considered

1. Keep `texts: list[str]` and pass identities through a side channel
   (parallel id list, or provider peeking at global state).
2. **`embed_documents(chunks: list[Chunk])`** — the provider receives
   the canonical objects and uses what it needs: the precomputed
   adapter reads `(document_id, position)`, a live-model adapter reads
   `chunk.text`.
3. Two different provider interfaces (one for lookup adapters, one for
   encoding adapters), dispatched by the orchestrator.

## Decision

Option 2. `embed_query(text: str)` stays text-based — queries have no
identity, only text.

## Reason

Option 1's side channels are the interface lying about its inputs;
option 3 is the `if precomputed:` branch of ADR-002 wearing a
different coat — the orchestrator would again know which kind of
provider it holds. Option 2 keeps one seam, and the design already allows
it: "these are architectural boundaries, not mandatory signatures",
with the reason recorded here.

## Consequences

- Every provider sees Chunks; text-only providers just read
  `chunk.text` — no cost beyond the parameter type.
- The precomputed adapter fails loudly (`EmbeddingError`) when a chunk
  has no official vector: a miss means chunker and vector table
  disagree about the corpus; it is never silently re-encoded.
- The lookup key contract (`Chunk.position` == publisher `chunk_idx`)
  is what makes this work; it is guaranteed by the publisher_offsets
  chunker and pinned by tests on both sides of the seam.
- `embed_query` on the precomputed provider raises until the retrieval
  increment brings the live encoder — ingestion never
  embeds queries, so ingestion is unaffected; the deferral is
  loud (NotImplementedError with the reason), not hidden.
