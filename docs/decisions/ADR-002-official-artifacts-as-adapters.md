# ADR-002 — Official OWI chunks and embeddings enter as adapters of existing seams

Increments: chunking, embeddings

## Problem

The OWI corpus ships its own chunk boundaries (`chunk_offsets`) and its
own embeddings (one vector per window), and using them instead of
recomputing is the point of building on this corpus. The naive design
is a special path: `if precomputed: skip chunking and embedding`. That branch
would live in the orchestrator, grow conditions with every corpus, and
quietly defeat the research question — a pipeline whose orchestration
changes per data source is exactly the thing that cannot "evolve by
replacing components".

## Options considered

1. Conditional branch in the orchestrator (`if precomputed: ...`).
2. A separate ingestion pipeline for publisher-chunked corpora.
3. **Official artifacts become ordinary adapters of the existing seams**:
   a `publisher_offsets` strategy of the Chunker, and a `precomputed`
   adapter of the EmbeddingProvider (whose `embed_query()` delegates to
   the live jina-v5 encoder, since queries can never be precomputed).

## Decision

Option 3. The orchestrator wires loader → chunker → embedding provider →
store identically for every configuration; which chunker and provider
run is decided in `configs/*.yaml`, once, for the whole corpus.

## Reason

The orchestrator cannot tell publisher artifacts from computed ones —
which is the replaceability thesis demonstrated in miniature,
and the argument presented at review. Options 1 and 2 duplicate
orchestration logic and make every new corpus a code change instead of a
config change.

## Consequences

- **`Chunk.position` must equal the publisher's window index**
  (`chunk_idx` in the embeddings parquet): it is the lookup key the
  precomputed provider uses. Therefore a rejected window never
  renumbers its successors, pinned by unit test.
- **Chunk text is byte-exact `main_content[start:end]`** — no cleaning,
  no normalization on this path. The official vectors were computed
  over exactly these strings. Verified against the corpus: all 6,240 chunks of
  the spa 2026-07-28 slice are byte-identical to the independently
  produced hydration table. (Text cleaning belongs only to the
  `recursive` strategy, where we also produce the vectors ourselves.)
- **Overhang clip policy** (measured, not assumed): 1,949/6,240 real
  windows (and 1,053/3,195 on the 2026-08-08 day) end ≤10 characters
  past `len(main_content)` — a known publisher-pipeline artifact.
  Window ends overhanging by ≤ `MAX_OVERHANG = 10` are clipped to the
  text end and counted (`windows_clipped`, chunk metadata `clipped:
  True`); larger overhangs are rejected as corruption. The clipped text
  is what the chunk id hashes — deterministic either way. `MAX_OVERHANG`
  is deliberately code, not configuration: the rule that chunking
  parameters live in config covers values an experimenter may vary, and
  this is a validity threshold measured from the corpus — varying it
  would mean the corpus itself changed. For the same reason publisher
  windows are never content-filtered (even a whitespace-only window
  keeps its official embedding); the "avoid empty chunks" guidance
  applies to strategies that choose their own boundaries.
- The embedding half (precomputed provider keyed by `(document_id,
  position)`; `embed_query` via live jina-v5) inherits these guarantees;
  its details land with the embeddings increment.
