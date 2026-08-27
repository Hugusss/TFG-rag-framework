# ADR-009 — Query encoding runs locally, byte-exact with the rack reference

Increment: query encoding

## Problem

Retrieval needs `embed_query`, and queries can never be precomputed.
The live jina-v5 encoder must run somewhere, and its recipe must
reproduce the embedding space of the official corpus vectors — the
July +1024 investigation proved that "almost the same recipe" produces
plausible garbage.

## Options considered

1. **Local CPU encoder** (sentence-transformers on the laptop).
2. Remote encoding on compute6 (SSH to `venv-jina5`, the reference
   encoder that validated the corpus in July).
3. A hosted embedding API.

## Decision

Option 1, with option 2 retained as the *verification reference*:
`LocalEmbeddingProvider` reproduces the reference recipe exactly —

    SentenceTransformer(model_id, trust_remote_code=True, device="cpu")
    model.encode(texts, task="retrieval", normalize_embeddings=True)

— and the recipe is pinned by a unit test (a recording stub asserts the
exact call). The precomputed provider's `embed_query` delegates to a
lazily built local encoder that must declare the same `model_id`.

## Reason

- Feasibility was verified first: torch 2.13.0 / sentence-transformers
  5.7.0 / peft 0.20.0 install and run on Python 3.14.
- Correctness was verified against the reference: identical strings
  encode to **cos ≥ 0.9999** of compute6's output — the recipes are
  equivalent in practice.
- Local encoding removes the SSH dependency from every retrieval
  experiment (latency ~81 ms/query warm on CPU vs seconds over SSH),
  which matters for the latency measurements: the measured system
  must not include an accidental network hop.
- Option 3 cannot reproduce the corpus's embedding space at all.

## Consequences

- Dependencies: `sentence-transformers` (capped list) plus **`peft`**,
  which the jina-v5 model code requires at runtime — this ADR is the
  required record for adding a dependency beyond the capped list.
- **`trust_remote_code=True` executes third-party model code** — an
  accepted, documented risk inherited from the reference recipe (the
  official pipeline and compute6 both run the same code).
- The HF model revision is not pinned in code (matching the reference);
  today's re-download matched the July-cached reference at cos 0.9999+,
  and run manifests record encoder-stack versions. If upstream ever
  drifts, the recorded alignment check is the tripwire to
  re-run.
- Live-vs-official alignment on real chunks is 0.92–0.9999 (negative
  control 0.18): an upstream publisher-pipeline property, identical for
  the reference encoder — not local drift. Retrieval-quality
  conclusions must not assume live re-encoding reproduces official
  vectors exactly.
- Model load (~40 s cold, ~600 MB cache) is paid lazily on first
  query, never at ingest.
