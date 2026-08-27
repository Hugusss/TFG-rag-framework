# ADR-007 — BlocksDBStore adapter (reserved; not executed)

Increment: (reserved, then descoped)

## Problem

The previous project phase built a FAISS-IVF vector store on
serverless functions ("BlocksDB"). This decision number was reserved
for its return as a `BlocksDBStore` adapter behind the VectorStore
seam — the store-side counterpart of the replaceability thesis —
conditional on the core scope being finished first.

## Decision

**Not built.** Correctness and scalability evaluation were worth more
to the research question than a second backend, and the time available
did not fit both. The number is kept rather than silently reused, so
the sequence of decisions stays auditable.

## What survives instead

- The seam is proven twice over (Chroma adapter; partitioned
  composite), and `docs/adding-a-vector-store.md` documents the exact
  contract and the decisions an integer-keyed FAISS-style backend
  forces (id mapping, ANN parameters as index identity, remote index
  size, paged vector export).
- The prior phase's artifacts remain intact as read-only reference and
  as prior-work material for the thesis.

## Consequences

A future `BlocksDBStore` starts from a documented checklist instead of
this ADR; if it is ever built, it gets a new number and this record
stays as the trace of a deliberate descoping rather than an omission.
