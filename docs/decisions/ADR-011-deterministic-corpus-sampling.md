# ADR-011 — Deterministic, nested corpus sampling for dataset-size scaling

Increment: dataset-size experiment

## Problem

Measuring how the system behaves as the dataset grows means holding the
retrieval configuration constant and varying only the corpus size. For
that to be a controlled variable, "10 %, 25 %, 50 %, 100 %" must name
*the same documents* on every run and every machine, and the subset must
be a property of the configuration, so that a report can never present a
subset as the whole corpus.

## Options

(a) **First N documents in loader order** — simplest, but order is file
and partition-directory order: 10 % would be "the first parquet
shard", a biased slice (one crawl day, one language partition), and
any change in file layout changes the subset.
(b) **Seeded random sample** — reproducible only as long as the RNG
implementation, the seed and the iteration order all stay fixed; the
subset is an accident of `random`'s internals.
(c) **Hash sampling**: keep a document iff
`stable_bucket(document_id, 100) < percent`, with the same SHA-256
bucket function partitioning uses (ADR-006).

## Decision

**(c)**, as a decorator on the loader seam (`loaders/sample.py`,
`SampledLoader`) selected by `dataset.sample_percent` (1–100, default
100, validated, recorded in manifests). The bucket function moves to
`models.stable_bucket` so partitioning and sampling share one
definition of "stable".

## Reasons

- Deterministic across processes, machines and Python versions;
  independent of file order and layout.
- **Nested**: every document of the 10 % subset is in the 25 % subset,
  which is in the 50 %. Growth curves then measure *growth*, not the
  luck of four unrelated samples.
- A decorator, not a loader branch: the OWI loader stays the only
  place that parses OWI; any future loader is sampled the same way.
  The same move as ADR-002 and the embedding cache.
- Dropped documents are counted (`documents_sampled_out`) and
  reported; the subset size is a measured quantity, not an intention.

## Consequences

- Subset sizes are statistical (10 % gave 253 of 2,363 documents,
  10.7 %); the report carries the exact counts.
- The evaluation queries were written against the full corpus, so
  retrieval quality on a subset mostly measures how many relevant
  documents the subset still holds; the benchmark reports that
  coverage next to recall so the two are never confused.
- Each subset is built fresh in a per-run directory (index sizes are
  only comparable between fresh builds); the
  benchmark never deletes, because the backend caches clients per
  path and a deleted directory under a live client corrupts the run.
