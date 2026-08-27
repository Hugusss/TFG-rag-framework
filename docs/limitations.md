# Limitations

Stated as measured on the evaluation corpus (OWI v2.0.0, Spanish,
2026-07-28: 2,363 documents, 6,240 official chunks) on one laptop
(8 threads, 32 GB), unless noted.

## Scope and deployment

- The collective mode is a **local simulation**: P Chroma collections
  on one machine, searched by threads. Nothing here has run across
  machines; the executor and store seams are where a distributed
  deployment would plug in, and that remains a claim, not a result.
- Only a **thread** executor exists. Process workers would need a
  partition descriptor that a child process can reopen; a
  function-as-a-service executor would need the same plus serialisable
  query state.
- Some store operations assume that reaching a partition is free.
  `PartitionedVectorStore.count()` and `partition_counts()` ask every
  inner store in turn, which is one local call per partition today and
  would become P network round trips against a remote backend. The same
  holds for `iter_vectors()`, which streams the partitions one after
  another. Nothing measured here is affected — the cost is invisible on
  one machine — but a remote store would want these batched or cached,
  and that is a change in the adapter, not in the seam.
- Generation ships two adapters: a deterministic **extractive mock**
  (the default, dependency-free) and an **Ollama-backed generator**
  (greedy decoding against a local endpoint — in practice an SSH
  tunnel to a GPU host). The real model was smoke-tested (grounded
  cited answers in 0.3–2.1 s warm, ~30 s first call while the model
  loads; fixed-sentence refusal on empty or insufficient context)
  but no systematic generation-quality evaluation was run — answers
  are attributable (provider and model in every report) yet not
  bit-reproducible across hardware. Retrieval-only operation remains
  the measured path.

## Performance

- At 1.36M vectors (25 GB index) the machine leaves the in-memory
  regime: single-collection search rises from ~3 ms to 65–74 ms
  (page-cache misses on a 32 GB laptop), and parallel partition
  searches contend for the same disk — 8 threads reach 91 ms wall
  while a *single* partition search costs only 14–16 ms. Collective
  retrieval on one machine never beats the monolith at any measured
  scale; the per-partition numbers are the measured case for
  distributing memory across machines, which this project does not do.

- **Query encoding dominates latency** (~85 ms of ~90 ms on CPU), so no
  retrieval-mode change moves end-to-end latency by more than a few
  percent here. A GPU encoder would change the constants.
- At 6,240 vectors a partition search costs 1.5–2.8 ms whatever its
  size (logarithmic HNSW plus per-call overhead), so **collective
  search is slower than the single collection at every partition
  count** (2.9–6.4 ms vs 2.7 ms). Threads scale about 2× at 8 workers
  while per-worker time doubles under contention.
- Live encoding on CPU runs at 0.2–1.1 chunks/s (hours per
  configuration); the chunk-size results for self-chunked
  configurations were produced once and one configuration used a
  partial embedding cache (declared in its report).

## Data and evaluation

- The official publisher chunks cover **66 % of the text** (a five-window
  cap per document); the self-chunked configurations cover all of it.
  Queries were written against the visible text.
- Larger `minimum_tokens` floors exclude short documents entirely
  (20 / 214 / 491 documents at 200 / 500 / 1000 words); the large
  configuration loses three evaluation-relevant documents before
  encoding.
- Chunk sizes are **whitespace words** (2.33 subword tokens per word
  measured); results are labelled accordingly.
- The **evaluation set has 30 queries** (26 answerable, 4 unanswerable):
  enough to catch a broken configuration, not to rank close ones. On
  dataset subsets recall tracks relevant-document coverage. Under
  corpus growth that *contains* the original corpus the set stays
  valid, and the measured true effect is moderate: exact Recall@10
  0.891 → 0.882 (×10 Spanish distractors) → 0.848 (×220, 75
  languages — the multilingual embedding space admits cross-language
  distractors).
- Unanswerable queries' top scores (0.29–0.55) **overlap** answerable
  ones; no score threshold separates them.

## Correctness and measurement

- Vector search is **approximate** (HNSW), and the cost of the default
  search width grows with scale: at the backend default width, queries
  losing true neighbours went from 1/30 at 6k vectors to 8/30 at 1.36M,
  where measured recall (0.834) fell below the exact reference (0.848).
  Raising the width (`vector_store.ef_search`, 800) restored
  exact-reference agreement at document level on every corpus measured,
  at 1.1–2× search latency, and brought 1.36M down to 4/30 affected
  queries; the 63k corpus was only ever measured at 800, where it loses
  nothing (0/30). At 1.36M a small chunk-level gap remains at equal
  width (overlap 0.977 monolithic, 0.990 partitioned): smaller
  per-partition graphs are consistently more exact.
- Index-size numbers are only comparable between **freshly built**
  indexes; a re-ingested directory carries storage history.
- Several result files carry a `-dirty` git commit, meaning the working
  tree held uncommitted changes when the run was measured. The marker is
  the honest part: it says the recorded hash does not by itself pin the
  code that produced the numbers. What each result does pin on its own is
  the rest of the manifest — effective configuration, dataset version,
  machine and dependency versions — and the run remains reproducible by
  re-executing the same configuration on the same corpus.
