# ADR-006 — Collective retrieval core: partition assignment and global merge rules

Increment: collective core

## Problem

Collective retrieval rests on two rules that must be settled before any
partitioned index exists: which partition owns a document, and how the
per-partition candidate lists reduce to one top-k. The assignment has to
be deterministic — a rebuild with the same input, configuration and P
must reproduce the same partitions — and the merge has to handle empty
and short lists, duplicate chunk ids, tied scores and invalid scores
with a deterministic order. Changing either rule later invalidates every
existing partitioned collection and every comparison made with it.

## Decision 1 — `partition_id = int(sha256(document_id)[:8 bytes]) mod P`

The eight are bytes of the digest read big-endian, not hex characters.

Options: (a) Python `hash(document_id)`, the obvious reading;
(b) SHA-256 of the UTF-8 id, first 8 bytes as an integer, mod P;
(c) range partitioning over sorted document ids; (d) chunk-level
round-robin.

Chosen: **(b)**, document level.

Reasons: (a) is *salted per process* for strings in CPython
(`PYTHONHASHSEED`), so two runs — or two workers — would assign
documents differently, which breaks the determinism
requirement. (b) is stable across machines, Python versions and
processes, costs microseconds (OWI ids are already SHA-256 hex, but the
rule must not depend on that — ADR-001's fallback ids are arbitrary
strings). (c) needs global knowledge of all ids before assigning the
first chunk, which breaks streaming ingest. (d) balances better but
splits documents across workers. Note what (d) does *not* cost: with
exact search, every chunk of the global top-k is also in its own
partition's top-k, so the merged result is identical whichever way
chunks are partitioned — correctness is not the argument. The case for
whole documents gives (i) locality of a document's evidence,
(ii) the invariant "a chunk
id exists in exactly one partition", which turns a duplicate into a
detectable fault instead of a legitimate event, and (iii) document
locality: all of a document's evidence — and any future per-document
operation (filters, deletes, context assembly) — lives on one worker.

Consequences: balance is statistical (≈ binomial; measured spread on
10k synthetic ids at P = 8 is ±5 %, pinned loosely by test) and is
itself an Experiment-B metric ("imbalance between partitions"); the
constants are pinned by golden tests so any change to the rule is a
visible, deliberate break; P is part of a partitioned index's identity
(collection naming carries the layout), never a runtime knob over an
existing index.

## Decision 2 — merge order is (score desc, chunk_id asc); duplicates keep the best candidate; invalid scores raise

- **Total order.** Sorting by score alone leaves ties to arrival
  order, which in collective mode is *thread scheduling* — results
  would differ run to run with equal data. `chunk_id` is unique,
  deterministic and available on every result, so it is the
  tiebreaker, so ordering stays deterministic for equal scores
  without any dependence on partition order
  (property-tested by shuffling partitions and their contents).
- **Duplicates.** With document-level partitioning a chunk id can
  appear in one partition only; a duplicate therefore signals
  overlapping partitions upstream. The merge keeps the best-scoring
  candidate (equal scores: lowest `(partition_id, worker_id)`, so even
  this does not depend on arrival order), never spends two top-k slots
  on one chunk, and **logs a warning** with the count — handled
  correctly, but never silently. The collective retriever surfaces the
  count as a metric.
- **Invalid scores** (NaN, ±inf, non-numeric, bool) raise
  `RetrievalError` naming the chunk, for *every* candidate, not only
  those that would reach the top-k: NaN compares false with everything
  and would silently scramble a sort; a backend producing it is broken
  and must be seen.
- `k` must be a positive int; nothing else is accepted.

Consequences: `merge_top_k` is pure and backend-free, so the same
function reduces thread, process and (later) remote partials, and
Experiment E compares baseline vs collective knowing the reduction
step itself is exhaustively tested.
