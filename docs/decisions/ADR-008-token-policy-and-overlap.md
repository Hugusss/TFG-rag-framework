# ADR-008 — Recursive chunking: token policy and overlap placement

Increment: recursive chunker

## Problem

Chunking is configured in "tokens" — target, overlap, minimum — but
"token" has no single meaning: whitespace words and model subword units
differ by a factor that matters when reporting chunk sizes. Where the
configurable overlap applies is equally open. Both need a concrete,
recorded choice, or the chunk-size experiment reports a number nobody
can interpret.

## Decision 1 — token = whitespace-delimited word

Options: (a) whitespace words (stdlib); (b) the jina-v5 subword
tokenizer (transformers is already installed); (c) a generic tokenizer
(tiktoken-style — new dependency).

Chosen: **(a)**, plainly documented everywhere as "token ≈ word".

Reasons: the dependency policy is stdlib-first; (b) would couple the
chunking seam to the embedding stack for precision the goal —
*documenting a trade-off on one dataset, not tuning a model budget* —
does not need, and is ~30× slower; (c) buys nothing over (a) here.
Consequences: word counts understate model tokens by roughly 1.3–2×
for Spanish; chunk-size experiment labels must say "words"; if a
future experiment needs the model's real budget, a tokenizer-backed
counter can be injected without changing the strategy's shape.

## Decision 2 — overlap applies only at arbitrary cuts

Overlap is prepended between consecutive *windows of an oversized
paragraph* (stride = target − overlap); packed-paragraph seams get no
overlap.

Reason: overlap exists to protect content that a cut splits
mid-thought. A paragraph boundary is the author's own semantic cut —
duplicating text across it would blur exactly the boundary meant to
retain. An arbitrary window cut inside a long paragraph is ours, and
gets the guard. Measured on the real corpus, most text arrives as few
huge paragraphs, so overlap is exercised heavily (duplication ≈ 0.4%
of corpus words at 500/50).

## Consequences

- Fully deterministic and fast (0.3 s corpus-wide at any config).
- Chunks below `minimum_tokens` are dropped **and counted**
  (`chunks_dropped_below_minimum`); short documents can drop to zero
  chunks and are counted in `documents_without_chunks` — at 1000/100
  that excludes 491 of 2,363 docs, a real experimental variable that
  Experiment D must report, not hide.
- Positions are assigned before the minimum filter, so changing
  `minimum_tokens` never renumbers surviving chunks (pinned by test).
- This strategy owns its text: whitespace normalization is safe
  because its vectors are always computed live from exactly this text
  (byte-exactness only binds the precomputed path — ADR-002); the
  factory refuses `recursive` + `precomputed` at construction.
