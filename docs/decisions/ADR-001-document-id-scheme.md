# ADR-001 — Document identifier scheme: adopt publisher ids

Increment: models

## Problem

Document ids must be deterministic, so that re-ingesting unchanged input
produces the same ids and therefore no duplicates. The obvious scheme is
`document_id = hash(normalized source identifier)`. The OWI corpus,
however, already ships a stable per-document identifier:
`record_id`, a sha-256 string that is also the join key between the
embeddings parquet, the records parquet, and the text parquet.

## Options considered

1. **Always hash ourselves**, even when the source has a stable id
   (`sha256(record_id)`).
2. **Adopt the publisher id when one exists; hash the normalized source
   identifier only when there is none** (fixture corpora, ad-hoc files).
3. Always adopt whatever identifier the source has, hashing nothing.

## Decision

Option 2. The OWI loader uses `record_id` as `document_id` verbatim.
`make_document_id(source_identifier)` (sha-256 of the NFC- and
whitespace-normalized identifier; blank identifiers rejected with
`ValueError`) exists for sources without a stable id. Chunk ids are always
ours, over an **injective, length-prefixed payload**:
`sha256(len(document_id) + ":" + document_id + ":" + position + ":" +
normalized text)`. The length prefix matters because adopted publisher ids
are arbitrary strings that may themselves contain the separator; without
it, two different (document_id, position, text) triples can encode to the
same payload — e.g. `("doc", 1, "2:x")` and `("doc:1", 2, "x")` — and the
store's duplicate-id protection would silently drop a real chunk.

## Reason

Re-hashing an id that is already a content hash adds no determinism — it
only destroys direct joinability: with option 1, looking up a document's
official embedding row, its text row, or its original file would require an
extra translation table. With option 2, `document_id == record_id` navigates
straight back to every original OWI artifact, which is what makes a
result auditable against the published files, and the precomputed embedding
provider can key vectors by `(document_id, chunk position)` against the
official parquet with no mapping layer. Option 3 fails for sources with no
identifier at all.

## Consequences

- Loaders own the branch: OWI loader → adopt; fixture loader → hash. No
  other component knows where an id came from.
- Document ids are not homogeneous in origin (some publisher hashes, some
  ours) but all are deterministic strings; nothing downstream cares.
- If a future source has unstable or colliding "stable" ids, its loader
  must fall back to hashing — documented here as the rule.
