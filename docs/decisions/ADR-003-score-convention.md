# ADR-003 — Score convention: similarity, higher is better

Increment: models

## Problem

Whether a larger score means a better result has to be settled once and
for all, because backends disagree: Chroma returns cosine
*distance* (lower is better), FAISS L2 likewise, while inner-product
scores are higher-better. `merge_top_k` and every consumer of
`SearchResult` need one convention.

## Options considered

1. Pass through the backend's native value plus a `higher_is_better` flag.
2. **Normalize at the adapter boundary: `score` is always a similarity,
   higher is better; each vector-store adapter converts its native
   metric** (e.g. cosine distance `d` → similarity `1 − d`).
3. Standardize on distance, lower is better.

## Decision

Option 2, recorded in the `SearchResult` docstring in `models.py`.

## Consequences

- `merge_top_k` stays a pure, backend-agnostic sort-descending function —
  the property its exhaustive unit tests rely on.
- Every adapter documents its conversion formula; a wrong or missing
  conversion inverts result ordering, so adapter tests must assert
  ordering against known vectors.
- Scores from different backends are comparable in reports only when the
  underlying metric family matches; benchmark writeups must state the
  metric per collection.
- Option 1 was rejected because it pushes an if-branch into every
  consumer; option 3 because "bigger bar = better hit" reads naturally in
  plots and reports.
