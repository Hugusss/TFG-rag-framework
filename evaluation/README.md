# Evaluation query set

`queries.jsonl` holds 30 Spanish queries over the pinned corpus
(`owi-v2.0.0-gpu-spa-2026-07-28`), mixing the types required by the
project specification: direct factual (q001–q008), specific-document
(q009–q013), multi-document (q014–q017), terminology/acronym
(q018–q020), unanswerable (q021–q024), and paraphrase pairs
(q025–q027, linked by `paraphrase_of`).

## Methodology and integrity

Relevance is judged at **document level**. Ground truth was established
by reading corpus documents directly and by exhaustive keyword search
over the text inside publisher chunk windows — **never by running the
retrieval system under evaluation**. Every judgment was reviewed and
validated by hand (2026-08-17); each entry's `notes` field records the
evidence or the set-defining criterion.

Known corpus properties reflected in the set: all targets lie inside
publisher-covered windows (only 66% of corpus text is covered — see
`docs/dataset.md`); byte-identical duplicate pages exist, so exact
score ties are legitimate; `answerable: false` queries were verified to
have zero topic occurrences in any covered window.

The set is small; retrieval-quality numbers computed from it support
trend statements, not strong statistical conclusions.
