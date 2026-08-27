# Evaluation query set

`queries.jsonl` holds 30 Spanish queries over the pinned corpus
(`owi-v2.0.0-gpu-spa-2026-07-28`), mixing the types required by the
project specification: direct factual (q001–q008, eight), specific
document (q009–q013, five), multi-document (q014–q017, four),
terminology and acronyms (q018–q020, three), unanswerable (q021–q024,
four) and three paraphrase pairs, which carry an `a`/`b` suffix
(q025a/q025b, q026a/q026b, q027a/q027b, six entries linked by
`paraphrase_of`). Eight plus five plus four plus three plus four plus
six is thirty.

Each line is one query with these fields:

| Field | Meaning |
|---|---|
| `query_id` | Stable identifier, the one every result file reports. |
| `query` | The question text, as the encoder receives it. |
| `answerable` | Whether the corpus can answer it. |
| `relevant_document_ids` | The judged documents; an empty list for the unanswerable ones, and that emptiness is what makes a hit on them a false positive by construction. |
| `notes` | The evidence for the judgement, including the membership criterion for the multi-document queries. |
| `paraphrase_of` | Present only on the `b` half of each pair. |

The benchmark scripts read these keys by name, so a renamed field
breaks them.

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
