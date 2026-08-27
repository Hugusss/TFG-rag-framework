# Dataset: OWI v2.0.0 Spanish slice

Corpus pinned for this project: **Open Web Index (OWI), `gpu` collection,
schema v2.0.0, language `spa`, crawl 2026-07-28** — dataset version string
`owi-v2.0.0-gpu-spa-2026-07-28` (as recorded in every result manifest).
A second day (2026-08-08) is held locally as an independent check that the
pipeline is day-agnostic; it is not part of the pinned corpus.

All numbers below were computed from the raw files with this repository's
loader and chunker (2026-08-17) and are reproducible by
re-running them over `data/owi/`.

## 1. File formats and layout

One corpus day consists of two dataset directories that share a hive
partition scheme (`year=Y/month=M/day=D/language=L`) and pair by shard
index `N`:

| File | Dataset | Schema (columns read by this pipeline in bold) | Role |
|---|---|---|---|
| `metadata_N_records.parquet` | embeddings (`gpu.owie`) | **`id`**, **`chunk_offsets: list<struct{start,end:int32}>`**, `schema_metadata`, `p_phishing`, `p_llm_generated`, `p_advertising` | **Defines the corpus**: exactly the documents the publisher chunked and embedded |
| `metadata_N_embeddings.parquet` | embeddings (`gpu.owie`) | **`record_id`**, **`chunk_idx`**, **`embedding: list<halffloat>`** | One official vector per publisher chunk; explicit per-row keys |
| `metadata_N.parquet` | text (`gpu.owi`) | **`id`**, **`main_content`**, **`url`**, **`title`**, **`language`** (+ ~58 more) | Document text and metadata; joined on `id == record_id` |

`schema_metadata` declares `schema_version: 2.0.0` and `embedding_model:
jinaai/jina-embeddings-v5-text-small` (1024-dimensional, unit-normalized,
stored fp16). Chunk text is reconstructed as `main_content[start:end]`;
the embeddings row for a chunk is keyed `(record_id, chunk_idx)` where
`chunk_idx` is the window's index in `chunk_offsets`.

The pinned slice has one shard (`N = 0`) per dataset. Raw size: records
0.5 MB + embeddings 15.0 MB + text 13.4 MB = **28.9 MB**.

## 2. Record counts

| | 2026-07-28 (pinned) | 2026-08-08 (check) |
|---|---|---|
| Records (= documents) | **2,363** | 1,258 |
| Publisher chunks (= vectors) | **6,240** | 3,195 |
| Text rows | 2,365 | 1,258 |
| Loader/chunker rejections | 0 | 0 |

The two extra text rows on 07-28 (`07d96745ca…`, `98c583e3c3…`) are pages
crawled but not embedded; they have no records entry and are therefore
not corpus members (the records file is authoritative).

Raw parquet size on disk: 28.9 MB (2026-07-28) and 16.8 MB
(2026-08-08), summed over the records, embeddings and text files of the
Spanish partition.

## 3. Metadata fields

Preserved into the canonical model when present: `url`, `title`,
`language`, crawl date (reconstructed from the hive partition),
`chunk_offsets`, source file names, ingestion version. Missing values are
represented as absent keys, never invented. The text file carries ~60
further columns (quality signals, link graphs, WARC references) that are
not needed by this pipeline and are not read.

- Null `title`: 4 documents (07-28), 5 (08-08). No null `url`,
  `language`, or `main_content` anywhere.
- `language` is `spa` for every row (single-language partition), matching
  the hive path.

## 4. Duplicates

- Duplicate ids: **zero** in both records and text files, both days.
- Duplicate *content* under distinct ids exists: 92 distinct chunk texts
  are byte-identical across 215 chunks (07-28) — typically the same page
  crawled under two URLs (verified pair: one Zaragoza tourism page under
  two ids; their official vectors are identical, cosine 1.0). Retrieval
  evaluation must expect exact score ties from these.

## 5. Distributions (2026-07-28)

Document length (characters of `main_content`):

| min | p25 | median | p75 | p95 | max | mean |
|---|---|---|---|---|---|---|
| 250 | 1,191 | 2,586 | 5,771 | 17,612 | 150,587 | 5,193 |

Publisher windows per document: median 2, maximum **5** (a publisher
cap; 529 documents sit at it). Distribution: 1 window ×757, 2 ×582,
3 ×306, 4 ×189, 5 ×529.

Chunk length (characters): min 47, median 1,465, p95 1,957, max 2,613,
mean 1,340.

**Partial coverage of long documents**: 429 documents (18%) have text
beyond their last window — 412 of them at the 5-window cap — and only
**66%** of all corpus text falls inside a publisher window (extreme
case: a 150,587-char document with 142,423 chars uncovered). Content in
uncovered tails cannot be found by dense retrieval over the official
vectors. This is a property of the published dataset, not of this
pipeline; any retrieval-quality analysis must account for it.

The cap is **universal across the OWI embeddings line**, not a quirk of
this slice (verified 2026-08-17 by reading `chunk_offsets` from the raw
records files): the 2026-03-13 v0.3.0 slice (57,072 documents, 169,677
chunks) maxes at 5 windows with 27% of documents at the cap, and the
German partition of 2026-08-08 (16,692 documents) shows the same
maximum and a matching distribution shape. It spans schema versions,
months, and languages — a constant of the publisher's embedding
pipeline. Pulling a newer or different-language dataset therefore
cannot remove it; full-text coverage requires computing embeddings
locally (the `recursive` chunking strategy with the `local` embedding
provider).

## 6. Known quirks and parsing hazards

Each hazard below is handled by a documented pipeline policy; none is
silently absorbed.

1. **Window overhang**: 1,949 of 6,240 windows (31%) declare an `end` up
   to 10 characters past `len(main_content)` — 1,920 of them by exactly
   10 (signature of a fixed-width truncation in the publisher pipeline;
   same pattern on 08-08: 1,047/1,053 at exactly 10). Policy: clip to
   the text end and count when the overhang is ≤ 10; reject the window
   as corrupt beyond that. No window starts out of bounds; none is
   inverted.
2. **Raw HTML in `main_content`**: the text is not cleaned prose; tags
   and attributes appear throughout. The official embeddings were
   computed over exactly these strings, so the publisher-chunk path
   must never clean or normalize text (vector alignment and
   deterministic chunk ids both depend on byte-exactness).
3. **Tiny row groups** in the embeddings parquet (~42 rows per group,
   148 groups in a 6,240-row file): harmless for whole-file reads, but
   row-group-level streaming would be inefficient.
4. **fp16 storage**: vectors are stored as half precision and converted
   on read; unit norms hold to ~1e-3.
5. The corpus is **small** (6,240 vectors). Scaling experiments must use
   controlled subsets or clearly labeled replication, and additional real
   days are available (see below).

## 7. Obtaining the data

The public OWI distribution channel is the `owilix` CLI (LEXIS/iRODS).
Datasets are published per day, collection, and resource type; the
embeddings resource (`gpu.owie`) lags the text resource (`gpu.owi`) by a
few days, so a usable day needs both:

```bash
owilix remote ls  'it4i:latest#7/collectionName=gpu'
owilix remote pull 'it4i:2026-07-28/collectionName=gpu' --language spa --yes
```

This populates `~/.owi/public/gpu/<dataset-uuid>/year=…/language=spa/`
with the paired parquet files. Point `dataset.path` in the pipeline
configuration at a directory containing both dataset trees (the loader
discovers and pairs shards recursively); this project uses `./data/owi/`,
which is never committed.

Publication cadence (checked 2026-08-12): near-daily `gpu.owie` datasets
from it4i since 2026-07-28, 210k–1.2M records per day across all
languages. Occasional zero-byte orphan catalogue entries exist (e.g. two
on 2026-07-31); verify record counts after pulling.

### The larger corpora

Four configurations use corpora bigger than the pinned day, and they are
built the same way, only pulling more:

- `./data/owi-spa-growth` (`spa_growth.yaml`): the Spanish partition of
  every day between 2026-07-28 and 2026-08-13 that publishes both
  resources. Pull each day into the same tree; the loader pairs shards
  recursively and the vector provider keeps the first copy of a chunk
  seen across files, counting the recrawled duplicates. Result: 22,227
  unique documents and 63,281 vectors.
- `./data/owi-fullday-0728` (`fullday_0728*.yaml`): the whole pinned day
  without the `--language` filter, all 75 languages. Result: 520,987
  documents and 1,361,288 vectors, and a 25 GB index once ingested, so
  check the disk before starting.

Both contain the pinned Spanish day, which is what keeps the evaluation
set valid across them.

**Official embeddings exist only for the `gpu` collection** (verified
over 30 days of it4i publications, 2026-08-17: `gpu.owie` is the sole
`owie` resource; `main`, `legal`, `curlie_full`, and `licenses` publish
text-only `owi` datasets). RAG over any other collection would require
locally computed embeddings and a loader for the manifest-less text
format.

## 8. Access and parsing problems encountered

- Historical (pre-v2.0.0) embeddings datasets used positional
  id↔vector association and one complete public dataset was corrupted by
  a constant +1024 offset (reported upstream and confirmed; root cause a
  GPU-pipeline race). v2.0.0's explicit `(record_id, chunk_idx)` keys
  remove that class of failure; this project verifies id↔vector
  alignment independently (join completeness at load, count equality at
  ingest, self-recovery queries against the built index).
- The `owilix` version installed in early July (5.5.1) predates the
  v2.0.0 publication wave and its command syntax; version 5.10.12 works
  as documented above.
- No other access or parsing failures: load with zero
  rejected records.
