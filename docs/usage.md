# Using the library

How to drive the pipeline from Python and from the command line, and
what every configuration key does. The design behind these interfaces
is in `architecture.md`; this document is the reference you keep open
while writing a config.

## 1. Install

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"     # add ",plots" to regenerate figures
.venv/bin/python -m pytest            # no dataset required
```

The distribution is `rag-framework`; the importable package is
`rag_framework`. There is no console script: the command line is
`python -m rag_framework`.

## 2. Five minutes

Ingest a corpus and ask one question, using the shipped baseline
configuration:

```bash
python -m rag_framework ingest --config configs/local_chroma.yaml
python -m rag_framework query  --config configs/local_chroma.yaml \
    --question "monumentos de la Alhambra" --retrieval-only --k 5
```

The same thing from Python:

```python
from rag_framework.orchestration.pipeline import RAGPipeline

pipeline = RAGPipeline.from_config("configs/local_chroma.yaml")
report = pipeline.ingest()
print(report.chunks_created, report.final_vector_count)

result = pipeline.query("monumentos de la Alhambra", k=5, retrieval_only=True)
for hit in result.sources:
    print(f"{hit.score:.4f}  {hit.document_id[:12]}  {hit.text[:60]}")
```

Switching to collective retrieval is a configuration change, not a code
change: use `configs/collective_8.yaml` instead.

## 3. From Python

### Entry points

The package deliberately exposes no re-exports: `from rag_framework
import RAGPipeline` does not work, and every import names its module.

```python
from rag_framework.config import load_config, PipelineConfig, ConfigError
from rag_framework.orchestration.pipeline import RAGPipeline
```

| Call | Signature | Notes |
|---|---|---|
| `load_config` | `(path: str \| Path) -> PipelineConfig` | Parses, validates and freezes. Raises `ConfigError` (a `ValueError`) and never returns a partially valid config. |
| `RAGPipeline` | `(config: PipelineConfig)` | The constructor does real work: it builds all seven components and times that in `.setup_seconds`. With `embedding.provider: precomputed` it reads and indexes the whole embeddings parquet here. |
| `RAGPipeline.from_config` | `(path) -> RAGPipeline` | Shorthand for `RAGPipeline(load_config(path))`. |
| `.ingest()` | `() -> IngestionReport` | No arguments: the corpus is always `dataset.path`. Idempotent, streams in batches of 512 chunks, times embedding and indexing separately. |
| `.query()` | `(question: str, k: int \| None = None, *, retrieval_only: bool = False) -> RAGResult` | `k=None` uses `retrieval.k`. Opens the collection lazily. Querying an empty collection is an error, never an empty list. |

After construction the pipeline exposes `.config`, `.loader`,
`.chunker`, `.embedding_provider`, `.vector_store`, `.retriever`,
`.generator` and `.setup_seconds`.

### Building components yourself

Seven factories build one component each from a config. The benchmarks
use them to share a single query encoder across configurations, which
avoids reloading the model and duplicating the vector table:

```python
from rag_framework.orchestration.pipeline import (
    build_loader, build_chunker, build_embedding_provider,
    build_vector_store, build_retriever, build_executor, build_generator,
)

provider = build_embedding_provider(config)          # loaded once
store    = build_vector_store(config, provider)
retriever = build_retriever(config, provider, store)  # reuses both
```

Each factory raises `ConfigError` listing the names it knows.

### Varying one setting

`PipelineConfig` and its seven section dataclasses are frozen, so
variations are made with `dataclasses.replace`. This is how the scaling
experiments sweep workers without touching a YAML file:

```python
from dataclasses import replace

for workers in (1, 2, 4, 8):
    variant = replace(config, retrieval=replace(config.retrieval, workers=workers))
```

### Decorators reachable only from Python

Two wrappers have no configuration name and are applied in code:

- `CachedEmbeddingProvider(inner, cache_dir)` persists vectors as
  parquet shards so an interrupted encode resumes where it stopped.
- `SampledLoader` is not named either, but it is applied for you when
  `dataset.sample_percent < 100`.

## 4. From the command line

`python -m rag_framework` takes one of two subcommands.

| Subcommand | Flags | |
|---|---|---|
| `ingest` | `--config PATH` | required |
| `query` | `--config PATH` | required |
| | `--question TEXT` | required |
| | `--k N` | optional; defaults to `retrieval.k` |
| | `--retrieval-only` | skips generation even if the config enables it |

Both print a JSON report to stdout and then write it under
`metrics.output`. Exit codes: `0` success, `1` a typed user error
(config, loader, embedding, vector store, generation), `2` argparse
usage, `3` the work succeeded but the report file could not be written.

The `benchmarks/` scripts are documented in `experiment-guide.md`.

## 5. Configuration reference

A configuration is a YAML mapping of sections. Six are required:
`dataset`, `chunking`, `embedding`, `vector_store`, `retrieval` and
`metrics`. `generation` is optional and defaults to a disabled mock.

Four rules hold everywhere. Unknown sections and unknown keys are
rejected by name. Duplicate keys are rejected instead of last-wins.
Types are exact, so `k: "10"` is an error rather than the integer ten.
And optional means absent: writing `key: null` is rejected like any
other wrong type, so "not set" has exactly one spelling.

### `dataset`

| Key | Type | Default | Required | Notes |
|---|---|---|---|---|
| `loader` | str | | yes | `owi` is the only registered name. |
| `path` | str | | yes | Directory holding the paired parquet trees. Also where `precomputed` reads the official vectors. |
| `version` | str | | no | Informational; recorded in the manifest as `dataset_version`. |
| `sample_percent` | int | `100` | no | 1 to 100. Below 100 the loader is wrapped in a deterministic, nested hash sample. |

### `chunking`

| Key | Type | Default | Required | Notes |
|---|---|---|---|---|
| `strategy` | str | | yes | `publisher_offsets` or `recursive`. |
| `target_tokens` | int | | with `recursive` | Positive. Rejected with `publisher_offsets`. |
| `overlap_tokens` | int | | with `recursive` | At least 0 and smaller than `target_tokens`. |
| `minimum_tokens` | int | | with `recursive` | At least 0 and at most `target_tokens`. It has no default on purpose: a silent default would silently drop documents. |

A token here is a whitespace-delimited word, not a model subword; the
measured ratio is 2.33 subwords per word.

### `embedding`

| Key | Type | Default | Required | Notes |
|---|---|---|---|---|
| `provider` | str | | yes | `precomputed` (official vectors) or `local` (encode here). |
| `model` | str | | yes | The corpus model is `jinaai/jina-embeddings-v5-text-small`. A query encoder declaring another model is refused. |
| `normalize` | bool | `true` | no | Only `true` is supported; `false` is rejected rather than accepted and ignored. |
| `batch_size` | int | | no | Positive. The live encoder uses 32 when unset. |

### `vector_store`

| Key | Type | Default | Required | Notes |
|---|---|---|---|---|
| `type` | str | | yes | `chroma` is the only registered name; the set is open by design. |
| `path` | str | | yes | Persistent client directory. Also what `index_size_bytes` measures. |
| `collection` | str | | yes | Base name. In collective mode the physical names are `<collection>-p{i:02d}of{P:02d}`. |
| `ef_search` | int | | no | HNSW search width, at least 1. Unset means the backend default, which is 100 in the pinned version. Raise it as the corpus grows. |

### `retrieval`

| Key | Type | Default | Required | Notes |
|---|---|---|---|---|
| `mode` | str | | yes | `sequential` or `collective`. |
| `k` | int | `10` | no | Positive. |
| `partitions` | int | `1` | no | Must be 1 when the mode is sequential. |
| `workers` | int | `1` | no | Must be 1 when the mode is sequential, and 1 when the executor is serial. |
| `executor` | str | `threads` | no | `serial` or `threads`. The key must be absent in sequential mode, even spelled with its default value. |

### `generation`

| Key | Type | Default | Required | Notes |
|---|---|---|---|---|
| `enabled` | bool | `false` | no | When false, `RAGResult.answer` is `None`. |
| `provider` | str | `mock` | no | `mock` or `ollama`. |
| `model` | str | | with `ollama` | Ollama model id, for example `ministral-3`. Rejected with any other provider. |
| `endpoint` | str | `http://localhost:11434` | no | A local port, in practice an SSH tunnel. Rejected with any other provider. |
| `timeout_seconds` | int | `120` | no | At least 1. Rejected with any other provider. |

### `metrics`

| Key | Type | Default | Required | Notes |
|---|---|---|---|---|
| `output` | str | | yes | Directory for the JSON reports; created if missing. |

### A configuration that encodes its own vectors

The shipped configs all use the official vectors. Chunking the text
yourself and encoding it locally is a supported path, and looks like
this:

```yaml
dataset:
  loader: owi
  path: ./data/owi
chunking:
  strategy: recursive
  target_tokens: 500
  overlap_tokens: 50
  minimum_tokens: 50
embedding:
  provider: local          # precomputed would be refused: see below
  model: jinaai/jina-embeddings-v5-text-small
vector_store:
  type: chroma
  path: ./state/chroma-recursive500
  collection: owi-spa0728-rec500-jinav5small-v1
retrieval:
  mode: sequential
  k: 10
metrics:
  output: ./results
```

Expect hours rather than seconds: encoding on CPU is what the official
vectors save you.

## 6. Combinations that are refused

Validation happens in three layers, and which layer catches a mistake
tells you how far the run got.

**On load, before touching disk or models.** `publisher_offsets` with
any chunk size, because the sizes come from the corpus; `recursive`
missing any of its three sizes; an overlap not smaller than the target,
or a minimum larger than it; `sequential` with more than one partition
or worker; `sequential` with an `executor` key at all; `serial` with
more than one worker; generation keys such as `model` or `endpoint`
with a provider other than `ollama`; and `ollama` without a model.

**On construction, before any work.** The important one is `recursive`
chunking with `precomputed` embeddings: official vectors exist only for
the publisher's chunks, so the pair is refused up front rather than
failing halfway through an ingest. Also caught here: `normalize: false`,
any unregistered name, and a generation provider outside `mock` and
`ollama` (the generator is built even when generation is disabled, so a
typo fails at startup and not at the first question).

**At run time.** Opening a collection whose sealed identity disagrees
with the current one in model, dimension, normalization or partition
layout; a collection whose distance space is not cosine; querying an
empty collection; a chunk with no official vector, which stops the
ingest instead of quietly re-encoding it; a query encoder from another
embedding space; and a non-finite score reaching the merge.

## 7. What is not configurable

Some constants live in code on purpose, so nobody goes looking for them
in YAML: the ingest batch of 512 chunks; the 10-character tolerance for
publisher windows that overshoot the text; the encoding recipe (CPU,
retrieval task, normalization on) that must match the corpus; the mock
snippet length; and the Ollama prompt limits, greedy decoding and fixed
refusal sentence.

## 8. The shipped configurations

| File | What it shows |
|---|---|
| `local_chroma.yaml` | The baseline: sequential, publisher chunks, official vectors. Ingests in seconds. |
| `collective_1/2/4/8.yaml` | Collective mode as a configuration change. `collective_1` exists so that the first point of a scaling curve runs the same code as P=8. |
| `spa_growth.yaml` | A ten-times larger Spanish corpus, and the only place `ef_search: 800` is demonstrated with its reason. |
| `fullday_0728.yaml` | Full scale, 75 languages, backend default width. |
| `fullday_0728_ef800.yaml` | The same index at width 800. It shares path and collection with the previous one on purpose: two widths over one index, not two indexes. |
| `fullday_0728_collective8.yaml` | Full scale split into eight partitions. |
| `generation_ollama.yaml` | Real generation. Retrieval is identical to the baseline and reuses its collection, so no re-ingest is needed. |
