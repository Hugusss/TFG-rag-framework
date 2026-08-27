# rag-framework

Modular Collective RAG Pipeline over the Open Web Index (OWI): ingestion,
chunking, embeddings, a replaceable vector-store backend (ChromaDB first),
sequential and partitioned-collective retrieval, optional generation, and
reproducible benchmarks.

Design goal: a modular, extensible RAG library that runs at small scale
on one machine and is ready to scale out later by replacing components,
without rewriting the application.

## Setup

Requires Python ≥ 3.12 and a Linux/macOS environment.

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"        # add ",plots" to regenerate figures
.venv/bin/python -m pytest               # 408 tests, no dataset needed
```

## Data

The corpus is the OWI v2.0.0 `gpu` collection, Spanish partition,
crawl 2026-07-28 (see `docs/dataset.md` for its full characterization
and for how to obtain any day with the `owilix` CLI). Place the paired
parquet trees under `./data/owi/` — the loader discovers and pairs
shards recursively. Data is never committed.

## Usage

Build the persistent index (idempotent — safe to run twice):

```bash
.venv/bin/python -m rag_framework ingest --config configs/local_chroma.yaml
```

Query it (retrieval-only; the first query downloads the embedding model
for query encoding, ~600 MB, one time):

```bash
.venv/bin/python -m rag_framework query \
  --config configs/local_chroma.yaml \
  --question "monumentos de la Alhambra de Granada" \
  --retrieval-only --k 5
```

Every run writes a JSON report under `metrics.output` (`./results` in
every shipped configuration) carrying the effective
configuration, git commit, machine, dependency versions, and dataset
version, so any result can be reproduced.

### Collective mode

The same corpus can be ingested into P partitions — one collection per
partition, documents assigned by a stable hash — and searched by P
workers with a global top-k merge. It is a config change only:

```bash
.venv/bin/python -m rag_framework ingest --config configs/collective_4.yaml
.venv/bin/python -m rag_framework query --config configs/collective_4.yaml \
  --question "monumentos de la Alhambra de Granada" --retrieval-only --k 5
```

Results carry `partition_id` / `worker_id` provenance and per-partition
timings. This is a *local simulation* of collective retrieval (threads
over local collections), measured, not a distributed system.

### Experiments and figures

`benchmarks/` holds one script per experiment: dataset-size scaling,
partition and worker scaling, chunk-size effect, corpus-growth quality
and collective correctness against an exact brute-force reference. Plus
`plot_results.py`, which regenerates `figures/` from the committed raw
results:

```bash
.venv/bin/python benchmarks/plot_results.py \
    --pin scaling-workers=scaling-workers-20260821T143459.302145Z.json \
    --pin correctness=correctness-20260821T141643.871410Z.json
```

The pins are not optional. Two kinds have a second committed campaign
at 1.36M vectors, and the script refuses to guess which one a figure
draws. Commands, measurement policy and caveats:
`docs/experiment-guide.md`.

## Documentation

- `docs/usage.md` — how to drive the pipeline from Python and from the
  command line, and every configuration key with its type, default and
  the combinations that are refused.
- `docs/architecture.md` — seams, data flow, boundaries, validation,
  what is and is not claimed.
- `docs/experiment-guide.md` — reproducing every experiment and figure.
- `docs/adding-a-vector-store.md` — the backend contract and the one
  factory branch to add.
- `docs/limitations.md` — measured limits of scope, performance, data
  and evaluation.
- `docs/dataset.md` — the corpus, its quirks and how to obtain it.
- `docs/decisions/` — the thirteen architecture decision records, with
  an index in `docs/decisions/README.md`.

## Layout

- `src/rag_framework/` — the library: `models` (canonical types),
  `config`, `loaders/`, `chunking/`, `embeddings/`, `vectorstores/`,
  `retrieval/`, `executors/`, `generation/`, `orchestration/`,
  `metrics/`. Each component sits behind a small interface; backends
  are selected in `configs/*.yaml`.
- `configs/` — ten pipeline configurations: the sequential baseline,
  collective layouts for 1, 2, 4 and 8 partitions, two larger corpora,
  two search widths at full scale and one with real generation. What
  each one demonstrates: `docs/usage.md`. `benchmarks/` — experiment
  and plot scripts. `evaluation/` — the evaluation query set.
  `results/` — committed raw run reports. `figures/` — regenerated
  plots. `tests/` — unit and integration suites (no dataset
  required).
