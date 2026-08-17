# rag-framework

Modular Collective RAG Pipeline over the Open Web Index (OWI): ingestion,
chunking, embeddings, a replaceable vector-store backend (ChromaDB first),
sequential and partitioned-collective retrieval, optional generation, and
reproducible benchmarks.

Research question: *can a local implementation evolve into a serverless,
distributed one by replacing components instead of rewriting the application?*

## Setup

Requires Python ≥ 3.12 and a Linux/macOS environment.

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest        # 191 tests, no dataset needed
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

Every run writes a JSON report under `results/` carrying the effective
configuration, git commit, machine, dependency versions, and dataset
version, so any result can be reproduced.

## Layout

- `src/rag_framework/` — the library: `models` (canonical types),
  `config`, `loaders/`, `chunking/`, `embeddings/`, `vectorstores/`,
  `retrieval/`, `orchestration/`, `metrics/`. Each component sits
  behind a small interface; backends are selected in `configs/*.yaml`.
- `configs/` — pipeline configurations. `docs/` — dataset and
  architecture documentation. `evaluation/` — the evaluation query
  set. `results/` — committed run reports. `tests/` — unit and
  integration suites (no dataset required).
