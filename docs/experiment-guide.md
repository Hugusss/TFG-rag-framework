# Experiment guide

Every experiment is a script under `benchmarks/` that measures through
the seams and writes **one raw JSON result file** through the metrics
layer. Figures are regenerated from those files and never drawn from
typed numbers. This guide gives the commands, the measurement policy
and the caveats needed to read the results.

## 0. Prerequisites

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev,plots]"   # plots extra only for figures
.venv/bin/python -m pytest                # no dataset needed
```

Place the corpus under `./data/owi/` (see `docs/dataset.md`). The
publisher configurations use the official vectors, so their ingests
take seconds; the first query downloads the query encoder once.

## 1. Build the indexes

```bash
.venv/bin/python -m rag_framework ingest --config configs/local_chroma.yaml    # sequential baseline
for p in 1 2 4 8; do
  .venv/bin/python -m rag_framework ingest --config configs/collective_$p.yaml # P partitions
done
```

Each layout has its own store directory (`state/chroma`,
`state/chroma-collective-p{P}`); the partition count is encoded in the
collection names. Ingest is idempotent, but see the index-size caveat
below before re-ingesting into a directory you intend to measure.

## 2. Run the experiments

| experiment | script | output prefix | possible duration |
|---|---|---|---|
| A — dataset-size scaling | `benchmarks/run_dataset_scaling.py` | `scaling-dataset` | ~2 min (builds 4 fresh subsets) |
| B — partition scaling, C — worker scaling | `benchmarks/run_scaling.py` (`--experiment partitions\|workers\|both`) | `scaling-partitions`, `scaling-workers` | ~5 min |
| D — chunk-size effect | `benchmarks/run_chunking.py` | `chunkexp-<configuration>` | seconds for `publisher`; **2.5–4 h of CPU encoding per self-chunked configuration** |
| E — collective correctness | `benchmarks/run_correctness.py` | `correctness` | ~1 min |
| figures | `benchmarks/plot_results.py` | `figures/NN-*.png` | seconds |

All scripts accept `--queries`, `--output`, `--k`, and where relevant
`--repetitions` and `--warm-up`; run any with `-h`.

Experiment D notes: the script runs **one child process per
configuration** (hours of CPU encoding fragment the allocator without
bound; a single long process was killed by the OOM killer at 19 GB),
wraps the live encoder in a disk cache under `state/chunkexp/<config>-
embedding-cache/` so an interrupted run resumes losing at most 512
chunks, and reports cache hits so a partially cached embedding time is
declared as such. Run it detached and with the allocator arena cap:

```bash
MALLOC_ARENA_MAX=2 systemd-run --user --unit=chunkexp --collect \
  -p WorkingDirectory=$PWD -p StandardOutput=append:$PWD/chunkexp.log -p StandardError=append:$PWD/chunkexp.log \
  $PWD/.venv/bin/python benchmarks/run_chunking.py
```

(or `nohup … &` on systems without a user manager). Prevent suspension
for the duration; a laptop that sleeps mid-run produces timings that
must be discarded.

## 3. Measurement policy

Every result file records what was measured and how:

- **Warm-up**: a stated number of unmeasured queries per configuration
  (default 3; the first also loads the query encoder). The policy text
  is in the file (`warm_up_policy`).
- **Repetitions**: latency is measured over all evaluation queries ×
  `repetitions` (default 5; Experiment D uses 2) and summarised as
  `n / min / median / p95 / max / mean / std`. Quality is scored once —
  retrieval is deterministic, repeating it would manufacture sample
  size.
- **One query encoder** is shared by every configuration within a run,
  so all configurations see the identical query vector and the model
  loads once.
- **Manifest**: effective configuration, dataset version, git commit
  (`-dirty` if the tree had uncommitted changes), hostname, platform,
  Python and dependency versions, timestamp. Figures print the source
  file and commit.
- **Fresh indexes for size and build-time numbers.** A directory that
  has been re-ingested carries storage history (the same 6,240-vector
  collection measured 114 MB freshly built and 138 MB after several
  re-ingests). Experiment A builds each subset in a new per-run
  directory under `state/dataset-scaling/<run-id>/` and never deletes;
  delete old runs by hand when they are no longer needed. Never delete
  a store directory while a process holds it open — the backend caches
  clients per path and the next write fails with a read-only-database
  error.
- **Exact reference for correctness.** Both retrieval modes search
  approximate (HNSW) indexes, so Experiment E compares each against a
  brute-force cosine top-k over the vectors the store actually holds
  (`VectorStore.iter_vectors`, pure Python); differences are classified
  as tie reorders, misses of one index, or candidates outside the exact
  top-k — never reported as bare "disagreement".

## 4. Reading the result files

- `ingest-*.json`: the `IngestionReport` plus per-seam details
  (rejections sample, documents without chunks, cache hits, sampled-out
  documents) and the manifest.
- `query-*.json`: metrics (per-stage seconds, collective stages when
  applicable), compact source rows with provenance, embedding identity,
  query-encoder identity.
- `chunkexp-*.json`, `scaling-*.json`, `correctness-*.json`: one file
  per experiment run; `rows` / `layouts` hold per-configuration blocks;
  per-query detail is kept so any aggregate can be recomputed.
- Committed files under `results/` are the runs the figures and the
  report quote; `python benchmarks/plot_results.py` uses the latest
  file of each kind.

## 5. Caveats that travel with the numbers

- The **evaluation set** (30 queries, 26 answerable) was written against
  the full corpus. On dataset subsets, recall tracks the fraction of
  relevant documents the subset still contains — that fraction is
  reported beside recall. Thirty queries detect a broken configuration;
  they do not rank close ones (one query moves a mean by 3.8 points).
- **Query encoding dominates latency** on CPU (~85 ms of ~90 ms), so
  retrieval-mode differences (2–13 ms) are a search-stage phenomenon;
  totals move by a few percent at most.
- **HNSW is approximate.** The single collection missed one true
  neighbour on one query; the partitioned layouts did not. Compare
  against the exact reference, not against the single collection.
- **Index size** depends on build history; compare fresh builds only.
- **Chunk sizes are in whitespace words**; the measured ratio is 2.33
  subword tokens per word for this corpus and tokenizer. Larger
  `minimum_tokens` floors exclude short documents from the index (491
  of 2,363 at 1000/100), a recall ceiling the chunk-size results report.
- The collective mode is a **single-machine simulation** (threads over
  local collections); worker times under contention and fan-out
  overheads are properties of this machine, not of a distributed
  deployment.

## 6. Adding an experiment

1. Measure through the seams (`RAGPipeline`, a retriever, a store) —
   never through a backend client directly.
2. Put any new metric or comparison as a pure function in
   `metrics/quality.py` (or a sibling module) with unit tests; the
   script only orchestrates.
3. State warm-up and repetitions in the payload; summarise with
   `metrics.quality.summarize`; keep per-query detail.
4. Write the payload with `metrics.manifest.write_report(payload,
   output_dir, "<prefix>")` and include `run_manifest(config)`.
5. Add a figure to `benchmarks/plot_results.py` that reads the file and
   stamps its source.
