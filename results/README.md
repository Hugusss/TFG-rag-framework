# Committed results

The files here are the runs the figures and the documentation quote. Each
carries a manifest with the effective configuration, the git commit it was
run from, the machine, the dependency versions and the dataset version
(`docs/experiment-guide.md` §4). This note says what the recorded commit
does and does not pin.

The commits recorded in the manifests belong to the incremental history
this repository was restored to; the table maps each one to its hash in the
present history and to the commit that committed the result. A commit marked
`-dirty` means the working tree held uncommitted changes when the run was
measured. Those runs fall in two classes:

- **A** (5 files): the code did not change between the recorded commit and
  the commit that added the result; the uncommitted change was the report of
  the run made seconds earlier, which `git status` counts as untracked. The
  code that ran is the recorded commit, by inference.
- **B** (15 files): scripts or configurations that the run names were committed
  later, so the recorded commit does not hold the code that produced the
  numbers. The commit that added the result is the earliest one that holds
  them; the exact working tree at run time is not recoverable.

The remaining 5 files were run from a clean tree.

| File | What it is | Recorded commit | Same commit here | Committed with | Class | Note |
|---|---|---|---|---|---|---|
| `chunkexp-large-20260820T233712.156905Z.json` | chunk-size experiment, 1000/100 | `8ca3010-dirty` | `e3f4086` | `a0c5338` | B | ran `benchmarks/run_chunking.py`, `src/rag_framework/embeddings/cache.py` before they were committed |
| `chunkexp-medium-20260820T194148.160857Z.json` | chunk-size experiment, 500/50 | `8ca3010-dirty` | `e3f4086` | `a0c5338` | B | ran `benchmarks/run_chunking.py`, `src/rag_framework/embeddings/cache.py` before they were committed |
| `chunkexp-publisher-20260819T165103.419735Z.json` | chunk-size experiment, publisher windows | `9b3b51f-dirty` | `f73e51f` | `a0c5338` | B | ran `benchmarks/run_chunking.py`, `src/rag_framework/embeddings/cache.py`, `src/rag_framework/metrics/quality.py` before they were committed |
| `chunkexp-small-20260820T124446.205203Z.json` | chunk-size experiment, 200/20 | `9b3b51f-dirty` | `f73e51f` | `a0c5338` | B | ran `benchmarks/run_chunking.py`, `src/rag_framework/embeddings/cache.py`, `src/rag_framework/metrics/quality.py` before they were committed |
| `correctness-20260821T141643.871410Z.json` | collective correctness, Spanish slice | `47ed0f0` | `3203e13` | `81bf9cd` | clean |  |
| `correctness-20260825T114104.618065Z.json` | collective correctness, full day | `0cbb8e6-dirty` | `cf3b964` | `260fb51` | B | ran `benchmarks/run_correctness.py`, `benchmarks/run_quality.py`, `configs/fullday_0728.yaml`, `configs/fullday_0728_collective8.yaml`, `configs/fullday_0728_ef800. before they were committed |
| `ingest-20260814T085932.953305Z.json` | ingest, Spanish slice | `0086688` | `3933532` | `6ce3558` | clean |  |
| `ingest-20260814T085940.407607Z.json` | ingest, Spanish slice | `0086688-dirty` | `3933532` | `6ce3558` | A | the tree held only the previous run's report, not yet committed |
| `ingest-20260821T135546.515714Z.json` | ingest, 2 partitions | `b94d334` | `ee219df` | `eac884e` | clean |  |
| `ingest-20260821T135554.717415Z.json` | ingest, 4 partitions | `b94d334-dirty` | `ee219df` | `eac884e` | A | the tree held only the previous run's report, not yet committed |
| `ingest-20260821T135604.050535Z.json` | ingest, 8 partitions | `b94d334-dirty` | `ee219df` | `eac884e` | A | the tree held only the previous run's report, not yet committed |
| `ingest-20260821T143228.072357Z.json` | ingest, 1 partition | `ad51698` | `6c797a5` | `ee47992` | clean |  |
| `ingest-20260824T150433.258139Z.json` | ingest, growth series | `0cbb8e6-dirty` | `cf3b964` | `260fb51` | B | ran `benchmarks/run_correctness.py`, `benchmarks/run_quality.py`, `configs/fullday_0728.yaml`, `configs/fullday_0728_collective8.yaml`, `configs/fullday_0728_ef800. before they were committed |
| `ingest-20260825T083509.493877Z.json` | ingest, full day | `0cbb8e6-dirty` | `cf3b964` | `260fb51` | B | ran `benchmarks/run_correctness.py`, `benchmarks/run_quality.py`, `configs/fullday_0728.yaml`, `configs/fullday_0728_collective8.yaml`, `configs/fullday_0728_ef800. before they were committed |
| `ingest-20260825T102720.550058Z.json` | ingest, full day, 8 partitions | `0cbb8e6-dirty` | `cf3b964` | `260fb51` | B | ran `benchmarks/run_correctness.py`, `benchmarks/run_quality.py`, `configs/fullday_0728.yaml`, `configs/fullday_0728_collective8.yaml`, `configs/fullday_0728_ef800. before they were committed |
| `quality-fullday_0728-20260825T085753.511636Z.json` | corpus quality, full day | `0cbb8e6-dirty` | `cf3b964` | `260fb51` | B | ran `benchmarks/run_correctness.py`, `benchmarks/run_quality.py`, `configs/fullday_0728.yaml`, `configs/fullday_0728_collective8.yaml`, `configs/fullday_0728_ef800. before they were committed |
| `quality-fullday_0728_collective8-20260825T103744.107611Z.json` | corpus quality, full day, 8 partitions | `0cbb8e6-dirty` | `cf3b964` | `260fb51` | B | ran `benchmarks/run_correctness.py`, `benchmarks/run_quality.py`, `configs/fullday_0728.yaml`, `configs/fullday_0728_collective8.yaml`, `configs/fullday_0728_ef800. before they were committed |
| `quality-fullday_0728_ef800-20260825T091632.430150Z.json` | corpus quality, full day, ef_search 800 | `0cbb8e6-dirty` | `cf3b964` | `260fb51` | B | ran `benchmarks/run_correctness.py`, `benchmarks/run_quality.py`, `configs/fullday_0728.yaml`, `configs/fullday_0728_collective8.yaml`, `configs/fullday_0728_ef800. before they were committed |
| `quality-local_chroma-20260825T083737.876189Z.json` | corpus quality, Spanish slice | `0cbb8e6-dirty` | `cf3b964` | `260fb51` | B | ran `benchmarks/run_correctness.py`, `benchmarks/run_quality.py`, `configs/fullday_0728.yaml`, `configs/fullday_0728_collective8.yaml`, `configs/fullday_0728_ef800. before they were committed |
| `quality-spa_growth-20260825T083817.682478Z.json` | corpus quality, growth series | `0cbb8e6-dirty` | `cf3b964` | `260fb51` | B | ran `benchmarks/run_correctness.py`, `benchmarks/run_quality.py`, `configs/fullday_0728.yaml`, `configs/fullday_0728_collective8.yaml`, `configs/fullday_0728_ef800. before they were committed |
| `query-20260817T133630.935019Z.json` | query, retrieval only | `009c995-dirty` | `c82dbec` | `1f659ae` | B | ran `src/rag_framework/__main__.py`, `src/rag_framework/embeddings/precomputed.py`, `src/rag_framework/metrics/manifest.py`, `src/rag_framework/orchestration/pipeli before they were committed |
| `scaling-dataset-20260821T145658.911056Z.json` | dataset-size scaling | `78de912` | `5bfe28d` | `6b0d821` | clean |  |
| `scaling-partitions-20260821T143346.750630Z.json` | partition scaling | `ad51698-dirty` | `6c797a5` | `ee47992` | A | the tree held only the previous run's report, not yet committed |
| `scaling-workers-20260821T143459.302145Z.json` | worker scaling, Spanish slice | `ad51698-dirty` | `6c797a5` | `ee47992` | A | the tree held only the previous run's report, not yet committed |
| `scaling-workers-20260825T104052.150672Z.json` | worker scaling, full day | `0cbb8e6-dirty` | `cf3b964` | `260fb51` | B | ran `benchmarks/run_correctness.py`, `benchmarks/run_quality.py`, `configs/fullday_0728.yaml`, `configs/fullday_0728_collective8.yaml`, `configs/fullday_0728_ef800. before they were committed |

Figures 1 to 8 read `scaling-dataset`, the four `chunkexp` files,
`scaling-workers-20260821T143459` (pinned), `scaling-partitions` and
`correctness-20260821T141643` (pinned); no figure reads the `ingest`,
`query` or `quality` files. New campaigns are measured from a clean tree at
a commit that is already published, so this table does not grow a class.
