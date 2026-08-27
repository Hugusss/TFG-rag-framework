# Adding a vector-store backend

A new backend is one module and one factory branch. `RAGPipeline`, the
retrievers, the partitioned store and the benchmarks do not change.

## 1. Implement the contract

Create `src/rag_framework/vectorstores/<backend>.py` with a class
deriving from `VectorStore`. That module is the **only** place the
backend's client library may be imported. The contract
(`vectorstores/base.py`):

- `create_or_open(collection_name)` — idempotent. Stamp the collection
  with the embedding identity you were constructed with (`model_id`,
  `dimension`, `normalized`, and the partition layout keys when
  present) and **refuse** to open an existing collection whose stored
  identity conflicts: an index built in one embedding space must never
  be queried in another.
- `add(chunks, embeddings)` — one vector per chunk; reject mismatched
  lengths and duplicate chunk ids within a batch; **upsert** so
  re-ingesting identical ids never duplicates rows. Stored per-row
  metadata is the chunk's metadata plus the reserved keys `document_id`
  and `position`; a reserved-key collision or a non-scalar value is an
  error, never a silent drop.
- `search(query_embedding, k, filters=None)` — at most `k` results,
  ordered by `SearchResult.score` as a **similarity, higher is better**.
  Convert the backend's native metric at this boundary (Chroma:
  `score = 1 − cosine distance`). `filters` is a flat `{key: value}`
  equality mapping; translate it to the native syntax here.
- `count()` — vectors currently stored.
- `reset()` — empty the collection, leave it usable.
- `iter_vectors()` — stream `(chunk_id, document_id, vector)` for every
  row in a stable order, vectors as Python lists. The correctness
  experiment computes its exact reference from this.

Wrap every backend exception in `VectorStoreError` with a message that
names the collection and the operation. Never let a client exception
escape raw.

## 2. Register it

In `orchestration/pipeline.py::build_vector_store`, add a branch on
`config.vector_store.type` that constructs your store with the
provider's identity metadata, mirroring the Chroma branch. Collective
mode wraps whatever the branch returns in `PartitionedVectorStore`
automatically, so your backend gets partitioned layouts for free —
provided it accepts the collection names `<name>-p{i:02d}of{P:02d}` and
the two extra metadata keys.

`vector_store.type` is an open set: no change to `config.py` is needed.
The factory's error message lists known types; add yours.

## 3. Test it

- Copy `tests/unit/test_chroma_store.py` and point it at your class:
  the contract tests (identity guard, idempotent add, duplicate
  rejection, metadata rules, similarity ordering, filters, reset,
  `iter_vectors`) are backend-agnostic by design.
- `tests/unit/test_partitioned_store.py::TestOnChroma` shows the
  partitioned layout test; run the same against your backend.
- One end-to-end run: `tests/integration/test_ingest.py` with
  `vector_store.type` switched to your backend on the synthetic corpus.

## 4. Decisions a remote or specialised backend must make

- **Index size.** `IngestionReport.index_size_bytes` is currently the
  size of the local store directory, computed by the pipeline. A remote
  backend has no directory; the measurement must become a store method
  (`index_size_bytes()`), and the pipeline's helper is the one place to
  change. This deferral is deliberate and documented in
  `orchestration/pipeline.py`.
- **`iter_vectors` over a network** must page; the exact reference
  loads the whole collection into memory (≈ 50 MB for 6,240 × 1,024
  floats), which is fine locally and needs a budget remotely.
- **Identity guard semantics** — where the backend keeps collection
  metadata, and what happens when it cannot (then refuse to open
  without an explicit override, never guess).
- **Integer ids.** Backends that key vectors by integers (FAISS-style)
  need a deterministic `chunk_id ↔ int` mapping stored alongside the
  index, and `iter_vectors` must return the chunk ids, not the ints.
- **Approximate search parameters** (`ef`, `nprobe`, …) are part of the
  index identity for correctness comparisons; record them in collection
  metadata and in the run manifest.
