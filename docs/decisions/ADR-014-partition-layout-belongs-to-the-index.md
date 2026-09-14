# ADR-014 — The partition layout belongs to the index; workers belong to the query

Increment: 26 (phase 2)

## Problem

Until this record, `retrieval.mode` chose two things at once: whether the
store factory built the partitioned layout (`PartitionedVectorStore`
with `retrieval.partitions`) and which retriever ran. A partitioned
index could only be opened in `collective` mode, and `sequential` mode
refused any partition count other than one. So the partition count, an
ingest-time property sealed on disk in the collection names
`<collection>-pNNofPP` and their metadata, lived in the query section of
the configuration; and the legitimate case "search a P-partition index
one partition at a time" had to be spelled `mode: collective,
executor: serial, workers: 1`.

The review of the report put it plainly: partitions should not be tied
to workers; the partitions of an index are the same whoever queries it,
only the worker count may differ. The seams already agreed —
`CollectiveRetriever` takes P from the store and W from the executor, and
the worker-scaling experiment swept W over a fixed P = 8 — but the
configuration and the two factories that read it did not.

## Options considered

1. **Keep `mode`, relax the validation** (allow `partitions > 1` in
   sequential mode, meaning "P searches in series inside the store").
   Rejected: two code paths for the same measurement (the store's
   internal loop and the serial executor), and `mode` would still decide
   the layout.
2. **Discover the layout on disk** (list the `-pNNofPP` collections) and
   drop `partitions` from the configuration. Rejected: the same YAML
   serves `ingest`, which creates the index and cannot discover anything;
   the identity guard compares what the caller declares, so an
   undeclared P would weaken it; and enumerating collections is backend
   code that would leak into the orchestrator (Rule 2).
3. **Move `partitions` to `vector_store`, derive `mode`, keep
   `executor` and `workers` in `retrieval` as query-time choices.**
   Chosen.

## Decision

- `vector_store.partitions` (optional integer ≥ 1) declares the index
  layout. Absent means one monolithic collection; `1` or more means the
  partitioned layout `<collection>-pNNofPP`. `1` is not the monolith: it
  takes the partitioned code path, so the first point of a scaling curve
  runs the same code as the others.
- `retrieval` keeps `k`, `executor` (`serial` | `threads`) and
  `workers`, all optional. `executor` and `workers` stay `None` when
  absent, are only allowed on a partitioned index, and
  `resolved_execution()` turns absence into `threads` and `1`. `serial`
  admits `workers` absent or `1`; any other value is refused because the
  serial executor would ignore it and a manifest must not record a knob
  that did nothing.
- The factories follow the store: `build_vector_store` wraps the backend
  in `PartitionedVectorStore` if and only if `partitions` is set;
  `build_retriever` returns `CollectiveRetriever` if and only if the store
  is partitioned, `SequentialRetriever` otherwise; `build_executor` reads
  the resolved values and refuses, by name, a partitioned store opened by
  a configuration that declares no layout.
- `retrieval.mode` is no longer declared. `layout_mode(config)` derives
  `sequential` or `collective` for query manifests, benchmark rows and
  figures, so the vocabulary of the published result files is unchanged.
  The key is still accepted when it agrees with the derived value — the
  written report reproduces such files — and refused with a message when
  it contradicts it.
- `retrieval.partitions` is refused with a message naming its new home
  (`1` under the old schema was the monolith: delete the key; `N > 1`:
  declare `vector_store.partitions: N`).

## Reason

The partition count is part of the index identity (ADR-006: "P is part
of a partitioned index's identity, never a runtime knob over an existing
index"); the worker count is not. Putting each where it belongs makes
"same index, different concurrency" expressible in configuration
(`collective_8.yaml` and `collective_8_serial.yaml` share their
`vector_store` section byte for byte) and removes the only rule that
coupled them. A backend that partitions natively has the same split —
partition count fixed at build time, number of workers chosen per query
— so the configuration surface will not need to change again for it.

## Consequences

- Configurations written for the previous schema fail with a named
  message. The ten shipped files and the tests were migrated in the same
  increment; `collective_8_serial.yaml` was added.
- Result rows and query manifests keep the keys `mode`, `partitions`,
  `workers` and `executor`. For a monolithic index the last three are
  now `null` (previously `1`, `1` and `threads`, schema defaults rather
  than measured settings), which `plot_results.py` tolerates because it
  selects rows by `mode`. Published result files are not touched; they
  remain evidence of the runs that produced them.
- The written report describes the previous schema (`mode: sequential`;
  "sequential mode does not admit partitions"); the tag `v1.0-memoria`
  freezes the code it describes. Removing the accepted-when-consistent
  `mode` key is a later decision.
- ADR-004's example of a value-dependent shape ("retrieval mode") is
  superseded by the rules above; ADR-004 itself (validate the shape at
  load time, closed sets only where the shape changes) still holds. The
  shape-changing value is now whether `vector_store.partitions` is
  present.
- The schema tables and the passages that described the old rules
  (`docs/usage.md`, `architecture.md`, `adding-a-vector-store.md`,
  README, ADR-004's note) change in the same increment, so the
  repository never documents a rule the code no longer enforces.
