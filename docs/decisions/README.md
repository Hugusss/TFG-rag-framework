# Architecture decision records

One record per non-trivial decision, written when the decision was
taken, with its problem, the options considered, the choice and its
consequences. The code cites them by number where their effect is
visible.

| Record | Decision |
|---|---|
| [ADR-001](ADR-001-document-id-scheme.md) | Document identifier scheme: adopt publisher ids |
| [ADR-002](ADR-002-official-artifacts-as-adapters.md) | Official OWI chunks and embeddings enter as adapters of existing seams |
| [ADR-003](ADR-003-score-convention.md) | Score convention: similarity, higher is better |
| [ADR-004](ADR-004-config-validation.md) | Configuration: stdlib dataclasses + explicit validation |
| [ADR-005](ADR-005-embedding-interface-takes-chunks.md) | `embed_documents` takes Chunks, not bare texts |
| [ADR-006](ADR-006-partition-assignment-and-merge-order.md) | Collective retrieval core: partition assignment and global merge rules |
| [ADR-007](ADR-007-blocksdb-adapter-reserved.md) | BlocksDBStore adapter (reserved; not executed) |
| [ADR-008](ADR-008-token-policy-and-overlap.md) | Recursive chunking: token policy and overlap placement |
| [ADR-009](ADR-009-query-encoding-local.md) | Query encoding runs locally, byte-exact with the rack reference |
| [ADR-010](ADR-010-generation-mock-first.md) | Generation: mock-first, real backend deferred |
| [ADR-011](ADR-011-deterministic-corpus-sampling.md) | Deterministic, nested corpus sampling for dataset-size scaling |
| [ADR-012](ADR-012-matplotlib-for-plots.md) | matplotlib as an optional extra for the required plots |
| [ADR-013](ADR-013-ollama-generation.md) | Real generation through Ollama; model choice |

ADR-007 is reserved rather than implemented: it scopes a second vector
store that the project deferred and lists as future work. ADR-010 and
ADR-013 are read together, the first for why generation shipped mocked
and the second for the real model added beside it.
