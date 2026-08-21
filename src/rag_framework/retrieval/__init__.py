"""Retrieval seam: queries become ranked SearchResults here.

Retrievers own the query flow (embed, search, merge) and its per-stage
timing. They contain no backend calls of their own — vector search
happens behind the VectorStore seam — and no encoding logic — vectors
come from the EmbeddingProvider seam.

The collective mode's two pure building blocks live here as well:
``partitioning`` (which partition owns a document) and ``merge``
(global top-k over per-partition candidates). Neither touches a
backend or a thread, so both are tested without one.
"""
