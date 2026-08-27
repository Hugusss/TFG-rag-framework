"""Retrieval seam: queries become ranked SearchResults here.

Retrievers own the query flow (embed, search, merge) and its per-stage
timing. They make no backend calls of their own and hold no encoding
logic: both live behind their own seams. The collective mode's two pure
building blocks live here too — ``partitioning`` (which partition owns a
document) and ``merge`` (global top-k over per-partition candidates).
"""
