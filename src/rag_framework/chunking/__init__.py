"""Chunking seam: Documents become retrievable Chunks here.

Chunkers are the only components that decide chunk boundaries. Which
strategy runs (publisher-provided windows or computed splits) is a
configuration choice made once for the whole corpus — never a
per-document branch.
"""
