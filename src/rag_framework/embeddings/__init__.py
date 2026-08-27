"""Embedding seam: chunks and queries become vectors here.

Providers are the only components that know models, vector files, or
encoding details. Whether vectors are looked up from the corpus's
official embeddings or computed by a local model is a configuration
choice — invisible to every other component.
"""
