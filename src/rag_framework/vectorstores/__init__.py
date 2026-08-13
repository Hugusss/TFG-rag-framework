"""Vector-store seam: chunks and their vectors persist and search here.

Store adapters are the only components that know a backend's client,
its distance metric, or its storage layout. Everything else speaks
Chunks in and SearchResults out — which is what makes the backend
replaceable (the research question).
"""
