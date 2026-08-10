# rag-framework

Modular Collective RAG Pipeline over the Open Web Index (OWI): ingestion,
chunking, embeddings, a replaceable vector-store backend (ChromaDB first),
sequential and partitioned-collective retrieval, optional generation, and
reproducible benchmarks.

Research question: *can a local implementation evolve into a serverless,
distributed one by replacing components instead of rewriting the application?*

**Status: planning.** Requirements are kept outside the repository.
Implementation starts with Week 1: dataset ingestion into a persistent
ChromaDB index.
