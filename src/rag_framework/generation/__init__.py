"""Generation seam: retrieved context becomes an answer here.

Generators own prompt construction and model access. The
pipeline hands them the query and the retrieved SearchResults and
receives text; nothing else in the system knows what a prompt is.
"""
