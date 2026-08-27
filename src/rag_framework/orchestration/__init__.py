"""Orchestration seam: wiring and per-stage timing, nothing else.

The pipeline builds components from configuration through
name-resolving factories and moves data between seams. It contains no
dataset parsing, no chunk logic, no vector math, and no backend calls —
each of those lives behind its own interface.
"""
