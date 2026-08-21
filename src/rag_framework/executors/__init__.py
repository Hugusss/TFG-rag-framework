"""Executor seam: the only place concurrency lives (spec Rule 2).

An Executor runs one function over many items and reports, per item,
what happened — result or captured failure, which worker ran it, how
long it took. Retrievers decide what to do with that; executors never
merge, retry, or hide an outcome. Local executors (serial, threads) are
a *local simulation* of collective execution (spec section 13.2): a
remote, function-as-a-service executor must fit the same interface.
"""
