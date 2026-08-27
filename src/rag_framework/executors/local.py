"""Local executors: serial (the reference) and a thread pool.

Whether threads actually help is left as something to measure, not to
assume: vector search runs in backend native code that may release the
GIL, while memory and cache contention can cancel the gain.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from rag_framework.executors.base import Executor, TaskOutcome


def _run(fn: Callable[[Any], Any], index: int, item: Any) -> TaskOutcome:
    """Run one item on the current worker, capturing its failure."""
    worker_id = threading.current_thread().name
    start = time.perf_counter()
    try:
        result = fn(item)
    except Exception as error:  # a worker failure is data, not a crash
        return TaskOutcome(
            index, worker_id, time.perf_counter() - start, error=error
        )
    return TaskOutcome(index, worker_id, time.perf_counter() - start, result)


class SerialExecutor(Executor):
    """Runs every item in the calling thread, one after another."""

    name = "serial"

    def __init__(self) -> None:
        super().__init__(workers=1)

    def map(self, fn, items: Sequence[Any]) -> list[TaskOutcome]:
        return [_run(fn, index, item) for index, item in enumerate(items)]


class ThreadExecutor(Executor):
    """A persistent pool of ``workers`` threads named ``w0..wN-1``."""

    name = "threads"

    def __init__(self, workers: int) -> None:
        super().__init__(workers)
        self._pool: ThreadPoolExecutor | None = None
        self._lock = threading.Lock()

    def map(self, fn, items: Sequence[Any]) -> list[TaskOutcome]:
        pool = self._pool_or_start()
        futures = [
            pool.submit(_run, fn, index, item)
            for index, item in enumerate(items)
        ]
        return [future.result() for future in futures]  # item order

    def _pool_or_start(self) -> ThreadPoolExecutor:
        with self._lock:
            if self._pool is None:
                self._pool = ThreadPoolExecutor(
                    max_workers=self.workers, thread_name_prefix="w"
                )
            return self._pool

    def close(self) -> None:
        with self._lock:
            if self._pool is not None:
                self._pool.shutdown(wait=True)
                self._pool = None
