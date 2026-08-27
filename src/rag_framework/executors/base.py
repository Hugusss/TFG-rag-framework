"""The Executor interface and its per-task outcome record."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TaskOutcome:
    """What happened to item ``index``: exactly one of ``result`` /
    ``error`` is meaningful, ``error`` being the captured exception."""

    index: int
    worker_id: str
    seconds: float
    result: Any = None
    error: Exception | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class Executor(ABC):
    """Runs ``fn`` over ``items`` and reports one outcome per item.

    Contract for every implementation:

    - :meth:`map` returns outcomes **in item order**, one per item, no
      matter how they were scheduled; it never raises for a failing
      item — the exception is captured in ``TaskOutcome.error`` so a
      worker failure is visible data rather than a lost result, and
      the caller decides whether the whole call fails.
    - ``worker_id`` identifies the worker that ran the item, stable for
      that worker's lifetime; ``seconds`` is the item's wall time on
      that worker, exclusive of queueing.
    - ``workers`` is the configured parallelism; ``name`` the executor
      kind, recorded in reports.
    - :meth:`close` releases workers; calling it twice is harmless.
    """

    name: str = "executor"

    def __init__(self, workers: int) -> None:
        if isinstance(workers, bool) or not isinstance(workers, int):
            raise ValueError(
                f"workers must be an int, got {type(workers).__name__}"
            )
        if workers < 1:
            raise ValueError(f"workers must be at least 1, got {workers}")
        self.workers = workers

    @abstractmethod
    def map(
        self, fn: Callable[[Any], Any], items: Sequence[Any]
    ) -> list[TaskOutcome]:
        """Apply ``fn`` to every item; outcomes in item order."""

    def close(self) -> None:
        """Release worker resources (no-op by default)."""
