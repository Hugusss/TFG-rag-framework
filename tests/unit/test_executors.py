"""Unit tests for the executor seam (serial and threads)."""

import threading
import time

import pytest

from rag_framework.executors.base import TaskOutcome
from rag_framework.executors.local import SerialExecutor, ThreadExecutor


def executors():
    return [SerialExecutor(), ThreadExecutor(1), ThreadExecutor(4)]


@pytest.mark.parametrize("executor", executors(), ids=lambda e: f"{e.name}{e.workers}")
class TestContract:
    def test_outcomes_in_item_order_whatever_the_timing(self, executor):
        def slow_square(n):
            time.sleep(0.02 * (4 - n))  # later items finish first
            return n * n

        outcomes = executor.map(slow_square, [0, 1, 2, 3])
        assert [o.index for o in outcomes] == [0, 1, 2, 3]
        assert [o.result for o in outcomes] == [0, 1, 4, 9]
        assert all(o.ok and o.error is None for o in outcomes)
        assert all(o.seconds >= 0.0 for o in outcomes)
        executor.close()

    def test_failure_is_captured_not_raised(self, executor):
        def flaky(n):
            if n == 1:
                raise ValueError("partition 1 exploded")
            return n

        outcomes = executor.map(flaky, [0, 1, 2])
        assert [o.ok for o in outcomes] == [True, False, True]
        assert isinstance(outcomes[1].error, ValueError)
        assert "exploded" in str(outcomes[1].error)
        assert outcomes[1].result is None
        assert outcomes[2].result == 2  # others still ran
        executor.close()

    def test_empty_items(self, executor):
        assert executor.map(lambda x: x, []) == []
        executor.close()

    def test_close_is_idempotent(self, executor):
        executor.map(lambda x: x, [1])
        executor.close()
        executor.close()
        # usable again after close: the pool restarts lazily
        assert executor.map(lambda x: x + 1, [1])[0].result == 2
        executor.close()


class TestWorkers:
    def test_serial_runs_in_the_caller(self):
        outcomes = SerialExecutor().map(lambda _: threading.get_ident(), [0, 1])
        assert {o.result for o in outcomes} == {threading.get_ident()}
        assert {o.worker_id for o in outcomes} == {threading.current_thread().name}
        assert SerialExecutor().workers == 1

    def test_threads_use_distinct_named_workers(self):
        executor = ThreadExecutor(4)
        barrier = threading.Barrier(4, timeout=5)

        def wait(_):
            barrier.wait()  # only passes if 4 run concurrently
            return threading.current_thread().name

        outcomes = executor.map(wait, range(4))
        names = {o.worker_id for o in outcomes}
        assert len(names) == 4
        assert all(name.startswith("w_") for name in names)
        assert names == {o.result for o in outcomes}
        executor.close()

    def test_single_thread_serializes(self):
        executor = ThreadExecutor(1)
        outcomes = executor.map(lambda _: threading.current_thread().name, range(5))
        assert len({o.worker_id for o in outcomes}) == 1
        executor.close()

    @pytest.mark.parametrize("bad", [0, -1, 2.0, True])
    def test_invalid_worker_count(self, bad):
        with pytest.raises(ValueError, match="workers"):
            ThreadExecutor(bad)

    def test_outcome_is_immutable(self):
        outcome = TaskOutcome(0, "w", 0.0, 1)
        with pytest.raises(AttributeError):
            outcome.result = 2
