from __future__ import annotations

from src.runtime.buffers import AtomicValue, BoundedQueue


def test_bounded_queue_drop_oldest_policy() -> None:
    queue = BoundedQueue[int](max_size=2, drop_policy="drop_oldest")

    assert queue.put(1)
    assert queue.put(2)
    assert queue.put(3)

    assert queue.get_nowait() == 2
    assert queue.get_nowait() == 3
    assert queue.get_nowait() is None

    stats = queue.stats()
    assert stats.dropped == 1
    assert stats.pushes == 3


def test_bounded_queue_drop_newest_policy() -> None:
    queue = BoundedQueue[int](max_size=2, drop_policy="drop_newest")

    assert queue.put(10)
    assert queue.put(20)
    assert queue.put(30) is False

    assert queue.get_nowait() == 10
    assert queue.get_nowait() == 20

    stats = queue.stats()
    assert stats.dropped == 1
    assert stats.pushes == 2


def test_atomic_value_roundtrip() -> None:
    value = AtomicValue[int]()
    assert value.get() is None
    value.set(42)
    assert value.get() == 42
