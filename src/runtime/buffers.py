from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading
import time
from typing import Deque, Generic, Optional, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class QueueStats:
    max_size: int
    size: int
    pushes: int
    pops: int
    dropped: int
    drop_policy: str


class BoundedQueue(Generic[T]):
    """Thread-safe bounded queue with explicit backpressure policy."""

    def __init__(self, max_size: int, drop_policy: str = "drop_oldest") -> None:
        if max_size <= 0:
            raise ValueError("max_size must be > 0")
        if drop_policy not in {"drop_oldest", "drop_newest"}:
            raise ValueError("drop_policy must be 'drop_oldest' or 'drop_newest'")

        self._max_size = max_size
        self._drop_policy = drop_policy
        self._data: Deque[T] = deque()
        self._cond = threading.Condition()

        self._pushes = 0
        self._pops = 0
        self._dropped = 0

    def put(self, item: T) -> bool:
        """Returns True if enqueued, False if item was dropped by policy."""
        with self._cond:
            if len(self._data) >= self._max_size:
                if self._drop_policy == "drop_oldest":
                    self._data.popleft()
                    self._dropped += 1
                else:
                    self._dropped += 1
                    return False

            self._data.append(item)
            self._pushes += 1
            self._cond.notify()
            return True

    def get(self, timeout: Optional[float] = None) -> Optional[T]:
        with self._cond:
            if timeout is None:
                while not self._data:
                    self._cond.wait()
            else:
                end_time = time.monotonic() + timeout
                while not self._data:
                    remaining = end_time - time.monotonic()
                    if remaining <= 0:
                        return None
                    self._cond.wait(remaining)

            item = self._data.popleft()
            self._pops += 1
            return item

    def get_nowait(self) -> Optional[T]:
        with self._cond:
            if not self._data:
                return None
            item = self._data.popleft()
            self._pops += 1
            return item

    def drain_latest(self) -> Optional[T]:
        """Return latest item and discard older buffered items."""
        with self._cond:
            if not self._data:
                return None
            latest = self._data[-1]
            popped_count = len(self._data)
            self._data.clear()
            self._pops += popped_count
            return latest

    def peek_latest(self) -> Optional[T]:
        with self._cond:
            if not self._data:
                return None
            return self._data[-1]

    def qsize(self) -> int:
        with self._cond:
            return len(self._data)

    def clear(self) -> None:
        with self._cond:
            self._data.clear()

    def stats(self) -> QueueStats:
        with self._cond:
            return QueueStats(
                max_size=self._max_size,
                size=len(self._data),
                pushes=self._pushes,
                pops=self._pops,
                dropped=self._dropped,
                drop_policy=self._drop_policy,
            )


class AtomicValue(Generic[T]):
    """Small lock-based latest-value container for cross-thread UI access."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._value: Optional[T] = None

    def set(self, value: T) -> None:
        with self._lock:
            self._value = value

    def get(self) -> Optional[T]:
        with self._lock:
            return self._value
