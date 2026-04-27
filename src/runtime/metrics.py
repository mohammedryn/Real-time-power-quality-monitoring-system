from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
import threading
import time
from typing import Deque, Dict, Iterator


@dataclass(frozen=True)
class StageStats:
    count: int
    mean_ms: float
    min_ms: float
    max_ms: float
    p95_ms: float


class RollingStats:
    def __init__(self, max_samples: int = 300) -> None:
        if max_samples <= 0:
            raise ValueError("max_samples must be > 0")
        self._samples: Deque[float] = deque(maxlen=max_samples)

    def add(self, value_ms: float) -> None:
        self._samples.append(float(value_ms))

    def snapshot(self) -> StageStats:
        if not self._samples:
            return StageStats(count=0, mean_ms=0.0, min_ms=0.0, max_ms=0.0, p95_ms=0.0)

        values = sorted(self._samples)
        count = len(values)
        p95_index = max(0, int(round(0.95 * (count - 1))))

        return StageStats(
            count=count,
            mean_ms=float(sum(values) / count),
            min_ms=float(values[0]),
            max_ms=float(values[-1]),
            p95_ms=float(values[p95_index]),
        )


class RuntimeMetrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._started_at = time.time()
        self._stages: Dict[str, RollingStats] = {}
        self._counters: Dict[str, int] = {
            "frames_ingested": 0,
            "frames_dropped_acq": 0,
            "frames_scored": 0,
            "results_dropped": 0,
        }

    def incr(self, key: str, amount: int = 1) -> None:
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + amount

    def record_stage(self, stage: str, duration_ms: float) -> None:
        with self._lock:
            if stage not in self._stages:
                self._stages[stage] = RollingStats()
            self._stages[stage].add(duration_ms)

    @contextmanager
    def time_stage(self, stage: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.record_stage(stage, elapsed_ms)

    def snapshot(self) -> dict:
        with self._lock:
            stage_snapshot = {name: stats.snapshot() for name, stats in self._stages.items()}
            counters = dict(self._counters)
            uptime_sec = max(0.0, time.time() - self._started_at)

        return {
            "uptime_sec": uptime_sec,
            "counters": counters,
            "stages": {
                key: {
                    "count": value.count,
                    "mean_ms": value.mean_ms,
                    "min_ms": value.min_ms,
                    "max_ms": value.max_ms,
                    "p95_ms": value.p95_ms,
                }
                for key, value in stage_snapshot.items()
            },
        }
