from __future__ import annotations

import time

from src.runtime.metrics import RuntimeMetrics


def test_runtime_metrics_stage_and_counters() -> None:
    metrics = RuntimeMetrics()

    metrics.incr("frames_ingested")
    metrics.incr("frames_scored", 3)
    metrics.record_stage("model_ms", 4.0)
    metrics.record_stage("model_ms", 6.0)

    snapshot = metrics.snapshot()

    assert snapshot["counters"]["frames_ingested"] == 1
    assert snapshot["counters"]["frames_scored"] == 3
    assert snapshot["stages"]["model_ms"]["count"] == 2
    assert snapshot["stages"]["model_ms"]["mean_ms"] == 5.0


def test_runtime_metrics_stage_timer() -> None:
    metrics = RuntimeMetrics()

    with metrics.time_stage("inference_total_ms"):
        time.sleep(0.01)

    snapshot = metrics.snapshot()
    stage = snapshot["stages"]["inference_total_ms"]

    assert stage["count"] == 1
    assert stage["mean_ms"] > 0.0
