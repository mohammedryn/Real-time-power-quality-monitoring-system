from __future__ import annotations

from types import SimpleNamespace
import time

import numpy as np
import serial

from src.dsp.preprocess import load_config
from src.runtime.pipeline import RuntimePipeline


class DummyPredictor:
    def __init__(self, n_classes: int) -> None:
        self._n_classes = n_classes

    def predict_proba(self, feature_vector: np.ndarray) -> np.ndarray:
        return np.full(self._n_classes, 1.0 / self._n_classes, dtype=np.float32)


def test_live_startup_retries_when_initial_serial_open_fails(monkeypatch) -> None:
    instances = []

    class FailingReceiver:
        def __init__(self, *args, **kwargs) -> None:
            self.ser = None
            self.stats = SimpleNamespace()
            self.open_attempts = 0
            self.close_calls = 0
            instances.append(self)

        def open(self) -> None:
            self.open_attempts += 1
            raise serial.SerialException("serial unavailable")

        def read_frame(self, frame_timeout: float = 1.0):
            raise AssertionError("read_frame must not run until open succeeds")

        def close(self) -> None:
            self.close_calls += 1

    monkeypatch.setattr("src.runtime.pipeline.SerialFrameReceiver", FailingReceiver)

    cfg = load_config("configs/default.yaml")
    pipeline = RuntimePipeline(
        cfg,
        DummyPredictor(len(cfg["classes"]["names"])),
        port="/dev/ttyACM0",
        serial_retry_delay=0.01,
    )

    pipeline.start()
    try:
        deadline = time.time() + 1.0
        while time.time() < deadline:
            counters = pipeline.metrics.snapshot()["counters"]
            if counters.get("serial_open_failures", 0) >= 2:
                break
            time.sleep(0.01)
        else:
            raise AssertionError("serial open retry counter did not advance")

        assert pipeline.source_exhausted is False
        assert instances[0].open_attempts >= 2
        assert instances[0].close_calls >= 1
    finally:
        pipeline.stop()

    assert pipeline.source_exhausted is True
