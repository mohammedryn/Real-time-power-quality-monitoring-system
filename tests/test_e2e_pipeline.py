from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from src.dsp.preprocess import load_config
from src.infer.offline_replay import load_replay_source
from src.io.frame_protocol import N_FEATURES, N_SAMPLES, pack_feature_frame, pack_frame
from src.runtime.pipeline import RuntimePipeline


class DummyPredictor:
    def __init__(self, n_classes: int) -> None:
        self._n_classes = n_classes

    def predict_proba(self, feature_vector: np.ndarray) -> np.ndarray:
        idx = int(abs(float(feature_vector[50]) * 1000.0)) % self._n_classes
        probs = np.zeros(self._n_classes, dtype=np.float32)
        probs[idx] = 1.0
        return probs


def _wait_for_pipeline_done(
    pipeline: RuntimePipeline,
    expected_frames: int,
    timeout_sec: float = 5.0,
) -> int:
    deadline = time.time() + timeout_sec
    scored = 0

    while time.time() < deadline:
        snapshot = pipeline.get_result(timeout=0.1)
        if snapshot is not None:
            scored += 1

        if scored >= expected_frames:
            return scored

    raise TimeoutError("pipeline did not finish replay processing within timeout")


def test_runtime_pipeline_replay_e2e(tmp_path: Path) -> None:
    cfg = load_config("configs/default.yaml")
    n_classes = len(cfg["classes"]["names"])

    rng = np.random.default_rng(123)
    feat_row = rng.standard_normal(N_FEATURES).astype(np.float32)

    raw_v = rng.integers(low=1700, high=2400, size=N_SAMPLES, dtype=np.int16)
    raw_i = rng.integers(low=1800, high=2300, size=N_SAMPLES, dtype=np.int16)

    replay_source = [
        {"seq": 1, "features": feat_row},
        {"seq": 2, "v_raw": raw_v.tolist(), "i_raw": raw_i.tolist()},
    ]

    session_log = tmp_path / "session.jsonl"
    pipeline = RuntimePipeline(
        cfg,
        DummyPredictor(n_classes),
        replay_source=replay_source,
        session_log_path=str(session_log),
    )

    pipeline.start()
    scored = _wait_for_pipeline_done(pipeline, expected_frames=2)
    pipeline.stop()

    assert scored == 2
    assert session_log.exists()

    lines = session_log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    record = json.loads(lines[0])
    assert "top1" in record
    assert "metrics" in record
    assert "health" in record


def test_offline_replay_loader_supports_npy_jsonl_bin(tmp_path: Path) -> None:
    rng = np.random.default_rng(77)

    # npy
    npy_path = tmp_path / "features.npy"
    np.save(npy_path, rng.standard_normal((3, N_FEATURES)).astype(np.float32))
    npy_frames = list(load_replay_source(str(npy_path)))
    assert len(npy_frames) == 3

    # jsonl
    jsonl_path = tmp_path / "frames.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as fp:
        fp.write(json.dumps({"seq": 10, "features": rng.standard_normal(N_FEATURES).tolist()}) + "\n")
        fp.write(json.dumps({"seq": 11, "features": rng.standard_normal(N_FEATURES).tolist()}) + "\n")
    jsonl_frames = list(load_replay_source(str(jsonl_path)))
    assert len(jsonl_frames) == 2

    # binary mixed raw + feature stream
    feature_bytes = pack_feature_frame(5, rng.standard_normal(N_FEATURES).astype(np.float32))
    raw_v = rng.integers(low=1700, high=2400, size=N_SAMPLES, dtype=np.int16)
    raw_i = rng.integers(low=1800, high=2300, size=N_SAMPLES, dtype=np.int16)
    raw_bytes = pack_frame(6, raw_v, raw_i)

    bin_path = tmp_path / "mixed.bin"
    bin_path.write_bytes(feature_bytes + raw_bytes)

    bin_frames = list(load_replay_source(str(bin_path)))
    assert len(bin_frames) == 2
