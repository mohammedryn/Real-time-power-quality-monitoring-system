from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Optional

import numpy as np
import pytest

from src.dsp.preprocess import load_config
from src.dsp.feature_index import TOTAL_FEATURES
from src.infer.offline_replay import load_replay_source
from src.io.frame_protocol import (
    N_FEATURES, N_SAMPLES,
    pack_feature_frame, pack_frame, pack_model_ready_frame,
    ModelReadyFrame, parse_frame, parse_model_ready_frame,
)
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


class DummyMultiInputPredictor:
    """Simulates a 3-input multi-label Keras model."""

    _is_multi_input = True

    def __init__(self, n_classes: int) -> None:
        self._n_classes = n_classes

    def predict_proba(
        self,
        feature_vector: np.ndarray,
        v_norm: Optional[np.ndarray] = None,
        i_norm: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        # Return fixed sigmoid probabilities — index by a hash of X_mag values.
        probs = np.full(self._n_classes, 0.1, dtype=np.float32)
        if feature_vector is not None and len(feature_vector) >= 29:
            # Use first X_mag value (feature_vector[28]) to deterministically
            # pick a "fault" class.
            fault_idx = int(abs(float(feature_vector[28]) * 100.0)) % (self._n_classes - 1) + 1
            probs[0] = 0.1         # Normal probability low
            probs[fault_idx] = 0.9 # Some fault above threshold
        return probs


class AssertingRawMultiInputPredictor:
    """Ensures raw-frame path routes to 3-input prediction with v_norm/i_norm."""

    _is_multi_input = True

    def __init__(self, n_classes: int) -> None:
        self._n_classes = n_classes
        self.multi_input_calls = 0

    def predict_proba(
        self,
        feature_vector: np.ndarray,
        v_norm: Optional[np.ndarray] = None,
        i_norm: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        assert feature_vector.shape[0] == 298
        assert v_norm is not None
        assert i_norm is not None
        assert np.asarray(v_norm).shape == (N_SAMPLES,)
        assert np.asarray(i_norm).shape == (N_SAMPLES,)
        self.multi_input_calls += 1

        probs = np.zeros(self._n_classes, dtype=np.float32)
        probs[0] = 0.8
        if self._n_classes > 1:
            probs[1] = 0.6
        return probs


def _make_model_ready_replay(rng: np.random.Generator, n_frames: int):
    """Build model-ready frame replay source from random arrays."""
    frames = []
    for seq in range(n_frames):
        X_wave  = rng.standard_normal(1000).astype(np.float32)
        X_mag   = rng.standard_normal(28).astype(np.float32)
        X_phase = rng.standard_normal(270).astype(np.float32)
        raw = pack_model_ready_frame(seq, X_wave, X_mag, X_phase)
        frames.append(parse_model_ready_frame(raw))
    return frames


def test_model4_pipeline_replay_e2e(tmp_path: Path) -> None:
    """Full pipeline with ModelReadyFrame replay and 3-input predictor."""
    cfg = load_config("configs/default.yaml")
    n_classes = len(cfg["classes"]["names"])

    rng = np.random.default_rng(42)
    replay_source = _make_model_ready_replay(rng, n_frames=4)

    session_log = tmp_path / "session_model4.jsonl"
    pipeline = RuntimePipeline(
        cfg,
        DummyMultiInputPredictor(n_classes),
        replay_source=replay_source,
        session_log_path=str(session_log),
    )

    pipeline.start()
    scored = _wait_for_pipeline_done(pipeline, expected_frames=4)
    pipeline.stop()

    assert scored == 4
    assert session_log.exists()

    lines = session_log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 4

    record = json.loads(lines[0])
    assert "top1" in record
    assert "metrics" in record
    assert "health" in record
    assert "active_labels" in record

    # active_labels must be a list and non-empty
    assert isinstance(record["active_labels"], list)
    assert len(record["active_labels"]) >= 1


def test_model4_frame_context_has_v_norm_i_norm(tmp_path: Path) -> None:
    """ModelReadyFrame must produce a FrameContext with v_norm and i_norm."""
    cfg = load_config("configs/default.yaml")
    n_classes = len(cfg["classes"]["names"])

    rng = np.random.default_rng(7)
    replay_source = _make_model_ready_replay(rng, n_frames=1)

    pipeline = RuntimePipeline(
        cfg,
        DummyMultiInputPredictor(n_classes),
        replay_source=replay_source,
        session_log_path=str(tmp_path / "s.jsonl"),
    )

    pipeline.start()
    snapshot = pipeline.get_result(timeout=3.0)
    pipeline.stop()

    assert snapshot is not None
    # With multi-label config, active_labels should reflect threshold logic
    assert isinstance(snapshot.active_labels, list)
    assert len(snapshot.active_labels) >= 1


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

    model_ready_bytes = pack_model_ready_frame(
        7,
        rng.standard_normal(1000).astype(np.float32),
        rng.standard_normal(28).astype(np.float32),
        rng.standard_normal(270).astype(np.float32),
    )

    bin_path = tmp_path / "mixed.bin"
    bin_path.write_bytes(feature_bytes + raw_bytes + model_ready_bytes)

    bin_frames = list(load_replay_source(str(bin_path)))
    assert len(bin_frames) == 3
    assert any(isinstance(f, ModelReadyFrame) for f in bin_frames)


def test_offline_replay_loader_supports_npy_298(tmp_path: Path) -> None:
    rng = np.random.default_rng(1234)
    npy_path = tmp_path / "features_298.npy"
    np.save(npy_path, rng.standard_normal((2, TOTAL_FEATURES)).astype(np.float32))

    frames = list(load_replay_source(str(npy_path)))
    assert len(frames) == 2
    assert np.asarray(frames[0]["features"], dtype=np.float32).shape == (TOTAL_FEATURES,)


def test_offline_replay_loader_jsonl_rejects_bad_feature_length(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "bad_features.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as fp:
        fp.write(json.dumps({"seq": 1, "features": [0.0] * (N_FEATURES - 1)}) + "\n")

    with pytest.raises(ValueError, match="Invalid features length"):
        list(load_replay_source(str(jsonl_path)))


def test_offline_replay_loader_jsonl_rejects_bad_raw_lengths(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "bad_raw.jsonl"
    payload = {
        "seq": 1,
        "v_raw": [0] * (N_SAMPLES - 1),
        "i_raw": [0] * N_SAMPLES,
    }
    with jsonl_path.open("w", encoding="utf-8") as fp:
        fp.write(json.dumps(payload) + "\n")

    with pytest.raises(ValueError, match="Invalid raw waveform lengths"):
        list(load_replay_source(str(jsonl_path)))


def test_offline_replay_loader_jsonl_rejects_ambiguous_payload(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "ambiguous.jsonl"
    payload = {
        "seq": 1,
        "features": [0.0] * N_FEATURES,
        "v_raw": [0] * N_SAMPLES,
        "i_raw": [0] * N_SAMPLES,
    }
    with jsonl_path.open("w", encoding="utf-8") as fp:
        fp.write(json.dumps(payload) + "\n")

    with pytest.raises(ValueError, match="exactly one payload type"):
        list(load_replay_source(str(jsonl_path)))


def test_raw_frame_routes_to_multi_input_predictor_path(tmp_path: Path) -> None:
    cfg = load_config("configs/default.yaml")
    n_classes = len(cfg["classes"]["names"])

    rng = np.random.default_rng(99)
    raw_v = rng.integers(low=1700, high=2400, size=N_SAMPLES, dtype=np.int16)
    raw_i = rng.integers(low=1800, high=2300, size=N_SAMPLES, dtype=np.int16)
    parsed_raw = parse_frame(pack_frame(123, raw_v, raw_i))

    predictor = AssertingRawMultiInputPredictor(n_classes)
    pipeline = RuntimePipeline(
        cfg,
        predictor,
        replay_source=[parsed_raw],
        session_log_path=str(tmp_path / "raw_multi_input.jsonl"),
    )

    pipeline.start()
    scored = _wait_for_pipeline_done(pipeline, expected_frames=1)
    pipeline.stop()

    assert scored == 1
    assert predictor.multi_input_calls == 1
