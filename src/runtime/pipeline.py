from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import threading
import time
from typing import Iterable, Optional, Protocol, Union

import numpy as np

from src.dsp.features import extract_features
from src.dsp.preprocess import preprocess_frame
from src.io.frame_protocol import FeatureFrame, ParsedFrame
from src.io.serial_receiver import SerialFrameReceiver
from src.runtime.buffers import AtomicValue, BoundedQueue
from src.runtime.metrics import RuntimeMetrics


def _read_device_temp_c() -> Optional[float]:
    try:
        import psutil
    except Exception:
        return None

    try:
        sensors = psutil.sensors_temperatures(fahrenheit=False)
    except Exception:
        return None

    if not sensors:
        return None

    for entries in sensors.values():
        for entry in entries:
            current = getattr(entry, "current", None)
            if current is not None:
                return float(current)
    return None


class PredictorProtocol(Protocol):
    def predict_proba(self, feature_vector: np.ndarray) -> np.ndarray:
        ...


@dataclass
class FrameContext:
    seq: int
    timestamp: float
    features: np.ndarray
    v_phys: Optional[np.ndarray] = None
    i_phys: Optional[np.ndarray] = None


@dataclass
class InferenceSnapshot:
    seq: int
    timestamp: float
    class_names: list[str]
    probabilities: list[float]
    top1_label: str
    top1_confidence: float
    metrics: dict
    health: dict
    harmonics_v: list[float]
    harmonics_i: list[float]
    v_phys: Optional[list[float]] = None
    i_phys: Optional[list[float]] = None
    event: Optional[dict] = None


class SessionLogger:
    def __init__(self, output_path: str | Path) -> None:
        self._path = Path(output_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = self._path.open("w", encoding="utf-8")
        self._lock = threading.Lock()

    def write(self, snapshot: InferenceSnapshot) -> None:
        record = {
            "seq": snapshot.seq,
            "timestamp": snapshot.timestamp,
            "top1": snapshot.top1_label,
            "confidence": snapshot.top1_confidence,
            "probabilities": snapshot.probabilities,
            "metrics": snapshot.metrics,
            "health": snapshot.health,
            "event": snapshot.event,
        }
        with self._lock:
            self._fp.write(json.dumps(record) + "\n")
            self._fp.flush()

    def close(self) -> None:
        with self._lock:
            if not self._fp.closed:
                self._fp.close()


class ArtifactPredictor:
    """Thin integration layer for external model/scaler artifacts.

    This intentionally avoids implementing model architecture/training logic.
    """

    def __init__(
        self,
        class_names: list[str],
        model_path: Optional[str] = None,
        scaler_path: Optional[str] = None,
    ) -> None:
        self.class_names = class_names
        self._model = None
        self._scaler = None
        self._model_kind = "none"

        if scaler_path:
            self._scaler = self._load_joblib(scaler_path)
        if model_path:
            self._model = self._load_model(model_path)

    def _load_joblib(self, path: str):
        from joblib import load

        model_path = Path(path)
        if not model_path.exists():
            raise FileNotFoundError(f"Scaler artifact not found: {model_path}")
        return load(model_path)

    def _load_model(self, path: str):
        model_path = Path(path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {model_path}")

        suffix = model_path.suffix.lower()
        if suffix in {".joblib", ".pkl"}:
            self._model_kind = "sklearn"
            return self._load_joblib(path)

        if suffix in {".keras", ".h5"}:
            try:
                from tensorflow.keras.models import load_model
            except Exception as exc:  # pragma: no cover - env dependent
                raise RuntimeError("TensorFlow is required to load keras/h5 model artifacts") from exc
            self._model_kind = "tensorflow"
            return load_model(model_path)

        raise ValueError(f"Unsupported model artifact format: {model_path.name}")

    def predict_proba(self, feature_vector: np.ndarray) -> np.ndarray:
        x = np.asarray(feature_vector, dtype=np.float32).reshape(1, -1)
        if self._scaler is not None:
            x = self._scaler.transform(x)

        if self._model is None:
            return np.full(len(self.class_names), 1.0 / len(self.class_names), dtype=np.float32)

        if hasattr(self._model, "predict_proba"):
            probs = np.asarray(self._model.predict_proba(x), dtype=np.float32).reshape(-1)
        elif hasattr(self._model, "predict"):
            if self._model_kind == "tensorflow":
                probs = np.asarray(self._model.predict(x, verbose=0), dtype=np.float32).reshape(-1)
            else:
                probs = np.asarray(self._model.predict(x), dtype=np.float32).reshape(-1)
        else:
            raise TypeError("Loaded model does not expose predict_proba or predict")

        if probs.size != len(self.class_names):
            raise ValueError(
                f"Model output size {probs.size} does not match class count {len(self.class_names)}"
            )

        probs = np.maximum(probs, 0.0)
        denom = float(np.sum(probs))
        if denom <= 0.0:
            return np.full(len(self.class_names), 1.0 / len(self.class_names), dtype=np.float32)
        return probs / denom


class RuntimePipeline:
    def __init__(
        self,
        cfg: dict,
        predictor: PredictorProtocol,
        *,
        port: Optional[str] = None,
        receiver_mode: str = "feature",
        replay_source: Optional[Iterable[Union[FeatureFrame, ParsedFrame, dict, np.ndarray]]] = None,
        session_log_path: Optional[str] = None,
        serial_timeout: float = 1.0,
        baud: int = 115200,
    ) -> None:
        self.cfg = cfg
        self.predictor = predictor
        self.class_names = list(cfg["classes"]["names"])
        self.normal_label = self.class_names[0]

        runtime_cfg = cfg.get("runtime", {})
        queue_size = int(runtime_cfg.get("max_queue_size", 64))
        drop_policy = str(runtime_cfg.get("drop_policy", "drop_oldest"))

        self._acq_queue: BoundedQueue[Union[FeatureFrame, ParsedFrame, dict, np.ndarray]] = BoundedQueue(
            max_size=queue_size, drop_policy=drop_policy
        )
        self._result_queue: BoundedQueue[InferenceSnapshot] = BoundedQueue(
            max_size=queue_size, drop_policy=drop_policy
        )
        self._latest = AtomicValue[InferenceSnapshot]()
        self._metrics = RuntimeMetrics()

        self._expected_n = int(cfg["signal"]["samples_per_frame"])
        self._mains_freq = float(cfg["signal"].get("mains_frequency_hz", 50.0))

        self._receiver: Optional[SerialFrameReceiver] = None
        if replay_source is None:
            if not port:
                raise ValueError("port must be provided for live mode")
            self._receiver = SerialFrameReceiver(
                port=port,
                baud=baud,
                timeout=serial_timeout,
                mode=receiver_mode,
            )

        self._replay_source = replay_source
        self._stop_event = threading.Event()
        self._source_exhausted = threading.Event()
        self._threads: list[threading.Thread] = []

        self._logger = SessionLogger(session_log_path) if session_log_path else None

    @property
    def metrics(self) -> RuntimeMetrics:
        return self._metrics

    @property
    def source_exhausted(self) -> bool:
        return self._source_exhausted.is_set()

    def pending_results(self) -> int:
        return self._result_queue.qsize()

    def get_latest_snapshot(self) -> Optional[InferenceSnapshot]:
        return self._latest.get()

    def get_result(self, timeout: Optional[float] = None) -> Optional[InferenceSnapshot]:
        return self._result_queue.get(timeout=timeout)

    def start(self) -> None:
        if self._threads:
            return

        self._stop_event.clear()
        self._source_exhausted.clear()

        acq_thread = threading.Thread(target=self._acquisition_loop, name="pq-acquisition", daemon=True)
        infer_thread = threading.Thread(target=self._inference_loop, name="pq-inference", daemon=True)
        self._threads = [acq_thread, infer_thread]

        for thread in self._threads:
            thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        for thread in self._threads:
            thread.join(timeout=2.0)
        self._threads.clear()

        if self._receiver is not None:
            self._receiver.close()
        if self._logger is not None:
            self._logger.close()

    def _acquisition_loop(self) -> None:
        try:
            if self._replay_source is not None:
                for frame in self._replay_source:
                    if self._stop_event.is_set():
                        break
                    self._metrics.incr("frames_ingested")
                    if not self._acq_queue.put(frame):
                        self._metrics.incr("frames_dropped_acq")
                self._source_exhausted.set()
                return

            assert self._receiver is not None
            self._receiver.open()

            while not self._stop_event.is_set():
                with self._metrics.time_stage("acquisition_ms"):
                    frame = self._receiver.read_frame(frame_timeout=0.5)
                if frame is None:
                    continue
                self._metrics.incr("frames_ingested")
                if not self._acq_queue.put(frame):
                    self._metrics.incr("frames_dropped_acq")
        finally:
            self._source_exhausted.set()

    def _inference_loop(self) -> None:
        while not self._stop_event.is_set():
            frame = self._acq_queue.get(timeout=0.2)
            if frame is None:
                if self._source_exhausted.is_set() and self._acq_queue.qsize() == 0:
                    return
                continue

            with self._metrics.time_stage("inference_total_ms"):
                context = self._frame_to_context(frame)
                with self._metrics.time_stage("model_ms"):
                    probs = self.predictor.predict_proba(context.features)
                snapshot = self._build_snapshot(context, probs)

            self._metrics.incr("frames_scored")
            self._latest.set(snapshot)
            if not self._result_queue.put(snapshot):
                self._metrics.incr("results_dropped")

            if self._logger is not None:
                self._logger.write(snapshot)

    def _frame_to_context(self, frame: Union[FeatureFrame, ParsedFrame, dict, np.ndarray]) -> FrameContext:
        now = time.time()

        if isinstance(frame, FeatureFrame):
            features = np.asarray(frame.features, dtype=np.float32)
            return FrameContext(seq=int(frame.seq), timestamp=now, features=features)

        if isinstance(frame, ParsedFrame):
            processed = preprocess_frame(frame.v_raw, frame.i_raw, self.cfg, expected_n=self._expected_n)
            features = extract_features(processed["v_phys"], processed["i_phys"])
            return FrameContext(
                seq=int(frame.seq),
                timestamp=now,
                features=np.asarray(features, dtype=np.float32),
                v_phys=np.asarray(processed["v_phys"], dtype=np.float32),
                i_phys=np.asarray(processed["i_phys"], dtype=np.float32),
            )

        if isinstance(frame, np.ndarray):
            features = np.asarray(frame, dtype=np.float32).reshape(-1)
            return FrameContext(seq=-1, timestamp=now, features=features)

        if isinstance(frame, dict):
            seq = int(frame.get("seq", -1))
            if "features" in frame:
                features = np.asarray(frame["features"], dtype=np.float32).reshape(-1)
                return FrameContext(seq=seq, timestamp=now, features=features)

            if "v_raw" in frame and "i_raw" in frame:
                v_raw = np.asarray(frame["v_raw"], dtype=np.int16)
                i_raw = np.asarray(frame["i_raw"], dtype=np.int16)
                processed = preprocess_frame(v_raw, i_raw, self.cfg, expected_n=self._expected_n)
                features = extract_features(processed["v_phys"], processed["i_phys"])
                return FrameContext(
                    seq=seq,
                    timestamp=now,
                    features=np.asarray(features, dtype=np.float32),
                    v_phys=np.asarray(processed["v_phys"], dtype=np.float32),
                    i_phys=np.asarray(processed["i_phys"], dtype=np.float32),
                )

        raise TypeError(f"Unsupported frame type: {type(frame)!r}")

    def _build_snapshot(self, context: FrameContext, probs: np.ndarray) -> InferenceSnapshot:
        probs = np.asarray(probs, dtype=np.float32).reshape(-1)
        top_idx = int(np.argmax(probs))

        top1_label = self.class_names[top_idx]
        top1_conf = float(probs[top_idx])

        features = context.features
        rms_v = float(features[2])
        rms_i = float(features[14])
        thd_v = float(features[50])
        thd_i = float(features[51])

        cross_sin_h1 = float(features[104])
        cross_cos_h1 = float(features[117])
        phase_h1 = float(np.arctan2(cross_sin_h1, cross_cos_h1))

        dpf = float(np.cos(phase_h1))
        distortion_factor = 1.0 / float(np.sqrt(1.0 + thd_i * thd_i + 1e-12))
        pf = float(dpf * distortion_factor)

        metrics = {
            "rms_v": rms_v,
            "rms_i": rms_i,
            "thd_v": thd_v,
            "thd_i": thd_i,
            "dpf": dpf,
            "pf": pf,
            "frequency_hz": self._mains_freq,
        }

        health = {
            "runtime": self._metrics.snapshot(),
            "acq_queue": self._acq_queue.stats().__dict__,
            "result_queue": self._result_queue.stats().__dict__,
        }
        if self._receiver is not None:
            health["receiver"] = self._receiver.stats.__dict__
            health["serial_status"] = (
                "connected" if (self._receiver.ser is not None and self._receiver.ser.is_open) else "disconnected"
            )

        total_stats = health["runtime"].get("stages", {}).get("inference_total_ms", {})
        health["inference_latency_ms_mean"] = float(total_stats.get("mean_ms", 0.0))
        health["inference_latency_ms_p95"] = float(total_stats.get("p95_ms", 0.0))

        temp_c = _read_device_temp_c()
        if temp_c is not None:
            health["device_temp_c"] = temp_c

        event = None
        if top1_label != self.normal_label:
            if top1_conf >= 0.9:
                severity = "high"
            elif top1_conf >= 0.7:
                severity = "medium"
            else:
                severity = "low"
            event = {
                "label": top1_label,
                "confidence": top1_conf,
                "severity": severity,
                "timestamp": context.timestamp,
            }

        return InferenceSnapshot(
            seq=context.seq,
            timestamp=context.timestamp,
            class_names=self.class_names,
            probabilities=[float(x) for x in probs.tolist()],
            top1_label=top1_label,
            top1_confidence=top1_conf,
            metrics=metrics,
            health=health,
            harmonics_v=[float(x) for x in features[24:37].tolist()],
            harmonics_i=[float(x) for x in features[37:50].tolist()],
            v_phys=context.v_phys.tolist() if context.v_phys is not None else None,
            i_phys=context.i_phys.tolist() if context.i_phys is not None else None,
            event=event,
        )
