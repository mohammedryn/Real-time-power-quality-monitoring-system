from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import threading
import time
from typing import Iterable, List, Optional, Protocol, Union

import numpy as np
import serial

from src.dsp.features import extract_features
from src.dsp.preprocess import preprocess_frame
from src.io.frame_protocol import FeatureFrame, ModelReadyFrame, ParsedFrame
from src.io.serial_receiver import SerialFrameReceiver
from src.runtime.buffers import AtomicValue, BoundedQueue
from src.runtime.metrics import RuntimeMetrics

# ---- Feature index constants for the 298-element model-4 vector -------------
# X_phase = feat[0:28] ++ feat[56:214] ++ feat[214:298]
# X_mag   = feat[28:56]
_IDX_RMS_V        = 2
_IDX_RMS_I        = 14
_IDX_THD_V        = 54   # within the 298-vector (power metrics inserted at [24:28])
_IDX_THD_I        = 55
_IDX_CROSS_SIN_H1 = 108
_IDX_CROSS_COS_H1 = 121
_IDX_HARM_V       = slice(28, 41)   # 13 voltage harmonic magnitudes
_IDX_HARM_I       = slice(41, 54)   # 13 current harmonic magnitudes

# Default per-class sigmoid thresholds (overridden by ml_inference.thresholds in config)
_DEFAULT_THRESHOLDS = [0.50, 0.50, 0.35, 0.50, 0.50, 0.35, 0.50]


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
    features: np.ndarray          # 298-element for model4; 282 for legacy
    v_phys: Optional[np.ndarray] = None
    i_phys: Optional[np.ndarray] = None
    v_norm: Optional[np.ndarray] = None   # peak-normalised voltage (500,)
    i_norm: Optional[np.ndarray] = None   # peak-normalised current (500,)


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
    active_labels: list[str] = field(default_factory=list)
    active_probs: list[float] = field(default_factory=list)
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
            "active_labels": snapshot.active_labels,
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

    Supports single-input (sklearn / single-input Keras) and
    three-input Keras models (model_4: wave_input, mag_input, phase_input).
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
        self._is_multi_input = False   # set True for 3-input Keras models

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
            loaded = load_model(model_path)
            # Detect 3-input architecture (model_4: wave, mag, phase inputs)
            try:
                if isinstance(loaded.input, list) and len(loaded.input) == 3:
                    self._is_multi_input = True
            except Exception:
                pass
            return loaded

        raise ValueError(f"Unsupported model artifact format: {model_path.name}")

    def predict_proba(
        self,
        feature_vector: np.ndarray,
        v_norm: Optional[np.ndarray] = None,
        i_norm: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return per-class probabilities.

        For 3-input (multi-label) models the 298-element feature_vector is
        split into X_mag and X_phase, and v_norm/i_norm form X_wave.
        Probabilities are NOT normalised for multi-label (sigmoid outputs are
        independent per class).
        """
        if self._is_multi_input and v_norm is not None and i_norm is not None:
            return self._predict_multi_input(feature_vector, v_norm, i_norm)

        # --- Single-input path (sklearn or single-input Keras) ---
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

    def _predict_multi_input(
        self,
        features: np.ndarray,
        v_norm: np.ndarray,
        i_norm: np.ndarray,
    ) -> np.ndarray:
        """Call a 3-input Keras model (model_4 architecture).

        features is the 298-element vector:
          X_mag   = features[28:56]
          X_phase = features[0:28] ++ features[56:214] ++ features[214:298]
          X_wave  = stack([v_norm, i_norm], axis=-1).reshape(1, 500, 2)
        """
        feat = np.asarray(features, dtype=np.float32)
        X_wave = np.stack(
            [np.asarray(v_norm, dtype=np.float32),
             np.asarray(i_norm, dtype=np.float32)],
            axis=-1
        ).reshape(1, 500, 2)

        X_mag = feat[28:56].reshape(1, 28)
        X_phase = np.concatenate([feat[0:28], feat[56:214], feat[214:298]]).reshape(1, 270)

        probs = np.asarray(
            self._model.predict([X_wave, X_mag, X_phase], verbose=0),
            dtype=np.float32,
        ).reshape(-1)

        if probs.size != len(self.class_names):
            raise ValueError(
                f"Model output size {probs.size} does not match class count {len(self.class_names)}"
            )

        return np.maximum(probs, 0.0)


class RuntimePipeline:
    def __init__(
        self,
        cfg: dict,
        predictor: PredictorProtocol,
        *,
        port: Optional[str] = None,
        receiver_mode: str = "model4",
        replay_source: Optional[Iterable[Union[FeatureFrame, ParsedFrame, ModelReadyFrame, dict, np.ndarray]]] = None,
        session_log_path: Optional[str] = None,
        serial_timeout: float = 1.0,
        serial_retry_delay: float = 1.0,
        baud: int = 115200,
    ) -> None:
        if serial_retry_delay <= 0.0:
            raise ValueError("serial_retry_delay must be > 0")

        self.cfg = cfg
        self.predictor = predictor
        self.class_names = list(cfg["classes"]["names"])
        self.normal_label = self.class_names[0]

        runtime_cfg = cfg.get("runtime", {})
        queue_size = int(runtime_cfg.get("max_queue_size", 64))
        drop_policy = str(runtime_cfg.get("drop_policy", "drop_oldest"))

        self._acq_queue: BoundedQueue[Union[FeatureFrame, ParsedFrame, ModelReadyFrame, dict, np.ndarray]] = BoundedQueue(
            max_size=queue_size, drop_policy=drop_policy
        )
        self._result_queue: BoundedQueue[InferenceSnapshot] = BoundedQueue(
            max_size=queue_size, drop_policy=drop_policy
        )
        self._latest = AtomicValue[InferenceSnapshot]()
        self._metrics = RuntimeMetrics()

        self._expected_n = int(cfg["signal"]["samples_per_frame"])
        self._mains_freq = float(cfg["signal"].get("mains_frequency_hz", 50.0))

        # Multi-label inference config
        ml_cfg = cfg.get("ml_inference", {})
        self._multi_label: bool = bool(ml_cfg.get("multi_label", False))
        thresh_map: dict = ml_cfg.get("thresholds", {})
        if thresh_map:
            self._class_thresholds: List[float] = [
                float(thresh_map.get(name, 0.50)) for name in self.class_names
            ]
        else:
            n = len(self.class_names)
            self._class_thresholds = _DEFAULT_THRESHOLDS[:n] if n <= len(_DEFAULT_THRESHOLDS) else [0.50] * n

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
        self._serial_retry_delay = float(serial_retry_delay)

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
            while not self._stop_event.is_set():
                try:
                    self._receiver.open()
                    break
                except (serial.SerialException, OSError):
                    self._metrics.incr("serial_open_failures")
                    self._receiver.close()
                    if self._stop_event.wait(self._serial_retry_delay):
                        return

            while not self._stop_event.is_set():
                try:
                    with self._metrics.time_stage("acquisition_ms"):
                        frame = self._receiver.read_frame(frame_timeout=0.5)
                except (serial.SerialException, OSError):
                    self._metrics.incr("serial_read_failures")
                    self._receiver.close()
                    self._stop_event.wait(self._serial_retry_delay)
                    continue
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
                    # Route to 3-input call when predictor supports it and
                    # the frame provided normalised waveforms.
                    if (getattr(self.predictor, "_is_multi_input", False)
                            and context.v_norm is not None
                            and context.i_norm is not None):
                        probs = self.predictor.predict_proba(
                            context.features, context.v_norm, context.i_norm
                        )
                    else:
                        probs = self.predictor.predict_proba(context.features)
                snapshot = self._build_snapshot(context, probs)

            self._metrics.incr("frames_scored")
            self._latest.set(snapshot)
            if not self._result_queue.put(snapshot):
                self._metrics.incr("results_dropped")

            if self._logger is not None:
                self._logger.write(snapshot)

    def _frame_to_context(
        self,
        frame: Union[FeatureFrame, ParsedFrame, ModelReadyFrame, dict, np.ndarray],
    ) -> FrameContext:
        now = time.time()

        if isinstance(frame, ModelReadyFrame):
            # Reconstruct the 298-element feature vector:
            #   features = X_phase[0:28] ++ X_mag ++ X_phase[28:]
            # This is the inverse of the slicing done on the Teensy.
            features = np.concatenate([
                frame.X_phase[:28],      # feat[0:28]  – time-domain stats
                frame.X_mag,             # feat[28:56] – harmonic mags + THD
                frame.X_phase[28:],      # feat[56:298] – phase, power, wavelet
            ]).astype(np.float32)
            return FrameContext(
                seq=int(frame.seq),
                timestamp=now,
                features=features,
                v_norm=np.asarray(frame.v_norm, dtype=np.float32),
                i_norm=np.asarray(frame.i_norm, dtype=np.float32),
            )

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
                v_norm=np.asarray(processed["v_norm"], dtype=np.float32),
                i_norm=np.asarray(processed["i_norm"], dtype=np.float32),
            )

        if isinstance(frame, np.ndarray):
            features = np.asarray(frame, dtype=np.float32).reshape(-1)
            return FrameContext(seq=-1, timestamp=now, features=features)

        if isinstance(frame, dict):
            seq = int(frame.get("seq", -1))
            if "features" in frame:
                features = np.asarray(frame["features"], dtype=np.float32).reshape(-1)
                return FrameContext(seq=seq, timestamp=now, features=features)

            if all(k in frame for k in ("X_wave", "X_mag", "X_phase")):
                x_wave = np.asarray(frame["X_wave"], dtype=np.float32).reshape(-1)
                x_mag = np.asarray(frame["X_mag"], dtype=np.float32).reshape(-1)
                x_phase = np.asarray(frame["X_phase"], dtype=np.float32).reshape(-1)

                if x_wave.size != 1000 or x_mag.size != 28 or x_phase.size != 270:
                    raise ValueError(
                        "Model-ready dict requires X_wave(1000), X_mag(28), X_phase(270)"
                    )

                features = np.concatenate([
                    x_phase[:28],
                    x_mag,
                    x_phase[28:],
                ]).astype(np.float32)

                return FrameContext(
                    seq=seq,
                    timestamp=now,
                    features=features,
                    v_norm=np.asarray(x_wave[:500], dtype=np.float32),
                    i_norm=np.asarray(x_wave[500:], dtype=np.float32),
                )

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
                    v_norm=np.asarray(processed["v_norm"], dtype=np.float32),
                    i_norm=np.asarray(processed["i_norm"], dtype=np.float32),
                )

        raise TypeError(f"Unsupported frame type: {type(frame)!r}")

    def _build_snapshot(self, context: FrameContext, probs: np.ndarray) -> InferenceSnapshot:
        probs = np.asarray(probs, dtype=np.float32).reshape(-1)
        top_idx = int(np.argmax(probs))

        top1_label = self.class_names[top_idx]
        top1_conf = float(probs[top_idx])

        # Extract metrics from the feature vector.
        # Indices are valid for both 298-element (model4) and 282-element
        # (legacy) vectors — all accesses are within 0..281.
        features = context.features
        n_feat = len(features)

        rms_v = float(features[_IDX_RMS_V]) if n_feat > _IDX_RMS_V else 0.0
        rms_i = float(features[_IDX_RMS_I]) if n_feat > _IDX_RMS_I else 0.0
        thd_v = float(features[_IDX_THD_V]) if n_feat > _IDX_THD_V else 0.0
        thd_i = float(features[_IDX_THD_I]) if n_feat > _IDX_THD_I else 0.0

        cross_sin_h1 = float(features[_IDX_CROSS_SIN_H1]) if n_feat > _IDX_CROSS_SIN_H1 else 0.0
        cross_cos_h1 = float(features[_IDX_CROSS_COS_H1]) if n_feat > _IDX_CROSS_COS_H1 else 0.0
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
                "connected"
                if (self._receiver.ser is not None and self._receiver.ser.is_open)
                else "disconnected"
            )

        total_stats = health["runtime"].get("stages", {}).get("inference_total_ms", {})
        health["inference_latency_ms_mean"] = float(total_stats.get("mean_ms", 0.0))
        health["inference_latency_ms_p95"] = float(total_stats.get("p95_ms", 0.0))

        temp_c = _read_device_temp_c()
        if temp_c is not None:
            health["device_temp_c"] = temp_c

        # Harmonic magnitudes (valid for both 282 and 298 feature vectors)
        harm_v_end = _IDX_HARM_V.stop if n_feat >= _IDX_HARM_V.stop else n_feat
        harm_i_end = _IDX_HARM_I.stop if n_feat >= _IDX_HARM_I.stop else n_feat
        harmonics_v = [float(x) for x in features[_IDX_HARM_V.start:harm_v_end].tolist()]
        harmonics_i = [float(x) for x in features[_IDX_HARM_I.start:harm_i_end].tolist()]

        # Active labels (multi-label: threshold-based; single-label: top1 only)
        active_labels: list[str] = []
        active_probs_list: list[float] = []
        if self._multi_label:
            for name, prob, thresh in zip(self.class_names, probs.tolist(), self._class_thresholds):
                if prob >= thresh:
                    active_labels.append(name)
                    active_probs_list.append(float(prob))
            if not active_labels:
                active_labels = [self.normal_label]
                active_probs_list = [float(probs[0])]
        else:
            active_labels = [top1_label]
            active_probs_list = [top1_conf]

        # Event detection
        event = None
        if self._multi_label:
            non_normal = [(l, p) for l, p in zip(active_labels, active_probs_list)
                          if l != self.normal_label]
            if non_normal:
                max_conf = max(p for _, p in non_normal)
                severity = "high" if max_conf >= 0.9 else ("medium" if max_conf >= 0.7 else "low")
                event = {
                    "labels": [l for l, _ in non_normal],
                    "label": non_normal[0][0],
                    "confidence": max_conf,
                    "severity": severity,
                    "timestamp": context.timestamp,
                }
        else:
            if top1_label != self.normal_label:
                severity = "high" if top1_conf >= 0.9 else ("medium" if top1_conf >= 0.7 else "low")
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
            harmonics_v=harmonics_v,
            harmonics_i=harmonics_i,
            active_labels=active_labels,
            active_probs=active_probs_list,
            v_phys=context.v_phys.tolist() if context.v_phys is not None else None,
            i_phys=context.i_phys.tolist() if context.i_phys is not None else None,
            event=event,
        )
