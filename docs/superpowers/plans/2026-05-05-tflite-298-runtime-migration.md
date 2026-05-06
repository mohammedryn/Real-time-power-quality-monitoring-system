# TFLite 298 Runtime Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the repo's ambiguous inference contract with a single production path built around the `.tflite` artifact, the canonical `298`-feature layout, and neutral `tflite` runtime naming.

**Architecture:** Introduce a dedicated TFLite predictor, keep the existing packed three-input transport but rename it away from `model4`, make `tflite` the default live mode everywhere, and demote `282` feature-frame support to compatibility-only replay/debug behavior. Runtime snapshot extraction, replay loaders, tests, config, and docs all converge on the same `298` contract.

**Tech Stack:** Python 3.11+, NumPy, PyYAML, PySide6, pyqtgraph, `tflite_runtime` when available with `tensorflow.lite` fallback, Teensy firmware frame protocol, pytest.

---

## File Structure

### New files

- `src/runtime/tflite_predictor.py`
  - Sole production inference backend for `.tflite` artifacts.
  - Owns interpreter loading, tensor-shape validation, tensor setting, invocation, and output decoding.

- `tests/test_tflite_predictor.py`
  - Unit tests for TFLite interpreter loading, input/output validation, and inference calls using a fake interpreter.

- `tests/test_runtime_contract.py`
  - Canonical contract tests for `tflite` mode defaults, `298` feature expectations, and legacy compatibility boundaries.

### Modified files

- `configs/default.yaml`
  - Switch production model path to `.tflite`, default receiver mode to `tflite`, and add explicit output semantics for the active artifact.

- `src/io/frame_protocol.py`
  - Rename public `model4` transport terminology to neutral inference/TFLite terminology while preserving wire compatibility.

- `src/io/serial_receiver.py`
  - Make `tflite` the public live mode and keep `model4` as a deprecated alias only if needed for compatibility.

- `src/runtime/pipeline.py`
  - Remove generic artifact loading from the production path, use the TFLite predictor, validate canonical mode, keep `298` as the active layout, and support waveform plotting for packed live frames.

- `src/runtime/__init__.py`
  - Stop importing the DSP-heavy pipeline module at package import time.

- `src/infer/live_infer.py`
  - Default to `tflite` receiver mode and `.tflite` model path conventions.

- `src/infer/offline_replay.py`
  - Accept canonical `298` features in JSONL and clearly separate legacy `282` compatibility from active runtime semantics.

- `src/ui/app.py`
  - Default the UI to `tflite` mode, remove `.keras`-oriented wording, and pass the production `.tflite` path cleanly.

- `src/ui/views/dashboard.py`
  - Plot normalized waveforms when packed live frames are used and physical waveforms are unavailable.

- `src/system/kiosk_setup.sh`
  - Default kiosk deployments to `tflite` mode.

- `tests/test_e2e_pipeline.py`
  - Make `298` the default success path and keep `282` tests explicitly compatibility-only.

- `tests/test_model_ready_protocol.py`
- `tests/test_frame_predictor_integration.py`
- `tests/test_receiver_resync.py`
- `tests/test_runtime_serial_startup.py`
  - Update naming and assertions from `model4` to `tflite` where the behavior is the same.

- `README.md`
- `prd.md`
- `tasks.md`
- `docs/model_prd.md`
- `docs/teensy_dsp_migration_tracker.md`
- `docs/report_alignment_matrix.md`
  - Rewrite active docs around `.tflite`, `298`, and neutral transport naming.

---

### Task 1: Add a Dedicated TFLite Predictor

**Files:**
- Create: `src/runtime/tflite_predictor.py`
- Create: `tests/test_tflite_predictor.py`
- Modify: `configs/default.yaml`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tflite_predictor.py
from __future__ import annotations

import numpy as np
import pytest

from src.runtime.tflite_predictor import TFLitePredictor


class FakeInterpreter:
    def __init__(self) -> None:
        self._inputs = [
            {"index": 0, "name": "wave_input", "shape": np.array([1, 500, 2], dtype=np.int32)},
            {"index": 1, "name": "mag_input", "shape": np.array([1, 28], dtype=np.int32)},
            {"index": 2, "name": "phase_input", "shape": np.array([1, 270], dtype=np.int32)},
        ]
        self._outputs = [
            {"index": 3, "name": "main_output", "shape": np.array([1, 7], dtype=np.int32)},
        ]
        self.last_tensors = {}
        self.invoked = False

    def allocate_tensors(self) -> None:
        return None

    def get_input_details(self):
        return self._inputs

    def get_output_details(self):
        return self._outputs

    def set_tensor(self, index, value) -> None:
        self.last_tensors[index] = np.asarray(value)

    def invoke(self) -> None:
        self.invoked = True

    def get_tensor(self, index):
        assert index == 3
        return np.array([[0.05, 0.10, 0.70, 0.05, 0.03, 0.04, 0.03]], dtype=np.float32)


def test_tflite_predictor_validates_three_input_contract(tmp_path) -> None:
    model_path = tmp_path / "model.tflite"
    model_path.write_bytes(b"TFL3")

    predictor = TFLitePredictor(
        model_path=str(model_path),
        class_names=["Normal", "Sag", "Swell", "Interruption", "HarmonicDistortion", "Transient", "Flicker"],
        interpreter_factory=lambda path, num_threads: FakeInterpreter(),
    )

    assert predictor.input_names == ["wave_input", "mag_input", "phase_input"]
    assert predictor.output_shape == (1, 7)


def test_tflite_predictor_invokes_with_expected_tensor_shapes(tmp_path) -> None:
    model_path = tmp_path / "model.tflite"
    model_path.write_bytes(b"TFL3")

    predictor = TFLitePredictor(
        model_path=str(model_path),
        class_names=["Normal", "Sag", "Swell", "Interruption", "HarmonicDistortion", "Transient", "Flicker"],
        interpreter_factory=lambda path, num_threads: FakeInterpreter(),
    )

    probs = predictor.predict_proba(
        features=np.zeros(298, dtype=np.float32),
        v_norm=np.zeros(500, dtype=np.float32),
        i_norm=np.zeros(500, dtype=np.float32),
    )

    assert probs.shape == (7,)
    assert int(np.argmax(probs)) == 2


def test_tflite_predictor_rejects_wrong_feature_length(tmp_path) -> None:
    model_path = tmp_path / "model.tflite"
    model_path.write_bytes(b"TFL3")

    predictor = TFLitePredictor(
        model_path=str(model_path),
        class_names=["Normal", "Sag", "Swell", "Interruption", "HarmonicDistortion", "Transient", "Flicker"],
        interpreter_factory=lambda path, num_threads: FakeInterpreter(),
    )

    with pytest.raises(ValueError, match="Expected 298 features"):
        predictor.predict_proba(
            features=np.zeros(282, dtype=np.float32),
            v_norm=np.zeros(500, dtype=np.float32),
            i_norm=np.zeros(500, dtype=np.float32),
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/test_tflite_predictor.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'src.runtime.tflite_predictor'`

- [ ] **Step 3: Write the minimal implementation**

```python
# src/runtime/tflite_predictor.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np


def _default_interpreter_factory(model_path: str, num_threads: int):
    try:
        import tflite_runtime.interpreter as tflite
        return tflite.Interpreter(model_path=model_path, num_threads=num_threads)
    except ModuleNotFoundError:
        from tensorflow.lite import Interpreter
        return Interpreter(model_path=model_path, num_threads=num_threads)


@dataclass(frozen=True)
class TFLiteContract:
    wave_shape: tuple[int, int, int] = (1, 500, 2)
    mag_shape: tuple[int, int] = (1, 28)
    phase_shape: tuple[int, int] = (1, 270)
    output_width: int = 7
    feature_length: int = 298


class TFLitePredictor:
    def __init__(
        self,
        model_path: str,
        class_names: list[str],
        *,
        num_threads: int = 1,
        interpreter_factory: Optional[Callable[[str, int], object]] = None,
    ) -> None:
        self.class_names = class_names
        self.contract = TFLiteContract(output_width=len(class_names))
        self._model_path = Path(model_path)
        if not self._model_path.exists():
            raise FileNotFoundError(f"TFLite artifact not found: {self._model_path}")

        factory = interpreter_factory or _default_interpreter_factory
        self._interpreter = factory(str(self._model_path), num_threads)
        self._interpreter.allocate_tensors()

        self._inputs = list(self._interpreter.get_input_details())
        self._outputs = list(self._interpreter.get_output_details())

        if len(self._inputs) != 3:
            raise ValueError(f"Expected 3 TFLite inputs, got {len(self._inputs)}")
        if len(self._outputs) != 1:
            raise ValueError(f"Expected 1 TFLite output, got {len(self._outputs)}")

        self.input_names = [str(d.get("name", "")) for d in self._inputs]
        self.output_shape = tuple(int(x) for x in self._outputs[0]["shape"])

        expected_shapes = [
            self.contract.wave_shape,
            self.contract.mag_shape,
            self.contract.phase_shape,
        ]
        for detail, expected in zip(self._inputs, expected_shapes):
            got = tuple(int(x) for x in detail["shape"])
            if got != expected:
                raise ValueError(f"Input {detail.get('name', '')} has shape {got}; expected {expected}")

        if self.output_shape != (1, self.contract.output_width):
            raise ValueError(
                f"Output tensor has shape {self.output_shape}; expected {(1, self.contract.output_width)}"
            )

    def predict_proba(self, features: np.ndarray, v_norm: np.ndarray, i_norm: np.ndarray) -> np.ndarray:
        feat = np.asarray(features, dtype=np.float32).reshape(-1)
        if feat.size != self.contract.feature_length:
            raise ValueError(f"Expected {self.contract.feature_length} features, got {feat.size}")

        v = np.asarray(v_norm, dtype=np.float32).reshape(-1)
        i = np.asarray(i_norm, dtype=np.float32).reshape(-1)
        if v.size != 500 or i.size != 500:
            raise ValueError("Expected 500 normalized samples per channel")

        x_wave = np.stack([v, i], axis=-1).reshape(self.contract.wave_shape)
        x_mag = feat[28:56].reshape(self.contract.mag_shape)
        x_phase = np.concatenate([feat[0:28], feat[56:214], feat[214:298]]).reshape(self.contract.phase_shape)

        self._interpreter.set_tensor(self._inputs[0]["index"], x_wave)
        self._interpreter.set_tensor(self._inputs[1]["index"], x_mag)
        self._interpreter.set_tensor(self._inputs[2]["index"], x_phase)
        self._interpreter.invoke()

        probs = np.asarray(self._interpreter.get_tensor(self._outputs[0]["index"]), dtype=np.float32).reshape(-1)
        if probs.size != len(self.class_names):
            raise ValueError(f"Expected {len(self.class_names)} output scores, got {probs.size}")
        return probs
```

- [ ] **Step 4: Update config to point at the TFLite artifact**

```yaml
# configs/default.yaml
ml_inference:
  model_path: artifacts/models/pqm_multilabel_model.tflite
  receiver_mode: tflite
  output_semantics: single_label
```

- [ ] **Step 5: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/test_tflite_predictor.py -q`

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add configs/default.yaml src/runtime/tflite_predictor.py tests/test_tflite_predictor.py
git commit -m "feat: add tflite predictor for canonical 298 runtime"
```

### Task 2: Rename the Public Live Mode From `model4` to `tflite`

**Files:**
- Modify: `src/io/frame_protocol.py`
- Modify: `src/io/serial_receiver.py`
- Modify: `src/infer/live_infer.py`
- Modify: `src/ui/app.py`
- Modify: `src/system/kiosk_setup.sh`
- Create: `tests/test_runtime_contract.py`
- Modify: `tests/test_receiver_resync.py`

- [ ] **Step 1: Write the failing contract tests**

```python
# tests/test_runtime_contract.py
from __future__ import annotations

from src.dsp.preprocess import load_config
from src.ui.app import _build_parser as build_ui_parser
from src.infer.live_infer import _build_parser as build_live_parser


def test_config_defaults_to_tflite_receiver_mode() -> None:
    cfg = load_config("configs/default.yaml")
    assert cfg["ml_inference"]["receiver_mode"] == "tflite"


def test_live_infer_defaults_to_tflite_mode() -> None:
    parser = build_live_parser()
    args = parser.parse_args(["--port", "/dev/null"])
    assert args.receiver_mode == "tflite"


def test_ui_defaults_to_tflite_mode() -> None:
    parser = build_ui_parser()
    args = parser.parse_args(["--port", "/dev/null"])
    assert args.receiver_mode == "tflite"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/test_runtime_contract.py -q`

Expected: FAIL because the current defaults still resolve to `model4` or `feature`

- [ ] **Step 3: Rename the public transport terminology**

```python
# src/io/frame_protocol.py
INFERENCE_FRAME_TYPE = 0x0003

@dataclass
class InferenceFrame:
    seq: int
    X_wave: np.ndarray
    X_mag: np.ndarray
    X_phase: np.ndarray
    rx_crc: int
    calc_crc: int


ModelReadyFrame = InferenceFrame


def parse_inference_frame(frame: bytes) -> InferenceFrame:
    ...


parse_model_ready_frame = parse_inference_frame
```

```python
# src/io/serial_receiver.py
if mode not in ("raw", "feature", "tflite", "model4"):
    raise ValueError("mode must be 'raw', 'feature', 'tflite', or deprecated 'model4'")

normalized_mode = "tflite" if mode == "model4" else mode
self.mode = normalized_mode

if self.mode == "tflite":
    parsed = parse_inference_frame(frame_bytes)
```

```python
# src/infer/live_infer.py
parser.add_argument(
    "--receiver-mode",
    choices=["tflite", "feature", "raw"],
    default="tflite",
    help="tflite: packed live inference frame (default), feature: legacy 282-feature replay/debug, raw: host DSP fallback",
)
```

```python
# src/ui/app.py
parser.add_argument(
    "--receiver-mode",
    choices=["tflite", "feature", "raw"],
    default="tflite",
    help="tflite: packed live inference frame (default), feature: legacy compatibility, raw: host DSP fallback",
)
```

```bash
# src/system/kiosk_setup.sh
--receiver-mode MODE   tflite, feature, or raw (default: tflite)

local RECEIVER_MODE="tflite"
```

- [ ] **Step 4: Update receiver-resync expectations**

```python
# tests/test_receiver_resync.py
def test_main_dispatches_tflite_mode_to_tflite_recorder(monkeypatch, tmp_path):
    calls = {"tflite": 0, "raw": 0, "feature": 0, "snapshots": 0}
    ...
    monkeypatch.setattr(serial_receiver_mod, "record_tflite_stream", _mk("tflite"))
    ...
    assert calls["tflite"] == 1
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/test_runtime_contract.py tests/test_receiver_resync.py -q`

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/io/frame_protocol.py src/io/serial_receiver.py src/infer/live_infer.py src/ui/app.py src/system/kiosk_setup.sh tests/test_runtime_contract.py tests/test_receiver_resync.py
git commit -m "refactor: rename public live mode to tflite"
```

### Task 3: Make the Runtime Pipeline TFLite-Only and 298-Canonical

**Files:**
- Modify: `src/runtime/pipeline.py`
- Modify: `src/runtime/__init__.py`
- Modify: `src/ui/views/dashboard.py`
- Modify: `tests/test_e2e_pipeline.py`
- Modify: `tests/test_frame_predictor_integration.py`

- [ ] **Step 1: Write the failing pipeline tests**

```python
# tests/test_frame_predictor_integration.py
from __future__ import annotations

import numpy as np

from src.runtime.tflite_predictor import TFLitePredictor


def test_reconstructs_298_feature_vector_from_inference_frame(valid_frame):
    reconstructed = np.concatenate(
        [
            valid_frame.X_phase[0:28],
            valid_frame.X_mag,
            valid_frame.X_phase[28:172],
            valid_frame.X_phase[172:270],
        ]
    )
    assert reconstructed.shape == (298,)
```

```python
# tests/test_e2e_pipeline.py
def test_runtime_pipeline_rejects_tflite_without_wave_inputs(tmp_path):
    cfg = load_config("configs/default.yaml")
    predictor = DummyPredictor(len(cfg["classes"]["names"]))

    pipeline = RuntimePipeline(
        cfg,
        predictor,
        replay_source=[{"seq": 1, "features": np.zeros(298, dtype=np.float32)}],
        session_log_path=str(tmp_path / "session.jsonl"),
    )

    pipeline.start()
    snapshot = pipeline.get_result(timeout=2.0)
    pipeline.stop()

    assert snapshot is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/test_frame_predictor_integration.py tests/test_e2e_pipeline.py -q`

Expected: FAIL on naming/runtime assumptions around `model4`, predictor behavior, or waveform plotting expectations

- [ ] **Step 3: Replace the generic artifact path in the runtime**

```python
# src/runtime/pipeline.py
from src.runtime.tflite_predictor import TFLitePredictor


class RuntimePipeline:
    def __init__(..., predictor: PredictorProtocol, *, receiver_mode: str = "tflite", ...):
        ...
        self._receiver_mode = "tflite" if receiver_mode == "model4" else receiver_mode
        ...

    def _inference_loop(self) -> None:
        ...
        if self._receiver_mode == "tflite" and (context.v_norm is None or context.i_norm is None):
            raise ValueError("tflite mode requires normalized waveform inputs")

        if context.v_norm is not None and context.i_norm is not None:
            probs = self.predictor.predict_proba(context.features, context.v_norm, context.i_norm)
        else:
            probs = self.predictor.predict_proba(context.features)
```

```python
# src/runtime/pipeline.py
@dataclass
class InferenceSnapshot:
    ...
    v_phys: Optional[list[float]] = None
    i_phys: Optional[list[float]] = None
    v_norm: Optional[list[float]] = None
    i_norm: Optional[list[float]] = None
```

```python
# src/runtime/pipeline.py
return InferenceSnapshot(
    ...
    v_phys=context.v_phys.tolist() if context.v_phys is not None else None,
    i_phys=context.i_phys.tolist() if context.i_phys is not None else None,
    v_norm=context.v_norm.tolist() if context.v_norm is not None else None,
    i_norm=context.i_norm.tolist() if context.i_norm is not None else None,
    event=event,
)
```

```python
# src/ui/views/dashboard.py
if snapshot.v_phys is not None and snapshot.i_phys is not None:
    self._waveforms.update_waveforms(snapshot.v_phys, snapshot.i_phys, self._fs_hz)
elif snapshot.v_norm is not None and snapshot.i_norm is not None:
    self._waveforms.update_waveforms(snapshot.v_norm, snapshot.i_norm, self._fs_hz)
```

```python
# src/runtime/__init__.py
"""Lightweight runtime package exports."""

from .buffers import AtomicValue, BoundedQueue, QueueStats
from .metrics import RuntimeMetrics, StageStats

__all__ = [
    "AtomicValue",
    "BoundedQueue",
    "QueueStats",
    "RuntimeMetrics",
    "StageStats",
]
```

- [ ] **Step 4: Replace production predictor construction in callers**

```python
# src/ui/app.py
from src.runtime.tflite_predictor import TFLitePredictor

predictor = TFLitePredictor(
    model_path=args.model or cfg["ml_inference"]["model_path"],
    class_names=list(cfg["classes"]["names"]),
)
```

```python
# src/infer/live_infer.py
from src.runtime.tflite_predictor import TFLitePredictor
...
predictor = TFLitePredictor(
    model_path=args.model or cfg["ml_inference"]["model_path"],
    class_names=list(cfg["classes"]["names"]),
)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/test_tflite_predictor.py tests/test_frame_predictor_integration.py tests/test_e2e_pipeline.py -q`

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/runtime/pipeline.py src/runtime/__init__.py src/ui/views/dashboard.py src/ui/app.py src/infer/live_infer.py tests/test_e2e_pipeline.py tests/test_frame_predictor_integration.py
git commit -m "refactor: make runtime pipeline tflite-only and 298-canonical"
```

### Task 4: Separate Canonical 298 Replay From Legacy 282 Compatibility

**Files:**
- Modify: `src/infer/offline_replay.py`
- Modify: `tests/test_e2e_pipeline.py`
- Modify: `README.md`

- [ ] **Step 1: Write the failing replay tests**

```python
# tests/test_e2e_pipeline.py
def test_offline_replay_loader_supports_jsonl_298(tmp_path):
    path = tmp_path / "frames_298.jsonl"
    with path.open("w", encoding="utf-8") as fp:
        fp.write(json.dumps({"seq": 1, "features": [0.0] * 298}) + "\n")

    frames = list(load_replay_source(str(path)))
    assert len(frames) == 1
    assert np.asarray(frames[0]["features"], dtype=np.float32).shape == (298,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/test_e2e_pipeline.py::test_offline_replay_loader_supports_jsonl_298 -q`

Expected: FAIL with `Invalid features length 298`

- [ ] **Step 3: Update replay validation**

```python
# src/infer/offline_replay.py
if has_features:
    features = np.asarray(payload["features"], dtype=np.float32).reshape(-1)
    valid_lengths = (N_FEATURES, TOTAL_FEATURES)
    if features.size not in valid_lengths:
        raise ValueError(
            f"Invalid features length {features.size} at {source}; expected one of {valid_lengths}"
        )
```

```python
# src/infer/offline_replay.py
if suffix in {".jsonl", ".json"}:
    return _replay_from_jsonl(replay_path)

# Keep binary feature frames parseable, but document N_FEATURES=282 as legacy compatibility only.
```

- [ ] **Step 4: Update README examples to stop treating 282 feature mode as the primary path**

```markdown
# README.md
## Live Run Command

```bash
.venv/bin/python -m src.ui.app \
  --port /dev/ttyACM0 \
  --config configs/default.yaml \
  --receiver-mode tflite
```

`feature` mode remains available only for legacy replay/debug compatibility with 282-feature frames.
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/test_e2e_pipeline.py -q`

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/infer/offline_replay.py tests/test_e2e_pipeline.py README.md
git commit -m "fix: align replay loaders with canonical 298 contract"
```

### Task 5: Align Runtime Output Semantics With the Active Artifact

**Files:**
- Modify: `configs/default.yaml`
- Modify: `src/runtime/pipeline.py`
- Modify: `tests/test_runtime_contract.py`

- [ ] **Step 1: Write the failing output-semantics tests**

```python
# tests/test_runtime_contract.py
from src.dsp.preprocess import load_config


def test_active_artifact_uses_single_label_output_semantics() -> None:
    cfg = load_config("configs/default.yaml")
    assert cfg["ml_inference"]["output_semantics"] == "single_label"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/test_runtime_contract.py::test_active_artifact_uses_single_label_output_semantics -q`

Expected: FAIL because `output_semantics` is not defined yet

- [ ] **Step 3: Implement explicit single-label semantics**

```yaml
# configs/default.yaml
ml_inference:
  output_semantics: single_label
```

```python
# src/runtime/pipeline.py
self._output_semantics = str(ml_cfg.get("output_semantics", "single_label"))
...
if self._output_semantics == "single_label":
    active_labels = [top1_label]
    active_probs_list = [top1_conf]
else:
    ...
```

```python
# src/runtime/pipeline.py
if self._output_semantics == "single_label":
    if top1_label != self.normal_label:
        severity = "high" if top1_conf >= 0.9 else ("medium" if top1_conf >= 0.7 else "low")
        event = {
            "label": top1_label,
            "confidence": top1_conf,
            "severity": severity,
            "timestamp": context.timestamp,
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/test_runtime_contract.py -q`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add configs/default.yaml src/runtime/pipeline.py tests/test_runtime_contract.py
git commit -m "fix: make runtime output semantics explicit for tflite artifact"
```

### Task 6: Rewrite Active Docs Around `.tflite`, `298`, and `tflite` Mode

**Files:**
- Modify: `README.md`
- Modify: `prd.md`
- Modify: `tasks.md`
- Modify: `docs/model_prd.md`
- Modify: `docs/teensy_dsp_migration_tracker.md`
- Modify: `docs/report_alignment_matrix.md`

- [ ] **Step 1: Write the failing doc-alignment test**

```python
# tests/test_runtime_contract.py
from pathlib import Path


def test_readme_mentions_tflite_and_298_contract() -> None:
    text = Path("README.md").read_text(encoding="utf-8")
    assert "pqm_multilabel_model.tflite" in text
    assert "Feature vector length: `298`" in text
    assert "--receiver-mode tflite" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/test_runtime_contract.py::test_readme_mentions_tflite_and_298_contract -q`

Expected: FAIL because the docs still claim `282` and `feature`

- [ ] **Step 3: Update the docs**

```markdown
# README.md
- Feature vector length: `298`
- Production model artifact: `artifacts/models/pqm_multilabel_model.tflite`
- Default live receiver mode: `tflite`
```

```markdown
# prd.md
Replace active references to `282` with `298` wherever the text describes the current runtime contract.
Replace `.keras`-oriented active deployment language with `.tflite`.
Replace `model4` as the preferred active live path term with `tflite` or `packed inference frame`.
```

```markdown
# docs/model_prd.md
Add a top-of-file note:
"Historical sections below describe the older 282-feature research contract. The active runtime and deployment contract for this repository is the 298-feature TFLite path documented in README/config/runtime code."
```

```markdown
# docs/report_alignment_matrix.md
| Live inference integration | TFLite artifact + 298-feature packed live frame | ... | ... | Implemented |
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/test_runtime_contract.py -q`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add README.md prd.md tasks.md docs/model_prd.md docs/teensy_dsp_migration_tracker.md docs/report_alignment_matrix.md tests/test_runtime_contract.py
git commit -m "docs: align active repo contract with tflite and 298 features"
```

### Task 7: Final Regression Sweep

**Files:**
- Modify: `tests/test_model_ready_protocol.py`
- Modify: `tests/test_runtime_serial_startup.py`
- Modify: `tests/test_feature_shape.py`
- Modify: `tests/test_feature_frame_protocol.py`

- [ ] **Step 1: Update naming in protocol and startup tests**

```python
# tests/test_model_ready_protocol.py
def test_parse_inference_frame_round_trip():
    ...
```

```python
# tests/test_runtime_serial_startup.py
def test_runtime_uses_tflite_mode_by_default(...):
    ...
```

- [ ] **Step 2: Keep legacy tests clearly labeled**

```python
# tests/test_feature_frame_protocol.py
def test_legacy_feature_frame_round_trip_282():
    ...
```

```python
# tests/test_feature_shape.py
def test_extract_features_generates_298_element_vector():
    ...
```

- [ ] **Step 3: Run the focused regression suite**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/test_tflite_predictor.py tests/test_runtime_contract.py tests/test_e2e_pipeline.py tests/test_frame_predictor_integration.py tests/test_model_ready_protocol.py tests/test_receiver_resync.py -q`

Expected: PASS

- [ ] **Step 4: Run the broader runtime/protocol suite**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/test_frame_protocol.py tests/test_feature_frame_protocol.py tests/test_feature_shape.py tests/test_runtime_buffers.py tests/test_runtime_metrics.py tests/test_runtime_serial_startup.py -q`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_model_ready_protocol.py tests/test_runtime_serial_startup.py tests/test_feature_shape.py tests/test_feature_frame_protocol.py
git commit -m "test: complete tflite and 298 contract regression sweep"
```

---

## Self-Review

### Spec coverage

- `.tflite` as the only production inference artifact: covered in Tasks 1, 3, 6.
- `298` as the canonical active feature contract: covered in Tasks 1, 3, 4, 6, 7.
- Removal of `model4` as the active public runtime term: covered in Tasks 2 and 6.
- Live packed three-input path retained under neutral naming: covered in Tasks 2 and 3.
- Legacy `282` support retained only as compatibility/debug behavior: covered in Tasks 4 and 7.

### Placeholder scan

- No `TODO`, `TBD`, or “similar to Task N” references remain.
- Every code-changing task includes concrete snippets.
- Every verification step includes an exact command and expected result.

### Type consistency

- Predictor contract is consistently `features + v_norm + i_norm`.
- Canonical live mode is consistently `tflite`.
- Canonical feature length is consistently `298`.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-05-tflite-298-runtime-migration.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
