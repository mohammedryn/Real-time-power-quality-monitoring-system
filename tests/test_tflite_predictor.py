from __future__ import annotations

import builtins
import sys
import types

import numpy as np
import pytest

from src.runtime.tflite_predictor import TFLitePredictor
import src.runtime.tflite_predictor as predictor_mod


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
        self.last_tensors: dict[int, np.ndarray] = {}
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


def test_default_factory_falls_back_to_ai_edge_litert(monkeypatch, tmp_path) -> None:
    model_path = tmp_path / "model.tflite"
    model_path.write_bytes(b"TFL3")

    class FakeLiteRTInterpreter:
        def __init__(self, *, model_path: str, num_threads: int | None = None) -> None:
            self.model_path = model_path
            self.num_threads = num_threads

    fake_pkg = types.ModuleType("ai_edge_litert")
    fake_interpreter_mod = types.ModuleType("ai_edge_litert.interpreter")
    fake_interpreter_mod.Interpreter = FakeLiteRTInterpreter
    fake_pkg.interpreter = fake_interpreter_mod

    monkeypatch.setitem(sys.modules, "ai_edge_litert", fake_pkg)
    monkeypatch.setitem(sys.modules, "ai_edge_litert.interpreter", fake_interpreter_mod)

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "tflite_runtime.interpreter":
            raise ModuleNotFoundError("No module named 'tflite_runtime'")
        if name == "tensorflow":
            raise ModuleNotFoundError("No module named 'tensorflow'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    interpreter = predictor_mod._default_interpreter_factory(str(model_path), 4)

    assert isinstance(interpreter, FakeLiteRTInterpreter)
    assert interpreter.model_path == str(model_path)
    assert interpreter.num_threads == 4
