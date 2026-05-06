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
        try:
            from ai_edge_litert.interpreter import Interpreter as LiteRTInterpreter

            # LiteRT is the current Google-distributed lightweight runtime.
            # Some releases accept num_threads, while others only accept model_path.
            try:
                return LiteRTInterpreter(model_path=model_path, num_threads=num_threads)
            except TypeError:
                return LiteRTInterpreter(model_path=model_path)
        except ModuleNotFoundError:
            pass

        try:
            import tensorflow as tf
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "TFLite inference requires tflite_runtime, ai-edge-litert, or tensorflow to be installed"
            ) from exc
        return tf.lite.Interpreter(model_path=model_path, num_threads=num_threads)


@dataclass(frozen=True)
class TFLiteContract:
    wave_shape: tuple[int, int, int] = (1, 500, 2)
    mag_shape: tuple[int, int] = (1, 28)
    phase_shape: tuple[int, int] = (1, 270)
    feature_length: int = 298


class TFLitePredictor:
    """Production predictor for the canonical 298-feature TFLite artifact."""

    _is_multi_input = True

    def __init__(
        self,
        model_path: str,
        class_names: list[str],
        *,
        num_threads: int = 1,
        interpreter_factory: Optional[Callable[[str, int], object]] = None,
    ) -> None:
        self.class_names = list(class_names)
        self.contract = TFLiteContract()
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

        self.input_names = [str(detail.get("name", "")) for detail in self._inputs]
        self.output_shape = tuple(int(x) for x in self._outputs[0]["shape"])

        expected_shapes = (
            self.contract.wave_shape,
            self.contract.mag_shape,
            self.contract.phase_shape,
        )
        for detail, expected in zip(self._inputs, expected_shapes):
            got = tuple(int(x) for x in detail["shape"])
            if got != expected:
                raise ValueError(
                    f"Input {detail.get('name', '')!r} has shape {got}; expected {expected}"
                )

        expected_output_shape = (1, len(self.class_names))
        if self.output_shape != expected_output_shape:
            raise ValueError(
                f"Output tensor has shape {self.output_shape}; expected {expected_output_shape}"
            )

    def predict_proba(
        self,
        features: np.ndarray,
        v_norm: np.ndarray | None = None,
        i_norm: np.ndarray | None = None,
    ) -> np.ndarray:
        feat = np.asarray(features, dtype=np.float32).reshape(-1)
        if feat.size != self.contract.feature_length:
            raise ValueError(f"Expected {self.contract.feature_length} features, got {feat.size}")

        if v_norm is None or i_norm is None:
            raise ValueError("TFLite predictor requires v_norm and i_norm inputs")

        v = np.asarray(v_norm, dtype=np.float32).reshape(-1)
        i = np.asarray(i_norm, dtype=np.float32).reshape(-1)
        if v.size != 500 or i.size != 500:
            raise ValueError("Expected 500 normalized samples per channel")

        x_wave = np.stack([v, i], axis=-1).reshape(self.contract.wave_shape)
        x_mag = feat[28:56].reshape(self.contract.mag_shape)
        x_phase = np.concatenate([feat[0:28], feat[56:214], feat[214:298]]).reshape(
            self.contract.phase_shape
        )

        self._interpreter.set_tensor(self._inputs[0]["index"], x_wave)
        self._interpreter.set_tensor(self._inputs[1]["index"], x_mag)
        self._interpreter.set_tensor(self._inputs[2]["index"], x_phase)
        self._interpreter.invoke()

        probs = np.asarray(
            self._interpreter.get_tensor(self._outputs[0]["index"]),
            dtype=np.float32,
        ).reshape(-1)
        if probs.size != len(self.class_names):
            raise ValueError(f"Expected {len(self.class_names)} output scores, got {probs.size}")
        return probs
