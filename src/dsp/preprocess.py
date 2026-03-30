from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_config(path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as fp:
        cfg = yaml.safe_load(fp)
    if not isinstance(cfg, dict):
        raise ValueError("Config file must load to a mapping")
    return cfg


def _extract_calibration(cfg: dict[str, Any]) -> tuple[float, float, float, float]:
    calibration = cfg.get("calibration")
    if not isinstance(calibration, dict):
        raise ValueError("Missing calibration section in config")

    try:
        v_mid = float(calibration["v_adc_midpoint"])
        i_mid = float(calibration["i_adc_midpoint"])
        v_scale = float(calibration["v_counts_to_volts"])
        i_scale = float(calibration["i_counts_to_amps"])
    except KeyError as exc:
        raise ValueError(f"Missing calibration key: {exc}") from exc

    return v_mid, i_mid, v_scale, i_scale


def adc_to_physical(v_raw: np.ndarray, i_raw: np.ndarray, cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    v_mid, i_mid, v_scale, i_scale = _extract_calibration(cfg)

    v_arr = np.asarray(v_raw, dtype=np.float64)
    i_arr = np.asarray(i_raw, dtype=np.float64)

    v_phys = (v_arr - v_mid) * v_scale
    i_phys = (i_arr - i_mid) * i_scale
    return v_phys, i_phys


def remove_dc_offset(v_phys: np.ndarray, i_phys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    v_dc = np.asarray(v_phys, dtype=np.float64) - float(np.mean(v_phys))
    i_dc = np.asarray(i_phys, dtype=np.float64) - float(np.mean(i_phys))
    return v_dc, i_dc


def normalize_waveforms(v_wave: np.ndarray, i_wave: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    v = np.asarray(v_wave, dtype=np.float64)
    i = np.asarray(i_wave, dtype=np.float64)

    v_scale = max(float(np.max(np.abs(v))), eps)
    i_scale = max(float(np.max(np.abs(i))), eps)

    return v / v_scale, i / i_scale


def preprocess_frame(
    v_raw: np.ndarray,
    i_raw: np.ndarray,
    cfg: dict[str, Any],
    expected_n: int | None = None,
) -> dict[str, np.ndarray]:
    v_arr = np.asarray(v_raw)
    i_arr = np.asarray(i_raw)

    if v_arr.ndim != 1 or i_arr.ndim != 1:
        raise ValueError("Input waveforms must be 1D arrays")
    if v_arr.shape != i_arr.shape:
        raise ValueError(f"Waveform shape mismatch: v={v_arr.shape}, i={i_arr.shape}")

    if expected_n is None:
        signal_cfg = cfg.get("signal", {}) if isinstance(cfg, dict) else {}
        if isinstance(signal_cfg, dict) and "samples_per_frame" in signal_cfg:
            expected_n = int(signal_cfg["samples_per_frame"])

    if expected_n is not None and len(v_arr) != int(expected_n):
        raise ValueError(f"Expected {expected_n} samples per frame, got {len(v_arr)}")

    v_phys, i_phys = adc_to_physical(v_arr, i_arr, cfg)
    v_phys, i_phys = remove_dc_offset(v_phys, i_phys)
    v_norm, i_norm = normalize_waveforms(v_phys, i_phys)

    for name, arr in {
        "v_phys": v_phys,
        "i_phys": i_phys,
        "v_norm": v_norm,
        "i_norm": i_norm,
    }.items():
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Non-finite values found in {name}")

    return {
        "v_phys": v_phys,
        "i_phys": i_phys,
        "v_norm": v_norm,
        "i_norm": i_norm,
    }