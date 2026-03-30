from __future__ import annotations

import numpy as np

from src.dsp.preprocess import load_config, preprocess_frame


def test_preprocess_outputs_shape_and_finite_values():
    cfg = load_config("configs/default.yaml")
    n = int(cfg["signal"]["samples_per_frame"])

    t = np.linspace(0.0, 1.0, n, endpoint=False)
    v_raw = (2071 + 120 * np.sin(2.0 * np.pi * 5.0 * t)).astype(np.int16)
    i_raw = (2048 + 80 * np.sin(2.0 * np.pi * 5.0 * t + 0.2)).astype(np.int16)

    out = preprocess_frame(v_raw, i_raw, cfg)

    assert out["v_phys"].shape == (n,)
    assert out["i_phys"].shape == (n,)
    assert out["v_norm"].shape == (n,)
    assert out["i_norm"].shape == (n,)

    assert np.all(np.isfinite(out["v_phys"]))
    assert np.all(np.isfinite(out["i_phys"]))
    assert np.all(np.isfinite(out["v_norm"]))
    assert np.all(np.isfinite(out["i_norm"]))

    assert np.max(np.abs(out["v_norm"])) <= 1.0 + 1e-9
    assert np.max(np.abs(out["i_norm"])) <= 1.0 + 1e-9
    assert abs(float(np.mean(out["v_phys"]))) < 1e-9
    assert abs(float(np.mean(out["i_phys"]))) < 1e-9


def test_preprocess_rejects_wrong_frame_length():
    cfg = load_config("configs/default.yaml")
    n = int(cfg["signal"]["samples_per_frame"])

    v_raw = np.zeros(n - 1, dtype=np.int16)
    i_raw = np.zeros(n - 1, dtype=np.int16)

    try:
        preprocess_frame(v_raw, i_raw, cfg, expected_n=n)
    except ValueError as exc:
        assert "Expected" in str(exc)
    else:
        raise AssertionError("Expected ValueError for wrong frame length")


def test_preprocess_handles_near_zero_signal_without_non_finite():
    cfg = load_config("configs/default.yaml")
    n = int(cfg["signal"]["samples_per_frame"])

    v_raw = np.full(n, int(cfg["calibration"]["v_adc_midpoint"]), dtype=np.int16)
    i_raw = np.full(n, int(cfg["calibration"]["i_adc_midpoint"]), dtype=np.int16)

    out = preprocess_frame(v_raw, i_raw, cfg, expected_n=n)

    assert np.all(np.isfinite(out["v_norm"]))
    assert np.all(np.isfinite(out["i_norm"]))
    assert np.max(np.abs(out["v_norm"])) == 0.0
    assert np.max(np.abs(out["i_norm"])) == 0.0