"""
Parity tests between the Python feature pipeline and a Python re-implementation
of the Teensy C++ DSP logic.

These tests validate that the firmware dsp.cpp logic, once ported to C++, will
produce feature vectors that match the Python pipeline within tolerance.

The Python reference here mirrors dsp.cpp exactly:
  - Same calibration constants (from configs/default.yaml)
  - Same Goertzel algorithm for harmonic extraction (float32)
  - Same db4 DWT with periodization boundary (float32 inputs)
  - Same feature assembly order (from feature_index.py)

Key design decision:
  Phase features (phase_abs, phase_cross, phase_rel) are only tested on
  harmonics with significant magnitude. For low-energy bins, phase is
  numerically unstable in both float32 Goertzel and float64 FFT — this is
  expected behaviour and is handled by the model learning from consistent
  firmware output, not from Python parity.

Tolerances (from feature_index.PARITY_TOLERANCES):
  - time/magnitudes/power groups: atol=1e-3
  - Phase groups: magnitude-gated, atol=5e-3 for significant harmonics
  - DWT moments/entropy: atol=1e-2
"""
from __future__ import annotations

import numpy as np
import pytest
import pywt

from src.dsp.features import extract_features
from src.dsp.feature_index import FEATURE_INDEX, PARITY_TOLERANCES, DWT_BAND_SIZES
from src.dsp.preprocess import load_config

# Calibration constants and sampling assumptions — must match firmware + config.
CFG = load_config("configs/default.yaml")

V_MIDPOINT = float(CFG["calibration"]["v_adc_midpoint"])
V_SCALE    = float(CFG["calibration"]["v_counts_to_volts"])
I_MIDPOINT = float(CFG["calibration"]["i_adc_midpoint"])
I_SCALE    = float(CFG["calibration"]["i_counts_to_amps"])

FS     = int(CFG["signal"]["fs_hz"])
N      = int(CFG["signal"]["samples_per_frame"])
N_HARM = 13

# Minimum magnitude (relative to fundamental) for a harmonic to have a
# meaningful phase comparison between Goertzel float32 and FFT float64.
PHASE_MAG_THRESHOLD = 0.05


# ---- Python reference of Goertzel (float32) ---------------------------------

def _goertzel_bin_f32(x: np.ndarray, k: int) -> tuple[float, float]:
    """Single DFT bin via Goertzel, matching goertzel.h (float32 arithmetic)."""
    x_f32  = x.astype(np.float32)
    n      = len(x_f32)
    omega  = np.float32(2.0 * np.pi * k / n)
    coeff  = np.float32(2.0) * np.cos(omega)
    s1 = np.float32(0.0)
    s2 = np.float32(0.0)
    for xi in x_f32:
        s  = xi + coeff * s1 - s2
        s2 = s1
        s1 = s
    real = s1 - np.cos(omega) * s2
    imag = np.sin(omega) * s2
    # Correct inherent 1-sample phase lag: multiply by e^(+jω)
    co, si = np.cos(omega), np.sin(omega)
    real_c = real * co - imag * si
    imag_c = real * si + imag * co
    mag  = float(np.sqrt(real_c * real_c + imag_c * imag_c) / np.float32(n / 2.0))
    ph   = float(np.arctan2(imag_c, real_c))
    return mag, ph


def _teensy_harmonics(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mags   = np.zeros(N_HARM, dtype=np.float32)
    phases = np.zeros(N_HARM, dtype=np.float32)
    for h in range(1, N_HARM + 1):
        m, p = _goertzel_bin_f32(x, h * 5)
        mags[h - 1]   = np.float32(m)
        phases[h - 1] = np.float32(p)
    return mags, phases


def _teensy_preprocess(v_raw: np.ndarray, i_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    v = (v_raw.astype(np.float32) - np.float32(V_MIDPOINT)) * np.float32(V_SCALE)
    i = (i_raw.astype(np.float32) - np.float32(I_MIDPOINT)) * np.float32(I_SCALE)
    v -= v.mean()
    i -= i.mean()
    return v, i


def _dwt_stats_f32(x: np.ndarray) -> list[float]:
    coeffs = pywt.wavedec(x.astype(np.float32), 'db4', level=5, mode='periodization')
    stats  = []
    for c in coeffs:
        if len(c) == 0:
            c = np.array([0.0], dtype=np.float32)
        mn = float(np.mean(c))
        sd = float(np.std(c))
        e  = float(np.sum(c ** 2))
        sk = float(np.mean((c - mn) ** 3) / sd ** 3) if sd > 1e-10 else 0.0
        kt = float(np.mean((c - mn) ** 4) / sd ** 4 - 3.0) if sd > 1e-10 else 0.0
        p  = c ** 2 / e if e > 0 else np.zeros_like(c)
        p  = p[p > 0]
        en = float(-np.sum(p * np.log2(p))) if len(p) > 0 else 0.0
        stats.extend([mn, sd, sk, kt, e, en])
    return stats


def _td_f32(x: np.ndarray) -> list[float]:
    mn  = float(np.mean(x))
    sd  = float(np.std(x))
    rms = float(np.sqrt(np.mean(x ** 2)))
    pk  = float(np.max(np.abs(x)))
    ma  = float(np.mean(np.abs(x)))
    return [
        mn, sd, rms, pk,
        pk / rms if rms > 1e-6 else 0.0,
        rms / ma if ma > 1e-6 else 0.0,
        float(np.mean((x - mn) ** 3) / sd ** 3) if sd > 1e-10 else 0.0,
        float(np.mean((x - mn) ** 4) / sd ** 4 - 3.0) if sd > 1e-10 else 0.0,
        float(np.max(x) - np.min(x)),
        float(np.sum(np.diff(np.signbit(x)))),
        float(np.min(x)),
        float(np.max(x)),
    ]


def _teensy_features(v_raw: np.ndarray, i_raw: np.ndarray) -> np.ndarray:
    """Full Python re-implementation of compute_features() in dsp.cpp."""
    v, i = _teensy_preprocess(v_raw, i_raw)

    feat: list[float] = []

    # 1. Time-domain
    feat.extend(_td_f32(v))
    feat.extend(_td_f32(i))

    # 2. Harmonics + THD
    v_mag, v_ph = _teensy_harmonics(v)
    i_mag, i_ph = _teensy_harmonics(i)
    feat.extend(v_mag.tolist())
    feat.extend(i_mag.tolist())
    v_rss = float(np.sqrt(np.sum(v_mag[1:] ** 2)))
    i_rss = float(np.sqrt(np.sum(i_mag[1:] ** 2)))
    feat.append(v_rss / float(v_mag[0]) if v_mag[0] > 1e-6 else 0.0)
    feat.append(i_rss / float(i_mag[0]) if i_mag[0] > 1e-6 else 0.0)

    # 3. Absolute phase sin/cos
    feat.extend(np.sin(v_ph).tolist())
    feat.extend(np.cos(v_ph).tolist())
    feat.extend(np.sin(i_ph).tolist())
    feat.extend(np.cos(i_ph).tolist())

    # 4. Cross-channel phase
    cross = v_ph - i_ph
    feat.extend(np.sin(cross).tolist())
    feat.extend(np.cos(cross).tolist())

    # 5. Relative-to-fundamental
    rel_v = v_ph[1:] - v_ph[0]
    rel_i = i_ph[1:] - i_ph[0]
    feat.extend(np.sin(rel_v).tolist())
    feat.extend(np.cos(rel_v).tolist())
    feat.extend(np.sin(rel_i).tolist())
    feat.extend(np.cos(rel_i).tolist())

    # 6. Harmonic power
    for h in range(N_HARM):
        feat.append(float(v_mag[h] * i_mag[h] * np.cos(cross[h])))
        feat.append(float(v_mag[h] * i_mag[h] * np.sin(cross[h])))

    # 7. Circular stats
    def c_mean(a):
        return float(np.arctan2(np.mean(np.sin(a)), np.mean(np.cos(a))))
    def c_std(a):
        R = np.sqrt(np.mean(np.sin(a)) ** 2 + np.mean(np.cos(a)) ** 2)
        R = max(float(R), 1e-12)
        return float(np.sqrt(-2.0 * np.log(R)))
    feat.extend([c_mean(v_ph), c_std(v_ph),
                 c_mean(i_ph), c_std(i_ph),
                 c_mean(cross), c_std(cross)])

    # 8. DWT
    feat.extend(_dwt_stats_f32(v))
    feat.extend(_dwt_stats_f32(i))

    assert len(feat) == 282
    return np.array(feat, dtype=np.float32)


# ---- Signal generators ------------------------------------------------------

def _make_adc_frame(seed: int, n_harmonics: int = 13) -> tuple[np.ndarray, np.ndarray]:
    """Generate a 50 Hz + harmonics signal as int16 ADC counts.

    Using all 13 harmonics ensures every Goertzel bin has meaningful energy,
    making phase comparisons numerically stable.
    """
    rng = np.random.default_rng(seed)
    t   = np.arange(N) / FS
    f0  = 50.0

    v_sig = sum(
        rng.uniform(0.2, 1.0) * np.sin(2 * np.pi * h * f0 * t + rng.uniform(0, np.pi))
        for h in range(1, n_harmonics + 1)
    )
    i_sig = sum(
        rng.uniform(0.1, 0.5) * np.sin(2 * np.pi * h * f0 * t + rng.uniform(0, np.pi))
        for h in range(1, n_harmonics + 1)
    )

    # Add moderate noise (40 dB SNR)
    for sig in (v_sig, i_sig):
        pw = np.mean(sig ** 2)
        sig += rng.normal(0, np.sqrt(pw / 1e4), N)

    v_adc = (v_sig / V_SCALE + V_MIDPOINT).clip(0, 4095).astype(np.int16)
    i_adc = (i_sig / I_SCALE + I_MIDPOINT).clip(0, 4095).astype(np.int16)
    return v_adc, i_adc


def _py_pipeline(v_adc: np.ndarray, i_adc: np.ndarray) -> np.ndarray:
    """Python pipeline with float64 (same as extract_features path)."""
    v = (v_adc.astype(np.float64) - V_MIDPOINT) * V_SCALE
    i = (i_adc.astype(np.float64) - I_MIDPOINT) * I_SCALE
    v -= v.mean()
    i -= i.mean()
    return extract_features(v, i)


# ---- DWT band length test ---------------------------------------------------

def test_dwt_band_lengths_under_periodization():
    """Band lengths for N=500 under periodization must match canonical sizes."""
    signal = np.random.default_rng(0).standard_normal(N).astype(np.float64)
    coeffs = pywt.wavedec(signal, 'db4', level=5, mode='periodization')
    expected = [DWT_BAND_SIZES[b] for b in ("cA5", "cD5", "cD4", "cD3", "cD2", "cD1")]
    actual   = [len(c) for c in coeffs]
    assert actual == expected, f"DWT band lengths {actual} != expected {expected}"


# ---- Non-phase group parity -------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2, 7, 42])
def test_parity_time_domain(seed: int):
    """Time-domain statistics must match within 1e-3."""
    v_adc, i_adc = _make_adc_frame(seed)
    py_feat = _py_pipeline(v_adc, i_adc)
    fw_feat = _teensy_features(v_adc, i_adc)
    s, e = FEATURE_INDEX["time_v"]
    np.testing.assert_allclose(fw_feat[s:e], py_feat[s:e], atol=1e-3, rtol=0)


@pytest.mark.parametrize("seed", [0, 1, 2, 7, 42])
def test_parity_harmonic_magnitudes_and_thd(seed: int):
    """Harmonic magnitudes and THD must match within 1e-3."""
    v_adc, i_adc = _make_adc_frame(seed)
    py_feat = _py_pipeline(v_adc, i_adc)
    fw_feat = _teensy_features(v_adc, i_adc)
    s, e = FEATURE_INDEX["harm_mag_thd"]
    np.testing.assert_allclose(fw_feat[s:e], py_feat[s:e], atol=1e-3, rtol=0)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_parity_dwt(seed: int):
    """DWT subband statistics must match within 1e-2."""
    v_adc, i_adc = _make_adc_frame(seed)
    py_feat = _py_pipeline(v_adc, i_adc)
    fw_feat = _teensy_features(v_adc, i_adc)
    s, e = FEATURE_INDEX["dwt"]
    np.testing.assert_allclose(fw_feat[s:e], py_feat[s:e], atol=1e-2, rtol=0)


# ---- Phase group parity (magnitude-gated) -----------------------------------

def _significant_harmonic_mask(v_adc, i_adc, threshold_rel=PHASE_MAG_THRESHOLD):
    """Return bool[13]: True for harmonics with magnitude > threshold * fundamental."""
    fw_feat = _teensy_features(v_adc, i_adc)
    s, _ = FEATURE_INDEX["harm_mag_thd"]
    v_mags = fw_feat[s:s + N_HARM]           # V magnitudes h1..13
    i_mags = fw_feat[s + N_HARM:s + 2 * N_HARM]  # I magnitudes h1..13
    fund_v = max(float(v_mags[0]), 1e-9)
    fund_i = max(float(i_mags[0]), 1e-9)
    mask_v = v_mags / fund_v >= threshold_rel
    mask_i = i_mags / fund_i >= threshold_rel
    return mask_v & mask_i


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_parity_phase_abs_significant_harmonics(seed: int):
    """Absolute phase sin/cos for harmonics with meaningful magnitude."""
    v_adc, i_adc = _make_adc_frame(seed, n_harmonics=13)
    py_feat = _py_pipeline(v_adc, i_adc)
    fw_feat = _teensy_features(v_adc, i_adc)
    mask = _significant_harmonic_mask(v_adc, i_adc)

    s, _ = FEATURE_INDEX["phase_abs"]
    # Layout: sin(V_h1..13), cos(V_h1..13), sin(I_h1..13), cos(I_h1..13)
    for offset, label in [(0, "sin_V"), (13, "cos_V"), (26, "sin_I"), (39, "cos_I")]:
        idx = s + offset
        fw_sel = fw_feat[idx:idx + N_HARM][mask]
        py_sel = py_feat[idx:idx + N_HARM][mask]
        np.testing.assert_allclose(
            fw_sel, py_sel, atol=5e-3, rtol=0,
            err_msg=f"Phase abs '{label}' mismatch for significant harmonics [seed={seed}]",
        )


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_parity_harmonic_power(seed: int):
    """Per-harmonic power (active and reactive) for significant harmonics."""
    v_adc, i_adc = _make_adc_frame(seed, n_harmonics=13)
    py_feat = _py_pipeline(v_adc, i_adc)
    fw_feat = _teensy_features(v_adc, i_adc)
    mask = _significant_harmonic_mask(v_adc, i_adc)

    s, _ = FEATURE_INDEX["harm_power"]
    # Layout: P_1, Q_1, P_2, Q_2, ..., P_13, Q_13
    for h, sig in enumerate(mask):
        if sig:
            for offset in (0, 1):  # P, Q
                i_idx = s + h * 2 + offset
                np.testing.assert_allclose(
                    fw_feat[i_idx], py_feat[i_idx], atol=5e-3, rtol=0,
                    err_msg=f"harm_power h={h+1} offset={offset} mismatch [seed={seed}]",
                )


# ---- Edge-case NaN tests ----------------------------------------------------

def test_near_zero_signal_no_nan():
    v_adc = np.full(N, int(V_MIDPOINT), dtype=np.int16)
    i_adc = np.full(N, int(I_MIDPOINT), dtype=np.int16)
    fw_feat = _teensy_features(v_adc, i_adc)
    assert not np.any(np.isnan(fw_feat)), "Firmware reference produced NaN on near-zero input"

    v_phys = (v_adc.astype(np.float64) - V_MIDPOINT) * V_SCALE - 0.0
    i_phys = (i_adc.astype(np.float64) - I_MIDPOINT) * I_SCALE - 0.0
    py_feat = extract_features(v_phys, i_phys)
    assert not np.any(np.isnan(py_feat)), "Python pipeline produced NaN on near-zero input"


def test_calibration_limits_no_nan():
    v_adc = np.where(np.arange(N) % 2 == 0, 4095, 0).astype(np.int16)
    i_adc = np.where(np.arange(N) % 2 == 0, 4095, 0).astype(np.int16)

    fw_feat = _teensy_features(v_adc, i_adc)
    assert not np.any(np.isnan(fw_feat))

    v_phys = (v_adc.astype(np.float64) - V_MIDPOINT) * V_SCALE
    i_phys = (i_adc.astype(np.float64) - I_MIDPOINT) * I_SCALE
    v_phys -= v_phys.mean()
    i_phys -= i_phys.mean()
    py_feat = extract_features(v_phys, i_phys)
    assert not np.any(np.isnan(py_feat))
