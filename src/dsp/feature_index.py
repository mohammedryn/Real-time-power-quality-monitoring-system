from __future__ import annotations

# Single source of truth for the 282-element feature vector layout.
# Firmware, Python pipeline, tests, and model code all reference these slices.
# Do not reorder entries without updating firmware dsp.cpp offsets in lockstep.

FEATURE_INDEX: dict[str, tuple[int, int]] = {
    # Indices 0–23: time-domain statistics, V then I (12 stats each)
    # Order per channel: mean, std, rms, peak, crest_factor, form_factor,
    #                    skewness, kurtosis, ptp, zero_crossings, min, max
    "time_v":       (0,   24),

    # Indices 24–51: harmonic magnitudes h1–13 for V (13) + h1–13 for I (13) + THD-V + THD-I
    "harm_mag_thd": (24,  52),

    # Indices 52–103: absolute phase sin/cos encoding
    # [sin(V_phi_1..13), cos(V_phi_1..13), sin(I_phi_1..13), cos(I_phi_1..13)]
    "phase_abs":    (52,  104),

    # Indices 104–129: cross-channel phase differences V_phi_h - I_phi_h for h=1..13
    # [sin(cross_1..13), cos(cross_1..13)]
    "phase_cross":  (104, 130),

    # Indices 130–177: phase relative to fundamental for h=2..13 (12 harmonics each)
    # [sin(relV_2..13), cos(relV_2..13), sin(relI_2..13), cos(relI_2..13)]
    "phase_rel":    (130, 178),

    # Indices 178–203: per-harmonic active and reactive power
    # [P_1, Q_1, P_2, Q_2, ..., P_13, Q_13]
    "harm_power":   (178, 204),

    # Indices 204–209: circular phase statistics
    # [circmean_V, circstd_V, circmean_I, circstd_I, circmean_cross, circstd_cross]
    "circ_stats":   (204, 210),

    # Indices 210–281: DWT db4 level-5 subband features (periodization boundary)
    # 36 features for V + 36 for I
    # Per channel: [cA5, cD5, cD4, cD3, cD2, cD1] x [mean, std, skew, kurt, energy, entropy]
    "dwt":          (210, 282),
}

TOTAL_FEATURES = 282

# DWT band sizes under periodization mode (ceil(N/2) at each level, N=500)
DWT_BAND_SIZES = {
    "cA5": 16,
    "cD5": 16,
    "cD4": 32,
    "cD3": 63,
    "cD2": 125,
    "cD1": 250,
}

# Tolerance groups for parity tests (Python float64 vs firmware float32)
PARITY_TOLERANCES: dict[str, float] = {
    "time_v":       1e-3,
    "harm_mag_thd": 1e-3,
    "phase_abs":    1e-3,
    "phase_cross":  1e-3,
    "phase_rel":    1e-3,
    "harm_power":   1e-3,
    "circ_stats":   1e-3,
    "dwt":          1e-2,
}


def slice_of(group: str) -> slice:
    start, stop = FEATURE_INDEX[group]
    return slice(start, stop)
