from __future__ import annotations

# Single source of truth for the 298-element feature vector layout (model_4 spec).
# Firmware, Python pipeline, tests, and model code all reference these slices.
# Do not reorder entries without updating firmware dsp.cpp offsets in lockstep.
#
# Vector layout:
#   [0:24]     Time-domain features (12V + 12I)
#   [24:28]    Overall power metrics (S, P, Q, PF)
#   [28:56]    Harmonic magnitudes & THD (13V + 13I + 2 THD)
#   [56:108]   Phase self sin/cos (52)
#   [108:134]  Phase cross sin/cos (26)
#   [134:182]  Phase relative-to-fundamental sin/cos (48)
#   [182:208]  Per-harmonic power P/Q (26)
#   [208:214]  Circular phase statistics (6)
#   [214:256]  DWT voltage (42 = 36 standard + 6 transient-boosters)
#   [256:298]  DWT current (42 = 36 standard + 6 transient-boosters)

FEATURE_INDEX: dict[str, tuple[int, int]] = {
    # Indices 0–23: time-domain statistics, V then I (12 stats each)
    # Order per channel: mean, std, rms, peak, crest_factor, form_factor,
    #                    skewness, kurtosis, ptp, zero_crossings, min, max
    "time_v":       (0,   24),

    # Indices 24–27: overall power metrics
    # [apparent_power, active_power, reactive_power, power_factor]
    "power_metrics": (24, 28),

    # Indices 28–55: harmonic magnitudes h1–13 for V (13) + h1–13 for I (13) + THD-V + THD-I
    "harm_mag_thd": (28,  56),

    # Indices 56–107: absolute phase sin/cos encoding
    # [sin(V_phi_1..13), cos(V_phi_1..13), sin(I_phi_1..13), cos(I_phi_1..13)]
    "phase_abs":    (56,  108),

    # Indices 108–133: cross-channel phase differences V_phi_h - I_phi_h for h=1..13
    # [sin(cross_1..13), cos(cross_1..13)]
    "phase_cross":  (108, 134),

    # Indices 134–181: phase relative to fundamental for h=2..13 (12 harmonics each)
    # [sin(relV_2..13), cos(relV_2..13), sin(relI_2..13), cos(relI_2..13)]
    "phase_rel":    (134, 182),

    # Indices 182–207: per-harmonic active and reactive power
    # [P_1, Q_1, P_2, Q_2, ..., P_13, Q_13]
    "harm_power":   (182, 208),

    # Indices 208–213: circular phase statistics
    # [circmean_V, circstd_V, circmean_I, circstd_I, circmean_cross, circstd_cross]
    "circ_stats":   (208, 214),

    # Indices 214–297: DWT db4 level-5 subband features (symmetric boundary)
    # 42 features for V (36 standard + 6 transient-boosters) + 42 for I
    # Per channel: [cA5, cD5, cD4, cD3, cD2, cD1] x [mean, std, skew, kurt, energy, entropy]
    #              + 6 transient-booster metrics [D1_energy_ratio, D2_energy_ratio, D1_max_abs,
    #                                              D2_max_abs, TKEO_D1_max, TKEO_D1_mean]
    "dwt":          (214, 298),
}

TOTAL_FEATURES = 298

# DWT band sizes under symmetric mode (PyWavelets default, N=500)
# These are used to validate DWT decomposition correctness in tests
DWT_BAND_SIZES = {
    "cA5": 22,
    "cD5": 22,
    "cD4": 37,
    "cD3": 68,
    "cD2": 130,
    "cD1": 253,
}

# Tolerance groups for parity tests (Python float64 vs firmware float32)
PARITY_TOLERANCES: dict[str, float] = {
    "time_v":       1e-3,
    "power_metrics": 1e-2,  # Power calculations more sensitive to precision
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
