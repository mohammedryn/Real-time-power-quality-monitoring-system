#pragma once
#include <stdint.h>

// 298-element feature vector layout (matches Jafed's model_4 training DSP):
//   [0:12]   time_v        – 12 time-domain stats for voltage
//   [12:24]  time_i        – 12 time-domain stats for current
//   [24:28]  power_metrics – apparent, active, reactive, power_factor
//   [28:56]  mag_feats     – 13 V mags + 13 I mags + THD_V + THD_I
//   [56:108] phase_self    – sin/cos of all 13 V and I phases
//   [108:134] phase_cross  – sin/cos of per-harmonic V-I phase diff
//   [134:182] phase_rel    – sin/cos of phases relative to fundamental
//   [182:208] power_harm   – interleaved active/reactive per harmonic
//   [208:214] circ_stats   – circular mean/std of V, I, cross phases
//   [214:256] dwt_v        – 36 standard + 6 transient-booster wavelet stats for V
//   [256:298] dwt_i        – 36 standard + 6 transient-booster wavelet stats for I
//
// X_mag   = feat[28:56]   (28 floats)
// X_phase = feat[0:28] ++ feat[56:214] ++ feat[214:298]  (270 floats)

static constexpr int N_FEATURES     = 298;
static constexpr int N_WAVE_SAMPLES = 500;

// Calibration — must stay in sync with configs/default.yaml
static constexpr float V_MIDPOINT = 2071.0f;
static constexpr float V_SCALE    = 0.579f;    // volts per ADC count
static constexpr float I_MIDPOINT = 2048.0f;
static constexpr float I_SCALE    = 0.030518f; // amps per ADC count

// Compute the model-ready frame from raw ADC data.
// Outputs:
//   feat_out    – 298 float32 features in Jafed's canonical order
//   v_norm_out  – 500 peak-normalised voltage samples  [-1, 1]
//   i_norm_out  – 500 peak-normalised current samples  [-1, 1]
void compute_model4_frame(
    const int16_t* v_raw, const int16_t* i_raw,
    float* feat_out,
    float* v_norm_out,
    float* i_norm_out);
