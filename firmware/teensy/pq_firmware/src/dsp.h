#pragma once
#include <stdint.h>

// Full 282-element feature vector extractor.
// Matches Python pipeline: preprocess_frame() -> extract_features()
// Feature index layout: see src/dsp/feature_index.py (canonical source of truth)
//
// Calibration constants must match configs/default.yaml:
//   calibration.v_adc_midpoint, calibration.v_counts_to_volts
//   calibration.i_adc_midpoint, calibration.i_counts_to_amps

static constexpr int N_FEATURES = 282;

// Calibration — must stay in sync with configs/default.yaml
static constexpr float V_MIDPOINT = 2071.0f;
static constexpr float V_SCALE    = 0.579f;    // volts per ADC count
static constexpr float I_MIDPOINT = 2048.0f;
static constexpr float I_SCALE    = 0.030518f; // amps per ADC count

// Compute 282-element float32 feature vector from a raw ADC frame.
// v_raw and i_raw are N_SAMPLES_PQ (500) int16 ADC counts.
// features_out must point to a buffer of at least N_FEATURES floats.
void compute_features(const int16_t* v_raw, const int16_t* i_raw, float* features_out);
