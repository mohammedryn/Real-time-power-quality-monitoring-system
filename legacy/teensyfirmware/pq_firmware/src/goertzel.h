#pragma once
#include <math.h>
#include <stdint.h>

// Goertzel algorithm for extracting a single DFT bin from a real signal.
//
// For fs=5000, N=500: bin_k = harmonic_order * 5
//   h=1 (50 Hz)  -> bin 5
//   h=13 (650 Hz) -> bin 65
//
// Magnitude is normalised by N/2 to match numpy rfft convention:
//   mags[k] = abs(rfft(x))[k] / (N/2)
// (DC bin is halved again in Python but we never extract bin 0 here.)

struct GoertzelResult {
    float magnitude;
    float phase;     // radians in (-pi, pi]
};

// Compute one DFT bin k from signal x of length n.
static inline GoertzelResult goertzel_bin(const float* x, int n, int k) {
    const float omega = 2.0f * (float)M_PI * k / n;
    const float coeff = 2.0f * cosf(omega);
    const float sin_w = sinf(omega);
    const float cos_w = cosf(omega);

    float s_prev2 = 0.0f;
    float s_prev1 = 0.0f;

    for (int i = 0; i < n; i++) {
        float s = x[i] + coeff * s_prev1 - s_prev2;
        s_prev2 = s_prev1;
        s_prev1 = s;
    }

    // X[k] = s_prev1 - e^{-j*omega} * s_prev2
    float real = s_prev1 - cos_w * s_prev2;
    float imag =           sin_w * s_prev2;

    // Correct the inherent 1-sample phase lag: multiply by e^{+j*omega}
    float real_c = real * cos_w - imag * sin_w;
    float imag_c = real * sin_w + imag * cos_w;

    float mag = sqrtf(real_c * real_c + imag_c * imag_c) / ((float)n / 2.0f);
    float ph  = atan2f(imag_c, real_c);

    return {mag, ph};
}

static constexpr int N_HARMONICS  = 13;
static constexpr int N_SAMPLES_PQ = 500;

// Extract all 13 harmonic magnitudes and phases for one channel.
// bins: k = h*5 for h = 1..13  (50 Hz .. 650 Hz at fs=5000, N=500)
static inline void goertzel_harmonics(
    const float* x,
    float mag_out[N_HARMONICS],
    float phase_out[N_HARMONICS])
{
    for (int h = 1; h <= N_HARMONICS; h++) {
        GoertzelResult r = goertzel_bin(x, N_SAMPLES_PQ, h * 5);
        mag_out[h - 1]   = r.magnitude;
        phase_out[h - 1] = r.phase;
    }
}
