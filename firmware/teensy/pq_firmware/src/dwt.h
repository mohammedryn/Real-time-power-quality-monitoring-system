#pragma once
#include <math.h>
#include <stdint.h>
#include <string.h>

// DWT decomposition using Daubechies-4 wavelet (db4), level 5.
// Boundary mode: periodization (circular wrap) — matches Python
//   pywt.wavedec(x, 'db4', level=5, mode='periodization')
//
// db4 filter length: 8 taps (4 vanishing moments => 2*4 = 8 coefficients)
// Output band lengths at each level for N=500 input (ceil(N/2)):
//   cD1: 250, cD2: 125, cD3: 63, cD4: 32, cD5: 16, cA5: 16
//   Total: 502 coefficients per channel.
//
// Band order in output array: [cA5 | cD5 | cD4 | cD3 | cD2 | cD1]
// Sizes:                        16    16    32    63   125   250

// db4 decomposition low-pass filter (dec_lo), 8 taps.
// Source: PyWavelets Wavelet('db4').dec_lo
static constexpr float DB4_LO[8] = {
    -0.010597401784997278f,
     0.032883011666982945f,
     0.030841381835986965f,
    -0.187034811718881134f,
    -0.027983769416983850f,
     0.630880767929590380f,
     0.714846570552541500f,
     0.230377813308855230f,
};

// db4 decomposition high-pass filter (dec_hi), 8 taps.
// dec_hi[k] = (-1)^k * dec_lo[L-1-k] where L=8
static constexpr float DB4_HI[8] = {
    -0.230377813308855230f,
     0.714846570552541500f,
    -0.630880767929590380f,
    -0.027983769416983850f,
     0.187034811718881134f,
     0.030841381835986965f,
    -0.032883011666982945f,
    -0.010597401784997278f,
};

static constexpr int DWT_FILTER_LEN = 8;
static constexpr int DWT_LEVELS     = 5;

// Band sizes for N=500 input under periodization (ceil at each level).
static constexpr int DWT_BAND_SIZES[DWT_LEVELS + 1] = {16, 16, 32, 63, 125, 250};
// Index 0 = cA5 size, indices 1..5 = cD5..cD1 sizes (detail, finest last).

// Maximum working buffer needed: 500 floats (level-0 approximation copy).
static constexpr int DWT_WORK_BUF = 500;

// Convolve src (length n) with filter h (length flen) with circular (periodic)
// boundary, downsample by 2, write ceil(n/2) outputs into dst.
static inline void dwt_level(
    const float* src, int n,
    const float* h, int flen,
    float* dst)
{
    int out_len = (n + 1) / 2;  // ceil(n/2)
    for (int i = 0; i < out_len; i++) {
        float acc = 0.0f;
        int src_idx = 2 * i;
        for (int k = 0; k < flen; k++) {
            // Circular wrap: index modulo n (positive modulo)
            int idx = ((src_idx - k) % n + n) % n;
            acc += h[k] * src[idx];
        }
        dst[i] = acc;
    }
}

// Full DWT decomposition.
// Input:  x[500]
// Output: out[502] packed as [cA5(16) | cD5(16) | cD4(32) | cD3(63) | cD2(125) | cD1(250)]
//
// work[] must be at least DWT_WORK_BUF (500) floats — caller provides it to
// avoid stack pressure inside this function.
static inline void dwt_db4_level5(
    const float* x,
    float* out,
    float* work)
{
    // We need approx buffers for levels 1..5 and detail buffers to write into out.
    // Level sizes: 500 -> 250 -> 125 -> 63 -> 32 -> 16
    static float approx[6][250];   // approx[0] unused; approx[i] = cA_i, max 250
    static float detail_buf[5][250]; // temp detail at each level, max 250

    // Level 1: input x (500) -> approx[1](250), detail_buf[0](250)
    // Level 2: approx[1](250) -> approx[2](125), detail_buf[1](125)
    // ...
    // Level 5: approx[4](32) -> approx[5](16), detail_buf[4](16)

    int cur_len = N_SAMPLES_PQ;  // 500
    const float* cur = x;

    for (int lv = 0; lv < DWT_LEVELS; lv++) {
        int out_len = (cur_len + 1) / 2;
        dwt_level(cur, cur_len, DB4_LO, DWT_FILTER_LEN, approx[lv + 1]);
        dwt_level(cur, cur_len, DB4_HI, DWT_FILTER_LEN, detail_buf[lv]);
        cur     = approx[lv + 1];
        cur_len = out_len;
    }

    // Pack into out: [cA5 | cD5 | cD4 | cD3 | cD2 | cD1]
    // cA5 = approx[5], size 16
    // cD5 = detail_buf[4], size 16
    // cD4 = detail_buf[3], size 32
    // cD3 = detail_buf[2], size 63
    // cD2 = detail_buf[1], size 125
    // cD1 = detail_buf[0], size 250
    float* p = out;
    memcpy(p, approx[5],      16  * sizeof(float)); p += 16;
    memcpy(p, detail_buf[4],  16  * sizeof(float)); p += 16;
    memcpy(p, detail_buf[3],  32  * sizeof(float)); p += 32;
    memcpy(p, detail_buf[2],  63  * sizeof(float)); p += 63;
    memcpy(p, detail_buf[1],  125 * sizeof(float)); p += 125;
    memcpy(p, detail_buf[0],  250 * sizeof(float));

    (void)work;  // provided by caller for future in-place variant
}
