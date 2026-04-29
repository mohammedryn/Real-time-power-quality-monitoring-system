#pragma once
#include <math.h>
#include <stdint.h>
#include <string.h>

// DWT decomposition using Daubechies-4 wavelet (db4), level 5.
// Boundary mode: symmetric (half-sample symmetric, SYMH) — matches Python
//   pywt.wavedec(x, 'db4', level=5)   [default mode is 'symmetric']
//
// db4 filter length: 8 taps (4 vanishing moments => 2*4 = 8 coefficients)
// Output band lengths at each level for N=500 input using floor((n+flen-1)/2):
//   Level 1: floor((500+7)/2) = 253  -> cD1
//   Level 2: floor((253+7)/2) = 130  -> cD2
//   Level 3: floor((130+7)/2) = 68   -> cD3
//   Level 4: floor((68+7)/2)  = 37   -> cD4
//   Level 5: floor((37+7)/2)  = 22   -> cD5, cA5
//   Total: 22+22+37+68+130+253 = 532 coefficients per channel.
//
// Band order in output array: [cA5 | cD5 | cD4 | cD3 | cD2 | cD1]
// Sizes:                        22    22    37    68   130   253

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

// Band sizes for N=500 input under symmetric boundary.
// Index 0 = cA5 size, indices 1..5 = cD5..cD1 sizes (detail, finest last).
static constexpr int DWT_BAND_SIZES[DWT_LEVELS + 1] = {22, 22, 37, 68, 130, 253};

// Total DWT output coefficients per channel.
static constexpr int DWT_TOTAL_COEFFS = 532;  // 22+22+37+68+130+253

// Maximum working buffer: 500 floats (level-0 approximation copy).
static constexpr int DWT_WORK_BUF = 500;

// Symmetric (half-sample symmetric, SYMH) boundary index.
// Maps an out-of-bounds index to its mirror position within [0, n).
// Single-level reflection is sufficient for db4 (flen=8) when n >= 8.
static inline int sym_idx(int i, int n) {
    if (i < 0)  return -1 - i;        // left mirror: -1->0, -2->1, ...
    if (i >= n) return 2 * n - 1 - i; // right mirror: n->n-1, n+1->n-2, ...
    return i;
}

// Convolve src (length n) with filter h (length flen) using symmetric
// boundary extension, downsample by 2, write floor((n+flen-1)/2) outputs.
static inline void dwt_level(
    const float* src, int n,
    const float* h, int flen,
    float* dst)
{
    int out_len = (n + flen - 1) / 2;
    for (int i = 0; i < out_len; i++) {
        float acc = 0.0f;
        int src_idx = 2 * i;
        for (int k = 0; k < flen; k++) {
            int idx = sym_idx(src_idx - k, n);
            acc += h[k] * src[idx];
        }
        dst[i] = acc;
    }
}

// Full DWT decomposition.
// Input:  x[500]
// Output: out[532] packed as [cA5(22)|cD5(22)|cD4(37)|cD3(68)|cD2(130)|cD1(253)]
//
// work[] provided by caller (DWT_WORK_BUF floats) — reserved for future use.
static inline void dwt_db4_level5(
    const float* x,
    float* out,
    float* work)
{
    // Approximation buffers: approx[lv] holds cA_lv.
    // Level 1 output is 253 floats — largest approximation buffer needed.
    static float approx[6][260];
    // Detail buffers: detail_buf[lv-1] holds cD_lv.
    // Level 1 detail (cD1) is 253 floats — largest detail buffer needed.
    static float detail_buf[5][260];

    // Level 1: input x (500)  -> approx[1](253), detail_buf[0](253)
    // Level 2: approx[1](253) -> approx[2](130), detail_buf[1](130)
    // Level 3: approx[2](130) -> approx[3](68),  detail_buf[2](68)
    // Level 4: approx[3](68)  -> approx[4](37),  detail_buf[3](37)
    // Level 5: approx[4](37)  -> approx[5](22),  detail_buf[4](22)

    int cur_len = N_SAMPLES_PQ;  // 500
    const float* cur = x;

    for (int lv = 0; lv < DWT_LEVELS; lv++) {
        dwt_level(cur, cur_len, DB4_LO, DWT_FILTER_LEN, approx[lv + 1]);
        dwt_level(cur, cur_len, DB4_HI, DWT_FILTER_LEN, detail_buf[lv]);
        cur_len = (cur_len + DWT_FILTER_LEN - 1) / 2;
        cur     = approx[lv + 1];
    }

    // Pack into out: [cA5(22) | cD5(22) | cD4(37) | cD3(68) | cD2(130) | cD1(253)]
    // cA5 = approx[5], cD5 = detail_buf[4], ..., cD1 = detail_buf[0]
    float* p = out;
    memcpy(p, approx[5],      22  * sizeof(float)); p += 22;
    memcpy(p, detail_buf[4],  22  * sizeof(float)); p += 22;
    memcpy(p, detail_buf[3],  37  * sizeof(float)); p += 37;
    memcpy(p, detail_buf[2],  68  * sizeof(float)); p += 68;
    memcpy(p, detail_buf[1],  130 * sizeof(float)); p += 130;
    memcpy(p, detail_buf[0],  253 * sizeof(float));

    (void)work;
}
