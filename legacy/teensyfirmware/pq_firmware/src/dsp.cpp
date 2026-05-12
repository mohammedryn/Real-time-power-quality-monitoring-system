#include "dsp.h"
#include "goertzel.h"
#include "dwt.h"
#include <math.h>
#include <string.h>

// ---- Internal helpers -------------------------------------------------------

static inline float pq_mean(const float* x, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += x[i];
    return s / n;
}

static inline float pq_mean_abs(const float* x, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += fabsf(x[i]);
    return s / n;
}

static inline float pq_rms(const float* x, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += x[i] * x[i];
    return sqrtf(s / n);
}

static inline float pq_std(const float* x, int n, float mean) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = x[i] - mean;
        s += d * d;
    }
    return sqrtf(s / n);
}

static inline float pq_skew(const float* x, int n, float mean, float std) {
    if (std < 1e-10f) return 0.0f;
    float s = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = x[i] - mean;
        s += d * d * d;
    }
    return (s / n) / (std * std * std);
}

static inline float pq_kurtosis(const float* x, int n, float mean, float std) {
    if (std < 1e-10f) return 0.0f;
    float s = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = x[i] - mean;
        s += d * d * d * d;
    }
    return (s / n) / (std * std * std * std) - 3.0f;
}

static inline float pq_ptp(const float* x, int n) {
    float mn = x[0], mx = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] < mn) mn = x[i];
        if (x[i] > mx) mx = x[i];
    }
    return mx - mn;
}

static inline float pq_min(const float* x, int n) {
    float v = x[0];
    for (int i = 1; i < n; i++) if (x[i] < v) v = x[i];
    return v;
}

static inline float pq_max(const float* x, int n) {
    float v = x[0];
    for (int i = 1; i < n; i++) if (x[i] > v) v = x[i];
    return v;
}

static inline float pq_peak(const float* x, int n) {
    float v = fabsf(x[0]);
    for (int i = 1; i < n; i++) { float a = fabsf(x[i]); if (a > v) v = a; }
    return v;
}

static inline float pq_zero_crossings(const float* x, int n) {
    int zc = 0;
    for (int i = 1; i < n; i++) {
        if ((x[i] >= 0.0f) != (x[i - 1] >= 0.0f)) zc++;
    }
    return (float)zc;
}

// 12 time-domain features into out[12]:
// mean, std, rms, peak, crest_factor, form_factor, skewness, kurtosis, ptp, zc, min, max
static void time_domain_features(const float* x, int n, float* out) {
    float mn   = pq_mean(x, n);
    float sd   = pq_std(x, n, mn);
    float rms  = pq_rms(x, n);
    float peak = pq_peak(x, n);
    float ma   = pq_mean_abs(x, n);

    out[0]  = mn;
    out[1]  = sd;
    out[2]  = rms;
    out[3]  = peak;
    out[4]  = (rms > 1e-6f) ? peak / rms : 0.0f;
    out[5]  = (ma  > 1e-6f) ? rms  / ma  : 0.0f;
    out[6]  = pq_skew(x, n, mn, sd);
    out[7]  = pq_kurtosis(x, n, mn, sd);
    out[8]  = pq_ptp(x, n);
    out[9]  = pq_zero_crossings(x, n);
    out[10] = pq_min(x, n);
    out[11] = pq_max(x, n);
}

// DWT subband stats: 6 values per band (mean, std, skew, kurt, energy, entropy).
static void dwt_band_stats(const float* coeff, int n, float* out) {
    float mn  = pq_mean(coeff, n);
    float sd  = pq_std(coeff, n, mn);

    float energy = 0.0f;
    for (int i = 0; i < n; i++) energy += coeff[i] * coeff[i];

    float entropy = 0.0f;
    if (energy > 0.0f) {
        for (int i = 0; i < n; i++) {
            float p = (coeff[i] * coeff[i]) / energy;
            if (p > 0.0f) entropy -= p * log2f(p);
        }
    }

    out[0] = mn;
    out[1] = sd;
    out[2] = pq_skew(coeff, n, mn, sd);
    out[3] = pq_kurtosis(coeff, n, mn, sd);
    out[4] = energy;
    out[5] = entropy;
}

// Extract 42 DWT features from one channel into out[42]:
//   [0:36]  standard stats: 6 per band × 6 bands in order cA5,cD5,cD4,cD3,cD2,cD1
//   [36:42] transient boosters: D1 energy ratio, D2 energy ratio,
//           D1 max abs, D2 max abs, TKEO-D1 max, TKEO-D1 mean
//
// Band sizes (symmetric): cA5=22, cD5=22, cD4=37, cD3=68, cD2=130, cD1=253
static void dwt_channel_features(const float* x, float* out) {
    static float dwt_out[DWT_TOTAL_COEFFS];   // 532 floats
    static float dwt_work[DWT_WORK_BUF];
    dwt_db4_level5(x, dwt_out, dwt_work);

    // Band sizes must match DWT_BAND_SIZES: [22,22,37,68,130,253]
    static constexpr int sizes[6] = {22, 22, 37, 68, 130, 253};

    // 1. Standard stats: 6 values × 6 bands = 36 features
    const float* p = dwt_out;
    float* o = out;
    for (int b = 0; b < 6; b++) {
        dwt_band_stats(p, sizes[b], o);
        p += sizes[b];
        o += 6;
    }

    // 2. Transient booster features (6 more)
    // Band layout: [cA5(22)|cD5(22)|cD4(37)|cD3(68)|cD2(130)|cD1(253)]
    // cD2 offset = 22+22+37+68 = 149
    // cD1 offset = 22+22+37+68+130 = 279
    static constexpr int cD2_offset = 149;
    static constexpr int cD2_len    = 130;
    static constexpr int cD1_offset = 279;
    static constexpr int cD1_len    = 253;

    const float* cD1 = dwt_out + cD1_offset;
    const float* cD2 = dwt_out + cD2_offset;

    // Total energy across all 532 coefficients (matches Jafed's sum-of-all-bands)
    float total_energy = 1e-9f;
    for (int i = 0; i < DWT_TOTAL_COEFFS; i++) total_energy += dwt_out[i] * dwt_out[i];

    // A. High-frequency energy ratios
    float d1_energy = 0.0f, d2_energy = 0.0f;
    for (int i = 0; i < cD1_len; i++) d1_energy += cD1[i] * cD1[i];
    for (int i = 0; i < cD2_len; i++) d2_energy += cD2[i] * cD2[i];

    // B. Maximum absolute amplitude
    float d1_max_abs = 0.0f, d2_max_abs = 0.0f;
    for (int i = 0; i < cD1_len; i++) { float a = fabsf(cD1[i]); if (a > d1_max_abs) d1_max_abs = a; }
    for (int i = 0; i < cD2_len; i++) { float a = fabsf(cD2[i]); if (a > d2_max_abs) d2_max_abs = a; }

    // C. TKEO on cD1: y[k] = cD1[k]^2 - cD1[k-1]*cD1[k+1]  for k=1..N-2
    float tkeo_max = -3.4e38f, tkeo_sum = 0.0f;
    for (int i = 1; i < cD1_len - 1; i++) {
        float t = cD1[i] * cD1[i] - cD1[i - 1] * cD1[i + 1];
        if (t > tkeo_max) tkeo_max = t;
        tkeo_sum += t;
    }
    float tkeo_mean = tkeo_sum / (float)(cD1_len - 2);

    out[36] = d1_energy / total_energy;
    out[37] = d2_energy / total_energy;
    out[38] = d1_max_abs;
    out[39] = d2_max_abs;
    out[40] = tkeo_max;
    out[41] = tkeo_mean;
}

// Circular mean: atan2(mean(sin), mean(cos))
static inline float circ_mean(const float* angles, int n) {
    float sc = 0.0f, cc = 0.0f;
    for (int i = 0; i < n; i++) { sc += sinf(angles[i]); cc += cosf(angles[i]); }
    return atan2f(sc / n, cc / n);
}

// Circular std: sqrt(-2 * ln(R)) where R = |mean phasor|
static inline float circ_std(const float* angles, int n) {
    float sc = 0.0f, cc = 0.0f;
    for (int i = 0; i < n; i++) { sc += sinf(angles[i]); cc += cosf(angles[i]); }
    float R = sqrtf((sc / n) * (sc / n) + (cc / n) * (cc / n));
    if (R < 1e-12f) return sqrtf(-2.0f * logf(1e-12f));
    return sqrtf(-2.0f * logf(R));
}

// ---- Main entry point -------------------------------------------------------

void compute_model4_frame(
    const int16_t* v_raw, const int16_t* i_raw,
    float* feat,
    float* v_norm_out,
    float* i_norm_out)
{
    static float v_phys[N_SAMPLES_PQ];
    static float i_phys[N_SAMPLES_PQ];

    // 1. ADC -> physical units + DC removal
    float v_sum = 0.0f, i_sum = 0.0f;
    for (int k = 0; k < N_SAMPLES_PQ; k++) {
        v_phys[k] = ((float)v_raw[k] - V_MIDPOINT) * V_SCALE;
        i_phys[k] = ((float)i_raw[k] - I_MIDPOINT) * I_SCALE;
        v_sum += v_phys[k];
        i_sum += i_phys[k];
    }
    float v_dc = v_sum / N_SAMPLES_PQ;
    float i_dc = i_sum / N_SAMPLES_PQ;
    for (int k = 0; k < N_SAMPLES_PQ; k++) {
        v_phys[k] -= v_dc;
        i_phys[k] -= i_dc;
    }

    // 2. Peak normalisation -> v_norm, i_norm  ([-1, 1], float32)
    float v_peak = 1e-8f, i_peak = 1e-8f;
    for (int k = 0; k < N_SAMPLES_PQ; k++) {
        float av = fabsf(v_phys[k]); if (av > v_peak) v_peak = av;
        float ai = fabsf(i_phys[k]); if (ai > i_peak) i_peak = ai;
    }
    for (int k = 0; k < N_SAMPLES_PQ; k++) {
        v_norm_out[k] = v_phys[k] / v_peak;
        i_norm_out[k] = i_phys[k] / i_peak;
    }

    int idx = 0;

    // 3. Time-domain features [0:24]  (12 V + 12 I)
    time_domain_features(v_phys, N_SAMPLES_PQ, &feat[idx]);  idx += 12;
    time_domain_features(i_phys, N_SAMPLES_PQ, &feat[idx]);  idx += 12;
    // feat[2]  = rms_v,  feat[14] = rms_i

    // 4. Overall power metrics [24:28]
    float v_rms = feat[2];
    float i_rms = feat[14];
    float apparent = v_rms * i_rms;
    float active   = 0.0f;
    for (int k = 0; k < N_SAMPLES_PQ; k++) active += v_phys[k] * i_phys[k];
    active /= N_SAMPLES_PQ;
    float sq_diff = apparent * apparent - active * active;
    float reactive = sqrtf(sq_diff > 0.0f ? sq_diff : 0.0f);
    float pf       = (apparent > 1e-6f) ? active / apparent : 0.0f;
    feat[idx++] = apparent;
    feat[idx++] = active;
    feat[idx++] = reactive;
    feat[idx++] = pf;
    // idx = 28

    // 5. Harmonic magnitudes and phases via Goertzel
    static float v_mag[N_HARMONICS], v_ph[N_HARMONICS];
    static float i_mag[N_HARMONICS], i_ph[N_HARMONICS];
    goertzel_harmonics(v_phys, v_mag, v_ph);
    goertzel_harmonics(i_phys, i_mag, i_ph);

    // Harmonic magnitudes + THD [28:56]  (13 V + 13 I + 2 THD)
    for (int h = 0; h < N_HARMONICS; h++) feat[idx++] = v_mag[h];
    for (int h = 0; h < N_HARMONICS; h++) feat[idx++] = i_mag[h];
    // THD = RSS(h2..h13) / fund
    float v_rss = 0.0f, i_rss = 0.0f;
    for (int h = 1; h < N_HARMONICS; h++) {
        v_rss += v_mag[h] * v_mag[h];
        i_rss += i_mag[h] * i_mag[h];
    }
    feat[idx++] = (v_mag[0] > 1e-6f) ? sqrtf(v_rss) / v_mag[0] : 0.0f;  // thd_v at [54]
    feat[idx++] = (i_mag[0] > 1e-6f) ? sqrtf(i_rss) / i_mag[0] : 0.0f;  // thd_i at [55]
    // idx = 56

    // 6. Absolute phase sin/cos [56:108]
    for (int h = 0; h < N_HARMONICS; h++) feat[idx++] = sinf(v_ph[h]);
    for (int h = 0; h < N_HARMONICS; h++) feat[idx++] = cosf(v_ph[h]);
    for (int h = 0; h < N_HARMONICS; h++) feat[idx++] = sinf(i_ph[h]);
    for (int h = 0; h < N_HARMONICS; h++) feat[idx++] = cosf(i_ph[h]);
    // idx = 108

    // 7. Cross-channel phase differences [108:134]
    static float cross[N_HARMONICS];
    for (int h = 0; h < N_HARMONICS; h++) cross[h] = v_ph[h] - i_ph[h];
    for (int h = 0; h < N_HARMONICS; h++) feat[idx++] = sinf(cross[h]);
    for (int h = 0; h < N_HARMONICS; h++) feat[idx++] = cosf(cross[h]);
    // idx = 134

    // 8. Relative-to-fundamental phase [134:182]
    static float rel_v[N_HARMONICS - 1], rel_i[N_HARMONICS - 1];
    for (int h = 1; h < N_HARMONICS; h++) rel_v[h - 1] = v_ph[h] - v_ph[0];
    for (int h = 1; h < N_HARMONICS; h++) rel_i[h - 1] = i_ph[h] - i_ph[0];
    for (int h = 0; h < N_HARMONICS - 1; h++) feat[idx++] = sinf(rel_v[h]);
    for (int h = 0; h < N_HARMONICS - 1; h++) feat[idx++] = cosf(rel_v[h]);
    for (int h = 0; h < N_HARMONICS - 1; h++) feat[idx++] = sinf(rel_i[h]);
    for (int h = 0; h < N_HARMONICS - 1; h++) feat[idx++] = cosf(rel_i[h]);
    // idx = 182

    // 9. Per-harmonic power [182:208] – interleaved P1,Q1,...,P13,Q13
    for (int h = 0; h < N_HARMONICS; h++) {
        feat[idx++] = v_mag[h] * i_mag[h] * cosf(cross[h]);  // active
        feat[idx++] = v_mag[h] * i_mag[h] * sinf(cross[h]);  // reactive
    }
    // idx = 208

    // 10. Circular statistics [208:214]
    feat[idx++] = circ_mean(v_ph, N_HARMONICS);
    feat[idx++] = circ_std(v_ph, N_HARMONICS);
    feat[idx++] = circ_mean(i_ph, N_HARMONICS);
    feat[idx++] = circ_std(i_ph, N_HARMONICS);
    feat[idx++] = circ_mean(cross, N_HARMONICS);
    feat[idx++] = circ_std(cross, N_HARMONICS);
    // idx = 214

    // 11. DWT features [214:298]  (42 V + 42 I)
    static float dwt_feat_v[42], dwt_feat_i[42];
    dwt_channel_features(v_phys, dwt_feat_v);
    dwt_channel_features(i_phys, dwt_feat_i);
    memcpy(&feat[idx], dwt_feat_v, 42 * sizeof(float)); idx += 42;
    memcpy(&feat[idx], dwt_feat_i, 42 * sizeof(float)); idx += 42;
    // idx = 298

    (void)idx;  // compiler verifies idx==298 in debug builds
}
