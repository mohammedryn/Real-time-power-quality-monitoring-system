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

// Count zero-crossings (sign changes between adjacent samples)
static inline float pq_zero_crossings(const float* x, int n) {
    int zc = 0;
    for (int i = 1; i < n; i++) {
        // signbit: 0 for >= 0, nonzero for < 0
        if ((x[i] >= 0.0f) != (x[i - 1] >= 0.0f)) zc++;
    }
    return (float)zc;
}

// Extract 12 time-domain features into out[12].
// Order: mean, std, rms, peak, crest_factor, form_factor,
//        skewness, kurtosis, ptp, zero_crossings, min, max
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
    out[4]  = (rms  > 1e-6f) ? peak / rms  : 0.0f;  // crest factor
    out[5]  = (ma   > 1e-6f) ? rms  / ma   : 0.0f;  // form factor
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

    // Log-energy entropy: -sum(p * log2(p)) where p = coeff[i]^2 / energy
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

// Extract 36 DWT features from one channel into out[36].
// Band order: cA5(16), cD5(16), cD4(32), cD3(63), cD2(125), cD1(250)
static void dwt_channel_features(const float* x, float* out) {
    static float dwt_out[502];
    static float dwt_work[DWT_WORK_BUF];
    dwt_db4_level5(x, dwt_out, dwt_work);

    // Band sizes: [16, 16, 32, 63, 125, 250]
    static constexpr int sizes[6] = {16, 16, 32, 63, 125, 250};
    const float* p = dwt_out;
    float* o = out;
    for (int b = 0; b < 6; b++) {
        dwt_band_stats(p, sizes[b], o);
        p += sizes[b];
        o += 6;
    }
}

// Circular mean: atan2(mean(sin), mean(cos))
static inline float circ_mean(const float* angles, int n) {
    float sc = 0.0f, cc = 0.0f;
    for (int i = 0; i < n; i++) { sc += sinf(angles[i]); cc += cosf(angles[i]); }
    return atan2f(sc / n, cc / n);
}

// Circular std: sqrt(-2 * ln(R)) where R = resultant length
// R=1 -> std=0 (all identical); R->0 -> std->inf (uniformly spread)
// Guard: R < 1e-12 is the maximum-spread numerical edge (log would blow up)
static inline float circ_std(const float* angles, int n) {
    float sc = 0.0f, cc = 0.0f;
    for (int i = 0; i < n; i++) { sc += sinf(angles[i]); cc += cosf(angles[i]); }
    float R = sqrtf((sc / n) * (sc / n) + (cc / n) * (cc / n));
    if (R < 1e-12f) return sqrtf(-2.0f * logf(1e-12f));  // cap at extreme spread
    return sqrtf(-2.0f * logf(R));
}

// ---- Main entry point -------------------------------------------------------

void compute_features(const int16_t* v_raw, const int16_t* i_raw, float* feat) {
    // Working buffers (static to avoid stack pressure on M7)
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

    int idx = 0;

    // 2. Time-domain features [0-23]  (12 V + 12 I)
    time_domain_features(v_phys, N_SAMPLES_PQ, &feat[idx]);       idx += 12;
    time_domain_features(i_phys, N_SAMPLES_PQ, &feat[idx]);       idx += 12;

    // 3. Harmonic magnitudes via Goertzel
    static float v_mag[N_HARMONICS], v_ph[N_HARMONICS];
    static float i_mag[N_HARMONICS], i_ph[N_HARMONICS];
    goertzel_harmonics(v_phys, v_mag, v_ph);
    goertzel_harmonics(i_phys, i_mag, i_ph);

    // Harmonic magnitudes + THD [24-51]  (13 V + 13 I + 2 THD)
    for (int h = 0; h < N_HARMONICS; h++) feat[idx++] = v_mag[h];
    for (int h = 0; h < N_HARMONICS; h++) feat[idx++] = i_mag[h];

    // THD = RSS(h2..h13) / fund  (ratio, not percent)
    float v_rss = 0.0f, i_rss = 0.0f;
    for (int h = 1; h < N_HARMONICS; h++) {
        v_rss += v_mag[h] * v_mag[h];
        i_rss += i_mag[h] * i_mag[h];
    }
    feat[idx++] = (v_mag[0] > 1e-6f) ? sqrtf(v_rss) / v_mag[0] : 0.0f;
    feat[idx++] = (i_mag[0] > 1e-6f) ? sqrtf(i_rss) / i_mag[0] : 0.0f;

    // 4. Absolute phase sin/cos [52-103]
    // [sin(V_phi_1..13), cos(V_phi_1..13), sin(I_phi_1..13), cos(I_phi_1..13)]
    for (int h = 0; h < N_HARMONICS; h++) feat[idx++] = sinf(v_ph[h]);
    for (int h = 0; h < N_HARMONICS; h++) feat[idx++] = cosf(v_ph[h]);
    for (int h = 0; h < N_HARMONICS; h++) feat[idx++] = sinf(i_ph[h]);
    for (int h = 0; h < N_HARMONICS; h++) feat[idx++] = cosf(i_ph[h]);

    // 5. Cross-channel phase differences [104-129]
    // cross[h] = V_phi[h] - I_phi[h]
    // [sin(cross_1..13), cos(cross_1..13)]
    static float cross[N_HARMONICS];
    for (int h = 0; h < N_HARMONICS; h++) cross[h] = v_ph[h] - i_ph[h];
    for (int h = 0; h < N_HARMONICS; h++) feat[idx++] = sinf(cross[h]);
    for (int h = 0; h < N_HARMONICS; h++) feat[idx++] = cosf(cross[h]);

    // 6. Relative-to-fundamental phase [130-177]
    // rel_V[h] = V_phi[h] - V_phi[0]  for h=1..12  (harmonics 2..13)
    // [sin(relV), cos(relV), sin(relI), cos(relI)]
    static float rel_v[N_HARMONICS - 1], rel_i[N_HARMONICS - 1];
    for (int h = 1; h < N_HARMONICS; h++) rel_v[h - 1] = v_ph[h] - v_ph[0];
    for (int h = 1; h < N_HARMONICS; h++) rel_i[h - 1] = i_ph[h] - i_ph[0];
    for (int h = 0; h < N_HARMONICS - 1; h++) feat[idx++] = sinf(rel_v[h]);
    for (int h = 0; h < N_HARMONICS - 1; h++) feat[idx++] = cosf(rel_v[h]);
    for (int h = 0; h < N_HARMONICS - 1; h++) feat[idx++] = sinf(rel_i[h]);
    for (int h = 0; h < N_HARMONICS - 1; h++) feat[idx++] = cosf(rel_i[h]);

    // 7. Per-harmonic power [178-203]
    // [P_1,Q_1, ..., P_13,Q_13]
    for (int h = 0; h < N_HARMONICS; h++) {
        feat[idx++] = v_mag[h] * i_mag[h] * cosf(cross[h]);  // active
        feat[idx++] = v_mag[h] * i_mag[h] * sinf(cross[h]);  // reactive
    }

    // 8. Circular statistics [204-209]
    // [circmean_V, circstd_V, circmean_I, circstd_I, circmean_cross, circstd_cross]
    feat[idx++] = circ_mean(v_ph, N_HARMONICS);
    feat[idx++] = circ_std(v_ph, N_HARMONICS);
    feat[idx++] = circ_mean(i_ph, N_HARMONICS);
    feat[idx++] = circ_std(i_ph, N_HARMONICS);
    feat[idx++] = circ_mean(cross, N_HARMONICS);
    feat[idx++] = circ_std(cross, N_HARMONICS);

    // 9. DWT features [210-281]  (36 V + 36 I)
    dwt_channel_features(v_phys, &feat[idx]);  idx += 36;
    dwt_channel_features(i_phys, &feat[idx]);  idx += 36;

    // Sanity check — compiler will optimise this out in release builds
    (void)idx;  // idx should be 282 here
}
