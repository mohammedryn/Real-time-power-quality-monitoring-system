import numpy as np
from scipy.stats import skew, kurtosis, circmean, circstd
import pywt

def get_time_domain_features(sig):
    """Returns 12 exact time-domain statistical features per channel."""
    mean = np.mean(sig)
    std = np.std(sig)
    rms = np.sqrt(np.mean(sig**2))
    peak = np.max(np.abs(sig))
    
    crest_factor = peak / rms if rms > 1e-6 else 0.0
    mean_abs = np.mean(np.abs(sig))
    form_factor = rms / mean_abs if mean_abs > 1e-6 else 0.0
    
    skw = skew(sig)
    krt = kurtosis(sig)
    ptp = np.max(sig) - np.min(sig)
    
    # Optimized Zero crossings using boolean differences
    zc = float(np.count_nonzero(np.diff(np.signbit(sig))))
    s_min = np.min(sig)
    s_max = np.max(sig)
    
    return [
        mean, std, rms, peak, crest_factor, form_factor,
        0.0 if np.isnan(skw) else float(skw),
        0.0 if np.isnan(krt) else float(krt),
        ptp, zc, s_min, s_max
    ]

def get_fft_features(sig, fs=5000, n_samples=500):
    """
    Computes rFFT. Dynamically locates the fundamental bin to prevent
    spectral leakage issues if the grid frequency deviates from exactly 50Hz.
    Calculates THD using all available upper harmonics, but returns features
    only up to the 13th harmonic.
    """
    fft_vals = np.fft.rfft(sig)
    
    # RFFT scaling
    mags = np.abs(fft_vals) / (n_samples / 2.0)
    mags[0] = mags[0] / 2.0  # DC component
    phases = np.angle(fft_vals)
    
    # Dynamically find the fundamental frequency bin (search bins 3 to 8)
    # At 5kHz/500 samples, 50Hz should normally be bin 5.
    search_start = 3
    search_end = 8
    fund_bin = np.argmax(mags[search_start:search_end]) + search_start
    
    h_mags = []
    h_phases = []
    
    # Extract harmonics 1 to 13 
    for h in range(1, 14):
        bin_idx = h * fund_bin
        if bin_idx < len(mags):
            h_mags.append(mags[bin_idx])
            h_phases.append(phases[bin_idx])
        else:
            h_mags.append(0.0)
            h_phases.append(0.0)
            
    # Calculate True THD using the full available harmonic spectrum
    fund_mag = h_mags[0]
    all_harmonic_mags = []
    h = 2
    while (h * fund_bin) < len(mags):
        all_harmonic_mags.append(mags[h * fund_bin])
        h += 1
        
    harmonics_rss = np.sqrt(np.sum(np.array(all_harmonic_mags)**2))
    thd = (harmonics_rss / fund_mag) if fund_mag > 1e-6 else 0.0
    
    return thd, h_mags, h_phases   

def log_energy_entropy(x):
    """
    Computes Log-Energy Entropy safely handling empty arrays.
    """
    energy = x**2
    sum_energy = np.sum(energy)
    if sum_energy == 0.0:
        return 0.0
    p = energy / sum_energy
    p_nonzero = p[p > 0.0]
    
    # Guard against perfectly flat signal decomposition
    if len(p_nonzero) == 0:
        return 0.0
        
    return -np.sum(p_nonzero * np.log2(p_nonzero))

import pywt
import numpy as np
from scipy.stats import skew, kurtosis

def extract_dwt_features(signal, wavelet='db4', level=5):
    """
    Extracts 36 standard wavelet features + 6 Transient-specific booster features.
    Total features returned: 42
    """
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    features = []
    
    # Pre-calculate total energy for the ratio calculations below
    total_energy = np.sum([np.sum(c**2) for c in coeffs]) + 1e-9
    
    # ---------------------------------------------------------
    # 1. Standard Statistical Features (Your Original Code)
    # ---------------------------------------------------------
    for coeff in coeffs:
        if len(coeff) == 0:
            coeff = np.array([0.0])
            
        mean = np.mean(coeff)
        std = np.std(coeff)
        skw = skew(coeff) if len(coeff) > 2 else 0.0
        krt = kurtosis(coeff) if len(coeff) > 2 else 0.0
        energy = np.sum(coeff**2)
        entropy = log_energy_entropy(coeff) # Assuming this is defined elsewhere
        
        features.extend([
            0.0 if np.isnan(mean) else mean,
            0.0 if np.isnan(std) else std,
            0.0 if np.isnan(skw) else skw,
            0.0 if np.isnan(krt) else krt,
            0.0 if np.isnan(energy) else energy,
            0.0 if np.isnan(entropy) else entropy
        ])
        
    # ---------------------------------------------------------
    # 2. Transient Booster Features (NEW)
    # ---------------------------------------------------------
    # In pywt.wavedec, the array order is [cA5, cD5, cD4, cD3, cD2, cD1]
    # We explicitly grab the highest frequency details (cD1 and cD2)
    cD1 = coeffs[-1]
    cD2 = coeffs[-2]
    
    # A. High-Frequency Energy Ratios
    d1_energy_ratio = np.sum(cD1**2) / total_energy
    d2_energy_ratio = np.sum(cD2**2) / total_energy
    
    # B. Maximum Absolute Amplitude
    d1_max_abs = np.max(np.abs(cD1)) if len(cD1) > 0 else 0.0
    d2_max_abs = np.max(np.abs(cD2)) if len(cD2) > 0 else 0.0
    
    # C. Teager-Kaiser Energy Operator (TKEO) on D1
    if len(cD1) >= 3:
        tkeo_d1 = cD1[1:-1]**2 - cD1[:-2] * cD1[2:]
        tkeo_d1_max = np.max(tkeo_d1)
        tkeo_d1_mean = np.mean(tkeo_d1)
    else:
        tkeo_d1_max = 0.0
        tkeo_d1_mean = 0.0
        
    # Safely append the new features to your list
    features.extend([
        0.0 if np.isnan(d1_energy_ratio) else d1_energy_ratio,
        0.0 if np.isnan(d2_energy_ratio) else d2_energy_ratio,
        0.0 if np.isnan(d1_max_abs) else d1_max_abs,
        0.0 if np.isnan(d2_max_abs) else d2_max_abs,
        0.0 if np.isnan(tkeo_d1_max) else tkeo_d1_max,
        0.0 if np.isnan(tkeo_d1_mean) else tkeo_d1_mean
    ])
        
    return features

def extract_features(v_phys, i_phys):
    """
    Constructs the optimized, strictly ordered 286-element machine learning feature vector.
    Vectorized math ensures fast execution during mass dataset generation.
    """
    v_phys = np.array(v_phys, dtype=np.float64)
    i_phys = np.array(i_phys, dtype=np.float64)
    
    # 1. Time Domain (12 + 12 = 24 total)
    time_v = get_time_domain_features(v_phys)
    time_i = get_time_domain_features(i_phys)
    
    # 1b. Overall Power Metrics (4 total)
    v_rms = time_v[2]
    i_rms = time_i[2]
    apparent_power = v_rms * i_rms
    active_power = np.mean(v_phys * i_phys)
    # max() safeguards against floating point errors where active slightly exceeds apparent
    reactive_power = np.sqrt(max(0.0, apparent_power**2 - active_power**2))
    power_factor = active_power / apparent_power if apparent_power > 1e-6 else 0.0
    
    overall_power_metrics = [apparent_power, active_power, reactive_power, power_factor]
    
    # FFT Extraction base math
    v_thd, v_h_mags_list, v_h_phases_list = get_fft_features(v_phys)
    i_thd, i_h_mags_list, i_h_phases_list = get_fft_features(i_phys)
    
    # Convert lists to arrays once for fast vectorization
    v_h_phases = np.array(v_h_phases_list)
    i_h_phases = np.array(i_h_phases_list)
    v_h_mags = np.array(v_h_mags_list)
    i_h_mags = np.array(i_h_mags_list)
    
    # 2. Harmonic Magnitudes & THD (13 + 13 + 2 = 28 total)
    mag_feats = v_h_mags.tolist() + i_h_mags.tolist() + [v_thd, i_thd]
    
    # 3. Absolute Phase Encoding (Vectorized) (13*2 + 13*2 = 52 total)
    phase_self_v = np.concatenate([np.sin(v_h_phases), np.cos(v_h_phases)])
    phase_self_i = np.concatenate([np.sin(i_h_phases), np.cos(i_h_phases)])
    phase_self = np.concatenate([phase_self_v, phase_self_i]).tolist()
    
    # 4. Cross-Channel Phase (Vectorized) (13*2 = 26 total)
    cross_phases = v_h_phases - i_h_phases
    phase_cross = np.concatenate([np.sin(cross_phases), np.cos(cross_phases)]).tolist()
    
    # 5. Relative Phase to Fundamental (Vectorized) (12*2*2 = 48 total)
    v_fund_phase = v_h_phases[0]
    i_fund_phase = i_h_phases[0]
    rel_v_phases = v_h_phases[1:] - v_fund_phase
    rel_i_phases = i_h_phases[1:] - i_fund_phase
    
    phase_rel_fund = np.concatenate([
        np.sin(rel_v_phases), np.cos(rel_v_phases),
        np.sin(rel_i_phases), np.cos(rel_i_phases)
    ]).tolist()
    
    # 6. Harmonic Powers (Vectorized Interleaving) (13*2 = 26 total)
    p_h = v_h_mags * i_h_mags * np.cos(cross_phases)
    q_h = v_h_mags * i_h_mags * np.sin(cross_phases)
    
    power_harm = np.empty((p_h.size + q_h.size,), dtype=p_h.dtype)
    power_harm[0::2] = p_h
    power_harm[1::2] = q_h
    power_harm = power_harm.tolist()
        
    # 7. Circular Phase Statistics (6 total)
    circ_stats = [
        circmean(v_h_phases, high=np.pi, low=-np.pi),
        circstd(v_h_phases, high=np.pi, low=-np.pi),
        circmean(i_h_phases, high=np.pi, low=-np.pi),
        circstd(i_h_phases, high=np.pi, low=-np.pi),
        circmean(cross_phases, high=np.pi, low=-np.pi),
        circstd(cross_phases, high=np.pi, low=-np.pi)
    ]
    circ_stats = [0.0 if np.isnan(x) else float(x) for x in circ_stats]
    
    # 8. Wavelet DWT Subband Features (36 + 36 = 72)
    wave_v = extract_dwt_features(v_phys)
    wave_i = extract_dwt_features(i_phys)
    wavelet = wave_v + wave_i
    
    # === FINAL VECTOR ASSEMBLY ===
    vector = (
        time_v +                 # 12
        time_i +                 # 12
        overall_power_metrics +  # 4 (New Overall Power)
        mag_feats +              # 28
        phase_self +             # 52
        phase_cross +            # 26
        phase_rel_fund +         # 48
        power_harm +             # 26
        circ_stats +             # 6
        wavelet                  # 84 #added 12 more features to wavelet domain, now 42 per channel instead of 30
    )                            # Total: 298
    
    if len(vector) != 298:
        raise ValueError(f"CRITICAL ERROR: Generated vector length was {len(vector)} instead of exactly 298.")
        
    return np.array(vector, dtype=np.float32)