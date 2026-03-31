import numpy as np
from scipy.stats import skew, kurtosis, circmean, circstd
from .wavelet_features import extract_dwt_features

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
    
    # Count sign changes (Zero crossings)
    zc = float(len(np.where(np.diff(np.signbit(sig)))[0]))
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
    Computes rFFT targeting a 50Hz fundamental on a 500-sample array at 5kHz.
    Frequency Resolution = fs / N = 10Hz per bin.
    Harmonic N = bin N*5 (Since fundamental is 50Hz -> bin 5).
    """
    fft_vals = np.fft.rfft(sig)
    
    # RFFT scaling
    mags = np.abs(fft_vals) / (n_samples / 2.0)
    mags[0] = mags[0] / 2.0  # DC component
    phases = np.angle(fft_vals)
    
    h_mags = []
    h_phases = []
    # Extract harmonics 1 to 13 (50 Hz up to 650 Hz)
    for h in range(1, 14):
        bin_idx = h * 5
        if bin_idx < len(mags):
            h_mags.append(mags[bin_idx])
            h_phases.append(phases[bin_idx])
        else:
            h_mags.append(0.0)
            h_phases.append(0.0)
            
    # Calculate Total Harmonic Distortion (THD) 
    # Ratio of RSS of all upper harmonics to the fundamental magnitude
    fund_mag = h_mags[0]
    harmonics_rss = np.sqrt(np.sum([m**2 for m in h_mags[1:]]))
    thd = (harmonics_rss / fund_mag) if fund_mag > 1e-6 else 0.0
    
    return thd, h_mags, h_phases
    
def extract_features(v_phys, i_phys):
    """
    Constructs the exact, strictly ordered 282-element machine learning feature vector 
    as defined per the AI-Ready methodology in the report.
    Returns: A flat numpy array of float32, length exactly 282.
    """
    v_phys = np.array(v_phys, dtype=np.float64)
    i_phys = np.array(i_phys, dtype=np.float64)
    
    # 1. Time Domain (12 + 12 = 24 total)
    time_v = get_time_domain_features(v_phys)
    time_i = get_time_domain_features(i_phys)
    
    # FFT Extraction base math
    v_thd, v_h_mags, v_h_phases = get_fft_features(v_phys)
    i_thd, i_h_mags, i_h_phases = get_fft_features(i_phys)
    
    # 2. Harmonic Magnitudes & THD (13 + 13 + 2 = 28 total)
    mag_feats = v_h_mags + i_h_mags + [v_thd, i_thd]
    
    # 3. Absolute Phase Encoding (13*2 + 13*2 = 52 total)
    # sin/cos limits wrap boundaries to prevent neural net discontinuity
    phase_self_v = [np.sin(p) for p in v_h_phases] + [np.cos(p) for p in v_h_phases]
    phase_self_i = [np.sin(p) for p in i_h_phases] + [np.cos(p) for p in i_h_phases]
    phase_self = phase_self_v + phase_self_i
    
    # 4. Cross-Channel Phase (13*2 = 26 total)
    cross_phases = [v - i for v, i in zip(v_h_phases, i_h_phases)]
    phase_cross = [np.sin(p) for p in cross_phases] + [np.cos(p) for p in cross_phases]
    
    # 5. Relative Phase to Fundamental (12*2*2 = 48 total)
    v_fund_phase = v_h_phases[0]
    i_fund_phase = i_h_phases[0]
    rel_v_phases = [p - v_fund_phase for p in v_h_phases[1:]] # h=2..13
    rel_i_phases = [p - i_fund_phase for p in i_h_phases[1:]]
    
    phase_rel_fund = (
        [np.sin(p) for p in rel_v_phases] +
        [np.cos(p) for p in rel_v_phases] +
        [np.sin(p) for p in rel_i_phases] +
        [np.cos(p) for p in rel_i_phases]
    )
    
    # 6. Harmonic Powers (13*2 = 26 total)
    power_harm = []
    for h in range(13):
        p_h = v_h_mags[h] * i_h_mags[h] * np.cos(cross_phases[h])  # Active
        q_h = v_h_mags[h] * i_h_mags[h] * np.sin(cross_phases[h])  # Reactive
        power_harm.extend([p_h, q_h])
        
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
        time_v +            # 24
        time_i +            # ^ (bundled above)
        mag_feats +         # 28
        phase_self +        # 52
        phase_cross +       # 26
        phase_rel_fund +    # 48
        power_harm +        # 26
        circ_stats +        # 6
        wavelet             # 72
    )                       # Total: 282
    
    if len(vector) != 282:
        raise ValueError(f"CRITICAL ERROR: Generated vector length was {len(vector)} instead of exactly 282.")
        
    return np.array(vector, dtype=np.float32)
