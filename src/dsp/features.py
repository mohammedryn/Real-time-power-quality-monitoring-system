import numpy as np
from scipy.stats import skew, kurtosis, circmean, circstd
from .wavelet_features import extract_dwt_features


def get_fft_features(sig, fs=5000, n_samples=500, fundamental_hz: float = 50.0):
    """
    Computes rFFT with a fixed fundamental frequency.
    
    Args:
        sig: Input signal
        fs: Sampling frequency (default 5000 Hz)
        n_samples: Number of samples (default 500)
        fundamental_hz: Fundamental frequency in Hz. Defaults to 50.0 to match
                   the Teensy firmware harmonic extraction assumptions.
    
    Returns: (thd, h_mags, h_phases)
        thd: Total Harmonic Distortion
        h_mags: Magnitudes of harmonics 1-13
        h_phases: Phases of harmonics 1-13
    """
    fundamental_hz = float(fundamental_hz)
    freq_res = fs / n_samples  # Hz per bin
    bin_per_hz = 1.0 / freq_res  # bins per Hz
    fundamental_bin = fundamental_hz * bin_per_hz  # fractional bin for fundamental
    
    fft_vals = np.fft.rfft(sig)
    
    # RFFT scaling
    mags = np.abs(fft_vals) / (n_samples / 2.0)
    mags[0] = mags[0] / 2.0  # DC component
    phases = np.angle(fft_vals)
    
    h_mags = []
    h_phases = []
    # Extract harmonics 1 to 13 at the detected fundamental
    for h in range(1, 14):
        # Harmonic bin (may be fractional if fundamental is not exact)
        bin_idx = int(np.round(h * fundamental_bin))
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


def extract_features(v_phys, i_phys):
    """
    Constructs the exact, strictly ordered 298-element machine learning feature vector 
    matching firmware DSP and model_4 training specifications.
    
    Vector layout [298 elements total]:
        [0:24]      Time-domain features (12 voltage + 12 current)
        [24:28]     Overall power metrics (apparent, active, reactive, power_factor)
        [28:56]     Harmonic magnitudes & THD (13 voltage + 13 current + 2 THD)
        [56:108]    Phase self (sin/cos of 13 harmonics per channel, 52 total)
        [108:134]   Phase cross (sin/cos of V-I phase differences, 26 total)
        [134:182]   Phase relative-to-fundamental (sin/cos, 48 total)
        [182:208]   Per-harmonic power (active + reactive, 26 total)
        [208:214]   Circular phase statistics (6 total)
        [214:256]   DWT voltage features (36 standard + 6 transient-boosters = 42)
        [256:298]   DWT current features (36 standard + 6 transient-boosters = 42)
    
    Returns: A flat numpy array of float32, length exactly 298.
    """
    v_phys = np.array(v_phys, dtype=np.float64)
    i_phys = np.array(i_phys, dtype=np.float64)
    
    # 1. Time Domain Features (12 + 12 = 24 total) [0:24]
    time_v = get_time_domain_features(v_phys)
    time_i = get_time_domain_features(i_phys)
    
    # FFT Extraction base math
    v_thd, v_h_mags, v_h_phases = get_fft_features(v_phys)
    i_thd, i_h_mags, i_h_phases = get_fft_features(i_phys)
    
    # 2. Overall Power Metrics (4 total) [24:28]
    # Apparent power: product of RMS magnitudes
    v_rms = time_v[2]  # RMS voltage is at index 2 of time-domain features
    i_rms = time_i[2]  # RMS current is at index 2 of time-domain features
    apparent_power = float(v_rms * i_rms)
    # Active power: mean of instantaneous power v(t)*i(t)
    active_power = float(np.mean(v_phys * i_phys))
    # Reactive power: sqrt(S^2 - P^2), bounded at 0 for numerical stability
    reactive_power = float(np.sqrt(max(0.0, apparent_power**2 - active_power**2)))
    # Power factor: ratio of active to apparent power
    power_factor = float(active_power / apparent_power) if apparent_power > 1e-6 else 0.0
    
    power_metrics = [apparent_power, active_power, reactive_power, power_factor]
    
    # 3. Harmonic Magnitudes & THD (13 + 13 + 2 = 28 total) [28:56]
    mag_feats = v_h_mags + i_h_mags + [v_thd, i_thd]
    
    # 4. Absolute Phase Encoding (13*2 + 13*2 = 52 total) [56:108]
    # sin/cos wrap boundaries to prevent neural net discontinuity at phase=±π
    phase_self_v = [np.sin(p) for p in v_h_phases] + [np.cos(p) for p in v_h_phases]
    phase_self_i = [np.sin(p) for p in i_h_phases] + [np.cos(p) for p in i_h_phases]
    phase_self = phase_self_v + phase_self_i
    
    # 5. Cross-Channel Phase (13*2 = 26 total) [108:134]
    cross_phases = [v - i for v, i in zip(v_h_phases, i_h_phases)]
    phase_cross = [np.sin(p) for p in cross_phases] + [np.cos(p) for p in cross_phases]
    
    # 6. Relative Phase to Fundamental (12*2*2 = 48 total) [134:182]
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
    
    # 7. Harmonic Powers (13*2 = 26 total) [182:208]
    # Interleaved: [P1, Q1, P2, Q2, ..., P13, Q13]
    power_harm = []
    for h in range(13):
        p_h = v_h_mags[h] * i_h_mags[h] * np.cos(cross_phases[h])  # Active
        q_h = v_h_mags[h] * i_h_mags[h] * np.sin(cross_phases[h])  # Reactive
        power_harm.extend([p_h, q_h])
        
    # 8. Circular Phase Statistics (6 total) [208:214]
    circ_stats = [
        circmean(v_h_phases, high=np.pi, low=-np.pi),
        circstd(v_h_phases, high=np.pi, low=-np.pi),
        circmean(i_h_phases, high=np.pi, low=-np.pi),
        circstd(i_h_phases, high=np.pi, low=-np.pi),
        circmean(cross_phases, high=np.pi, low=-np.pi),
        circstd(cross_phases, high=np.pi, low=-np.pi)
    ]
    circ_stats = [0.0 if np.isnan(x) else float(x) for x in circ_stats]
    
    # 9. Wavelet DWT Subband Features (42 + 42 = 84) [214:298]
    # Each channel returns 42 features: 36 standard (6 bands × 6 stats) + 6 transient-boosters
    wave_v = extract_dwt_features(v_phys)
    wave_i = extract_dwt_features(i_phys)
    wavelet = wave_v + wave_i
    
    # === FINAL VECTOR ASSEMBLY ===
    vector = (
        time_v +                # 12 (voltage time-domain)
        time_i +                # 12 (current time-domain)
        power_metrics +         # 4 (NEW: overall power metrics)
        mag_feats +             # 28 (harmonic magnitudes + THD)
        phase_self +            # 52 (absolute phase sin/cos)
        phase_cross +           # 26 (cross-channel phase sin/cos)
        phase_rel_fund +        # 48 (relative-to-fundamental sin/cos)
        power_harm +            # 26 (per-harmonic active/reactive power)
        circ_stats +            # 6 (circular phase statistics)
        wavelet                 # 84 (42 per channel: 36 standard + 6 transient-boosters)
    )                           # Total: 298 elements
    
    if len(vector) != 298:
        raise ValueError(
            f"CRITICAL ERROR: Generated feature vector length was {len(vector)} "
            f"instead of exactly 298. Expected [24 time-domain + 4 power + 28 mag + "
            f"52 self-phase + 26 cross-phase + 48 rel-phase + 26 power-harm + 6 circ + 84 wavelet]."
        )
        
    return np.array(vector, dtype=np.float32)
