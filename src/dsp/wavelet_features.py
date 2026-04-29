import numpy as np
import pywt
from scipy.stats import skew, kurtosis

# For N=500, db4, level=5 and symmetric mode, wavedec returns:
# [cA5(22), cD5(22), cD4(37), cD3(68), cD2(130), cD1(253)]
EXPECTED_DB4_L5_BAND_SIZES_500 = (22, 22, 37, 68, 130, 253)

def log_energy_entropy(x):
    """
    Computes Log-Energy Entropy: sum(p * log2(p)) where p is the squared ratio of coefficients.
    """
    energy = x**2
    sum_energy = np.sum(energy)
    if sum_energy == 0.0:
        return 0.0
    p = energy / sum_energy
    # Remove zeros to avoid taking log of zero
    p = p[p > 0.0]
    return -np.sum(p * np.log2(p))

def extract_dwt_features(signal, wavelet='db4', level=5):
    """
    Extracts exactly 42 wavelet features from a 1D signal.
    
    Returns:
        [0:36]  Standard statistical features: 6 per band × 6 bands in order cA5,cD5,cD4,cD3,cD2,cD1
                - Per band: mean, std, skewness, kurtosis, energy, entropy
        [36:42] Transient-booster features (high-frequency sensitivity):
                - D1 energy ratio (cD1 energy / total energy)
                - D2 energy ratio (cD2 energy / total energy)
                - D1 maximum absolute amplitude
                - D2 maximum absolute amplitude
                - TKEO-D1 maximum (Teager-Kaiser Energy Operator)
                - TKEO-D1 mean

    DWT band sizes used by this project:
        For the runtime frame size (N=500), db4 level-5 in symmetric mode,
        the coefficient lengths are fixed at:
            cA5=22, cD5=22, cD4=37, cD3=68, cD2=130, cD1=253
        (sum = 532 coefficients).
    """
    coeffs = pywt.wavedec(signal, wavelet, level=level, mode='symmetric')

    # Guardrail to keep firmware/Python parity assumptions explicit.
    if wavelet == 'db4' and level == 5 and len(signal) == 500:
        got_sizes = tuple(len(c) for c in coeffs)
        if got_sizes != EXPECTED_DB4_L5_BAND_SIZES_500:
            raise ValueError(
                f"Unexpected db4 level-5 band sizes for N=500: {got_sizes}; "
                f"expected {EXPECTED_DB4_L5_BAND_SIZES_500}"
            )

    features = []
    
    # =========================================================================
    # SECTION 1: Standard Statistical Features (36 total)
    # =========================================================================
    for coeff in coeffs:
        if len(coeff) == 0:
            coeff = np.array([0.0])
            
        mean = np.mean(coeff)
        std = np.std(coeff)
        skw = skew(coeff) if len(coeff) > 2 else 0.0
        krt = kurtosis(coeff) if len(coeff) > 2 else 0.0
        energy = np.sum(coeff**2)
        entropy = log_energy_entropy(coeff)
        
        # Guard against zero-division NaN edge cases
        features.extend([
            0.0 if np.isnan(mean) else mean,
            0.0 if np.isnan(std) else std,
            0.0 if np.isnan(skw) else skw,
            0.0 if np.isnan(krt) else krt,
            0.0 if np.isnan(energy) else energy,
            0.0 if np.isnan(entropy) else entropy
        ])
    
    # =========================================================================
    # SECTION 2: Transient-Booster Features (6 total)
    # =========================================================================
    # Coefficient layout: [cA5, cD5, cD4, cD3, cD2, cD1]
    # cD1 (highest frequency) is most sensitive to transients/discontinuities
    cD1 = coeffs[-1]  # Last element is D1
    cD2 = coeffs[-2]  # Second-to-last is D2
    
    # Calculate total energy across all coefficients for normalization
    total_energy = np.sum([np.sum(c**2) for c in coeffs]) + 1e-9
    
    # A. Energy ratios (normalized by total energy)
    d1_energy = np.sum(cD1**2) if len(cD1) > 0 else 0.0
    d2_energy = np.sum(cD2**2) if len(cD2) > 0 else 0.0
    d1_energy_ratio = float(d1_energy / total_energy)
    d2_energy_ratio = float(d2_energy / total_energy)
    
    # B. Maximum absolute amplitudes
    d1_max_abs = float(np.max(np.abs(cD1))) if len(cD1) > 0 else 0.0
    d2_max_abs = float(np.max(np.abs(cD2))) if len(cD2) > 0 else 0.0
    
    # C. Teager-Kaiser Energy Operator (TKEO) on D1
    # TKEO[k] = cD1[k]^2 - cD1[k-1]*cD1[k+1]  for k=1..N-2
    # Sensitive to sudden amplitude changes and discontinuities
    if len(cD1) >= 3:
        tkeo_values = cD1[1:-1]**2 - cD1[:-2] * cD1[2:]
        tkeo_max = float(np.max(tkeo_values))
        tkeo_mean = float(np.mean(tkeo_values))
    else:
        tkeo_max = 0.0
        tkeo_mean = 0.0
    
    # Append transient-booster features
    transient_boosters = [
        0.0 if np.isnan(d1_energy_ratio) else d1_energy_ratio,
        0.0 if np.isnan(d2_energy_ratio) else d2_energy_ratio,
        0.0 if np.isnan(d1_max_abs) else d1_max_abs,
        0.0 if np.isnan(d2_max_abs) else d2_max_abs,
        0.0 if np.isnan(tkeo_max) else tkeo_max,
        0.0 if np.isnan(tkeo_mean) else tkeo_mean,
    ]
    features.extend(transient_boosters)
    
    return features
