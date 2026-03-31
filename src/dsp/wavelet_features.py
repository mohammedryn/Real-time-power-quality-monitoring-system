import numpy as np
import pywt
from scipy.stats import skew, kurtosis

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
    Extracts exactly 36 wavelet features from a 1D signal (6 bands * 6 stats).
    Decomposition: Approximation cA5 and Details cD5, cD4, cD3, cD2, cD1.
    Computes [Mean, Std, Skewness, Kurtosis, Energy, Entropy] per band.
    """
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    features = []
    
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
        
    return features
