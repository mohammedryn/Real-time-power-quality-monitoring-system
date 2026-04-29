import numpy as np
import pytest
from src.dsp.features import extract_features


def test_feature_vector_length():
    """
    Verify that extract_features generates exactly 298-element vectors.
    Layout [298 elements total]:
        [0:24]   Time-domain features (12V + 12I)
        [24:28]  Overall power metrics (S, P, Q, PF)
        [28:56]  Harmonic magnitudes & THD (13V + 13I + THD_V + THD_I)
        [56:108] Phase self sin/cos (52)
        [108:134] Phase cross sin/cos (26)
        [134:182] Phase relative-to-fundamental sin/cos (48)
        [182:208] Per-harmonic power P/Q (26)
        [208:214] Circular phase statistics (6)
        [214:256] DWT voltage (42 = 36 standard + 6 transient-boosters)
        [256:298] DWT current (42 = 36 standard + 6 transient-boosters)
    """
    v = np.random.randn(500) * 325.0
    i = np.random.randn(500) * 10.0
    features = extract_features(v, i)
    assert features.shape == (298,), f"Expected (298,), got {features.shape}"


def test_feature_vector_no_nan():
    """Verify no NaN values in the feature vector."""
    v = np.random.randn(500) * 325.0
    i = np.random.randn(500) * 10.0
    features = extract_features(v, i)
    assert not np.isnan(features).any(), "Feature vector contains NaN values"


def test_feature_vector_dtype():
    """Verify feature vector is float32 (model inference requirement)."""
    v = np.random.randn(500) * 325.0
    i = np.random.randn(500) * 10.0
    features = extract_features(v, i)
    assert features.dtype == np.float32, f"Expected float32, got {features.dtype}"


def test_feature_vector_finite():
    """Verify all feature values are finite (no inf)."""
    v = np.random.randn(500) * 325.0
    i = np.random.randn(500) * 10.0
    features = extract_features(v, i)
    assert np.all(np.isfinite(features)), "Feature vector contains infinite values"
