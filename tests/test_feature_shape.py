import numpy as np
import pytest
from src.dsp.features import extract_features


def test_feature_vector_length():
    v = np.random.randn(500) * 325.0
    i = np.random.randn(500) * 10.0
    features = extract_features(v, i)
    assert features.shape == (282,), f"Expected (282,), got {features.shape}"


def test_feature_vector_no_nan():
    v = np.random.randn(500) * 325.0
    i = np.random.randn(500) * 10.0
    features = extract_features(v, i)
    assert not np.isnan(features).any(), "Feature vector contains NaN values"


def test_feature_vector_dtype():
    v = np.random.randn(500) * 325.0
    i = np.random.randn(500) * 10.0
    features = extract_features(v, i)
    assert features.dtype == np.float32, f"Expected float32, got {features.dtype}"
